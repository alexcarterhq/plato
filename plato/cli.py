"""Command-line interface for verification, measured runs and evidence replay."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
from urllib.request import urlopen

from .worlds import make_world

RELEASE_BASE = "https://github.com/alexcarterhq/plato/releases/download/v0.1.0"
BUNDLE = "plato-observations.tar.gz"
METADATA = {"validation.json", "diagnostic.json", "interface-diagnostic.json", "factual.json", "factual-spec.json", "lock.json",
            "cost.json", "manifest.json", "factual-trials.jsonl", "screen.jsonl"}
SPLITS = {"test", "validation", "robustness", "diagnostics", "metering"}


def _publishable(relative):
    p = PurePosixPath(relative)
    if p.is_absolute() or ".." in p.parts or any(s.startswith(".") for s in p.parts):
        return False
    if p.parts == ("cost.json",):
        return True
    if not p.parts or p.parts[0] not in {"qwen", "pythia"}:
        return False
    sub = p.parts[1:]
    if sub[:1] == ("full",):
        sub = sub[1:]
    return ((len(sub) == 1 and sub[0] in METADATA)
            or (len(sub) == 2 and sub[0] in SPLITS and p.suffix in {".jsonl", ".json"}))


def create_bundle(input_dir, output):
    root, output = Path(input_dir), Path(output)
    files = sorted(p for p in root.rglob("*") if p.is_file() and not p.is_symlink()
                   and _publishable(p.relative_to(root).as_posix()))
    if not any(p.name == "factual.json" for p in files):
        raise ValueError("A measured factual-stage manifest is required for a release")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    with partial.open("wb") as dest, gzip.GzipFile(filename="", fileobj=dest, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w|") as archive:
            for path in files:
                item = tarfile.TarInfo(path.relative_to(root).as_posix())
                item.size = path.stat().st_size
                with path.open("rb") as src:
                    archive.addfile(item, src)
    partial.replace(output)
    with output.open("rb") as file:
        sha = hashlib.file_digest(file, "sha256").hexdigest()
    output.with_name(output.name + ".sha256").write_text(f"{sha}  {output.name}\n")
    return sha


def unpack_bundle(path, root):
    total = 0
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            if not member.isfile() or not _publishable(member.name):
                raise ValueError(f"Unexpected release entry {member.name}")
            total += member.size
            if total > 3_000_000_000:
                raise ValueError("Release expansion exceeds 3 GB")
            target = root / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as out, archive.extractfile(member) as src:
                shutil.copyfileobj(src, out)


def _reproduce(args):
    from .report import generate
    if args.input:
        return generate(args.input, args.out)
    with tempfile.TemporaryDirectory(prefix="plato-replay-") as directory:
        tmp = Path(directory)
        if args.archive:
            bundle = args.archive
        else:
            bundle = tmp / BUNDLE
            with urlopen(f"{RELEASE_BASE}/{BUNDLE}", timeout=60) as src, bundle.open("wb") as out:
                shutil.copyfileobj(src, out)
            with urlopen(f"{RELEASE_BASE}/{BUNDLE}.sha256", timeout=30) as src:
                expected = src.read(1024).decode().split()[0]
            with bundle.open("rb") as file:
                if hashlib.file_digest(file, "sha256").hexdigest() != expected:
                    raise ValueError("Release checksum mismatch")
        inputs = tmp / "data"
        inputs.mkdir()
        unpack_bundle(bundle, inputs)
        return generate(inputs, args.out)


def verify():
    import numpy as np
    from .models import Bridge
    from .trials import decode_packet, encode_packet
    for i in range(32):
        w = make_world("smoke", i)
        assert all(w.destination(0,v) != w.destination(1,v) and w.wrong[v] != v for v in range(8))
        for book in range(2):
            text, stop = w.receiver_text(book)
            assert text[:stop].endswith(" |") and sorted(w.books[book]) == list(range(1,9))
    rng = np.random.default_rng(12)
    x = rng.normal(size=(80,8))
    q, _ = np.linalg.qr(rng.normal(size=(8,8)))
    y = .6 * x @ q + 2.
    model = Bridge.fit(x, y)
    np.testing.assert_allclose(model.apply(x), y, atol=2e-6)
    assert len(encode_packet(y[0])) == 16
    np.testing.assert_allclose(decode_packet(encode_packet(y[0])), y[0], rtol=.004)
    return {"worlds_checked": 32, "bridge_and_serialization": "passed", "scope": "offline instrumentation"}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="plato")
    cmd = parser.add_subparsers(dest="command", required=True)
    cmd.add_parser("verify", help="Check world generation, mapping and packet encoding offline")
    run = cmd.add_parser("stage", help="Execute one stage on local model hardware")
    run.add_argument("stage", choices=("bench","facts","calibrate","validate","test","robustness"))
    run.add_argument("--sender", choices=("qwen","pythia"), default="qwen")
    run.add_argument("--out", type=Path, required=True, help="Directory specific to this pair and mode")
    run.add_argument("--data", type=Path)
    run.add_argument("--quick", action="store_true", help="Use a separate smoke-run directory")
    run.add_argument("--shard", type=int, default=0)
    run.add_argument("--shards", type=int, default=1)
    repro = cmd.add_parser("reproduce", help="Recompute metrics, figure and static replay from recorded evidence")
    repro.add_argument("--quick", action="store_true", help="Replay saved predictions without downloading model weights")
    source = repro.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path)
    source.add_argument("--archive", type=Path)
    repro.add_argument("--out", type=Path, default=Path("release"))
    pack = cmd.add_parser("bundle", help="Package public observations and their manifests")
    pack.add_argument("--input", type=Path, required=True)
    pack.add_argument("--out", type=Path, default=Path(BUNDLE))
    args = parser.parse_args(argv)
    if args.command == "verify":
        print(json.dumps(verify(), indent=2))
    elif args.command == "stage":
        from .pipeline import run_stage
        result = run_stage(args.stage, args.sender, args.out, args.data, args.shard, args.shards, args.quick)
        print(json.dumps(result, indent=2))
    elif args.command == "bundle":
        print(create_bundle(args.input, args.out))
    else:
        try:
            _reproduce(args)
        except OSError as exc:
            parser.exit(1, f"Could not load the measured release: {exc}. Use --input for a downloaded run.\n")
        print((args.out / "index.html").resolve())
