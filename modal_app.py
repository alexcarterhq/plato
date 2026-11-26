"""Credit-limited Modal runner. Start locally with python modal_app.py --help."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import time

from plato.subjects import COUNTERFACT_SHA256

from plato.budget import (BudgetError, CAPS, CORES, Ledger, MICRO, RAM_GIB,
                          SHUTDOWN_SECONDS, STARTUP_SECONDS, envelope, rate)


VOLUME = "plato-transfer-experiments"
MOUNT = Path("/plato-volume")
DATA_URL = "https://rome.baulab.info/data/dsets/counterfact.json"
GPUS = ("L4", "L40S")
POOL = {"bench": "pilot", "facts": "facts", "calibrate": "primary",
        "validate": "primary", "test": "primary", "robustness": "robustness"}
TIMEOUT = {"prepare": 1800, "bench": 600, "facts": 14400, "calibrate": 7200,
           "validate": 7200, "test": 7200, "robustness": 7200}


def claim(token, cutoff, timeout):
    from datetime import timezone
    remaining = (datetime.fromisoformat(cutoff) - datetime.now(timezone.utc)).total_seconds()
    if remaining <= timeout + SHUTDOWN_SECONDS:
        raise BudgetError("Queued work would outlive its verified funding period")
    import modal
    claims = modal.Dict.from_name("plato-transfer-attempts", create_if_missing=True)
    if not claims.put(token, True, skip_if_exists=True):
        raise RuntimeError("Interrupted input. Replay requires a new budget reservation.")


def remote_prepare(token, cutoff, run, sender, data_bytes):
    import modal
    from urllib.request import urlopen
    from huggingface_hub import list_repo_files, snapshot_download
    from plato.config import MODELS

    claim(token, cutoff, TIMEOUT["prepare"])
    volume = modal.Volume.from_name(VOLUME)
    volume.reload()
    root = MOUNT / "runs" / run / sender
    root.mkdir(parents=True, exist_ok=True)
    data_file = root / "counterfact.json"
    if data_bytes is None and not data_file.exists():
        with urlopen(DATA_URL, timeout=120) as source:
            data_bytes = source.read()
    if data_bytes is not None:
        if hashlib.sha256(data_bytes).hexdigest() != COUNTERFACT_SHA256:
            raise ValueError("Input differs from the pinned ROME CounterFact release")
        if data_file.exists() and data_file.read_bytes() != data_bytes:
            raise ValueError("Existing CounterFact data differ for this experiment")
        data_file.write_bytes(data_bytes)
    if hashlib.sha256(data_file.read_bytes()).hexdigest() != COUNTERFACT_SHA256:
        raise ValueError("Cached CounterFact file failed its release checksum")

    for key in (sender, "smol"):
        repo, revision = MODELS[key]
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("Every model requires a full frozen commit SHA")
        files = list_repo_files(repo, revision=revision)
        # Some checkpoint revisions ship legacy PyTorch weights only.
        weights = "*.safetensors" if any(f.endswith(".safetensors") for f in files if "/" not in f) else "pytorch_model*.bin"
        snapshot_download(repo, revision=revision, allow_patterns=[weights, "*.json", "*.txt", "*.model", "*.tiktoken"],
                          ignore_patterns=["*/*"])
    volume.commit()
    return {"data_sha256": hashlib.sha256(data_file.read_bytes()).hexdigest(), "sender": sender}


def remote_stage(token, cutoff, timeout, run, sender, stage, hardware, shard, shards, quick):
    import modal
    import torch
    from plato.pipeline import run_stage

    claim(token, cutoff, timeout)
    volume = modal.Volume.from_name(VOLUME)
    volume.reload()
    root = MOUNT / "runs" / run / sender
    out = root / ("quick" if quick else "full")
    if stage == "bench":
        out = out / "bench" / hardware
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = run_stage(stage, sender, out, data_path=root / "counterfact.json",
                       shard=shard, shards=shards, quick=quick)
    torch.cuda.synchronize()
    report = dict(report)
    report.update(worker_wall_seconds=time.monotonic() - started,
                  requested_gpu=hardware, actual_gpu=torch.cuda.get_device_name(),
                  peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                  shard=shard, shards=shards)
    if stage == "bench":
        worlds, seconds = report["worlds"], report["warm_seconds"]
        if worlds <= 0 or seconds <= 0:
            raise ValueError("Benchmark must time complete controlled worlds")
        report["estimated_warm_usd_per_world"] = float(rate(hardware)) * seconds / worlds
    reportdir = out / "metering"
    reportdir.mkdir(exist_ok=True)
    (reportdir / f"{stage}-{shard}-of-{shards}.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return report


def stage_l4(*args):
    return remote_stage(*args)


def stage_l40s(*args):
    return remote_stage(*args)


def make_backend(timeout=TIMEOUT["test"]):
    import modal

    app = modal.App("plato-transfer", tags={"project": "plato-transfer"})
    volume = modal.Volume.from_name(VOLUME, create_if_missing=True)
    image = (modal.Image.debian_slim(python_version="3.11")
             .pip_install("torch==2.6.0", "transformers==4.51.3", "numpy==1.26.4", "safetensors==0.5.3")
             .env({"HF_HOME": str(MOUNT / "hf")})
             .add_local_file(Path(__file__).resolve(), "/root/modal_app.py")
             .add_local_dir(Path(__file__).resolve().parent / "plato", "/root/plato",
                            ignore=["**/._*", "**/__pycache__/**", "._*"]))
    options = dict(image=image, include_source=False, volumes={str(MOUNT): volume},
                   cpu=(CORES, CORES), memory=(RAM_GIB * 1024, RAM_GIB * 1024),
                   retries=0, startup_timeout=STARTUP_SECONDS, max_containers=1)
    # Older installed clients call the same one-input lifecycle `max_inputs`.
    if hasattr(modal, "Workspace"):
        options["single_use_containers"] = True
    else:
        options["max_inputs"] = 1
    prepare = app.function(**options, timeout=TIMEOUT["prepare"])(remote_prepare)
    workers = {gpu: app.function(**options, timeout=timeout, gpu=gpu,
                                 env={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})(fn)
               for gpu, fn in (("L4", stage_l4), ("L40S", stage_l40s))}
    return app, prepare, workers


def workspace_check(config):
    import modal
    if hasattr(modal, "Workspace"):
        workspace = modal.Workspace.from_context()
        workspace.hydrate()
        workspace_name = workspace.name
    else:
        # Modal 1.2 uses this same workspace lookup in its profile CLI.
        import asyncio
        from modal.config import _lookup_workspace, config as credentials
        response = asyncio.run(_lookup_workspace(credentials.get("server_url"),
                                                 credentials.get("token_id"), credentials.get("token_secret")))
        workspace_name = response.username
    if workspace_name != config["workspace"]:
        raise BudgetError(f"Authenticated workspace {workspace_name!r} differs from verified workspace")


def attempt(ledger, key, pool, gpu, timeout, submit):
    existing = ledger.completed(key)
    if existing is not None:
        return existing
    token = ledger.reserve(key, pool, gpu, timeout)
    start = time.monotonic()
    print(f"Reserved {key}: {token}", flush=True)
    try:
        result = submit(token)
    except BaseException:
        # A network or controller failure cannot certify that the worker stopped.
        print(f"Attempt {token} remains fully reserved. Verify termination in Modal "
              "and reconcile it before retrying.", flush=True)
        raise
    elapsed = time.monotonic() - start
    # Cold start, queueing and image builds can all be in client elapsed time.
    # Keep conservative estimates until matched with provider usage.
    charged = envelope(gpu, elapsed + SHUTDOWN_SECONDS)
    result = {**result, "client_wall_seconds": elapsed, "estimated_usd": charged / MICRO}
    ledger.finish(token, result, charged)
    return result


def download(ledger, sender, destination, include_states=False):
    import modal
    from modal.volume import FileEntryType
    config = ledger.config()
    workspace_check(config)
    volume = modal.Volume.from_name(VOLUME)
    prefix = PurePosixPath("runs", config["run"], sender)
    dest = Path(destination)
    for entry in volume.iterdir(str(prefix), recursive=True):
        path = PurePosixPath(entry.path.lstrip("/"))
        if entry.type != FileEntryType.FILE or path.name == "counterfact.json":
            continue
        if not include_states and path.name in {"sender-states.npz", "receiver-states.npz"}:
            continue
        relative = path.relative_to(prefix)
        if ".." in relative.parts:
            raise ValueError("Unexpected path in Volume listing")
        local = dest / relative
        local.parent.mkdir(parents=True, exist_ok=True)
        partial = local.with_name(f"{local.name}.partial")
        with partial.open("wb") as handle:
            for chunk in volume.read_file(str(path)):
                handle.write(chunk)
        partial.replace(local)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", default=".plato/ledger.sqlite", help="Shared ledger for this experiment")
    verbs = parser.add_subparsers(dest="command", required=True)
    setup = verbs.add_parser("init", help="Record independently verified billing settings")
    for name in ("credit", "workspace"):
        setup.add_argument("--" + name, required=True)
    setup.add_argument("--net-limit", help="Verified net spend limit; omit when it is unknown")
    for name in ("gross-limit", "gross-used"):
        setup.add_argument("--" + name, help="Current-cycle gross limit and usage, required for prepaid-only admission")
    setup.add_argument("--expires", help="Optional verified grant expiry, ISO timestamp with offset; run cutoff defaults to the next UTC billing month")
    verbs.add_parser("status")
    settle = verbs.add_parser("reconcile", help="Record verified provider gross cost for a terminal attempt")
    settle.add_argument("--attempt", required=True)
    settle.add_argument("--gross-usd", required=True)
    settle.add_argument("--confirmed-stopped", action="store_true",
                        help="Mark an interrupted local controller attempt terminal after checking Modal")
    for verb in ("prepare", "bench", "run", "download"):
        cmd = verbs.add_parser(verb)
        cmd.add_argument("--sender", choices=("qwen", "pythia"), required=True)
        if verb == "prepare":
            cmd.add_argument("--data", type=Path, help="Original CounterFact JSON; default downloads on remote CPU")
        if verb == "download":
            cmd.add_argument("--out", required=True)
            cmd.add_argument("--include-states", action="store_true", help="Include large raw activation caches")
        if verb in ("bench", "run"):
            cmd.add_argument("--quick", action="store_true", help="Isolate smoke artifacts from full experiments")
            cmd.add_argument("--timeout", type=int, help="Seconds per shard within its budget envelope")
        if verb == "run":
            cmd.add_argument("--stage", required=True, choices=tuple(s for s in POOL if s != "bench"))
            cmd.add_argument("--shards", type=int, default=1)
    return parser


def main():
    args = make_parser().parse_args()
    ledger = Ledger(args.ledger)
    cmd = args.command
    if cmd == "init":
        ledger.initialize(args.credit, args.gross_limit, args.gross_used, args.net_limit,
                          datetime.fromisoformat(args.expires) if args.expires else None, args.workspace)
        return
    if cmd == "status":
        config = ledger.config()
        rows = ledger.attempts()
        print(json.dumps({"config": config, "phase_budgets_usd": CAPS,
                          "reserved_or_estimated_usd": sum(r["charged"] for r in rows) / MICRO,
                          "attempts": rows}, indent=2))
        return
    if cmd == "reconcile":
        if args.confirmed_stopped:
            statuses = {r["id"]: r["status"] for r in ledger.attempts()}
            if statuses.get(args.attempt) == "running":
                ledger.finish(args.attempt)
        ledger.reconcile(args.attempt, args.gross_usd)
        return
    if cmd == "download":
        download(ledger, args.sender, args.out, args.include_states)
        return

    stage = args.stage if cmd == "run" else cmd
    shards = getattr(args, "shards", 1)
    if shards < 1 or (shards != 1 and stage not in {"test", "robustness"}):
        raise BudgetError("Only test and robustness allow multiple independent shards")
    sender = args.sender
    config = ledger.config()
    workspace_check(config)
    run = config["run"]
    src = Path(__file__).resolve().parent / "plato"
    state = hashlib.sha256()
    for file in sorted(src.glob("*.py")):
        if not file.name.startswith("._") and file.name not in {"report.py", "budget.py"}:
            state.update(file.name.encode() + b"\x00" + file.read_bytes())
    ledger.freeze("research-source-sha256", state.hexdigest())
    quick = getattr(args, "quick", False)
    mode = "quick" if quick else "full"
    timeout = getattr(args, "timeout", None) or TIMEOUT[stage]
    prepkey = f"prepare/{sender}"
    if stage != "prepare" and ledger.completed(prepkey) is None:
        raise BudgetError("Prepare this sender and receiver on CPU first")
    benchkey = lambda gpu: f"{sender}/{mode}/bench/{gpu}"
    if stage == "prepare":
        units = [("CPU", 0, prepkey, "pilot" if sender == "qwen" else "replication")]
    elif stage == "bench":
        units = [(gpu, 0, benchkey(gpu), "pilot") for gpu in GPUS]
    else:
        trials = [ledger.completed(benchkey(gpu)) for gpu in GPUS]
        if any(t is None for t in trials):
            raise BudgetError("Run both GPU benchmarks with this sender and mode first")
        ledger.freeze(f"{sender}/{mode}/{stage}/shards", str(shards))
        best = min(trials, key=lambda t: t["estimated_warm_usd_per_world"])["requested_gpu"]
        pool = "replication" if sender == "pythia" and stage != "robustness" else POOL[stage]
        units = [(best, index, f"{sender}/{mode}/{stage}/{index}-of-{shards}", pool) for index in range(shards)]
    data = args.data.read_bytes() if cmd == "prepare" and args.data else None
    if data is not None and hashlib.sha256(data).hexdigest() != COUNTERFACT_SHA256:
        raise BudgetError("Local data differs from the pinned ROME CounterFact release")
    import modal
    with ExitStack() as stack:
        backend = None

        def submit(token):
            nonlocal backend
            if backend is None:
                app, prepare, worker = make_backend(timeout)
                stack.enter_context(modal.enable_output())
                stack.enter_context(app.run())
                backend = prepare, worker
            prepare, worker = backend
            if stage == "prepare":
                return prepare.remote(token, config["expires"], run, sender, data)
            return worker[gpu].remote(
                token, config["expires"], timeout, run, sender, stage, gpu, shard, shards, quick)

        for gpu, shard, key, pool in units:
            result = attempt(ledger, key, pool, gpu, timeout, submit)
            print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
