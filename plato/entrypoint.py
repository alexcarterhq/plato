"""Dispatch the public offline reproduction command to the measured release format."""
import argparse
import hashlib
from pathlib import Path
import shutil
import sys
import tempfile
from urllib.error import HTTPError
from urllib.request import urlopen

from . import cli
from .control_release import BUNDLE, generate, is_control_archive, unpack


def main(argv=None):
    words = list(sys.argv[1:] if argv is None else argv)
    if not words or words[0] != "reproduce":
        return cli.main(words)
    parser = argparse.ArgumentParser(prog="plato reproduce")
    parser.add_argument("--quick", action="store_true", help="Rebuild from saved evidence")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--archive", type=Path)
    source.add_argument("--input", type=Path)
    parser.add_argument("--out", type=Path, default=Path("release"))
    args = parser.parse_args(words[1:])
    if args.input:
        if not (args.input/"gpu-controls").is_dir():
            return cli.main(words)
        generate(args.input,args.out)
    else:
        with tempfile.TemporaryDirectory(prefix="plato-replay-") as directory:
            tmp = Path(directory)
            bundle = args.archive
            if bundle is None:
                base = "https://github.com/alexcarterhq/plato/releases/latest/download"
                bundle = tmp / BUNDLE
                try:
                    with urlopen(f"{base}/{BUNDLE}",timeout=60) as response, bundle.open("wb") as out:
                        shutil.copyfileobj(response,out)
                except HTTPError as exc:
                    if exc.code == 404:
                        return cli.main(words)
                    raise
                with urlopen(f"{base}/{BUNDLE}.sha256",timeout=30) as response:
                    expected = response.read(1024).decode().split()[0]
                with bundle.open("rb") as file:
                    if hashlib.file_digest(file,"sha256").hexdigest() != expected:
                        raise ValueError("Control release checksum mismatch")
            if not is_control_archive(bundle):
                return cli.main(words)
            unpack(bundle,tmp/"raw")
            generate(tmp/"raw",args.out)
    print((args.out/"index.html").resolve())
