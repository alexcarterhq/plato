"""Budgeted CPU verification of two validation controls independent of both fitted maps."""
import argparse
import hashlib
import json
from pathlib import Path

from modal_app import MOUNT, VOLUME, attempt, claim, workspace_check
from plato.budget import Ledger, STARTUP_SECONDS

TIMEOUT = 3600


def remote(token, cutoff, run_id, shard, shards):
    import modal
    import torch
    from plato.readiness import run

    claim(token, cutoff, TIMEOUT)
    volume = modal.Volume.from_name(VOLUME)
    volume.reload()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    return run("qwen", MOUNT / "runs" / run_id / "qwen" / "full",
               shard=shard, shards=shards, on_world=volume.commit)


def main():
    import modal
    options = argparse.ArgumentParser(description=__doc__)
    options.add_argument("--ledger", default=".plato/ledger.sqlite")
    options.add_argument("--shard", type=int, required=True)
    options.add_argument("--shards", type=int, default=4)
    args = options.parse_args()
    if not 0 <= args.shard < args.shards <= 64:
        options.error("Shard must be in range, with 1–64 shards")
    ledger = Ledger(args.ledger)
    settings = ledger.config()
    workspace_check(settings)
    if not ledger.completed("prepare/qwen"):
        options.error("Complete Qwen and SmolLM2 weight preparation first")
    root = Path(__file__).resolve().parent
    ledger.freeze("readiness-shards", str(args.shards))
    # Freeze only the implementation of these two direct control measurements.
    for name in ("config.py", "models.py", "worlds.py", "readiness.py"):
        ledger.freeze("readiness-" + name, hashlib.sha256((root/"plato"/name).read_bytes()).hexdigest())

    def submit(token):
        app = modal.App("plato-readiness", tags={"project": "plato-transfer"})
        volume = modal.Volume.from_name(VOLUME)
        image = (modal.Image.debian_slim(python_version="3.11")
                 .pip_install("torch==2.6.0", "transformers==4.51.3", "numpy==1.26.4", "safetensors==0.5.3")
                 .env({"HF_HOME": str(MOUNT/"hf")})
                 .add_local_file(root/"modal_app.py", "/root/modal_app.py")
                 .add_local_file(root/"readiness_runner.py", "/root/readiness_runner.py")
                 .add_local_dir(root/"plato", "/root/plato", ignore=[
                     "._*", "**/._*", "**/__pycache__/**", "assets/**",
                     "readiness_report.py", "**/readiness_report.py", "report.py",
                 ]))
        lifecycle = {"single_use_containers": True} if hasattr(modal, "Workspace") else {"max_inputs": 1}
        worker = app.function(image=image, include_source=False, volumes={str(MOUNT):volume},
                              cpu=(4,4), memory=(32768,32768), timeout=TIMEOUT,
                              retries=0, startup_timeout=STARTUP_SECONDS, max_containers=1,
                              env={"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"}, **lifecycle)(remote)
        with modal.enable_output(), app.run():
            return worker.remote(token, settings["expires"], settings["run"], args.shard, args.shards)
    job = f"qwen/readiness/{args.shard}-of-{args.shards}"
    print(json.dumps(attempt(ledger, job, "pilot", "CPU-readiness", TIMEOUT, submit), indent=2))


if __name__ == "__main__":
    main()
