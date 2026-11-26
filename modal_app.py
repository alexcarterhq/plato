"""Credit-limited Modal runner. Start locally with python modal_app.py --help."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import time

from plato.subjects import COUNTERFACT_SHA256

from plato.budget import (BudgetError, CAPS, Ledger, MICRO,
                          SHUTDOWN_SECONDS, STARTUP_SECONDS, envelope, rate, resources)


VOLUME = "plato-transfer-experiments"
MOUNT = Path("/plato-volume")
DATA_URL = "https://rome.baulab.info/data/dsets/counterfact.json"
GPUS = ("L4", "L40S")
CPU_SCIENCE = "CPU-science"
POOL = {"bench": "pilot", "prescreen": "facts", "facts": "facts", "calibrate": "primary",
        "validate": "primary", "test": "primary", "robustness": "robustness"}
TIMEOUT = {"prepare": 1800, "bench": 600, "facts": 14400, "calibrate": 7200,
           "validate": 7200, "test": 7200, "robustness": 7200}
CPU_TIMEOUT = {**TIMEOUT, "bench": 1800, "prescreen": 14400, "facts": 43200,
               "calibrate": 14400, "validate": 14400, "robustness": 14400}


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


def remote_stage(token, cutoff, timeout, run, sender, stage, hardware, shard, shards, quick,
                 prescreen_spec=None, provider=None):
    import modal
    import resource
    from importlib.metadata import version
    import torch
    from plato.pipeline import run_stage

    claim(token, cutoff, timeout)
    volume = modal.Volume.from_name(VOLUME)
    volume.reload()
    root = MOUNT / "runs" / run / sender
    cpu_only = hardware == CPU_SCIENCE
    cores, ram_gib = resources(hardware)
    out = (root / "cpu" if cpu_only else root) / ("quick" if quick else "full")
    if stage == "bench":
        out = out / "bench" / hardware
    out.mkdir(parents=True, exist_ok=True)
    if cpu_only:
        if torch.cuda.is_available():
            raise RuntimeError("CPU execution requires a CPU-only container")
        torch.set_num_threads(cores)
        torch.set_num_interop_threads(1)
    started = time.monotonic()
    cpu_start = time.process_time()
    data = root / "counterfact.json"
    if stage == "prescreen":
        from plato.prescreen import run as screen
        report = screen(sender, data, out, shard=shard, shards=shards,
                        batch_limit=prescreen_spec["batch_limit"], on_commit=volume.commit)
    else:
        if stage == "facts" and prescreen_spec:
            from plato.prescreen import merge
            merge(sender, data, out, prescreen_spec["shards"], batch_limit=prescreen_spec["batch_limit"])
        report = run_stage(stage, sender, out, data_path=data, shard=shard, shards=shards, quick=quick)
    if not cpu_only:
        torch.cuda.synchronize()
    cpu_name = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                     if line.startswith("model name") or line.startswith("Hardware")), "unknown")
    report = dict(report)
    report.update(worker_wall_seconds=time.monotonic() - started,
                  worker_cpu_seconds=time.process_time() - cpu_start,
                  peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  hardware=hardware, cpu_model=cpu_name, cpu_cores_reserved=cores, memory_gib_reserved=ram_gib,
                  dtype="float32" if cpu_only else ("bfloat16" if torch.cuda.is_bf16_supported() else "float32"),
                  provider=provider or {}, function_call_id=modal.current_function_call_id(),
                  dependencies={name: version(name) for name in ("torch","numpy","transformers","safetensors")},
                  requested_gpu=None if cpu_only else hardware,
                  actual_gpu=None if cpu_only else torch.cuda.get_device_name(),
                  peak_gpu_bytes=0 if cpu_only else torch.cuda.max_memory_allocated(),
                  shard=shard, shards=shards)
    if cpu_only:
        report["gpu_s"] = 0.0
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


def stage_cpu(*args):
    return remote_stage(*args)


def make_backend(hardware, timeout):
    import modal

    volume = modal.Volume.from_name(VOLUME, create_if_missing=True)
    image = (modal.Image.debian_slim(python_version="3.11")
             .pip_install("torch==2.6.0", "transformers==4.51.3", "numpy==1.26.4", "safetensors==0.5.3")
             .env({"HF_HOME": str(MOUNT / "hf")})
             .add_local_file(Path(__file__).resolve(), "/root/modal_app.py")
             .add_local_dir(Path(__file__).resolve().parent / "plato", "/root/plato",
                            ignore=["**/._*", "**/__pycache__/**", "._*", "readiness*.py",
                                    "report.py", "assets/readiness*", "assets/replay*", "assets/index.html"]))
    app = modal.App("plato-transfer", image=image, tags={"project": "plato-transfer"})
    cores, ram_gib = resources(hardware)
    options = dict(image=image, include_source=False, volumes={str(MOUNT): volume},
                   cpu=(cores, cores), memory=(ram_gib * 1024, ram_gib * 1024),
                   retries=0, startup_timeout=STARTUP_SECONDS, max_containers=1)
    # Older installed clients call the same one-input lifecycle `max_inputs`.
    if hasattr(modal, "Workspace"):
        options["single_use_containers"] = True
    else:
        options["max_inputs"] = 1
    if hardware == "CPU":
        return app, app.function(**options, timeout=TIMEOUT["prepare"])(remote_prepare)
    env = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    if hardware == CPU_SCIENCE:
        env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS=str(cores), MKL_NUM_THREADS=str(cores),
                   OPENBLAS_NUM_THREADS=str(cores))
        return app, app.function(**options, timeout=timeout, env=env)(stage_cpu)
    worker = {"L4": stage_l4, "L40S": stage_l40s}[hardware]
    return app, app.function(**options, timeout=timeout, gpu=hardware, env=env)(worker)



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


class ProviderRejectedConfiguration(RuntimeError):
    """Modal refused to create the requested Function before a worker was dispatched."""


def attempt(ledger, key, pool, gpu, timeout, submit):
    existing = ledger.completed(key)
    if existing is not None:
        return existing
    token = ledger.reserve(key, pool, gpu, timeout)
    start = time.monotonic()
    print(f"Reserved {key}: {token}", flush=True)
    try:
        result = submit(token)
    except ProviderRejectedConfiguration as exc:
        ledger.finish(token, error=str(exc))
        raise
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


def download(ledger, sender, destination, include_states=False, hardware="gpu"):
    import modal
    from modal.volume import FileEntryType
    config = ledger.config()
    workspace_check(config)
    volume = modal.Volume.from_name(VOLUME)
    prefix = PurePosixPath("runs", config["run"], sender)
    if hardware == "cpu":
        prefix /= "cpu"
    dest = Path(destination)
    for entry in volume.iterdir(str(prefix), recursive=True):
        path = PurePosixPath(entry.path.lstrip("/"))
        if entry.type != FileEntryType.FILE or path.name == "counterfact.json":
            continue
        if not include_states and path.name in {"sender-states.npz", "receiver-states.npz"}:
            continue
        relative = path.relative_to(prefix)
        if hardware != "cpu" and "cpu" in relative.parts:
            continue
        if not include_states and ("calibration" in relative.parts or "prescreen" in relative.parts):
            continue
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
    settle.add_argument("--gross-usd", help="Verified provider charge; omit to retain the full estimate")
    settle.add_argument("--reason", help="Provider message when closing a stopped reservation")
    settle.add_argument("--confirmed-stopped", action="store_true",
                        help="Mark an interrupted local controller attempt terminal after checking Modal")
    for verb in ("prepare", "bench", "run", "download"):
        cmd = verbs.add_parser(verb)
        cmd.add_argument("--sender", choices=("qwen", "pythia"), required=True)
        if verb == "prepare":
            cmd.add_argument("--data", type=Path, help="Original CounterFact JSON; default downloads on remote CPU")
        if verb == "bench":
            cmd.add_argument("--gpu", choices=GPUS, help="Benchmark one device in an isolated Modal app")
        if verb != "prepare":
            cmd.add_argument("--hardware", choices=("gpu", "cpu"), default="gpu",
                             help="CPU selects eight physical cores, 32 GiB and FP32 without requesting a GPU")
        if verb == "download":
            cmd.add_argument("--out", required=True)
            cmd.add_argument("--include-states", action="store_true", help="Include large raw activation caches")
        if verb in ("bench", "run"):
            cmd.add_argument("--quick", action="store_true", help="Isolate smoke artifacts from full experiments")
            cmd.add_argument("--timeout", type=int, help="Seconds per shard within its budget envelope")
        if verb == "run":
            cmd.add_argument("--stage", required=True, choices=tuple(s for s in POOL if s != "bench"))
            cmd.add_argument("--shards", type=int, default=1)
            cmd.add_argument("--shard", type=int, help="Run only this zero-based shard of the total")
            cmd.add_argument("--batch-limit", type=int, help="Prescreen only the first N global batches, resumable")
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
        if not args.confirmed_stopped and args.gross_usd is None:
            raise BudgetError("Specify verified gross usage or confirm that the worker stopped")
        if args.confirmed_stopped:
            statuses = {r["id"]: r["status"] for r in ledger.attempts()}
            if statuses.get(args.attempt) == "running":
                ledger.finish(args.attempt, error=args.reason)
            elif args.attempt not in statuses:
                raise BudgetError("No attempt with this identifier")
        if args.gross_usd is not None:
            ledger.reconcile(args.attempt, args.gross_usd)
        return
    if cmd == "download":
        download(ledger, args.sender, args.out, args.include_states, args.hardware)
        return

    stage = args.stage if cmd == "run" else cmd
    cpu_only = getattr(args, "hardware", "gpu") == "cpu"
    shards = getattr(args, "shards", 1)
    single_shard = getattr(args, "shard", None)
    limit = getattr(args, "batch_limit", None)
    if shards < 1 or (shards != 1 and stage not in {"prescreen", "test", "robustness"}):
        raise BudgetError("Only prescreen, test and robustness allow independent shards")
    if single_shard is not None and not 0 <= single_shard < shards:
        raise BudgetError("Shard index must be within the requested shard count")
    if limit is not None and (stage != "prescreen" or limit < 1):
        raise BudgetError("A positive global batch limit applies only to prescreen")
    if stage == "prescreen" and not cpu_only:
        raise BudgetError("Sharded prescreen is available with --hardware cpu")
    if cpu_only and getattr(args, "gpu", None):
        raise BudgetError("A GPU model cannot be selected for a CPU benchmark")
    sender = args.sender
    config = ledger.config()
    workspace_check(config)
    run = config["run"]
    src = Path(__file__).resolve().parent / "plato"
    state = hashlib.sha256()
    # Preserve the original primary experiment manifest across independent analyses.
    for name in ('__init__.py', '__main__.py', 'cli.py', 'config.py', 'facts.py', 'models.py', 'pipeline.py', 'subjects.py', 'trials.py', 'worlds.py'):
        state.update(name.encode() + b"\x00" + (src / name).read_bytes())
    ledger.freeze("research-source-sha256", state.hexdigest())
    quick = getattr(args, "quick", False)
    mode = "quick" if quick else "full"
    namespace = f"{sender}/cpu/{mode}" if cpu_only else f"{sender}/{mode}"
    timeout_default = (CPU_TIMEOUT if cpu_only else TIMEOUT)[stage]
    timeout = getattr(args, "timeout", None)
    timeout = timeout_default if timeout is None else timeout
    if not 1 <= timeout <= 86400:
        raise BudgetError("Function timeout must be between 1 and 86400 seconds")
    prepkey = f"prepare/{sender}"
    if stage != "prepare" and ledger.completed(prepkey) is None:
        raise BudgetError("Prepare this sender and receiver on CPU first")
    benchkey = lambda gpu: f"{namespace}/bench/{gpu}"
    prescreen_spec = None
    if stage == "prepare":
        units = [("CPU", 0, prepkey, "pilot" if sender == "qwen" else "replication")]
    elif stage == "bench":
        devices = (CPU_SCIENCE,) if cpu_only else ((args.gpu,) if args.gpu else GPUS)
        units = [(hw, 0, benchkey(hw), "pilot" if sender == "qwen" else "replication") for hw in devices]
    else:
        if cpu_only:
            best = CPU_SCIENCE
        else:
            trials = [ledger.completed(benchkey(gpu)) for gpu in GPUS]
            if any(t is None for t in trials):
                raise BudgetError("Run both GPU benchmarks with this sender and mode first")
            best = min(trials, key=lambda t: t["estimated_warm_usd_per_world"])["requested_gpu"]
        if stage == "prescreen":
            ledger.freeze(f"{namespace}/fact-source", "prescreen")
        ledger.freeze(f"{namespace}/{stage}/shards", str(shards))
        if stage == "prescreen":
            ledger.freeze(f"{namespace}/prescreen/source",
                          hashlib.sha256((src / "prescreen.py").read_bytes()).hexdigest())
            prescreen_spec = {"shards": shards, "batch_limit": limit}
        if cpu_only and stage == "facts":
            count = ledger.setting(f"{namespace}/prescreen/shards")
            source = ledger.setting(f"{namespace}/fact-source") or ("prescreen" if count else "inline")
            ledger.freeze(f"{namespace}/fact-source", source)
            if source == "prescreen" and not count:
                raise BudgetError("Wait for the prescreen shard configuration")
            if count:
                count = int(count)
                if any(ledger.completed(f"{namespace}/prescreen/all/{i}-of-{count}") is None for i in range(count)):
                    raise BudgetError("Complete all uncapped prescreen shards before factual fitting")
                ledger.freeze(f"{namespace}/prescreen/source",
                              hashlib.sha256((src / "prescreen.py").read_bytes()).hexdigest())
                prescreen_spec = {"shards": count, "batch_limit": None}
        pool = "replication" if sender == "pythia" and stage != "robustness" else POOL[stage]
        name = f"prescreen/{limit or 'all'}" if stage == "prescreen" else stage
        indices = range(shards) if single_shard is None else [single_shard]
        units = [(best, i, f"{namespace}/{name}/{i}-of-{shards}", pool) for i in indices]
    data = args.data.read_bytes() if cmd == "prepare" and args.data else None
    if data is not None and hashlib.sha256(data).hexdigest() != COUNTERFACT_SHA256:
        raise BudgetError("Local data differs from the pinned ROME CounterFact release")
    for gpu, shard, key, pool in units:
        def submit(token):
            import modal
            app, worker = make_backend(gpu, timeout)
            registered = False
            try:
                with modal.enable_output(), app.run():
                    registered = True
                    if stage == "prepare":
                        return worker.remote(token, config["expires"], run, sender, data)
                    provider = {"app_id": app.app_id, "function_id": worker.object_id,
                                "image_id": app.image.object_id}
                    return worker.remote(token, config["expires"], timeout, run, sender, stage,
                                         gpu, shard, shards, quick, prescreen_spec, provider)
            except Exception as exc:
                message = str(exc)
                if not registered and "add a payment method" in message.lower():
                    raise ProviderRejectedConfiguration(message) from exc
                raise

        result = attempt(ledger, key, pool, gpu, timeout, submit)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
