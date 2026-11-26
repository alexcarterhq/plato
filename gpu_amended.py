"""Verify one development-selected prompt format on 64 new validation worlds."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time
from importlib.metadata import version

import gpu_controls as original
from modal_app import MOUNT, VOLUME, attempt, claim, workspace_check
from plato.budget import Ledger, STARTUP_SECONDS

PROTOCOL = "fresh-amendment-bf16-v1"
FIRST = 10000
ROOT = Path(__file__).resolve().parent
EXTRA = ("gpu_amended.py", "plato/interface_development.py")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    return {**original.sources(), **{name: digest(ROOT/name) for name in EXTRA}}


def lock_amendment(ledger, sender, selection_path):
    from plato.interface_development import FORMATS

    failed = ledger.completed(f"{sender}/{original.PROTOCOL}")
    if not failed or failed["worlds"] != original.WORLDS:
        raise ValueError("Original control receipt is required")
    original_definitively_failed = (not failed["direct_pass"] or not any(failed["self_pass_by_layer"].values()))
    if not original_definitively_failed:
        raise ValueError("The original pair has a potentially qualifying layer")
    selection = json.loads(selection_path.read_text())
    fmt = selection["selected_format"]
    if fmt not in FORMATS:
        raise ValueError("Unknown development selection")
    if selection["source_sha256"]["plato/interface_development.py"] != digest(ROOT/EXTRA[1]):
        raise ValueError("Nominated prompt source differs from its development version")
    lock = dict(protocol=PROTOCOL, selected_format=fmt, selection_sha256=digest(selection_path),
                source_sha256=source_hashes(), validation_ids=[FIRST,FIRST+original.WORLDS-1],
                validation_split="validation", precision="bfloat16", identity_tolerance=1e-4,
                sender_threshold=.95, full_threshold=.95, self_threshold=.80)
    ledger.freeze("amendment/fresh-validation-v1",json.dumps(lock,sort_keys=True))
    return lock, failed["rows_sha256"]


def remote(token, cutoff, run_id, sender, lock, original_digest, provider):
    import modal
    import torch
    from plato.config import MODELS
    from plato.interface_development import adapt
    from plato.models import FrozenLM
    from plato.worlds import make_world

    claim(token, cutoff, original.TIMEOUT)
    if lock["source_sha256"] != source_hashes():
        raise ValueError("Fresh validation sources differ from the local lock")
    gpu = torch.cuda.get_device_name()
    if gpu != "NVIDIA " + original.GPUS[sender] or not torch.cuda.is_bf16_supported():
        raise ValueError("Unexpected GPU")
    volume = modal.Volume.from_name(VOLUME)
    volume.reload()
    folder = MOUNT / "runs" / run_id / "gpu-amended" / sender
    folder.mkdir(parents=True,exist_ok=True)
    lock_path = folder/"lock.json"
    if lock_path.exists() and json.loads(lock_path.read_text()) != lock:
        raise ValueError("Checkpoint directory belongs to a different amendment")
    lock_path.write_text(json.dumps(lock,indent=2)+"\n")
    volume.commit()
    start = time.monotonic()
    donor, receiver = [FrozenLM(*MODELS[key],device="cuda") for key in (sender,"smol")]
    if any(m.model.dtype != torch.bfloat16 for m in (donor,receiver)):
        raise ValueError("Unsupported precision")
    load_seconds = time.monotonic() - start
    fmt = lock["selected_format"]
    sample = adapt(make_world("validation",FIRST),fmt)
    ls, lr = original.candidates(donor), original.candidates(receiver)
    identity = {"sender":{str(l):donor.identity_error(sample.sender_text(0),l) for l in ls}}
    for b in range(2):
        text, end = sample.receiver_text(b)
        identity[f"receiver_book{b}"] = {str(l):receiver.identity_error(text,l,end) for l in lr}
    rows = []
    for i in range(FIRST,FIRST+original.WORLDS):
        checkpoint = folder / f"world-{i}.json.gz"
        if checkpoint.exists():
            with gzip.open(checkpoint,"rt") as f:
                rows.extend(json.load(f))
            continue
        world = adapt(make_world("validation", i),fmt)
        new = list(original.world_rows(world,donor,receiver,sender,lr))
        for row in new:
            row["interface"] = fmt
            row["protocol"] = PROTOCOL
        tmp = checkpoint.with_suffix(".partial")
        with gzip.open(tmp,"wt") as file:
            json.dump(new,file,separators=(",",":"))
        tmp.replace(checkpoint)
        rows.extend(new)
        if (i-FIRST+1)%8 == 0:
            volume.commit()
            print(f"{sender}: {i-FIRST+1}/{original.WORLDS} complete",flush=True)
    # Reuse the original cardinality/duplicate checks on a zero-based index.
    scores = original.aggregate(({**r,"world":r["world"]-FIRST} for r in rows),lr)
    result_path = folder/"observations.jsonl.gz"
    with gzip.open(result_path,"wb") as f:
        for r in rows:
            f.write(json.dumps(r,sort_keys=True,separators=(",",":")).encode()+b"\n")
    direct = all(hit/n >= .95 for key,(hit,n) in scores.items() if not key.startswith("self_"))
    self_pass = {str(l): all(scores[original.scorekey("self",b,l)][0] / (8*original.WORLDS) >= .8
                             for b in range(2)) for l in lr}
    max_error = max(x for values in identity.values() for x in values.values())
    result = dict(**lock,sender=sender,split="validation",worlds=original.WORLDS,rows=len(rows),
                  original_experiment=original.PROTOCOL,original_rows_sha256=original_digest,
                  sender_revision=MODELS[sender],receiver_revision=MODELS["smol"],
                  provider=provider,function_call_id=modal.current_function_call_id(),actual_gpu=gpu,dtype="bfloat16",
                  identity_by_layer=identity,identity_max_abs=max_error,counts=scores,direct_pass=direct,
                  self_pass_by_layer=self_pass,rows_sha256=digest(result_path),load_seconds=load_seconds,
                  worker_seconds=time.monotonic()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                  cuda_version=torch.version.cuda,versions={n:version(n) for n in ("torch","transformers","tokenizers","numpy")})
    (folder/"receipt.json").write_text(json.dumps(result,indent=2)+"\n")
    volume.commit()
    return result


def main():
    import modal
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sender",choices=original.GPUS,required=True)
    parser.add_argument("--selection",type=Path,default=Path("results/gpu-development/receipt.json"))
    parser.add_argument("--out",type=Path,default=Path("results/gpu-amended"))
    parser.add_argument("--ledger",default=".plato/ledger.sqlite")
    args = parser.parse_args()
    sender = args.sender
    ledger = Ledger(args.ledger)
    settings = ledger.config()
    workspace_check(settings)
    gpu = original.GPUS[sender]
    lock, original_digest = lock_amendment(ledger,sender,args.selection)
    volume = modal.Volume.from_name(VOLUME)
    def submit(token):
        image = (modal.Image.debian_slim(python_version="3.11")
                 .pip_install("torch==2.6.0","transformers==4.51.3","numpy==1.26.4","safetensors==0.5.3")
                 .env({"HF_HOME":str(MOUNT/"hf")})
                 .add_local_file(ROOT/"modal_app.py","/root/modal_app.py")
                 .add_local_file(ROOT/"gpu_controls.py","/root/gpu_controls.py")
                 .add_local_file(ROOT/"gpu_amended.py","/root/gpu_amended.py")
                 .add_local_dir(ROOT/"plato","/root/plato",ignore=["._*","**/._*","**/__pycache__/**","assets/**"]))
        app = modal.App("plato-fresh-gpu-controls",image=image,tags={"project":"plato-transfer"})
        life = {"single_use_containers":True} if hasattr(modal,"Workspace") else {"max_inputs":1}
        worker = app.function(image=image,include_source=False,volumes={str(MOUNT):volume},gpu=gpu,
                              cpu=(2,2),memory=(16384,16384),timeout=original.TIMEOUT,retries=0,
                              startup_timeout=STARTUP_SECONDS,max_containers=1,
                              env={"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"},**life)(remote)
        with modal.enable_output(),app.run(detach=True):
            provider = dict(app_id=app.app_id,function_id=worker.object_id,image_id=image.object_id)
            return worker.remote(token,settings["expires"],settings["run"],sender,lock,original_digest,provider)
    receipt = attempt(ledger,f"{sender}/{PROTOCOL}","robustness",gpu,original.TIMEOUT,submit)
    dest = args.out/sender
    dest.mkdir(parents=True,exist_ok=True)
    tmp = dest/"observations.partial"
    with tmp.open("wb") as f:
        path = f"runs/{settings['run']}/gpu-amended/{sender}/observations.jsonl.gz"
        for chunk in volume.read_file(path):
            f.write(chunk)
    if digest(tmp) != receipt["rows_sha256"]:
        raise ValueError("Downloaded observations failed checksum")
    tmp.replace(dest/"observations.jsonl.gz")
    (dest/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps({k:receipt[k] for k in ("sender","actual_gpu","rows","counts","identity_max_abs","estimated_usd")}))

if __name__ == "__main__":
    main()
