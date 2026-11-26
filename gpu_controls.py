"""Run the original frozen validation controls without opening any test split."""
import argparse
import gzip
import hashlib
import json
import os
import time
from importlib.metadata import version
from pathlib import Path

from modal_app import MOUNT, VOLUME, attempt, claim, workspace_check
from plato.budget import Ledger, STARTUP_SECONDS

TIMEOUT = 1200
WORLDS = 64
GPUS = {"qwen": "L40S", "pythia": "L4"}
ROOT = Path(__file__).resolve().parent
SOURCES = ("gpu_controls.py", "plato/config.py", "plato/models.py", "plato/worlds.py", "plato/trials.py")
PROTOCOL = "original-bf16-v1"


def sources():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}


def candidates(model):
    return sorted({round((model.depth - 1) * f) for f in (0.22, 0.32, 0.40)})


def scorekey(arm, book, layer):
    return "_".join(map(str, (arm, book, layer)))


def world_rows(world, donor, receiver, sender, layers):
    from plato.trials import _timed, encode_packet, decode_packet
    from plato.worlds import exact_answer, first_answer

    variants = range(8)
    private = [world.sender_text(i) for i in variants]
    states, capture_seconds, _ = _timed(lambda: receiver.capture_many(private, layers, [None]*8), receiver)
    packets = {layer: [encode_packet(v) for v in states[layer]] for layer in layers}

    def generate(arm, book, layer, model, prompts, patches=None):
        answers, batch_seconds, _ = _timed(lambda: model.answer_many(prompts, patches), model)
        if len(answers) != 8:
            raise ValueError("Incomplete generation")
        for i in variants:
            target = world.category(i) if arm == "sender" else world.destination(book, i)
            packet = packets[layer][i] if arm == "self" else None
            yield dict(split="validation", sender=sender, world=world.index, variant=i,
                       condition=arm, rulebook=book, layer=layer, category=world.category(i),
                       prompt=prompts[i], source_prompt=private[i] if packet is not None else None,
                       packet_sha256=hashlib.sha256(packet).hexdigest() if packet is not None else None,
                       packet_bytes=len(packet) if packet is not None else None,
                       target=target, answer=answers[i], first_answer=first_answer(answers[i]),
                       correct=exact_answer(answers[i], target), batch_size=8, batch_seconds=batch_seconds,
                       capture_all_layers_batch_seconds=capture_seconds if packet is not None else None)

    yield from generate("sender", None, None, donor, [world.retrieval_text(i) for i in variants])
    for book in range(2):
        yield from generate("full", book, None, receiver,
                            [world.receiver_text(book, evidence=world.evidence(i))[0] for i in variants])
        plain, end = world.receiver_text(book)
        for layer in layers:
            patches = [(layer, end, decode_packet(packet)) for packet in packets[layer]]
            yield from generate("self", book, layer, receiver, [plain]*8, patches)


def aggregate(rows, layers):
    expected = [("sender", None, None)]
    for book in range(2):
        expected.append(("full", book, None))
        expected.extend(("self", book, layer) for layer in layers)
    counts = {scorekey(*key): [0, 0] for key in expected}
    seen = set()
    for row in rows:
        arm = row["condition"], row["rulebook"], row["layer"]
        key = row["world"], row["variant"], arm
        if key in seen or row["world"] not in range(WORLDS) or row["variant"] not in range(8):
            raise ValueError("Unexpected control observation")
        seen.add(key)
        count = counts[scorekey(*arm)]
        count[0] += int(row["correct"])
        count[1] += 1
    if any(n != WORLDS * 8 for _, n in counts.values()):
        raise ValueError("Missing observations")
    return counts


def run_remote(token, cutoff, run_id, sender, expected_sources, provider):
    import modal
    import torch
    from plato.config import MODELS
    from plato.models import FrozenLM
    from plato.worlds import make_world

    claim(token, cutoff, TIMEOUT)
    if sources() != expected_sources:
        raise ValueError("Worker source differs from reserved protocol")
    volume = modal.Volume.from_name(VOLUME)
    volume.reload()
    gpu = torch.cuda.get_device_name()
    if gpu != "NVIDIA " + GPUS[sender] or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Unexpected GPU or precision")
    folder = MOUNT / "runs" / run_id / "gpu-controls" / sender
    folder.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    donor, receiver = [FrozenLM(*MODELS[m], device="cuda") for m in (sender, "smol")]
    if any(m.model.dtype != torch.bfloat16 for m in (donor, receiver)):
        raise ValueError("Expected BF16 checkpoints on the worker")
    load_seconds = time.monotonic() - start
    layers = candidates(receiver)
    identity = {}
    sample = make_world("validation", 0)
    prompt, end = sample.receiver_text(0)
    for label, model, text, marker in (("sender", donor, sample.sender_text(0), None),
                                        ("receiver", receiver, prompt, end)):
        identity[label] = {str(l): model.identity_error(text, l, marker) for l in candidates(model)}
    rows = []
    for i in range(WORLDS):
        worldfile = folder / f"world-{i:02}.json.gz"
        if worldfile.exists():
            with gzip.open(worldfile, "rt") as f:
                rows.extend(json.load(f))
            continue
        new_rows = list(world_rows(make_world("validation", i), donor, receiver, sender, layers))
        tmp = worldfile.with_suffix(".partial")
        with gzip.open(tmp, "wt") as f:
            json.dump(new_rows, f, separators=(",", ":"))
        tmp.replace(worldfile)
        rows.extend(new_rows)
        if (i+1) % 8 == 0:
            volume.commit()
            print(f"{sender}: {i+1}/{WORLDS} complete", flush=True)
    counts = aggregate(rows, layers)
    observations = folder / "observations.jsonl.gz"
    with gzip.open(observations, "wb") as out:
        for row in rows:
            out.write(json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n")
    max_error = max(e for errors in identity.values() for e in errors.values())
    direct_pass = all(hit / n >= .95 for key, (hit,n) in counts.items() if not key.startswith("self_"))
    self_pass = {str(l): all(counts[scorekey("self", b, l)][0] / (8*WORLDS) >= .8
                             for b in (0,1)) for l in layers}
    result = dict(protocol=PROTOCOL, sender=sender, sender_revision=MODELS[sender],
                  receiver_revision=MODELS["smol"], split="validation", worlds=WORLDS, rows=len(rows),
                  provider=provider, function_call_id=modal.current_function_call_id(),
                  actual_gpu=gpu, dtype="bfloat16", identity_max_abs=max_error, identity_by_layer=identity,
                  counts=counts, direct_pass=direct_pass, self_pass_by_layer=self_pass,
                  control_interpretation="Self intervention evaluated at all original factual-layer candidates before factual model selection.",
                  source_sha256=expected_sources, rows_sha256=hashlib.sha256(observations.read_bytes()).hexdigest(),
                  load_seconds=load_seconds, worker_seconds=time.monotonic()-start,
                  peak_gpu_bytes=torch.cuda.max_memory_allocated(), cuda_version=torch.version.cuda,
                  versions={pkg:version(pkg) for pkg in ("torch","transformers","tokenizers","numpy")})
    (folder / "receipt.json").write_text(json.dumps(result, indent=2)+"\n")
    volume.commit()
    return result


def main():
    import modal
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sender", choices=GPUS, required=True)
    parser.add_argument("--ledger", default=".plato/ledger.sqlite")
    parser.add_argument("--out", type=Path, default=Path("results/gpu-controls"))
    args = parser.parse_args()
    sender = args.sender
    gpu = GPUS[sender]
    ledger = Ledger(args.ledger)
    settings = ledger.config()
    workspace_check(settings)
    if not ledger.completed(f"prepare/{sender}"):
        parser.error("Prepare both pinned models first")
    source = sources()
    ledger.freeze("original-gpu-controls-source",json.dumps(source,sort_keys=True))
    volume = modal.Volume.from_name(VOLUME)

    def submit(token):
        image = (modal.Image.debian_slim(python_version="3.11")
                 .pip_install("torch==2.6.0","transformers==4.51.3","numpy==1.26.4","safetensors==0.5.3")
                 .env({"HF_HOME": str(MOUNT/"hf")})
                 .add_local_file(ROOT/"modal_app.py", "/root/modal_app.py")
                 .add_local_file(ROOT/"gpu_controls.py", "/root/gpu_controls.py")
                 .add_local_dir(ROOT/"plato", "/root/plato", ignore=["._*","**/._*","**/__pycache__/**","assets/**"]))
        app = modal.App("plato-original-gpu-controls", image=image, tags={"project":"plato-transfer"})
        life = {"single_use_containers":True} if hasattr(modal,"Workspace") else {"max_inputs":1}
        worker = app.function(image=image, include_source=False, volumes={str(MOUNT):volume}, gpu=gpu,
                              cpu=(2,2),memory=(16384,16384),timeout=TIMEOUT,retries=0,
                              startup_timeout=STARTUP_SECONDS,max_containers=1,
                              env={"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"}, **life)(run_remote)
        with modal.enable_output(), app.run(detach=True):
            info = dict(app_id=app.app_id,function_id=worker.object_id,image_id=image.object_id)
            return worker.remote(token,settings["expires"],settings["run"],sender,source,info)

    receipt = attempt(ledger, f"{sender}/{PROTOCOL}", "pilot" if sender=="qwen" else "replication",
                      gpu, TIMEOUT, submit)
    dest = args.out / sender
    dest.mkdir(parents=True,exist_ok=True)
    tmp = dest/"observations.partial"
    with tmp.open("wb") as file:
        for data in volume.read_file(f"runs/{settings['run']}/gpu-controls/{sender}/observations.jsonl.gz"):
            file.write(data)
    if hashlib.sha256(tmp.read_bytes()).hexdigest()!=receipt["rows_sha256"]:
        raise ValueError("Downloaded observations failed checksum")
    tmp.replace(dest/"observations.jsonl.gz")
    (dest/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps({k:receipt[k] for k in ("sender","actual_gpu","rows","counts","identity_max_abs","estimated_usd")}))

if __name__=="__main__":
    main()
