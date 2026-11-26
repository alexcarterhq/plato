"""Budgeted, preregistered development-only prompt comparison on one L40S."""
from collections import defaultdict
import gzip
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

from modal_app import MOUNT, VOLUME, attempt, claim, workspace_check
from plato.budget import Ledger, STARTUP_SECONDS
from plato.config import MODELS
from plato.interface_development import FORMATS, FRACTIONS, WORLD_IDS, adapt
from plato.trials import decode_packet, encode_packet
from plato.worlds import CATEGORIES, exact_answer, make_world

TIMEOUT = 1200
JOB = "development/interfaces-v1"
ROOT = Path(__file__).resolve().parent
SOURCES = ("GPU_DEVELOPMENT.md", "gpu_development.py", "plato/interface_development.py",
           "plato/models.py", "plato/trials.py", "plato/worlds.py", "plato/config.py")


def hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}


def evaluate_world(world, models, layers):
    rows = []
    for fmt in FORMATS:
        w = adapt(world, fmt)
        private = [w.sender_text(i) for i in range(8)]
        retrieval = [w.retrieval_text(i) for i in range(8)]

        def record(model, condition, prompts, answers, book=None, layer=None, packets=None):
            if not (len(answers) == len(prompts) == 8):
                raise ValueError("Incomplete batch")
            for v, (prompt, answer) in enumerate(zip(prompts, answers)):
                target = w.category(v) if book is None else w.destination(book, v)
                packet = None if packets is None else packets[v]
                rows.append(dict(format=fmt, split=w.split, world=w.index, variant=v,
                                 model=model, condition=condition, rulebook=book, prompt=prompt,
                                 source_prompt=private[v] if packet is not None else None,
                                 rulebook_table=None if book is None else list(w.books[book]),
                                 category=w.category(v), target=target, answer=answer,
                                 correct=exact_answer(answer, target), layer=layer,
                                 packet_bytes=0 if packet is None else len(packet),
                                 packet_sha256=None if packet is None else hashlib.sha256(packet).hexdigest(),
                                 dtype="bfloat16"))

        for sender in ("qwen", "pythia"):
            record(sender, "sender", retrieval, models[sender].answer_many(retrieval))
        receiver = models["receiver"]
        states = receiver.capture_many(private, layers, [None]*8)
        packets = {layer: [encode_packet(x) for x in states[layer]] for layer in layers}
        for book in range(2):
            full = [w.receiver_text(book, evidence=w.evidence(i))[0] for i in range(8)]
            record("receiver", "full", full, receiver.answer_many(full), book)
            plain, marker_end = w.receiver_text(book)
            for layer in layers:
                patches = [(layer, marker_end, decode_packet(buf)) for buf in packets[layer]]
                record("receiver", "self", [plain]*8, receiver.answer_many([plain]*8, patches),
                       book, layer, packets[layer])
    return rows


def summarize(rows, layers):
    counters = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in ("format", "model", "condition", "rulebook", "layer"))
        counters[key].append(row["correct"])
    counts = {key: [sum(marks), len(marks)] for key, marks in counters.items()}
    if any(n != len(WORLD_IDS)*8 for _, n in counts.values()):
        raise ValueError("Incomplete development condition")
    def rate(fmt, model, arm, book=None, layer=None):
        c, n = counts[fmt, model, arm, book, layer]
        return c / n
    formats = {}
    ranks = {}
    for idx, fmt in enumerate(FORMATS):
        donors = {sender: rate(fmt, sender, "sender") for sender in ("qwen", "pythia")}
        full = [rate(fmt, "receiver", "full", book) for book in range(2)]
        oracle = {layer: [rate(fmt, "receiver", "self", book, layer) for book in range(2)]
                  for layer in layers}
        layer = max(layers, key=lambda l: min(oracle[l]))
        scores = {s: min(donors[s]/0.95, min(full)/0.95, min(oracle[layer])/0.80) for s in donors}
        passes = sum(v >= 1 for v in scores.values())
        ranks[fmt] = (passes, max(scores.values()), sum(scores.values()), min(full), -idx)
        formats[fmt] = dict(retrieval=donors, full=full, oracle=oracle,
                            optimistic_development_layer=layer, scores=scores, passing_pairs=passes)
    selection = max(FORMATS[1:], key=ranks.get)
    return dict(selected_format=selection, grid=formats, selected_development_passes=formats[selection]["passing_pairs"],
                rows=len(rows), cases_per_condition=len(WORLD_IDS)*8)


def remote(token, cutoff, run_id, source, app_id):
    import modal
    import torch
    from plato.models import FrozenLM

    claim(token, cutoff, TIMEOUT)
    assert hashes() == source
    vol = modal.Volume.from_name(VOLUME)
    vol.reload()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or torch.cuda.get_device_name() != "NVIDIA L40S":
        raise RuntimeError("Development protocol requires an NVIDIA L40S in BF16")
    start = time.monotonic()
    models = {alias: FrozenLM(*MODELS[key], device="cuda")
              for alias, key in (("qwen", "qwen"), ("pythia", "pythia"), ("receiver", "smol"))}
    if any(next(lm.model.parameters()).dtype != torch.bfloat16 for lm in models.values()):
        raise RuntimeError("Weights have incorrect precision")
    layers = [round((models["receiver"].depth-1)*v) for v in FRACTIONS]
    folder = MOUNT / "runs" / run_id / "development" / "interfaces-v1"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    checksums = {}
    for i in WORLD_IDS:
        path = folder / f"{i}.jsonl.gz"
        if path.is_file():
            with gzip.open(path, "rt") as f:
                chunk = [json.loads(line) for line in f]
        else:
            chunk = evaluate_world(make_world("bench", i), models, layers)
            raw = b"".join(json.dumps(r, sort_keys=True).encode()+b"\n" for r in chunk)
            tmp = path.with_suffix(".partial")
            tmp.write_bytes(gzip.compress(raw, mtime=0))
            tmp.replace(path)
            vol.commit()
        if len(chunk) != len(FORMATS)*8*(2+2+2*len(layers)) or any(r["world"] != i for r in chunk):
            raise ValueError("Incomplete development world")
        checksums[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.extend(chunk)
        print(f"Completed development world {i}", flush=True)
    return dict(**summarize(rows, layers), split="bench", world_ids=WORLD_IDS,
                layers=layers, raw_chunks=checksums, model_revisions=MODELS, source_sha256=source,
                gpu=torch.cuda.get_device_name(), dtype="bfloat16", app_id=app_id,
                function_call_id=modal.current_function_call_id(), worker_seconds=time.monotonic()-start,
                deps={p:version(p) for p in ("torch","transformers","numpy","safetensors")})


def main():
    import modal
    ledger = Ledger(".plato/ledger.sqlite")
    settings = ledger.config()
    workspace_check(settings)
    if any(ledger.completed("prepare/"+m) is None for m in ("qwen", "pythia")):
        raise RuntimeError("Prepare both pinned donors before development")
    source = hashes()
    ledger.freeze(JOB+"/sources", json.dumps(source, sort_keys=True))
    volume = modal.Volume.from_name(VOLUME)
    def submit(token):
        image = (modal.Image.debian_slim(python_version="3.11")
                 .pip_install("torch==2.6.0", "transformers==4.51.3", "numpy==1.26.4", "safetensors==0.5.3")
                 .env({"HF_HOME":str(MOUNT/"hf")})
                 .add_local_file(ROOT/"modal_app.py","/root/modal_app.py")
                 .add_local_file(ROOT/"gpu_development.py","/root/gpu_development.py")
                 .add_local_file(ROOT/"GPU_DEVELOPMENT.md","/root/GPU_DEVELOPMENT.md")
                 .add_local_dir(ROOT/"plato","/root/plato", ignore=["._*","**/._*","**/__pycache__/**","assets/**"]))
        app = modal.App("plato-gpu-development", tags={"project":"plato-transfer"})
        lifecycle = {"single_use_containers":True} if hasattr(modal,"Workspace") else {"max_inputs":1}
        worker = app.function(image=image, include_source=False, volumes={str(MOUNT): volume},
                              cpu=(2,2), memory=(16384,16384), timeout=TIMEOUT, retries=0, gpu="L40S",
                              startup_timeout=STARTUP_SECONDS, max_containers=1, env={
                                  "HF_HUB_OFFLINE":"1", "TRANSFORMERS_OFFLINE":"1"}, **lifecycle)(remote)
        with modal.enable_output(), app.run():
            return worker.remote(token, settings["expires"], settings["run"], source, app.app_id)
    result = attempt(ledger, JOB, "robustness", "L40S", TIMEOUT, submit)
    folder = Path("results/gpu-development")
    folder.mkdir(exist_ok=True, parents=True)
    out = folder/"observations.jsonl.gz"
    tmp = folder/"observations.partial"
    with tmp.open("wb") as combined:
        for name, expected in sorted(result["raw_chunks"].items()):
            key = f"runs/{settings['run']}/development/interfaces-v1/{name}"
            chunk = b"".join(volume.read_file(key))
            if hashlib.sha256(chunk).hexdigest() != expected:
                raise ValueError("Remote development shard checksum mismatch")
            combined.write(chunk)
    tmp.replace(out)
    result["observations_sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
    (folder/"receipt.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ("app_id","selected_format","selected_development_passes","grid")}))

if __name__ == "__main__":
    main()

