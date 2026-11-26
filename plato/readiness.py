"""CPU-only validation of the two prerequisites that do not require a representation map."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

from .config import MODELS
from .models import FrozenLM
from .worlds import exact_answer, first_answer, make_world

WORLDS = 64
VARIANTS = 8
THRESHOLD = .95
ARMS = (("sender", None), ("full", 0), ("full", 1))
PACKAGE = Path(__file__).resolve().parent


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _write(path, data):
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def _config():
    return {"protocol": "plato-cpu-readiness-v1", "sender": list(MODELS["qwen"]), "receiver": list(MODELS["smol"]),
            "split": "validation", "worlds": WORLDS, "variants": VARIANTS,
            "condition_batches": [["sender", None], ["full", 0], ["full", 1]],
            "threshold": THRESHOLD, "precision": "float32", "device": "cpu", "batch_size": VARIANTS,
            "greedy": True, "max_new_tokens": 12, "attention": "eager", "frozen": True,
            "dependencies": {d: importlib.metadata.version(d) for d in ("numpy", "torch", "transformers", "tokenizers")},
            "source_sha256": {n: _sha((PACKAGE / n).read_bytes()) for n in ("worlds.py", "models.py", "config.py", "readiness.py")}}


def inputs(world, arm, book):
    if arm == "sender":
        return ([world.retrieval_text(v) for v in range(VARIANTS)],
                [world.category(v) for v in range(VARIANTS)])
    return ([world.receiver_text(book, evidence=world.evidence(v))[0] for v in range(VARIANTS)],
            [world.destination(book, v) for v in range(VARIANTS)])


def validate_world(rows, world_id, signature):
    world = make_world("validation", world_id)
    if len(rows) != len(ARMS) * VARIANTS:
        raise ValueError(f"Incomplete readiness world {world_id}")
    by_key = {(r["condition"], r["rulebook"], r["variant"]): r for r in rows}
    for arm, book in ARMS:
        prompts, targets = inputs(world, arm, book)
        for v in range(VARIANTS):
            r = by_key.get((arm, book, v), {})
            if (r.get("world") != world_id or r.get("split") != "validation" or r.get("sender") != "qwen"
                    or r.get("config_sha256") != signature or r.get("precision") != "float32"
                    or r.get("prompt") != prompts[v] or r.get("target") != targets[v]
                    or r.get("category") != world.category(v) or not isinstance(r.get("answer"), str)
                    or type(r.get("correct")) is not bool or r["correct"] != exact_answer(r["answer"], targets[v])
                    or r.get("first_answer") != first_answer(r["answer"])):
                raise ValueError(f"Invalid readiness observation for {(world_id, arm, book, v)}")


def read_shard(path, indices, signature):
    if not path.exists():
        return []
    completed = []
    with path.open(encoding="utf-8") as f:
        for world_id in indices:
            line = f.readline()
            if not line:
                break
            rows = [json.loads(line)] + [json.loads(f.readline()) for _ in range(len(ARMS)*VARIANTS-1)]
            validate_world(rows, world_id, signature)
            completed.append(rows)
        if f.read(1):
            raise ValueError(f"Unexpected extra observations in {path}")
    return completed


def run(sender, output_dir, shard=0, shards=1, on_world=None):
    """Commit one complete world at a time. An optional no-argument hook can commit a remote Volume."""
    if sender != "qwen" or type(shards) is not int or type(shard) is not int or not 0 <= shard < shards <= WORLDS:
        raise ValueError("Use the primary qwen sender and a valid shard number")
    folder = Path(output_dir) / "readiness"
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{shard:04d}-of-{shards:04d}"
    path = folder / (stem + ".jsonl")
    config_path = folder / (stem + ".config.json")
    indices = list(range(shard, WORLDS, shards))
    config = _config()
    signature = _sha(canonical(config))
    receipt = {"config": config, "config_sha256": signature, "shard": shard, "shards": shards,
               "machine": platform.machine(), "processor": platform.processor(), "platform": platform.system()}
    for item in folder.glob("*-of-*.config.json"):
        if not item.name.startswith("._") and json.loads(item.read_text())["shards"] != shards:
            raise ValueError("Select a new output directory to change the readiness shard count")
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        if previous["config_sha256"] != signature or (previous["shard"],previous["shards"]) != (shard,shards):
            raise ValueError("The saved readiness configuration differs from this invocation")
        receipt = previous
    else:
        _write(config_path, canonical(receipt) + b"\n")
    completed = read_shard(path, indices, signature)
    data = path.read_bytes() if path.exists() else b""
    if len(completed) < len(indices):
        models = {"sender": FrozenLM(*MODELS[sender], device="cpu"),
                  "full": FrozenLM(*MODELS["smol"], device="cpu")}
        if any(str(model.model.dtype) != "torch.float32" or model.device.type != "cpu" for model in models.values()):
            raise ValueError("CPU audit requires FP32 checkpoints on CPU")
        for world_id in indices[len(completed):]:
            world = make_world("validation", world_id)
            rows = []
            for arm, book in ARMS:
                prompts, targets = inputs(world, arm, book)
                started = time.monotonic()
                answers = models[arm].answer_many(prompts, max_new_tokens=12)
                elapsed = time.monotonic() - started
                if len(answers) != VARIANTS:
                    raise RuntimeError("Incomplete readiness generation batch")
                for v, (prompt, target, answer) in enumerate(zip(prompts, targets, answers)):
                    rows.append({"split": "validation", "sender":sender, "world":world_id, "variant":v,
                                 "condition":arm, "rulebook":book, "category":world.category(v), "target":target,
                                 "prompt":prompt, "answer":answer, "first_answer":first_answer(answer),
                                 "correct":exact_answer(answer,target), "device":"cpu", "precision":"float32",
                                 "batch_seconds":elapsed,"config_sha256":signature})
            validate_world(rows,world_id,signature)
            data += b"".join(canonical(row) + b"\n" for row in rows)
            _write(path, data)
            completed.append(rows)
            print(f"readiness {stem}: recorded {len(completed)}/{len(indices)} worlds", flush=True)
            if on_world is not None:
                on_world()
    result = {"status":"measured", "stage":"readiness", "split":"validation", "worlds":len(completed),
              "rows":len(completed)*len(ARMS)*VARIANTS, "shard":shard,"shards":shards,"complete":True,
              "device":"cpu", "precision":"float32", "config_sha256":signature, "rows_sha256":_sha(data)}
    _write(folder / (stem + ".receipt.json"), canonical(result) + b"\n")
    return result


def generate(root, output_dir):
    from .readiness_report import generate as build
    return build(root, output_dir)
