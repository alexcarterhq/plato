"""Cache raw CounterFact generations on disjoint CPU workers before factual fitting."""
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import time

from .config import MODELS
from .facts import _correct, _load_facts, _part
from .models import FrozenLM
from .subjects import COUNTERFACT_SHA256, SUBJECTS_SHA256

BATCH_SIZE = 8
SEED = 7381


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _write(path, contents):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(contents)
    temporary.replace(path)


def _load(sender_id, data_path):
    path = Path(data_path)
    if sender_id not in ("qwen", "pythia") or hashlib.sha256(path.read_bytes()).hexdigest() != COUNTERFACT_SHA256:
        raise ValueError("Prescreen requires a pinned donor and the original CounterFact release")
    facts = _load_facts(path)
    if len(set(f.id for f in facts)) != len(facts):
        raise ValueError("Duplicate CounterFact case IDs")
    random.Random(SEED).shuffle(facts)
    package = Path(__file__).parent
    signature = {
        "dataset": COUNTERFACT_SHA256, "subject_ids": SUBJECTS_SHA256, "seed": SEED,
        "donor": list(MODELS[sender_id]), "recipient": list(MODELS["smol"]), "batch_size": BATCH_SIZE,
        "dtype": "float32", "device": "cpu", "max_new_tokens": 12, "do_sample": False,
        "versions": {key: importlib.metadata.version(key) for key in ("numpy","torch","transformers","tokenizers")},
        "sources": {name: hashlib.sha256((package / name).read_bytes()).hexdigest()
                    for name in ("facts.py", "models.py", "subjects.py", "prescreen.py")},
    }
    return facts, signature


def _paths(out, shard, shards):
    stem = f"{shard:04d}-of-{shards:04d}"
    directory = Path(out) / "prescreen"
    return tuple(directory / (stem + suffix) for suffix in (".config.json", ".jsonl", ".receipt.json"))


def _indices(facts, shard, shards, limit):
    total = math.ceil(len(facts) / BATCH_SIZE)
    if not 0 <= shard < shards <= total:
        raise ValueError("Invalid prescreen shard index or count")
    if limit is not None and (type(limit) is not int or not 1 <= limit <= total):
        raise ValueError("Global batch limit must be between one and the full number of batches")
    return range(shard, total if limit is None else limit, shards)


def _records(path, facts, indices, *, recover=False):
    if not path.exists():
        return []
    batches = []
    with path.open("rb+" if recover else "rb") as stream:
        while True:
            pos = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                if not recover:
                    raise ValueError(f"Incomplete batch log {path.name}")
                stream.truncate(pos)
                break
            batch = json.loads(line)
            if len(batches) >= len(indices) or batch["batch"] != indices[len(batches)]:
                raise ValueError(f"Unexpected batch sequence in {path.name}")
            start = BATCH_SIZE * batch["batch"]
            expected = [f.id for f in facts[start:start+BATCH_SIZE]]
            if [r.get("id") for r in batch["raw"]] != expected:
                raise ValueError("Prescreen case identities disagree with the locked shuffled source")
            if any(not isinstance(r.get(k), str) for r in batch["raw"] for k in ("sender_answer","receiver_answer")):
                raise ValueError("Incomplete raw generation in prescreen cache")
            batches.append(batch)
    return batches


def _digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _config(signature, shard, shards):
    return dict(protocol=signature, shard=shard, shards=shards)


def run(sender_id, data_path, output_dir, shard=0, shards=1, *, batch_limit=None, on_commit=None):
    """Extend one shard to a global batch limit. Receipts retain the actual raw answers."""
    start = time.monotonic()
    facts, signature = _load(sender_id, data_path)
    config, path, receipt = _paths(output_dir, shard, shards)
    indices = _indices(facts, shard, shards, batch_limit)
    config.parent.mkdir(parents=True, exist_ok=True)
    payload = _config(signature, shard, shards)
    if not config.exists() and (path.exists() or receipt.exists()):
        raise ValueError("Prescreen evidence exists without its checkpoint and data signature")
    if config.exists() and json.loads(config.read_bytes()) != payload:
        raise ValueError("Prescreen source, checkpoints, dependencies, or shard configuration changed")
    _write(config, _json(payload))
    path.touch(exist_ok=True)
    saved = _records(path, facts, indices, recover=True)
    if receipt.exists():
        prior = json.loads(receipt.read_bytes())
        covered = int(prior["bytes"])
        with path.open("rb") as stream:
            if hashlib.sha256(stream.read(covered)).hexdigest() != prior["rows_sha256"]:
                raise ValueError("Previously committed prescreen generations changed")
    offset = len(saved)
    if offset < len(indices):
        sender = FrozenLM(*MODELS[sender_id], device="cpu")
        receiver = FrozenLM(*MODELS["smol"], device="cpu")
    setup = time.monotonic() - start
    for batch_index in indices[offset:]:
        first = BATCH_SIZE * batch_index
        chunk = facts[first:first+BATCH_SIZE]
        texts = [f.prompt()[0] for f in chunk]
        t = time.monotonic()
        a = sender.answer_many(texts, max_new_tokens=12)
        b = receiver.answer_many(texts, max_new_tokens=12)
        elapsed = time.monotonic() - t
        if len(a) != len(chunk) or len(b) != len(chunk):
            raise ValueError("Both checkpoints must generate one answer for every prompt")
        batch = {"batch": batch_index, "seconds": elapsed, "raw": [
            {"id": f.id, "sender_answer": x, "receiver_answer": y} for f,x,y in zip(chunk,a,b)]}
        with path.open("ab") as stream:
            stream.write(_json(batch) + b"\n")
        saved.append(batch)
        if (len(saved) - offset) % 8 == 0:
            print(f"prescreen {sender_id} {shard}/{shards}: {len(saved)}/{len(indices)} batches", flush=True)
            if on_commit is not None:
                on_commit()
    n = sum(len(batch["raw"]) for batch in saved)
    wall = sum(batch["seconds"] for batch in saved)
    warm_rows = saved[1:]
    warm = sum(batch["seconds"] for batch in warm_rows)
    known = {part: 0 for part in ("train","validation","test")}
    for batch in saved:
        first = batch["batch"] * BATCH_SIZE
        for f, raw in zip(facts[first:first+BATCH_SIZE], batch["raw"]):
            if _correct(raw["sender_answer"], f.answers) and _correct(raw["receiver_answer"], f.answers):
                known[_part(f.group, SEED)] += 1
    result = {"sender": sender_id, "shard": shard, "shards": shards, "batch_limit": batch_limit,
              "total_batches": math.ceil(len(facts)/BATCH_SIZE), "complete": True, "batches": len(saved),
              "cases": n, "jointly_known": known, "generation_seconds": wall,
              "cases_per_second": n/wall if wall else None, "warm_seconds": warm,
              "warm_cases_per_second": (sum(len(batch["raw"]) for batch in warm_rows)/warm if warm else None),
              "setup_seconds": setup, "bytes": path.stat().st_size, "rows_sha256": _digest(path)}
    _write(receipt, _json(result))
    if on_commit is not None:
        on_commit()
    return result


def merge(sender_id, data_path, output_dir, shards, *, batch_limit=None):
    """Validate every receipt and materialize the exact format expected by run_facts."""
    facts, signature = _load(sender_id, data_path)
    out = Path(output_dir)
    records = {}
    for shard in range(shards):
        config, path, receipt = _paths(out, shard, shards)
        indices = _indices(facts, shard, shards, batch_limit)
        r = json.loads(receipt.read_bytes())
        if (json.loads(config.read_bytes()) != _config(signature, shard, shards)
                or not r.get("complete") or (r["shard"], r["shards"], r["sender"]) != (shard,shards,sender_id)
                or r["rows_sha256"] != _digest(path) or r["bytes"] != path.stat().st_size):
            raise ValueError("Prescreen receipt, signature or content mismatch")
        batches = _records(path, facts, indices)
        if len(batches) != len(indices) or r["batches"] != len(indices):
            raise ValueError("Incomplete CounterFact prescreen coverage")
        for batch in batches:
            for row in batch["raw"]:
                records[row["id"]] = row
    target = out / "screen.jsonl"
    if target.exists():
        for line in target.read_bytes().splitlines():
            row = json.loads(line)
            if records.get(row["id"]) != row:
                raise ValueError("Existing screen cache disagrees with the verified CPU responses")
    covered = min(len(facts), BATCH_SIZE * batch_limit) if batch_limit is not None else len(facts)
    if len(records) != covered:
        raise ValueError("Missing unique case IDs")
    _write(target, b"".join(_json(records[f.id]) + b"\n" for f in facts[:covered]))
    return {"cases": covered, "full": covered == len(facts), "screen_sha256": _digest(target)}
