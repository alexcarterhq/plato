"""Run a single frozen-protocol experiment stage and validate its prerequisites."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np

from .config import MODELS
from .facts import run_facts
from .models import Bridge, FrozenLM
from .trials import decode_packet, encode_packet, evaluate_world
from .worlds import exact_answer, make_world


PACKAGE = Path(__file__).resolve().parent
LOCKED_SOURCES = ("protocol.md", "config.py", "worlds.py", "models.py", "facts.py",
                  "trials.py", "pipeline.py", "subjects.py", "assets/counterfact-subjects.tsv")
CONDITIONS = {"factual", "factual_wrong", "contextual", "contextual_wrong",
              "none", "self", "full", "word", "code", "sender"}
SHARDABLE = {"test", "robustness"}
STAGES = SHARDABLE | {"facts", "calibrate", "validate", "bench"}
SIZES = {False: {"train": 512, "validation": 64, "test": 250, "robustness": 64},
         True: {"train": 8, "validation": 2, "test": 4, "robustness": 2}}


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(path):
    return json.loads(path.read_text())


def _save(path, payload):
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_bytes(_canonical(payload) + b"\n")
    tmp.replace(path)


def _sources():
    return {name: _sha(PACKAGE / name) for name in LOCKED_SOURCES}


def _versions():
    return {name: importlib.metadata.version(name) for name in
            ("numpy", "torch", "transformers", "tokenizers")}


def _lock_payload(out, sender, quick):
    artifacts = ("factual.json", "factual-spec.json", "factual-bridge.npz", "contextual-bridge.npz")
    facts = _json(out / "factual.json")
    calibration_spec = _json(out / "factual-spec.json")
    expected = (list(MODELS[sender]), list(MODELS["smol"]))
    if (calibration_spec != facts["protocol"] or
            (calibration_spec.get("sender"),calibration_spec.get("receiver")) != expected
            or bool(facts.get("quick")) != bool(quick)):
        raise ValueError("The factual calibration belongs to a different model pair or mode")
    sources = _sources()
    if calibration_spec.get("source_sha256") != {k:sources[k] for k in ("facts.py","models.py","subjects.py","assets/counterfact-subjects.tsv")}:
        raise ValueError("Factual calibration code changed before the lock was frozen")
    return {
        "sender": sender, "receiver": "smol", "quick": bool(quick), "world_counts": SIZES[quick],
        "sender_checkpoint": list(MODELS[sender]), "receiver_checkpoint": list(MODELS["smol"]),
        "sender_layer": facts["sender_layer"], "receiver_layer": facts["receiver_layer"],
        "factual_gate_passed": facts["gate_passed"], "data_sha256": facts["protocol"]["data_sha256"],
        "artifacts": {name: _sha(out / name) for name in artifacts},
        "source_sha256": sources, "dependencies": _versions(),
        "controls": {"sender": .95, "full": .95, "self": .80, "identity_tolerance": 1e-4},
    }


def _lock(out, sender, quick, *, create=False):
    payload = _lock_payload(out, sender, quick)
    payload["lock_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    path = out / "lock.json"
    if path.exists():
        if _json(path) != payload:
            raise ValueError("Lock mismatch in source, artifacts, environment or protocol")
    elif create:
        _save(path, payload)
    else:
        raise FileNotFoundError("Calibrate and freeze the lock before evaluation")
    return payload


def _models(sender):
    return FrozenLM(*MODELS[sender]), FrozenLM(*MODELS["smol"])


def _synchronize():
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _bench(sender_id, quick):
    start = time.monotonic()
    sender, receiver = _models(sender_id)
    ls, lr = round((sender.depth-1)*.32), round((receiver.depth-1)*.32)
    warmup = make_world("bench", 0)
    text = warmup.sender_text(0)
    d1, d2 = len(sender.capture(text, ls)), len(receiver.capture(text, lr))
    rank = min(d1, d2)
    # Shape-matched full-width multiplications exercise the actual packet path.
    # These fixed synthetic maps have no scientific interpretation.
    mock = Bridge(np.zeros(d1, "f"), np.zeros(d2, "f"), np.eye(d1, rank, dtype="f"),
                  np.eye(rank, d2, dtype="f"), 1.)
    bridges = {"factual": mock, "contextual": mock}
    deadline_label = "benchmark-only"
    _synchronize()
    load_seconds = time.monotonic() - start
    evaluate_world(sender, receiver, warmup, sender_id, ls, lr, bridges, deadline_label)
    _synchronize()
    t = time.monotonic()
    world = make_world("bench", 1)
    sample = evaluate_world(sender, receiver, world, sender_id, ls, lr, bridges, deadline_label)
    _synchronize()
    seconds = time.monotonic() - t
    _validate_world(sample, "bench", 1, sender_id, deadline_label)
    return dict(status="timing_only", quick=quick, worlds=1, warm_seconds=seconds,
                load_seconds=load_seconds, timed_condition_rows=len(sample))


def _calibrate(out, sender_id, quick):
    if (out / "lock.json").exists():
        return _lock(out, sender_id, quick)
    facts = _json(out / "factual.json")
    if facts["sender_layer"] is None or facts["receiver_layer"] is None:
        return _diagnostic(out, "insufficient_eligible_facts", sender_id, quick)
    ls, lr = facts["sender_layer"], facts["receiver_layer"]
    sender, receiver = _models(sender_id)
    count = SIZES[quick]["train"]
    pieces = out / "calibration"
    pieces.mkdir(exist_ok=True)
    cache_key = hashlib.sha256(_canonical([_sources(), _versions(), _sha(out/"factual.json"),
                                            list(MODELS[sender_id]), list(MODELS["smol"]), quick])).hexdigest()
    x, y = [], []
    for start in range(0, count, 8):
        final = pieces / f"worlds-{start:04d}-{cache_key[:16]}.npz"
        if not final.exists():
            texts = [make_world("train", w).sender_text(v)
                     for w in range(start,min(count,start+8)) for v in range(8)]
            chunk_x, chunk_y = [], []
            for i in range(0,len(texts),8):
                prompts = texts[i:i+8]
                ends = [None] * len(prompts)
                chunk_x.append(sender.capture_many(prompts, [ls], ends)[ls])
                chunk_y.append(receiver.capture_many(prompts, [lr], ends)[lr])
            tmp = final.with_suffix(".partial")
            with tmp.open("wb") as f:
                np.savez(f, x=np.concatenate(chunk_x), y=np.concatenate(chunk_y))
            tmp.replace(final)
        with np.load(final, allow_pickle=False) as data:
            x.append(data["x"])
            y.append(data["y"])
    bridge = Bridge.fit(np.concatenate(x), np.concatenate(y))
    bridge.save(out / "contextual-bridge.npz")
    return _lock(out, sender_id, quick, create=True)


def _diagnostic(out, reason, sender_id, quick):
    result = dict(status="diagnostic", reason=reason, sender=sender_id, quick=bool(quick))
    _save(out / "diagnostic.json", result)
    return result


def _validate_world(rows, split, index, sender, digest):
    expected = {(book, v, lane) for book in (0, 1) for v in range(8) for lane in CONDITIONS}
    keys = set()
    for row in rows:
        if (row.get("world"), row.get("split"), row.get("sender"), row.get("lock_sha256")) != (
                index, split, sender, digest):
            raise ValueError("Prediction provenance disagrees with its frozen world")
        key = (row["rulebook"],row["variant"],row["condition"])
        if key in keys or type(row["correct"]) is not bool:
            raise ValueError("Duplicate prediction or invalid correctness field")
        keys.add(key)
    if keys != expected:
        raise ValueError(f"Incomplete evidence for {split} world {index}")


def _recover(path, indices, split, sender, digest, *, partial):
    """Read complete world-sized write units. A torn final world may be replayed."""
    nrows = len(CONDITIONS) * 16
    done = []
    if not path.exists():
        return done
    with path.open("rb+" if partial else "rb") as f:
        for index in indices:
            start = f.tell()
            block = [f.readline() for _ in range(nrows)]
            if any(not line.endswith(b"\n") for line in block):
                if partial:
                    f.truncate(start)
                    return done
                raise ValueError("Incomplete committed evidence shard")
            rows = [json.loads(line) for line in block]
            _validate_world(rows, split, index, sender, digest)
            done.append(rows)
        if f.read(1):
            raise ValueError("Unexpected extra rows in evidence shard")
    return done


def _scores(completed):
    rates = {}
    for book in (0, 1):
        rates[str(book)] = {}
        for name in ("sender","full","self"):
            answers = [row["correct"] for sample in completed for row in sample
                       if row["rulebook"] == book and row["condition"] == name]
            rates[str(book)][name] = sum(answers)/len(answers)
    return rates


def _valid_controls(rates, identity):
    return (np.isfinite(identity) and 0 <= identity <= 1e-4
            and all(r[lane] >= cutoff for r in rates.values()
                    for lane,cutoff in (("sender",.95),("full",.95),("self",.80))))


def _validation(out, lock, sender_id, quick):
    manifest_path = out / "validation.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Finish validation before unsealing the held-out worlds")
    manifest = _json(manifest_path)
    path = out / "validation" / "0000-of-0001.jsonl"
    indices = list(range(SIZES[quick]["validation"]))
    if (manifest["lock_sha256"] != lock["lock_sha256"] or manifest["sender"] != sender_id
            or manifest["quick"] != quick or manifest.get("complete") is not True
            or manifest["worlds"] != len(indices) or _sha(path) != manifest["rows_sha256"]):
        raise ValueError("Validation receipt disagrees with the sealed evidence")
    rows = _recover(path, indices, "validation", sender_id, lock["lock_sha256"], partial=False)
    if _scores(rows) != manifest["rates"]:
        raise ValueError("Validation summary disagrees with the recorded controls")
    if bool(_valid_controls(manifest["rates"],manifest["identity_max_abs"])) != manifest["controls_valid"]:
        raise ValueError("Validation decision disagrees with the registered thresholds")
    return manifest


# Fixed formatting demonstrations for diagnosing public-text prerequisites.
_RETRIEVAL_DEMOS = (
    "Astra is amber.\nRead Astra's category. |\nCategory: amber\n\n"
    "Beryl is blue.\nRead Beryl's category. |\nCategory: blue\n\n"
)
_ROUTING_DEMOS = (
    "Category -> destination:\namber -> 2\nblue -> 1\nRecords:\nAstra is amber.\n"
    "Read Astra's category. |\nDestination: 2\n\n"
    "Category -> destination:\namber -> 2\nblue -> 1\nRecords:\nBeryl is blue.\n"
    "Read Beryl's category. |\nDestination: 1\n\n"
)


def _diagnose_interface(out, sender_id, quick, lock, validation):
    """Explore the interface within known validation worlds, leaving primary artifacts intact."""
    summary = out / "interface-diagnostic.json"
    digest = lock["lock_sha256"]
    if summary.exists():
        previous = _json(summary)
        if previous.get("lock_sha256") != digest:
            raise ValueError("Diagnostic receipt belongs to another experiment")
        return previous

    folder = out / "diagnostics"
    folder.mkdir(exist_ok=True)
    sender, receiver = _models(sender_id)
    selected = lock["receiver_layer"]
    spec = _json(out/"factual-spec.json")
    candidates = sorted({i for i in (*spec.get("receiver_layers",[]),selected,selected-1,selected+1)
                         if 0 <= i < receiver.depth})
    count = min(8,SIZES[quick]["validation"])
    text_failed = any(r["sender"] < .95 or r["full"] < .95 for r in validation["rates"].values())
    source_path = out/"validation"/"0000-of-0001.jsonl"
    validation_rows = _recover(source_path, list(range(SIZES[quick]["validation"])), "validation",
                               sender_id, digest, partial=False)
    totals = {}
    for index in range(count):
        path = folder/f"world-{index:04d}.json"
        if path.exists():
            cached = _json(path)
            if cached["lock_sha256"] != digest or cached["world"] != index:
                raise ValueError("Diagnostic world belongs to another experiment")
            records = cached["records"]
        else:
            world = make_world("validation",index)
            records = []
            for row in validation_rows[index]:
                if row["condition"] in ("sender","full","word","self"):
                    records.append(dict(arm="reference_"+row["condition"], layer=None, readout=None,
                                        rulebook=row["rulebook"],variant=row["variant"],
                                        answer=row.get("answer"),target=row.get("target"),correct=row["correct"]))
            def measure(arm, layer, site, book, answers, targets):
                if len(answers) != 8:
                    raise ValueError("An exploratory condition omitted a counterfactual")
                for v,(answer,target) in enumerate(zip(answers,targets)):
                    records.append(dict(arm=arm,layer=layer,readout=site,rulebook=book,variant=v,
                                        answer=answer,target=target,correct=exact_answer(answer,target)))
            private = [world.sender_text(v) for v in range(8)]
            ends = []
            for v, text in enumerate(private):
                binding = f"{world.entity} is {world.category(v)}."
                if text.count(binding) != 1:
                    raise ValueError("Could not locate unique lexical endpoint in validation evidence")
                ends.append(text.index(binding) + len(binding) - 1)
            for site, positions in (("neutral_marker",[None]*8),("private_category",ends)):
                states = receiver.capture_many(private,candidates,positions)
                for layer in candidates:
                    packets = [decode_packet(encode_packet(x)) for x in states[layer]]
                    for book in (0,1):
                        prompt,end = world.receiver_text(book)
                        targets = [world.destination(book,v) for v in range(8)]
                        for label,order in (("self",range(8)),("self_wrong",world.wrong)):
                            patches = [(layer,end,packets[i]) for i in order]
                            answers = receiver.answer_many([prompt]*8,patches)
                            measure(label,layer,site,book,answers,targets)
            if text_failed:
                retrieval = sender.answer_many([_RETRIEVAL_DEMOS+world.retrieval_text(v) for v in range(8)])
                for book in (0,1):
                    measure("few_shot_sender",None,None,book,retrieval,[world.category(v) for v in range(8)])
                    prompts = [_ROUTING_DEMOS+world.receiver_text(book,evidence=world.evidence(v))[0]
                               for v in range(8)]
                    answers = receiver.answer_many(prompts)
                    measure("few_shot_full",None,None,book,answers,[world.destination(book,v) for v in range(8)])
            _save(path,dict(lock_sha256=digest,world=index,source_split="validation",records=records))
        for row in records:
            key = (row["arm"],row["layer"],row["readout"],row["rulebook"])
            successes, n = totals.get(key,(0,0))
            totals[key] = (successes + int(row["correct"]), n+1)
    arms = [dict(arm=arm,layer=layer,readout=site,rulebook=book,correct=success,items=n,
                 accuracy=success/n) for (arm,layer,site,book),(success,n) in totals.items()]
    result = dict(status="validation_interface_diagnostic",sender=sender_id,quick=bool(quick),
                  lock_sha256=digest,worlds=count,source_split="validation",heldout_opened=False,
                  reference_layer=selected,candidate_layers=candidates,
                  few_shot_text_run=text_failed,arms=arms)
    _save(summary,result)
    return result


def run_stage(stage, sender, output_dir, data_path=None, shard=0, shards=1, quick=False):
    """Run at most one stage/shard. Full and quick modes require separate roots."""
    if stage not in STAGES or sender not in {"qwen","pythia"}:
        raise ValueError("Unknown stage or sender")
    if not 0 <= shard < shards or (stage not in SHARDABLE and shards != 1):
        raise ValueError("Only held-out test and robustness stages may be sharded")
    if stage in SHARDABLE and shards > SIZES[quick][stage]:
        raise ValueError("Shard count exceeds the number of independent worlds")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if stage == "bench":
        result = _bench(sender, quick)
        _save(out/"benchmark.json", result)
        return result
    if stage == "facts":
        if data_path is None:
            raise ValueError("Provide counterfact.json for factual calibration")
        donor, receiver = _models(sender)
        return run_facts(donor, receiver, data_path, out, quick=quick)
    if stage == "calibrate":
        return _calibrate(out,sender,quick)
    if not (out / "lock.json").exists() and (out / "diagnostic.json").exists():
        return _json(out / "diagnostic.json")
    lock = _lock(out,sender,quick)
    digest = lock["lock_sha256"]
    if stage == "validate" and (out/"validation.json").exists():
        return _validation(out,lock,sender,quick)
    if stage in SHARDABLE:
        val = _validation(out,lock,sender,quick)
        if not val["controls_valid"]:
            if stage == "robustness":
                if shard:
                    return dict(status="skipped_diagnostic_shard", sender=sender, shard=shard)
                return _diagnose_interface(out,sender,quick,lock,val)
            return _diagnostic(out, "validation_controls_failed", sender, quick)
    split = "validation" if stage == "validate" else stage
    indices = list(range(shard,SIZES[quick][split],shards))
    folder = out / split
    folder.mkdir(exist_ok=True)
    final = folder / f"{shard:04d}-of-{shards:04d}.jsonl"
    for path in folder.iterdir():
        if path.name.startswith("._"):
            continue
        if "-of-" in path.name and path.name.split("-of-")[-1].split(".")[0] != f"{shards:04d}":
            raise ValueError("A different shard layout already exists in this run")
    partial = final.with_suffix(".partial")
    committed = final.exists()
    completed = _recover(final if committed else partial, indices, split, sender, digest,
                         partial=not committed)
    donor = receiver = None
    need_models = not committed or stage == "validate"
    if need_models:
        donor, receiver = _models(sender)
    ls, lr = lock["sender_layer"], lock["receiver_layer"]
    if not committed:
        bridges = {key:Bridge.load(out/f"{key}-bridge.npz") for key in ("factual","contextual")}
        with partial.open("ab") as f:
            for i in indices[len(completed):]:
                sample = evaluate_world(donor,receiver,make_world(split,i),sender,ls,lr,bridges,digest)
                _validate_world(sample,split,i,sender,digest)
                for row in sample:
                    row["quick"] = bool(quick)
                f.write(b"".join(_canonical(row)+b"\n" for row in sample))
                f.flush()
                os.fsync(f.fileno())
                completed.append(sample)
        partial.replace(final)
    result = dict(status="measured", split=split, quick=bool(quick), sender=sender,
                  worlds=len(completed), complete=True, lock_sha256=digest, rows_sha256=_sha(final))
    if stage == "validate":
        world = make_world("validation",0)
        identity = max(receiver.identity_error(*_patch_location(world,b,lr)) for b in (0,1))
        rates = _scores(completed)
        result.update(identity_max_abs=float(identity), rates=rates,
                      controls_valid=bool(_valid_controls(rates,identity)))
        _save(out/"validation.json",result)
        if not result["controls_valid"]:
            _diagnostic(out, "validation_controls_failed", sender, quick)
    return result


def _patch_location(world, book, layer):
    text, end = world.receiver_text(book)
    return text,layer,end
