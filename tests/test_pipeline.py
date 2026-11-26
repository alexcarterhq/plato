import hashlib
import json

import numpy as np
import pytest

from plato import pipeline
from plato.models import Bridge


class FakeLM:
    depth = 4

    def capture_many(self, texts, layers, ends):
        data = np.array([list(hashlib.sha256(t.encode()).digest()[:4]) for t in texts], dtype="f")
        return {layer: data for layer in layers}

    def capture(self, text, layer):
        return self.capture_many([text], [layer], [None])[layer][0]

    def identity_error(self, text, layer, char_end):
        return 0.

    def answer_many(self, texts, patches=None):
        return ["amber"] * len(texts)


@pytest.fixture
def environment(monkeypatch, tmp_path):
    # Stub only external model computation. All disk, lock, world and resume logic is real.
    monkeypatch.setattr(pipeline, "_models", lambda sender: (FakeLM(), FakeLM()))
    monkeypatch.setattr(pipeline, "_sources", lambda: {"protocol.md":"test source", "facts.py":"fact source", "models.py":"model source", "subjects.py":"identity code", "assets/counterfact-subjects.tsv":"identity table"})
    monkeypatch.setattr(pipeline, "_versions", lambda: {"numpy":np.__version__})
    monkeypatch.setattr(pipeline, "_synchronize", lambda: None)
    calls = []
    def trial(s, r, w, sender, ls, lr, bridges, digest):
        calls.append((w.split,w.index))
        assert set(bridges) == {"factual","contextual"}
        return [dict(world=w.index, split=w.split, sender=sender, lock_sha256=digest,
                     rulebook=b, variant=v, condition=lane, correct=True)
                for b in (0,1) for v in range(8) for lane in sorted(pipeline.CONDITIONS)]
    monkeypatch.setattr(pipeline, "evaluate_world", trial)
    x = np.random.default_rng(77).normal(size=(16,4))
    Bridge.fit(x,x).save(tmp_path/"factual-bridge.npz")
    protocol = {"data_sha256": "test data", "sender": list(pipeline.MODELS["qwen"]),
                "receiver": list(pipeline.MODELS["smol"]), "source_sha256": {
                "facts.py":"fact source", "models.py":"model source", "subjects.py":"identity code", "assets/counterfact-subjects.tsv":"identity table"}}
    (tmp_path/"factual-spec.json").write_text(json.dumps(protocol))
    (tmp_path/"factual.json").write_text(json.dumps({
        "sender_layer":1, "receiver_layer":2, "gate_passed":False,
        "protocol":protocol, "quick":True,
    }))
    return tmp_path,calls


def run(stage, path, **kw):
    return pipeline.run_stage(stage, "qwen", path, quick=True, **kw)


def test_lock_precedes_heldout_and_shards_are_repeatable(environment):
    path,calls = environment
    lock = run("calibrate",path)
    assert lock["factual_gate_passed"] is False
    assert run("calibrate",path) == lock
    with pytest.raises(FileNotFoundError, match="validation"):
        run("test",path)
    validation = run("validate",path)
    assert validation["complete"] and validation["controls_valid"]
    assert validation["quick"] and validation["worlds"] == 2
    assert run("validate",path) == validation
    initial = run("test",path,shard=0,shards=2)
    assert initial["worlds"] == 2 and initial["lock_sha256"] == lock["lock_sha256"]
    assert run("test",path,shard=0,shards=2) == initial
    run("test",path,shard=1,shards=2)
    assert sorted(w for split,w in calls if split=="test") == [0,1,2,3]
    with pytest.raises(ValueError,match="layout"):
        run("test",path,shard=0,shards=1)
    # Lock detection also applies to already completed shards.
    with (path/"contextual-bridge.npz").open("ab") as f:
        f.write(b"tampered")
    with pytest.raises(ValueError, match="Lock mismatch"):
        run("test",path,shard=1,shards=2)


def test_failed_validation_keeps_both_heldout_sets_sealed(environment,monkeypatch):
    path,calls = environment
    previous = pipeline.evaluate_world
    def weak(*args):
        rows = previous(*args)
        for row in rows:
            row["correct"] = row["condition"] not in {"self","sender"}
        return rows
    monkeypatch.setattr(pipeline, "evaluate_world", weak)
    run("calibrate",path)
    val = run("validate",path)
    assert val["complete"] and not val["controls_valid"]
    assert run("test",path)["reason"] == "validation_controls_failed"
    frozen = (path/"lock.json").read_bytes()
    diag = run("robustness",path)
    assert diag["status"] == "validation_interface_diagnostic" and diag["worlds"] == 2
    assert diag["source_split"] == "validation" and not diag["heldout_opened"]
    assert diag["few_shot_text_run"] and (path/"lock.json").read_bytes() == frozen
    assert {row["readout"] for row in diag["arms"]} == {None,"neutral_marker","private_category"}
    assert any(row["arm"] == "few_shot_full" and row["items"] == 16 for row in diag["arms"])
    assert all(row["items"] == 16 for row in diag["arms"])
    assert run("robustness",path) == diag
    assert not (path/"test").exists() and not (path/"robustness").exists()
    assert all(split == "validation" for split,_ in calls)


def test_restart_after_interruption_replays_only_partial_world(environment,monkeypatch):
    path,calls = environment
    run("calibrate",path)
    run("validate",path)
    previous = pipeline.evaluate_world
    def interrupted(*args):
        if args[2].split == "test" and args[2].index == 1:
            raise RuntimeError("simulated interruption")
        return previous(*args)
    monkeypatch.setattr(pipeline,"evaluate_world",interrupted)
    with pytest.raises(RuntimeError,match="simulated interruption"):
        run("test",path)
    partial = next((path/"test").glob("*.partial"))
    with partial.open("ab") as f:
        f.write(b'{"torn":true')
    monkeypatch.setattr(pipeline,"evaluate_world",previous)
    assert run("test",path)["worlds"] == 4
    assert [w for split,w in calls if split=="test"] == [0,1,2,3]
    with (path/"validation"/"0000-of-0001.jsonl").open("ab") as f:
        f.write(b" ")
    with pytest.raises(ValueError,match="Validation receipt"):
        run("robustness",path)


def test_bench_warms_up_a_different_world_and_rejects_invalid_shards(environment):
    path,calls = environment
    result = run("bench",path)
    assert result["status"] == "timing_only" and result["worlds"] == 1
    assert result["warm_seconds"] > 0 and result["timed_condition_rows"] == 160
    assert calls == [("bench",0),("bench",1)]
    with pytest.raises(ValueError,match="sharded"):
        run("calibrate",path,shards=2)
