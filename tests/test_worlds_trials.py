import hashlib
import json

import numpy as np
import pytest

from plato.cli import create_bundle, unpack_bundle
from plato.trials import decode_packet, encode_packet, evaluate_world
from plato.worlds import CATEGORIES, exact_answer, make_world


class Harness:
    def capture_many(self, texts, layers, ends):
        return {layers[0]: np.eye(8, dtype="f")}

    def answer_many(self, texts, patches=None, *, max_new_tokens=12):
        if patches is None:
            return ["  1\nrecord"]*len(texts)
        return [str(int(np.argmax(vector))+1) for _, _, vector in patches]


class Identity:
    def apply(self, states):
        return states


def test_randomization_and_records():
    w = make_world("smoke", 12)
    assert w == make_world("smoke", 12)
    assert w != make_world("validation", 12)
    assert all(w.wrong[i] != i and w.books[0][i] != w.books[1][i] for i in range(8))
    e0, e1 = w.evidence(0).splitlines(), w.evidence(1).splitlines()
    assert len([1 for a,b in zip(e0,e1) if a != b]) == 1
    assert w.entity in e0[w.slot]
    assert w.sender_text(0).endswith(" |")
    assert not exact_answer("1 or 2", "1") and exact_answer("\n 1.\nmore", "1")


def test_factual_packet_and_wrong_control_provenance():
    w = make_world("bench", 3)
    rows = evaluate_world(Harness(), Harness(), w, "qwen", 0, 0,
                          {"factual": Identity(), "contextual": Identity()}, "test-digest")
    index = {(r["condition"],r["rulebook"],r["variant"]):r for r in rows}
    assert len(index) == len(rows) == 160
    for v in range(8):
        correct = index["factual",0,v]
        assert correct["packet_sha256"] == index["factual",1,v]["packet_sha256"]
        assert correct["target"] != index["factual",1,v]["target"]
        assert index["factual_wrong",0,v]["packet_sha256"] == index["factual",0,w.wrong[v]]["packet_sha256"]
        assert index["code",0,v]["correct"] and index["code",0,v]["packet_bytes"] == 1
        assert index["word",0,v]["packet_sha256"] == hashlib.sha256(b"1").hexdigest()
        assert w.category(v) == CATEGORIES[v]
    assert len({r["receiver_prompt"] for r in rows if r["condition"]=="factual" and r["rulebook"]==0}) == 1


def test_wire_equals_torch_bfloat16_cast():
    torch = pytest.importorskip("torch")
    x = np.random.default_rng(42).normal(size=2048).astype(np.float32)
    raw = encode_packet(x)
    assert len(raw) == 4096
    np.testing.assert_array_equal(decode_packet(raw), torch.tensor(x).bfloat16().float().numpy())


def test_evidence_archive_is_reproducible_and_excludes_caches(tmp_path):
    source = tmp_path / "data" / "qwen" / "full"
    source.mkdir(parents=True)
    (source/"factual.json").write_text(json.dumps({"status":"fixture"}))
    (source/"factual-bridge.npz").write_bytes(b"omitted")
    a, b = tmp_path/"a.gz", tmp_path/"b.gz"
    assert create_bundle(source.parents[1], a) == create_bundle(source.parents[1], b)
    unpack_bundle(a, tmp_path/"out")
    assert (tmp_path/"out/qwen/full/factual.json").exists()
    assert not (tmp_path/"out/qwen/full/factual-bridge.npz").exists()
