import json
from pathlib import Path
from unittest.mock import patch

import pytest

from plato import prescreen
from plato.facts import Fact


class FixtureModel:
    def __init__(self, repo, revision, device):
        self.repo = repo
        assert device == "cpu"

    def answer_many(self, prompts, max_new_tokens=12):
        assert max_new_tokens == 12
        return [text.split()[-1] for text in prompts]


def inputs():
    return [Fact(str(i), f"S{i}", f"Q{i}", "R", "Capital {}", (f"S{i}",)) for i in range(26)]


def load_fixture(*_):
    return inputs(), {"fixture": "v1"}


@pytest.fixture
def frozen(monkeypatch):
    monkeypatch.setattr(prescreen, "_load", load_fixture)
    monkeypatch.setattr(prescreen, "FrozenLM", FixtureModel)


def test_progressive_shards_preserve_source_order(frozen, tmp_path):
    calls = []
    for i in range(3):
        prescreen.run("qwen", None, tmp_path, i, 3, batch_limit=2, on_commit=lambda: calls.append(i))
    pilot = prescreen.merge("qwen", None, tmp_path, 3, batch_limit=2)
    assert not pilot["full"] and pilot["cases"] == 16
    for i in range(3):
        stat = prescreen.run("qwen", None, tmp_path, i, 3)
        assert stat["cases"] == sum(stat["jointly_known"].values())
    full = prescreen.merge("qwen", None, tmp_path, 3)
    assert full["full"] and full["cases"] == 26
    observed = [json.loads(line) for line in (tmp_path/"screen.jsonl").read_bytes().splitlines()]
    assert observed == [dict(id=f.id, sender_answer=f.subject, receiver_answer=f.subject) for f in inputs()]
    assert calls == [0,1,2]
    with patch.object(prescreen, "FrozenLM", side_effect=AssertionError("unneeded load")):
        assert prescreen.run("qwen", None, tmp_path, 1, 3)["complete"]


def test_torn_and_modified_cache(frozen, tmp_path):
    prescreen.run("qwen", None, tmp_path, batch_limit=1)
    path = prescreen._paths(tmp_path, 0, 1)[1]
    with path.open("ab") as file:
        file.write(b'{"incomplete":')
    prescreen.run("qwen", None, tmp_path)
    path.write_bytes(path.read_bytes().replace(b'"sender_answer":"S0"', b'"sender_answer":"BAD"'))
    with pytest.raises(ValueError, match="mismatch"):
        prescreen.merge("qwen", None, tmp_path, 1)
    with pytest.raises(ValueError, match="changed"):
        prescreen.run("qwen", None, tmp_path)


def test_partial_shards_and_conflicting_existing_rows(frozen, tmp_path):
    prescreen.run("qwen", None, tmp_path, 0, 2)
    with pytest.raises(FileNotFoundError):
        prescreen.merge("qwen", None, tmp_path, 2)
    prescreen.run("qwen", None, tmp_path, 1, 2)
    (tmp_path/"screen.jsonl").write_text('{"id":"0","sender_answer":"bad","receiver_answer":"S0"}\n')
    with pytest.raises(ValueError, match="disagrees"):
        prescreen.merge("qwen", None, tmp_path, 2)
