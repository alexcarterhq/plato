"""No checkpoint or Modal resources are used by these deterministic CPU audit fixtures."""
from contextlib import redirect_stdout
import gzip
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from plato import readiness
from plato.readiness_report import generate, load, stats


class _FixtureModel:
    def __init__(self, repo, revision, device):
        assert device=="cpu" and len(revision)==40
        self.device=SimpleNamespace(type="cpu")
        self.model=SimpleNamespace(dtype="torch.float32")

    def answer_many(self,texts,max_new_tokens):
        assert len(texts)==8 and max_new_tokens==12
        return ["amber\nAnother line" if p.endswith("Category:") else "1!\nAnother line" for p in texts]


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch.object(readiness,"FrozenLM",_FixtureModel).start()
        patch.object(readiness.importlib.metadata,"version",return_value="qa-fixture").start()

    def test_resumes_after_each_complete_world_and_detects_changes(self):
        with TemporaryDirectory() as td, redirect_stdout(io.StringIO()):
            root=Path(td)
            with self.assertRaisesRegex(RuntimeError,"stop"):
                readiness.run("qwen",root,shard=0,shards=32,
                              on_world=lambda: (_ for _ in ()).throw(RuntimeError("stop")))
            rows,man=load(root)
            self.assertEqual(len(rows),24)
            self.assertEqual({r["world"] for r in rows},{0})
            self.assertFalse(man[0]["final_receipt"])
            partial=generate(root,root/"release")
            self.assertFalse(partial["complete"])
            self.assertEqual(partial["lanes"]["sender"]["low"],None)
            done=readiness.run("qwen",root,shard=0,shards=32)
            self.assertEqual(done["rows"],48)
            with patch.object(readiness,"FrozenLM",side_effect=AssertionError("reloaded models")):
                self.assertEqual(readiness.run("qwen",root,shard=0,shards=32),done)
            rows,man=load(root)
            self.assertEqual(set(r["world"] for r in rows),{0,32})
            self.assertTrue(man[0]["final_receipt"])
            with self.assertRaisesRegex(ValueError,"shard count"):
                readiness.run("qwen",root,shard=0,shards=2)
            f=root/"readiness"/"0000-of-0032.jsonl"
            row=json.loads(f.read_text().splitlines()[0])
            row["prompt"]="changed"
            f.write_text("\n".join([json.dumps(row)]+f.read_text().splitlines()[1:])+"\n")
            with self.assertRaisesRegex(ValueError,"Invalid readiness observation"):
                load(root)

    def test_full_cohort_and_registered_scoring(self):
        with TemporaryDirectory() as td, redirect_stdout(io.StringIO()):
            root=Path(td)
            readiness.run("qwen",root)
            result=generate(root,root/"release")
            self.assertEqual((result["worlds"],result["rows"]),(64,1536))
            self.assertEqual(result["decision"],"prerequisites_failed")
            for lane in result["lanes"].values():
                self.assertEqual(lane["observations"],512)
                self.assertEqual((lane["correct"],lane["low"],lane["high"]),(64,.125,.125))
            for filename in ("index.html","readiness.svg","readiness.png","readiness-data.js","provenance.json","release-note.md"):
                self.assertTrue((root/"release"/filename).exists())
            with gzip.open(root/"release"/"observations.jsonl.gz","rt") as f:
                raw=list(map(json.loads,f))
            self.assertIn("Another line",raw[0]["answer"])
            self.assertNotIn("Another line",raw[0]["first_answer"])
            self.assertEqual(stats([r for r in raw if r["world"]<5])["decision"],"partial_threshold_out_of_reach")
            # Different pass/fail branch under a controlled, explicitly hypothetical score vector.
            self.assertEqual(stats([dict(r,correct=True) for r in raw])["decision"],"prerequisites_met")
