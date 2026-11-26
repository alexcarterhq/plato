"""Report numerical and artifact tests. All predictions here are synthetic QA fixtures."""
import gzip
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from plato import report


def fixture(n=2):
    records = {}
    for world in range(n):
        for book in (0, 1):
            for variant in range(8):
                target = str((variant + book) % 8)
                alt = str((variant + 1 + book) % 8)
                for condition in report.CONDITIONS:
                    wanted = f"c{variant}" if condition == "sender" else target
                    answer = alt if condition.endswith("_wrong") else wanted
                    row = dict(split="test", sender="qwen", world=world, variant=variant,
                               rulebook=book, condition=condition, category=f"c{variant}",
                               wrong_category=f"c{(variant + 1) % 8}", wrong_target=alt,
                               target=wanted, answer=answer, correct=answer == wanted,
                               packet_bytes=64, latency_s=.2, gpu_s=.1, sender_prompt=f"Category c{variant}",
                               receiver_prompt=f"Independent book {book}", packet_sha256=f"{variant:064x}",
                               lock_sha256="a" * 64)
                    if condition.endswith("_wrong"):
                        row["wrong_correct"] = True
                        row["packet_sha256"] = f"{(variant + 1) % 8:064x}"
                    records[(world, book, variant, condition)] = row
    return records


VALIDATION = {"complete": True, "quick": False, "controls_valid": True, "identity_max_abs": 1e-5, "lock_sha256": "a" * 64}


class ReportTests(unittest.TestCase):
    def test_cluster_weights_and_pairing(self):
        stats = report._summary([(1, 1)] * 8 + [(2, 0)])
        self.assertEqual(stats, {"mean": .5, "low": 0., "high": 1., "worlds": 2, "observations": 9})
        self.assertEqual(report._summary([(1, 1)])["low"], None)
        rows = fixture()
        for (world, _, _, condition), row in rows.items():
            if world == 1 and condition == "factual":
                row["correct"] = False
        book = report._per_book(rows, 0)
        self.assertEqual(book["factual_lift"]["mean"], .5)
        self.assertEqual(book["factual_lift"]["observations"], 16)
        self.assertEqual(book["factual_alternate"]["mean"], 1)

    def test_complete_gate_checks_identities_shards_and_counterfactuals(self):
        rows = fixture()
        with patch.object(report, "EXPECTED_WORLDS", 2):
            def gate():
                return report._group_summary(rows, 0, VALIDATION, "test", lock_verified=True)["bridges"]["factual"]
            self.assertTrue(gate()["passes"])
            self.assertFalse(report._group_summary(rows, 1, VALIDATION, "test", lock_verified=True)["bridges"]["factual"]["passes"])
            self.assertFalse(report._group_summary(rows, 0, {}, "test")["bridges"]["factual"]["passes"])
            rows[0, 0, 0, "factual"]["receiver_prompt"] += " leaked"
            self.assertFalse(gate()["checks"]["receiver_input_fixed"])
            rows[0, 0, 0, "factual"]["receiver_prompt"] = "Independent book 0"
            rows[0, 0, 0, "factual_wrong"]["packet_sha256"] = "0" * 64
            self.assertFalse(gate()["checks"]["wrong_packet_verified"])
            rows[0, 0, 0, "factual_wrong"]["packet_sha256"] = f"{1:064x}"
            rows[0, 1, 0, "factual"]["packet_sha256"] = "b" * 64
            self.assertFalse(gate()["checks"]["same_packet_verified"])
            rows[0, 1, 0, "factual"]["packet_sha256"] = "0" * 64
            rows[0, 1, 0, "factual"]["target"] = "0"
            self.assertFalse(gate()["checks"]["targets_always_change"])
            rows[0, 1, 0, "factual"]["lock_sha256"] = "e" * 64
            self.assertFalse(gate()["checks"]["lock_verified"])
            del rows[0, 0, 0, "code"]
            self.assertFalse(gate()["checks"]["complete_heldout"])

    def test_lock_and_validation_file_receipts_and_quick_isolation(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            full = root / "qwen" / "full"
            quick = root / "qwen" / "quick"
            v = full / "validation" / "0000-of-0001.jsonl"
            v.parent.mkdir(parents=True)
            quick.mkdir(parents=True)
            v.write_bytes(b"")
            content = {"sender": "qwen", "quick": False}
            digest = hashlib.sha256(json.dumps(content, separators=(",", ":"), sort_keys=True).encode()).hexdigest()
            (full / "lock.json").write_text(json.dumps(dict(content, lock_sha256=digest)))
            (full / "validation.json").write_text(json.dumps(dict(VALIDATION, sender="qwen", lock_sha256=digest,
                                                                  rows_sha256=hashlib.sha256(b"").hexdigest())))
            (quick / "validation.json").write_text(json.dumps({"sender":"qwen", "quick":True, "lock_sha256":"incorrect"}))
            self.assertTrue(report._metadata(root)["qwen"]["lock_file_verified"])
            v.write_bytes(b" ")
            self.assertFalse(report._metadata(root)["qwen"]["lock_file_verified"])
            v.write_bytes(b"")
            (full / "lock.json").write_text(json.dumps(dict(content, sender="qwen", quick=True, lock_sha256=digest)))
            self.assertFalse(report._metadata(root)["qwen"]["lock_file_verified"])

    def test_identical_retries_and_conflicting_records(self):
        with TemporaryDirectory() as td:
            path = Path(td) / "rows.jsonl"
            row = next(iter(fixture(1).values()))
            path.write_text((json.dumps(row) + "\n") * 2)
            self.assertEqual(report._rows(Path(td))[1]["test", "qwen"], 1)
            row["answer"] = "conflict"
            row["correct"] = False
            with path.open("a") as file:
                file.write(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                report._rows(Path(td))

    def test_end_to_end_assets_on_empty_and_partial_input(self):
        with TemporaryDirectory() as td:
            root, out = Path(td) / "inputs", Path(td) / "outputs"
            root.mkdir()
            summary = report.generate(root, out)
            self.assertEqual(summary["groups"], {})
            self.assertIn("No held-out predictions", (out / "report.md").read_text())
            folder = root / "qwen"
            folder.mkdir()
            (folder / "test.jsonl").write_text("\n".join(map(json.dumps, fixture().values())))
            (folder / "validation.json").write_text(json.dumps(VALIDATION))
            (root / "cost.json").write_text(json.dumps({"actual_usd": 1.23, "source": "QA fixture"}))
            (folder / "interface-diagnostic.json").write_text(json.dumps(dict(
                sender="qwen", source_split="validation", worlds=2, heldout_opened=False,
                arms=[dict(arm="oracle", layer=1, readout="marker", rulebook=0, items=16, correct=12, accuracy=.75)])))
            val = [dict(r, split="validation") for r in fixture().values()]
            (folder / "validation.jsonl").write_text("\n".join(map(json.dumps, val)))
            result = report.generate(root, out)
            self.assertIn("| oracle | 1 | marker | 1 | 12 / 16 | 75.0% |", (out / "report.md").read_text())
            self.assertIn("factual-map", (out / "release-note.md").read_text().lower())
            with gzip.open(out / "predictions.jsonl.gz", "rt") as f:
                self.assertEqual(sum(1 for _ in f), len(fixture()) * 2)
            self.assertFalse(result["groups"]["test"]["qwen"]["bridges"]["factual"]["passes"])
            for name in ("manifest.js", "paired-lift.svg", "paired-lift.png", "condition-accuracy.svg", "validation-accuracy.svg", "worlds/qwen-1.js", "replay.js", "index.html"):
                self.assertTrue((out / name).is_file(), name)
            self.assertIn("$1.23", (out / "report.md").read_text())
            self.assertEqual(json.loads((out / "summary.json").read_text())["cost"]["source"], "QA fixture")
