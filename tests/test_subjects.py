import hashlib
from io import BytesIO
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from plato.subjects import SUBJECTS_SHA256, SUBJECTS_TABLE
from scripts.build_subject_ids import build


class SubjectIdentityTests(unittest.TestCase):
    def test_shipped_identity_map(self):
        raw = SUBJECTS_TABLE.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), SUBJECTS_SHA256)
        rows = [line.split("\t") for line in raw.decode().splitlines()]
        self.assertEqual(len({case for case, qid in rows}), 21495)
        self.assertEqual(len({qid for case, qid in rows}), 20028)

    def test_exact_join_keeps_aliases_and_excludes_ambiguity(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            facts = [(0, "Alice", "Q8"), (1, "Ally", "Q8"), (2, "Same", "Q8"), (3, "Gone", "Q8")]
            rows = [{"case_id": case, "requested_rewrite": {"relation_id": "P1", "subject": label,
                                                             "target_true": {"id": obj}}}
                    for case, label, obj in facts]
            cf = path / "counterfact.json"
            cf.write_text(json.dumps(rows))
            data = b"".join((json.dumps(dict(sub_label=label, sub_uri=qid, obj_uri="Q8", predicate_id="P1")) + "\n").encode()
                            for label, qid in [("Alice", "Q1"), ("Ally", "Q1"), ("Same", "Q2"), ("Same", "Q3")])
            source = path / "data.tar.gz"
            with tarfile.open(source, "w:gz") as archive:
                info = tarfile.TarInfo("TREx/P1.jsonl")
                info.size = len(data)
                archive.addfile(info, BytesIO(data))
            output = path / "ids.tsv"
            with patch("scripts.build_subject_ids.COUNTERFACT_SHA256", hashlib.sha256(cf.read_bytes()).hexdigest()), \
                 patch("scripts.build_subject_ids.LAMA_SHA256", hashlib.sha256(source.read_bytes()).hexdigest()):
                stats = build(cf, source, output)
            self.assertEqual(output.read_text(), "0\tQ1\n1\tQ1\n")
            self.assertEqual((stats["ambiguous"], stats["unmatched"], stats["entities_with_multiple_labels"]), (1, 1, 1))
