"""Verified subject identities for the published version of ROME CounterFact."""
import hashlib
from pathlib import Path


COUNTERFACT_SHA256 = "d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f"
SUBJECTS_SHA256 = "331bc7161e47f105f1de89b13530b257bb286b2d8aa20dc732bc74d32e773677"
SUBJECTS_TABLE = Path(__file__).parent / "assets" / "counterfact-subjects.tsv"


def official_subject_ids(path):
    """Return the pinned case ID to subject QID map; other datasets manage their own IDs."""
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != COUNTERFACT_SHA256:
        return None
    raw = SUBJECTS_TABLE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SUBJECTS_SHA256:
        raise ValueError("CounterFact subject identity table checksum mismatch")
    return dict(line.split("\t") for line in raw.decode("ascii").splitlines())
