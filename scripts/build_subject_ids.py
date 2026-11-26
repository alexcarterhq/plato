"""Build the case-to-Wikidata identity table from the two original public files.

Inputs (download separately):
https://rome.baulab.info/data/dsets/counterfact.json
https://dl.fbaipublicfiles.com/LAMA/negated_data.tar.gz

The ROME CounterFact supplement identifies ParaRel as the origin of its triples.
ParaRel uses the T-REx part of LAMA. Match the relation and true object QID in
addition to the original case-sensitive subject label. Ignore non-unique joins.
Only numeric case IDs and public Wikidata QIDs are written to the output.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import tarfile


COUNTERFACT_SHA256 = "d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f"
LAMA_SHA256 = "1a151058e6608e47983ea4c99c50bb69248c1c0763a04a3793b0a0b657aa0b61"


def verified(path, expected):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"Unexpected upstream checksum for {path.name}")


def build(counterfact, lama_archive, output):
    verified(counterfact, COUNTERFACT_SHA256)
    verified(lama_archive, LAMA_SHA256)
    targets = {}
    candidates = {}
    for record in json.loads(counterfact.read_text()):
        fact = record["requested_rewrite"]
        key = fact["relation_id"], fact["subject"], fact["target_true"]["id"]
        if record["case_id"] in targets:
            raise ValueError("Duplicate case ID")
        targets[record["case_id"]] = key
        candidates.setdefault(key, set())
    with tarfile.open(lama_archive, "r|gz") as archive:
        for member in archive:
            if not member.isfile() or not re.fullmatch(r"(?:\./)?TREx/P\d+\.jsonl", member.name):
                continue
            for line in archive.extractfile(member):
                triple = json.loads(line)
                key = triple["predicate_id"], triple["sub_label"], triple["obj_uri"]
                if key in candidates and re.fullmatch(r"Q[1-9]\d*", triple["sub_uri"]):
                    candidates[key].add(triple["sub_uri"])

    coverage = Counter()
    entity_labels, label_entities = defaultdict(set), defaultdict(set)
    rows = []
    for case_id, key in sorted(targets.items()):
        ids = candidates[key]
        if len(ids) != 1:
            coverage["ambiguous" if ids else "unmatched"] += 1
            continue
        qid = next(iter(ids))
        rows.append(f"{case_id}\t{qid}\n")
        entity_labels[qid].add(key[1])
        label_entities[key[1]].add(qid)
        coverage["linked"] += 1
    output.write_bytes("".join(rows).encode("ascii"))
    return dict(cases=len(targets), linked=coverage["linked"], ambiguous=coverage["ambiguous"],
                unmatched=coverage["unmatched"], distinct_entities=len(entity_labels),
                entities_with_multiple_labels=sum(len(v) > 1 for v in entity_labels.values()),
                labels_with_multiple_entities=sum(len(v) > 1 for v in label_entities.values()),
                sha256=hashlib.sha256(output.read_bytes()).hexdigest())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("counterfact", "lama_archive", "output"):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.counterfact, args.lama_archive, args.output), indent=2))
