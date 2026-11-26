"""Factual-key calibration and held-out recipient interventions on CounterFact."""
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .models import Bridge


def _key(s):
    return " ".join(unicodedata.normalize("NFKC", s).casefold().split())


def _correct(prediction, answers):
    # Compare the full answer phrase; "New York City" is distinct from "New York".
    lines = prediction.strip().splitlines()
    line = _key(lines[0] if lines else "").lstrip(" '\"")
    return any(re.match(re.escape(_key(a)) + r"(?:$|[.,;:!?\"\n])", line)
               for a in answers if a.strip())


@dataclass(frozen=True)
class Fact:
    id: str
    subject: str
    group: str
    relation: str
    template: str
    answers: tuple
    paraphrase: str | None = None

    def prompt(self, paraphrase=False):
        template = self.paraphrase if paraphrase else self.template
        before, after = template.split("{}")
        return before + self.subject + after, len(before + self.subject)


def _load_facts(path, alias_groups=None, *, identity_stats=None):
    from .subjects import official_subject_ids

    official_ids = official_subject_ids(path)
    records = json.loads(Path(path).read_text())
    if not isinstance(records, list):
        raise ValueError("Expected the original CounterFact JSON array")
    roots = {}
    def find(value):
        roots.setdefault(value, value)
        while roots[value] != value:
            roots[value] = roots[roots[value]]
            value = roots[value]
        return value
    def link(names):
        keys = [find(_key(str(x))) for x in names if x]
        if keys:
            for key in keys:
                roots[key] = min(keys)
    for alias, entity in (alias_groups or {}).items():
        link([alias, entity])
    rows = []
    for i, row in enumerate(records):
        row_id = str(row.get("case_id", i))
        if official_ids is not None and row_id not in official_ids:
            continue
        r = row.get("requested_rewrite", row)
        subject, template = r.get("subject", ""), r.get("prompt", "")
        target = r.get("target_true", {})
        if not subject or template.count("{}") != 1 or not target.get("str"):
            continue
        aliases = r.get("subject_aliases", row.get("subject_aliases", []))
        if isinstance(aliases, str):
            aliases = [aliases]
        link([subject, *aliases, r.get("subject_id"), r.get("sub_uri"),
              row.get("subject_id"), row.get("sub_uri")])
        ans_alias = target.get("aliases", [])
        if isinstance(ans_alias, str):
            ans_alias = [ans_alias]
        alternate = None
        for prompt in row.get("paraphrase_prompts", []):
            if prompt.count(subject) == 1 and "{" not in prompt and "}" not in prompt:
                candidate = prompt.replace(subject, "{}")
                if _key(candidate) != _key(template):
                    alternate = candidate
                    break
        rows.append((row_id, subject, str(r["relation_id"]),
                     template, tuple([target["str"], *ans_alias]), alternate))
    if identity_stats is not None:
        identity_stats.update(source_records=len(records), retained_records=len(rows),
                              grouping="Wikidata QID" if official_ids is not None else "provided aliases and subject IDs")
    def identity(index, name):
        return official_ids[index] if official_ids is not None else find(_key(name))
    return [Fact(i,s,identity(i,s),rel,tpl,ans,alt) for i,s,rel,tpl,ans,alt in rows]


def _part(group, seed):
    n = int.from_bytes(hashlib.sha256(f"{seed}:{group}".encode()).digest()[:8], "big") % 9
    return "train" if n < 6 else ("validation" if n == 6 else "test")


def _triples(facts, seed):
    """Three distinct answers per independent cluster; each subject used once."""
    rng = random.Random(seed)
    rows = list(facts)
    rng.shuffle(rows)
    used, relations = set(), defaultdict(lambda: defaultdict(list))
    for f in rows:
        if f.group not in used:
            relations[f.relation][_key(f.answers[0])].append(f)
            used.add(f.group)
    triples = []
    for answers in relations.values():
        while len(answers) >= 3:
            keys = sorted(answers, key=lambda k: len(answers[k]), reverse=True)[:3]
            triple = [answers[k].pop() for k in keys]
            if len({a for f in triple for a in map(_key, f.answers)}) == sum(len(set(map(_key, f.answers))) for f in triple):
                triples.append(triple)
            for key in keys:
                if not answers[key]:
                    del answers[key]
    rng.shuffle(triples)
    return triples


def _dump(path, obj):
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _cache_records(path):
    completed = {}
    if path.exists():
        with path.open("rb+") as f:
            while True:
                start = f.tell()
                line = f.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    f.truncate(start)
                    break
                row = json.loads(line)
                completed[row["id"]] = row
    return completed


def _states(model, rows, layers, path, batch_size):
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            if list(data["ids"]) != [r.id for r in rows]:
                raise ValueError("Cached activation rows mismatch")
            return {layer: data[str(layer)] for layer in layers}
    collected = {layer: [] for layer in layers}
    for start in range(0, len(rows), batch_size):
        entries = [r.prompt() for r in rows[start:start + batch_size]]
        text, ends = zip(*entries)
        for layer, states in model.capture_many(text, layers, ends).items():
            collected[layer].append(states)
    stacked = {layer: np.concatenate(batches) for layer, batches in collected.items()}
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("wb") as f:
        np.savez(f, ids=np.array([r.id for r in rows]), **{str(k): v for k, v in stacked.items()})
    tmp.replace(path)
    return stacked


def _decode(receiver, rows, vectors, layer, batch_size, *, paraphrase=False):
    out = []
    for start in range(0, len(rows), batch_size):
        entries = [r.prompt(paraphrase) for r in rows[start:start + batch_size]]
        texts, ends = zip(*entries)
        patches = [(layer, end, vector) for end, vector in zip(ends, vectors[start:])]
        out.extend(receiver.answer_many(texts, patches))
    return out


def _cycle_permutation(n, rng):
    order = rng.permutation(n)
    permutation = np.empty_like(order)
    permutation[order] = np.roll(order, 1)
    return permutation


def _intervals(groups, scores, seed):
    # Each triad comprises three rotating donor/receiver/wrong-subject assignments.
    keys = list(dict.fromkeys(groups))
    group_rows = np.array([[i for i, value in enumerate(groups) if value == k] for k in keys])
    arrays = {name: np.asarray(values, dtype=float)[group_rows].mean(axis=1)
              for name, values in scores.items()}
    sample = np.random.default_rng(seed).integers(0, len(keys), size=(2000, len(keys)))
    means = {name: values[sample].mean(axis=1) for name, values in arrays.items()}
    negative = np.max([v for k, v in means.items() if k not in ("mapped", "self")], axis=0)
    return {k: list(map(float, np.quantile(v, [.025, .975]))) for k, v in
            {**means, "mapped_minus_strongest_negative": means["mapped"] - negative}.items()}


def run_facts(sender, receiver, data_path, out_dir, quick=False, *, seed=7381,
              batch_size=8, splits=None, sender_layers=None, receiver_layers=None,
              alias_groups=None, screen_limit=None):
    """Returns factual.json. Absolute bridge_path points to the selected map if fit."""
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    targets = splits or ((96, 36, 60) if quick else (3000, 500, 1000))
    if len(targets) != 3 or min(targets) < 2 or batch_size < 1:
        raise ValueError("Need positive batch size and at least two rows per split")
    sl = list(dict.fromkeys(sender_layers if sender_layers is not None else
                            [round((sender.depth-1)*v) for v in (.22, .32, .40)]))
    rl = list(dict.fromkeys(receiver_layers if receiver_layers is not None else
                            [round((receiver.depth-1)*v) for v in (.22, .32, .40)]))
    cap = int(screen_limit) if screen_limit is not None else (3500 if quick else 100000)
    spec = dict(seed=seed, targets=list(targets), sender_layers=sl, receiver_layers=rl,
                screen_limit=cap, data_sha256=hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
                sender=[sender.repo, sender.revision], receiver=[receiver.repo, receiver.revision],
                aliases_sha256=hashlib.sha256(json.dumps(alias_groups or {},sort_keys=True).encode()).hexdigest(),
                source_sha256={name:hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                               for name in ("facts.py", "models.py","subjects.py","assets/counterfact-subjects.tsv")}, protocol_version=2)
    spec_path = out / "factual-spec.json"
    if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
        raise ValueError("This output directory contains a different factual protocol")
    _dump(spec_path, spec)
    summary_path = out / "factual.json"
    if summary_path.exists():
        previous = json.loads(summary_path.read_text())
        if previous.get("status") == "insufficient_eligible_facts":
            return previous
        if ((out / "factual-bridge.npz").exists() and (out / "factual-trials.jsonl").exists()):
            previous["bridge_path"] = str(out / "factual-bridge.npz")
            return previous
    identity_stats = {}
    rows = _load_facts(data_path, alias_groups, identity_stats=identity_stats)
    random.Random(seed).shuffle(rows)
    limits = dict(zip(("train","validation","test"), targets))
    eligible = {k: [] for k in limits}
    screened = _cache_records(out / "screen.jsonl")
    counts = defaultdict(int)
    with (out / "screen.jsonl").open("a") as log:
        for i in range(0, min(len(rows), cap), batch_size):
            batch = [r for r in rows[i:min(i+batch_size,cap)] if
                     len(eligible[_part(r.group,seed)]) < limits[_part(r.group,seed)]]
            missing = [r for r in batch if r.id not in screened]
            if missing:
                prompts = [r.prompt()[0] for r in missing]
                for r, a, b in zip(missing, sender.answer_many(prompts),receiver.answer_many(prompts)):
                    record = dict(id=r.id, sender_answer=a, receiver_answer=b)
                    screened[r.id] = record
                    log.write(json.dumps(record)+"\n")
                log.flush()
            for r in batch:
                part = _part(r.group, seed)
                counts[part] += 1
                rec = screened[r.id]
                if (len(eligible[part]) < limits[part] and _correct(rec["sender_answer"], r.answers)
                        and _correct(rec["receiver_answer"], r.answers)):
                    eligible[part].append(r)
            if all(len(eligible[k]) >= v for k,v in limits.items()):
                break
    report = dict(protocol=spec, identity_resolution=identity_stats, counts={k:len(v) for k,v in eligible.items()},
                  screened=dict(counts), sender_layer=None, receiver_layer=None,
                  bridge_path=None, gate_passed=False, quick=bool(quick))
    def finish():
        _dump(out / "factual.json", report)
        return report

    report["eligible_by_relation"] = {p:dict(Counter(r.relation for r in cohort))
                                      for p,cohort in eligible.items()}
    train = eligible["train"]
    triples = {p:_triples(eligible[p],seed+1) for p in ("validation","test")}
    if len(train)<2 or not all(triples.values()):
        report["status"] = "insufficient_eligible_facts"
        return finish()
    eligible_all = train + [r for p in triples.values() for group in p for r in group]
    offsets = {r.id:i for i,r in enumerate(eligible_all)}
    sx = _states(sender, eligible_all, sl,out / "sender-states.npz",batch_size)
    ry = _states(receiver, eligible_all,rl,out / "receiver-states.npz",batch_size)
    def tasks(part):
        records = [(g, t[i], t[(i+1)%3], t[(i+2)%3]) for g,t in enumerate(triples[part]) for i in range(3)]
        groups, src, dst, wrong = map(list, zip(*records))
        return groups, src, dst, wrong

    _, src, dst, _ = tasks("validation")
    src_indices = [offsets[r.id] for r in src]
    best, best_bridge = -1, None
    details = []
    for a in sl:
        for b in rl:
            bridge = Bridge.fit(sx[a][:len(train)], ry[b][:len(train)])
            preds = _decode(receiver, dst, bridge.apply(sx[a][src_indices]), b, batch_size)
            score = np.mean([_correct(ans,r.answers) for ans,r in zip(preds,src)])
            details.append(dict(sender_layer=a,receiver_layer=b,accuracy=float(score)))
            if score > best:
                best, best_bridge = score, bridge
                report.update(sender_layer=a,receiver_layer=b)
    a,b = report["sender_layer"],report["receiver_layer"]
    best_bridge.save(out / "factual-bridge.npz")
    report.update(bridge_path=str(out / "factual-bridge.npz"), validation=details,
                  effective_rank=best_bridge.left.shape[1])
    groups,src,dst,wrong = tasks("test")
    si, wi = [offsets[r.id] for r in src], [offsets[r.id] for r in wrong]
    rng = np.random.default_rng(seed)
    shuffled = Bridge.fit(sx[a][:len(train)],ry[b][:len(train)][_cycle_permutation(len(train),rng)])
    # Random signed permutation destroys learned coordinates while retaining geometry.
    perm = rng.permutation(sx[a].shape[1])
    signs = rng.choice([-1.,1.],len(perm)).astype("f")
    vectors = {"self":ry[b][si], "mapped":best_bridge.apply(sx[a][si]),
               "wrong_real":best_bridge.apply(sx[a][wi]),"shuffled_fit":shuffled.apply(sx[a][si]),
               "random_map":best_bridge.apply(best_bridge.source_mean+
                            (sx[a][si]-best_bridge.source_mean)[:,perm]*signs),
               "mean":np.broadcast_to(best_bridge.target_mean, (len(dst),len(best_bridge.target_mean)))}
    results = {"none":[screened[r.id]["receiver_answer"] for r in dst]}
    for name, vec in vectors.items():
        results[name] = _decode(receiver, dst, vec,b,batch_size)
    scores = {name:[int(bool(_correct(ans,r.answers))) for ans,r in zip(preds,src)]
              for name,preds in results.items()}
    accuracy = {k:float(np.mean(v)) for k,v in scores.items()}
    ci = _intervals(groups,scores,seed)
    strongest = max(v for k,v in accuracy.items() if k not in ("self","mapped"))
    marker_text, marker_end = dst[0].prompt()
    identity = receiver.identity_error(marker_text, b, marker_end)
    # Withhold the receiver's paraphrases from all earlier selection. Use whole
    # triads so the bootstrap has the same independent sampling unit.
    alt_ids = [i for i,g in enumerate(groups) if all(row.paraphrase for row in triples["test"][g])]
    alt = {"items":len(alt_ids), "passed":False}
    alternate_outputs = {}
    if alt_ids:
        alt_rows = [dst[i] for i in alt_ids]
        alt_groups = [groups[i] for i in alt_ids]
        alt_scores = {}
        for key in ("mapped","wrong_real"):
            preds = _decode(receiver,alt_rows,vectors[key][alt_ids],b,batch_size,paraphrase=True)
            alternate_outputs[key] = dict(zip(alt_ids,preds))
            alt_scores[key] = [int(_correct(pred,src[i].answers)) for i,pred in zip(alt_ids,preds)]
        lo,hi = _intervals(alt_groups,alt_scores,seed)["mapped_minus_strongest_negative"]
        alt.update(mapped=float(np.mean(alt_scores["mapped"])),wrong_real=float(np.mean(alt_scores["wrong_real"])),
                   paired_ci=[lo,hi],passed=bool(lo>0))
    report["heldout_recipient_paraphrase"] = alt
    report["test_by_relation"] = {}
    for relation in sorted({r.relation for r in src}):
        members = [i for i,r in enumerate(src) if r.relation == relation]
        report["test_by_relation"][relation] = {
            "items":len(members), "clusters":len({groups[i] for i in members}),
            "accuracy":{k:float(np.mean([v[i] for i in members])) for k,v in scores.items()}}
    report["answer_outcomes"] = {}
    for arm, predictions in results.items():
        donor, original = 0, 0
        for pred, source, recipient in zip(predictions, src, dst):
            if _correct(pred,source.answers):
                donor += 1
            elif _correct(pred,recipient.answers):
                original += 1
        report["answer_outcomes"][arm] = {"donor":donor, "original":original,
                                           "other":len(dst)-donor-original}
    report.update(status="measured",identity_logit_error=identity, accuracy=accuracy,
                  confidence_intervals=ci, test_clusters=len(triples["test"]),
                  test_items=len(dst), validation_items=len(src_indices))
    report["gate_passed"] = bool(alt["passed"] and identity < 1e-4 and accuracy["self"] >= .25 and
                             accuracy["mapped"] >= .6*accuracy["self"] and
                             accuracy["mapped"]-strongest >= .1 and ci["mapped_minus_strongest_negative"][0]>0)
    with (out/"factual-trials.jsonl").open("w") as log:
        for i,(g,s,d,w) in enumerate(zip(groups,src,dst,wrong)):
            log.write(json.dumps(dict(cluster=g, donor=s.id, recipient=d.id, wrong=w.id,
                                      relation=s.relation, donor_targets=s.answers, recipient_targets=d.answers,
                                      outputs={k:v[i] for k,v in results.items()},
                                      paraphrase_outputs={k:v[i] for k,v in alternate_outputs.items()
                                                          if i in v}))+"\n")
    return finish()
