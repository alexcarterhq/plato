"""Summarize recorded predictions and export an offline replay of each held-out world."""

from collections import defaultdict
from functools import lru_cache
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics

import numpy as np

from .worlds import exact_answer


CONDITIONS = {
    "factual": "Factual map",
    "factual_wrong": "Factual map · wrong packet",
    "contextual": "Contextual map",
    "contextual_wrong": "Contextual map · wrong packet",
    "none": "No message",
    "self": "Receiver’s own state",
    "full": "Full evidence",
    "word": "Generated word",
    "code": "Deterministic code",
    "sender": "Sender retrieval",
}
PRIMARY_SPLITS = {"test", "validation", "robustness", "bench"}
EXPECTED_WORLDS = 250
BOOTSTRAP_SEED = 73639
BOOTSTRAP_REPLICATES = 10000


@lru_cache(maxsize=6)
def _draws(n):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    return rng.integers(n, size=(BOOTSTRAP_REPLICATES, n), dtype=np.int32)


def _interval(world_means, n_observations):
    if not world_means:
        return {"mean": None, "low": None, "high": None, "worlds": 0, "observations": 0}
    values = np.asarray(world_means, dtype=np.float64)
    n = len(values)
    low = high = None
    if n > 1:
        low, high = np.quantile(values[_draws(n)].mean(axis=1), [.025, .975]).tolist()
    return {"mean": float(values.mean()), "low": low, "high": high,
            "worlds": n, "observations": n_observations}


def _summary(observations):
    """Each observation is (world, value); resample full worlds, weighting worlds equally."""
    worlds = defaultdict(list)
    for world, value in observations:
        worlds[world].append(value)
    return _interval([statistics.fmean(worlds[w]) for w in sorted(worlds)],
                     sum(map(len, worlds.values())))


def _number(value, low=None):
    return (isinstance(value, (float, int)) and not isinstance(value, bool)
            and math.isfinite(value) and (low is None or value >= low))


def _rows(data_dir):
    groups = defaultdict(dict)
    duplicate_keys = defaultdict(int)
    for path in sorted(data_dir.rglob("*.jsonl")):
        parts = path.relative_to(data_dir).parts
        if path.name.startswith("._") or "quick" in parts:
            continue
        with path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                # Run directories can also contain the Modal billing ledger and diagnostic logs.
                if not isinstance(row, dict) or row.get("split") not in PRIMARY_SPLITS:
                    continue
                if row.get("quick") is True:
                    continue
                if "condition" not in row and "correct" not in row:
                    continue
                try:
                    sender, rulebook, variant, condition = (row[k] for k in ("sender", "rulebook", "variant", "condition"))
                    if (sender not in {"qwen", "pythia"} or type(rulebook) is not int or rulebook not in (0, 1)
                            or type(variant) is not int or variant not in range(8)):
                        raise ValueError("unexpected sender, rulebook, or variant")
                    if type(row["world"]) is not int or type(row["correct"]) is not bool:
                        raise ValueError("world must be an integer and correct must be a boolean")
                    if condition not in CONDITIONS or not all(isinstance(row[k], str) for k in ["answer", "target"]):
                        raise ValueError("unexpected condition or answer type")
                    if row["correct"] != exact_answer(row["answer"], row["target"]):
                        raise ValueError("recorded correctness disagrees with the raw answer and target")
                    key = (row["world"], rulebook, variant, condition)
                except (ValueError, KeyError, TypeError) as problem:
                    raise ValueError(f"Invalid prediction {path}:{line_number} ({problem})") from problem
                split = row["split"]
                if "bench" in parts:
                    index = parts.index("bench")
                    if index + 1 < len(parts) - 1:
                        split = f"bench/{parts[index + 1]}"
                group = (split, sender)
                if key in groups[group]:
                    if groups[group][key] != row:
                        raise ValueError(f"Conflicting predictions for {group}, {key} in {path}:{line_number}")
                    duplicate_keys[group] += 1
                else:
                    groups[group][key] = row
    return groups, duplicate_keys


def _condition_summary(members):
    answer = _summary((row["world"], int(row["correct"])) for row in members)
    for key in ("latency_s", "gpu_s", "packet_bytes"):
        vals = [r[key] for r in members if _number(r.get(key), low=0)]
        answer[key + "_median"] = statistics.median(vals) if vals else None
        if key == "latency_s":
            answer["latency_s_p95"] = float(np.quantile(vals, .95)) if vals else None
    return answer


def _per_book(rows, book):
    members = defaultdict(list)
    for (_, rb, _, condition), row in rows.items():
        if rb == book:
            members[condition].append(row)
    metrics = {c: _condition_summary(members[c]) for c in CONDITIONS}
    for bridge in ("factual", "contextual"):
        differences, alternate = [], []
        for row in members[bridge]:
            w, v = row["world"], row["variant"]
            control = rows.get((w, book, v, bridge + "_wrong"))
            if control is None:
                continue
            differences.append((w, int(row["correct"]) - int(control["correct"])))
            follows_wrong = control.get("wrong_correct")
            if type(follows_wrong) is bool:
                alternate.append((w, int(follows_wrong)))
        metrics[bridge + "_lift"] = _summary(differences)
        metrics[bridge + "_alternate"] = _summary(alternate)
    return metrics


def _digest(value):
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value.lower())


def _fixed_receiver_prompts(rows):
    prompts = defaultdict(set)
    for (world, book, _, condition), row in rows.items():
        if condition in {"factual", "factual_wrong", "contextual", "contextual_wrong"}:
            prompts[world, book].add(row.get("receiver_prompt"))
    return bool(prompts) and all(len(items) == 1 and all(isinstance(v, str) and v for v in items)
                                 for items in prompts.values())


def _two_books(rows, bridge):
    joint, changed, digest_pairs = [], [], []
    for (world, book, variant, condition), left in rows.items():
        if book != 0 or condition != bridge:
            continue
        right = rows.get((world, 1, variant, condition))
        if right is None:
            continue
        joint.append((world, int(left["correct"] and right["correct"])))
        changed.append(left["target"] != right["target"])
        a, b = left.get("packet_sha256"), right.get("packet_sha256")
        digest_pairs.append(_digest(a) and a == b)
    # Each wrong arm must use the actual matched packet belonging to its alternate category.
    lookup = {(w, b, row.get("category")): row for (w, b, _, c), row in rows.items() if c == bridge}
    wrong_ok = []
    categories, destinations, alternatives = defaultdict(set), defaultdict(set), defaultdict(set)
    for (w, b, _, c), row in rows.items():
        if c == bridge:
            categories[w, b].add(row.get("category"))
            destinations[w, b].add(row["target"])
        if c != bridge + "_wrong":
            continue
        alternatives[w, b].add(row.get("wrong_category"))
        source = lookup.get((w, b, row.get("wrong_category")))
        wrong_ok.append(source is not None and row.get("wrong_category") != row.get("category")
                        and row.get("wrong_target") == source["target"]
                        and _digest(row.get("packet_sha256"))
                        and row["packet_sha256"] == source.get("packet_sha256"))
    return {"joint": _summary(joint), "targets_always_change": bool(changed) and all(changed),
            "wrong_packet_verified": bool(wrong_ok) and all(wrong_ok) and all(
                len(cats) == len(destinations[key]) == 8 and cats == alternatives[key]
                for key, cats in categories.items()),
            "same_packet_verified": bool(digest_pairs) and all(digest_pairs)}


def _json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _metadata(root):
    """An export can contain both donors and both quick and full Modal run directories."""
    metadata = {}
    for name in ("validation", "factual", "lock", "diagnostic", "interface-diagnostic"):
        for path in sorted(root.rglob(name + ".json")):
            parts = path.relative_to(root).parts
            if "quick" in parts or "bench" in parts:
                continue
            record = _json(path)
            if record.get("quick") is True:
                continue
            sender = record.get("sender") or next((p for p in reversed(path.parts) if p in {"qwen", "pythia"}), None)
            if sender not in {"qwen", "pythia"}:
                continue
            item = metadata.setdefault(sender, {})
            if name in item and item[name] != record:
                raise ValueError(f"Multiple {name} files for {sender}; select one run directory")
            item[name] = record
            if name == "validation":
                source = path.parent / "validation" / "0000-of-0001.jsonl"
                recorded_sha = record.get("rows_sha256")
                item["validation_rows_verified"] = False
                if source.is_file() and _digest(recorded_sha):
                    with source.open("rb") as file:
                        item["validation_rows_verified"] = hashlib.file_digest(file, "sha256").hexdigest() == recorded_sha
    for item in metadata.values():
        lock = dict(item.get("lock", {}))
        digest = lock.pop("lock_sha256", None)
        expected = hashlib.sha256(json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        validation = item.get("validation", {})
        item["lock_file_verified"] = (_digest(digest) and digest == expected == validation.get("lock_sha256")
                                     and item.get("validation_rows_verified", False))
    return metadata


def _coverage(rows, duplicate_count):
    worlds = sorted({key[0] for key in rows})
    required = set(CONDITIONS)
    counts = defaultdict(set)
    for world, book, variant, condition in rows:
        counts[world, book, variant].add(condition)
    complete = [world for world in worlds
                if all(counts[world, book, variant] == required for book in range(2) for variant in range(8))]
    return {"worlds": len(worlds), "complete_worlds": len(complete), "duplicates": duplicate_count,
            "expected_world_ids": worlds == list(range(EXPECTED_WORLDS)),
            "rows": len(rows), "expected_worlds": EXPECTED_WORLDS}


def _gte(value, bound):
    return value is not None and (value >= bound or math.isclose(value, bound, abs_tol=1e-12))


def _group_summary(rows, duplicate_count, validation, split, lock_verified=False):
    books = {str(book): _per_book(rows, book) for book in (0, 1)}
    coverage = _coverage(rows, duplicate_count)
    bridges = {name: _two_books(rows, name) for name in ("factual", "contextual")}
    ident = validation.get("identity_max_abs")
    checks = {
        "complete_heldout": (split == "test" and coverage["worlds"] == coverage["complete_worlds"] == EXPECTED_WORLDS
                             and coverage["expected_world_ids"] and duplicate_count == 0),
        "validation_complete": validation.get("complete") is True,
        "validation_controls_valid": validation.get("controls_valid") is True,
        "full_run": validation.get("quick") is False,
        "receiver_input_fixed": _fixed_receiver_prompts(rows),
        "identity_within_tolerance": _number(ident, low=0) and ident <= 1e-4,
    }
    digest = validation.get("lock_sha256")
    checks["lock_verified"] = (lock_verified and _digest(digest)
                               and all(r.get("lock_sha256") == digest for r in rows.values()))
    for name, threshold in (("sender", .95), ("full", .95), ("self", .80)):
        checks[name + "_prerequisite"] = all(_gte(b[name]["mean"], threshold) for b in books.values())
    for name, results in bridges.items():
        requirements = dict(checks)
        requirements["targets_always_change"] = results["targets_always_change"]
        requirements["same_packet_verified"] = results["same_packet_verified"]
        requirements["wrong_packet_verified"] = results["wrong_packet_verified"]
        requirements["accuracy_at_least_75pct"] = all(_gte(b[name]["mean"], .75) for b in books.values())
        requirements["paired_lift_at_least_20pp"] = all(_gte(b[name + "_lift"]["mean"], .20) for b in books.values())
        requirements["positive_lift_lower_bound"] = all(
            b[name + "_lift"]["low"] is not None and b[name + "_lift"]["low"] > 0 for b in books.values())
        requirements["joint_accuracy_at_least_60pct"] = _gte(results["joint"]["mean"], .60)
        results["checks"], results["passes"] = requirements, all(requirements.values())
    return {"coverage": coverage, "books": books, "bridges": bridges, "identity_max_abs": ident,
            "inference": "held-out" if split == "test" else "descriptive"}


def _percent(value):
    return "—" if value is None else f"{100 * value:.1f}%"


def _estimate(stats):
    mean = _percent(stats["mean"])
    if stats["low"] is None:
        return mean
    return f'{mean} [{_percent(stats["low"])}, {_percent(stats["high"])}]'


def _write(path, contents):
    path.write_text(contents, encoding="utf-8")


def _dump(data):
    return json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _replay(rows, output):
    """Each script holds one world. The replay can be opened via file:// without a server."""
    worlds = defaultdict(list)
    for (split, sender), predictions in sorted(rows.items()):
        if split != "test":
            continue
        for row in predictions.values():
            worlds[sender, row["world"]].append(row)
    entries = []
    folder = output / "worlds"
    folder.mkdir(exist_ok=True)
    for (sender, world), predictions in sorted(worlds.items()):
        key = f"{sender}-{world}"
        prompts, prompt_ids, records = [], {}, []
        for row in sorted(predictions, key=lambda x: (x["rulebook"], x["variant"], x["condition"])):
            record = {k: row.get(k) for k in ("rulebook", "variant", "condition", "category", "wrong_category",
                                               "target", "answer", "correct", "packet_bytes", "wrong_target")}
            for field in ("sender_prompt", "receiver_prompt"):
                prompt = row.get(field) or ""
                if prompt not in prompt_ids:
                    prompt_ids[prompt] = len(prompts)
                    prompts.append(prompt)
                record[field] = prompt_ids[prompt]
            records.append(record)
        payload = {"key": key, "world": world, "sender": sender, "prompts": prompts, "rows": records}
        script = "window.platoWorld(" + _dump(payload) + ");\n"
        _write(folder / (key + ".js"), script)
        entries.append({"key": key, "world": world, "sender": sender})
    current = {item["key"] + ".js" for item in entries}
    for sender in ("qwen", "pythia"):
        for prior in folder.glob(sender + "-*.js"):
            if prior.name not in current and prior.stem[len(sender) + 1:].lstrip("-").isdecimal():
                prior.unlink()
    return entries


def _report_md(result):
    lines = ["# Can one frozen model tell another a fact only it saw?", "",
             "The sender sees an entity’s private category. The receiver sees a shuffled category-to-destination rulebook. "
             "A correct handoff lets it name the destination. Each world has eight counterfactual variants and two rulebooks.", "",
             "The primary sender is Qwen2.5-1.5B. The replication sender is Pythia-1.4B-deduped. Both send to SmolLM2-1.7B.", "",
             "## Held-out results", ""]
    tests = result["groups"].get("test", {})
    if result.get("diagnostics") or result["groups"].get("validation"):
        lines = lines[:-2] + ["## Calibration and validation", ""]
        for sender in sorted(set(result.get("diagnostics", {})) | set(result["groups"].get("validation", {}))):
            diag = result.get("diagnostics", {}).get(sender, {})
            lines += [f"### {sender.title()} sender", ""]
            f = diag.get("factual", {})
            if f:
                lines.append("Factual substitution screening gate: " + ("passed." if f.get("gate_passed") else "not passed."))
                if isinstance(f.get("accuracy"), dict):
                    lines.append("Recorded screening accuracies: " + ", ".join(
                        f"{key} {_percent(value)}" for key, value in f["accuracy"].items() if _number(value)) + ".")
            v = diag.get("validation", {})
            if v:
                identity = v.get("identity_max_abs")
                identity_text = f"{identity:.3g}" if _number(identity) else "unavailable"
                lines.append(f"Validation completed: {v.get('complete')}. Controls met: {v.get('controls_valid')}. "
                             f"Identity maximum absolute difference: {identity_text}.")
            reason = diag.get("diagnostic", {}).get("reason")
            if reason:
                lines.append("Recorded diagnostic: " + str(reason))
            val = result["groups"].get("validation", {}).get(sender)
            if val:
                lines += ["", "| Validation control | Rulebook 1 | Rulebook 2 |", "|---|---:|---:|"]
                for key in ("sender", "full", "self", "factual", "factual_wrong", "contextual", "contextual_wrong"):
                    lines.append(f'| {CONDITIONS[key]} | {_estimate(val["books"]["0"][key])} | {_estimate(val["books"]["1"][key])} |')
            diagnostic = diag.get("interface-diagnostic", {})
            if diagnostic.get("arms"):
                lines += ["", f'Interface diagnostic on {diagnostic.get("worlds")} validation worlds.', "",
                          "| Arm | Layer | Readout | Rulebook | Correct / items | Accuracy |", "|---|---:|---|---:|---:|---:|"]
                for arm in diagnostic["arms"]:
                    lines.append(f'| {arm["arm"]} | {arm.get("layer")} | {arm.get("readout")} | {arm["rulebook"] + 1} '
                                 f'| {arm["correct"]} / {arm["items"]} | {_percent(arm.get("accuracy"))} |')
            lines += [""]
        if result["groups"].get("validation"):
            lines += ["![Validation accuracy by measured condition](validation-accuracy.svg)", ""]
        lines += ["## Held-out results", ""]
    if not tests:
        lines += ["No held-out predictions were found. No experimental result is available.", ""]
    if tests:
        lines += ["![Held-out accuracy by measured condition](condition-accuracy.svg)", "",
                  "![Paired gain over a mismatched packet](paired-lift.svg)", ""]
    for sender, group in sorted(tests.items(), key=lambda pair: pair[0] != "qwen"):
        c = group["coverage"]
        lines += [f"### {sender} sender", "", f'{c["complete_worlds"]} complete worlds out of {c["worlds"]} recorded. '
                  f'{c["rows"]:,} unique condition rows. Duplicate rows: {c["duplicates"]}.', "",
                  "| Condition | Rulebook 1 accuracy [95% CI] | Rulebook 2 accuracy [95% CI] |",
                  "|---|---:|---:|"]
        for key, title in CONDITIONS.items():
            lines.append(f'| {title} | {_estimate(group["books"]["0"][key])} | {_estimate(group["books"]["1"][key])} |')
        lines += ["", "| Mapped packet | Rulebook 1 lift over wrong packet | Rulebook 2 lift over wrong packet | Both books correct | Decision |",
                  "|---|---:|---:|---:|---|"]
        for bridge, entry in group["bridges"].items():
            b0, b1 = (group["books"][book][bridge + "_lift"] for book in ("0", "1"))
            decision = "Meets the registered gate" if entry["passes"] else "Gate not met"
            lines.append(f'| {CONDITIONS[bridge]} | {_estimate(b0)} | {_estimate(b1)} | {_estimate(entry["joint"])} | {decision} |')
        for name, entry in group["bridges"].items():
            missing = [k.replace("_", " ") for k,v in entry["checks"].items() if not v]
            if missing:
                lines.append(f'\n{CONDITIONS[name]} unmet checks: ' + ", ".join(missing) + ".")
        lines += ["", "| Rulebook 1 condition | Median payload (bytes) | Median latency (s) | p95 latency (s) |",
                  "|---|---:|---:|---:|"]
        for key, title in CONDITIONS.items():
            m = group["books"]["0"][key]
            vals = [m[k] for k in ("packet_bytes_median", "latency_s_median", "latency_s_p95")]
            values = ["—" if v is None else f"{v:.4g}" for v in vals]
            lines.append("| " + " | ".join([title] + values) + " |")
        lines.extend(["", "Gate checks are recorded individually in [the summary](summary.json). "
                      "Contextual calibration has its own result and cannot establish transfer of the frozen factual map.", ""])
    robust = result["groups"].get("robustness", {})
    if robust:
        lines += ["## Unseen-template robustness", "",
                  "These separate worlds test how the locked maps behave under the alternate prompt template.", "",
                  "| Sender and map | Worlds | Rulebook 1 lift | Rulebook 2 lift | Both correct |", "|---|---:|---:|---:|---:|"]
        for sender, group in sorted(robust.items()):
            for name, bridge in group["bridges"].items():
                gains = [_estimate(group["books"][b][name + "_lift"]) for b in ("0", "1")]
                lines.append(f'| {sender} · {CONDITIONS[name]} | {group["coverage"]["worlds"]} | {gains[0]} | {gains[1]} | {_estimate(bridge["joint"])} |')
        lines.append("")
    lines.extend(["## Measurement", "",
                  f"Estimates weight worlds equally. Paired lifts subtract the two outcomes for each identical world, rulebook and variant "
                  f"before aggregation. Intervals are the 2.5th and 97.5th percentiles of {BOOTSTRAP_REPLICATES:,} whole-world "
                  f"bootstrap replicates (seed {BOOTSTRAP_SEED}). Two rulebooks reuse the same packet. The joint column requires both answers to be correct. "
                  "A single world has no bootstrap interval. Incomplete groups remain descriptive.", "",
                  "The summary includes median packet bytes, median and p95 latency, median GPU time, condition denominators, "
                  "and alternate-answer following when recorded. GPU time attributed to examples can overlap across batched calls.", ""])
    receipt = result["cost"]
    if receipt:
        lines.append(f'Recorded Modal spend: ${receipt["actual_usd"]:.2f}. Source: {receipt["source"]}.')
    else:
        lines.append("A reconciled Modal charge was not supplied. GPU time and allocated credit are separate from billed spend.")
    lines.extend(["", "The timing fields describe warmed, synchronized batch stages and in-process transport. "
                  "Median and p95 values use recorded batch-stage latency. They do not measure single-request network latency.",
                  "", "## Related work", "",
                  "This experiment applies a closed-form residual-state mapping to donor-private information. "
                  "[Plato’s Key](https://hazyresearch.stanford.edu/blog/2026-11-25-platos-key) studied factual substitution. "
                  "[The mismatched-cache audit](https://arxiv.org/abs/2608.04893) tested example-specific information in KV relays. "
                  "[StateBridge](https://arxiv.org/abs/2608.13317) aligns states into an input-prefix interface.",
                  "", "The [individual predictions](predictions.jsonl.gz) contain the recorded prompt, answer, target, "
                  "intervention condition and packet measurements for each case."])
    return "\n".join(lines) + "\n"


def _release_note(result):
    lines = ["# Release copy", ""]
    tests = result["groups"].get("test", {})
    if not tests:
        parts = ["# Diagnostic release", "", "The report contains no held-out evaluation."]
        for sender, item in sorted(result.get("diagnostics", {}).items()):
            v = item.get("validation", {})
            f = item.get("factual", {})
            if f:
                parts.append(f"For the {sender} sender, the factual substitution screening gate "
                             f'{"passed" if f.get("gate_passed") else "was not met"}.')
            val_group = result["groups"].get("validation", {}).get(sender)
            if val_group:
                oracle = [_percent(val_group["books"][b]["self"]["mean"]) for b in ("0", "1")]
                parts.append(f"On {val_group['coverage']['worlds']} validation worlds, the receiver’s own state reached "
                             f"{oracle[0]} and {oracle[1]} on the two rulebooks. "
                             f'The validation controls {"passed" if v.get("controls_valid") else "were not met"}.')
            if item.get("diagnostic", {}).get("reason"):
                parts.append("Recorded decision: " + str(item["diagnostic"]["reason"]) + ".")
        if result.get("diagnostics"):
            parts.append("The report links the evidence for each measured stage.")
        return "\n\n".join(parts) + "\n"
    lines.append("Can one frozen model tell another a fact only it saw?")
    for sender, group in sorted(tests.items(), key=lambda pair: pair[0] != "qwen"):
        total = group["coverage"]["complete_worlds"]
        entry = group["bridges"]["factual"]
        d = group["books"]["0"]["factual_lift"]
        if d["mean"] is None:
            lines.append(f"Paired factual-map outcomes are unavailable for {sender}.")
            continue
        lines += ["", f"For the {sender} sender, we recorded {total} complete held-out worlds. "
                  f"With a single factual-map activation, the receiver chose the correct destination on "
                  f'{_percent(group["books"]["0"]["factual"]["mean"])} of cases under the first rulebook and '
                  f'{_percent(group["books"]["1"]["factual"]["mean"])} under the second. '
                  f'The first rulebook’s paired gain over a mismatched activation was {100*d["mean"]:+.1f} percentage points. '
                  f'The full preregistered transfer gate was {"met" if entry["passes"] else "not met"}.']
    lines += ["", "Explore the recorded worlds, change a sender’s category, and check the receiver’s answer under both rulebooks. "
              "The report includes every control and the full evaluation data."]
    return "\n".join(lines) + "\n"


def _figure(results, path):
    if not results:
        for name in ("paired-lift.svg", "paired-lift.png"):
            (path / name).unlink(missing_ok=True)
        return
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import FuncFormatter
    figure = Figure(figsize=(9.0, 2.0 + 2.0 * len(results)), layout="constrained")
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    labels, ys = [], []
    for sender, group in sorted(results.items()):
        for bridge, marker in (("factual", "o"), ("contextual", "s")):
            y = len(ys)
            labels.append(f"{sender.title()} · {CONDITIONS[bridge]}")
            ys.append(y)
            for book, color, dy in (("0", "#245e85", -.12), ("1", "#b95728", .12)):
                s = group["books"][book][bridge + "_lift"]
                mean = s["mean"]
                if mean is None:
                    continue
                err = None if s["low"] is None else [[max(0, mean - s["low"])], [max(0, s["high"] - mean)]]
                ax.errorbar(mean, y + dy, xerr=err, fmt=marker, color=color, ms=8, capsize=4,
                            label=f"Rulebook {int(book)+1}" if y == 0 else None)
    ax.axvline(0, color="#8a8a80", lw=1)
    ax.set_yticks(ys, labels)
    ax.set_ylim(len(ys) - .4, -.6)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{100 * value:g}"))
    ax.set_xlabel("Accuracy difference: matched minus mismatched packet (percentage points)")
    ax.set_title("Does the receiver use the sender’s specific message?", loc="left", fontsize=15, pad=18)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=.18)
    ax.tick_params(axis="y", length=0)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.21), ncol=2, frameon=False)
    figure.savefig(path / "paired-lift.svg", format="svg", bbox_inches="tight")
    figure.savefig(path / "paired-lift.png", dpi=240, bbox_inches="tight")


def _accuracy_figure(results, path, stem, phase):
    if not results:
        for ext in ("svg", "png"):
            (path / (stem + "." + ext)).unlink(missing_ok=True)
        return
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import PercentFormatter
    arms = ("factual", "factual_wrong", "self", "word")
    figure = Figure(figsize=(8.7, 1.7 + 2.8 * len(results)), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(len(results), 1, sharex=True, squeeze=False)[:, 0]
    for ax, (sender, group) in zip(axes, sorted(results.items())):
        title = f'{sender.title()} sender · {group["coverage"]["worlds"]} worlds'
        ax.set_title(title, fontsize=12, loc="left")
        for y, name in enumerate(arms):
            for book, color, dy, symbol in (("0", "#245e85", -.13, "o"), ("1", "#b95728", .13, "s")):
                s = group["books"][book][name]
                if s["mean"] is None:
                    continue
                x = s["mean"]
                err = None if s["low"] is None else [[max(0,x-s["low"])], [max(0,s["high"]-x)]]
                ax.errorbar(x, y + dy, xerr=err, fmt=symbol, color=color, ms=6, capsize=4,
                            label=f"Rulebook {int(book)+1}" if y == 0 else None)
        ax.axvline(.125, color="#82867f", ls="--", lw=1, label="Chance (12.5%)")
        ax.set_yticks(range(len(arms)), [CONDITIONS[c] for c in arms])
        ax.set_ylim(len(arms)-.4, -.6)
        ax.set_xlim(-.025, 1.025)
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", alpha=.18)
    axes[-1].set_xlabel("Destination exact-match accuracy with 95% whole-world bootstrap intervals")
    figure.suptitle(f"How often does the receiver answer correctly?  /  {phase}", fontsize=15)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[-1].legend(handles, labels, loc="upper center", bbox_to_anchor=(.5,-.3), ncol=3, frameon=False)
    for ext in ("svg", "png"):
        figure.savefig(path / (stem + "." + ext), dpi=240, bbox_inches="tight")


def generate(input_dir, output_dir):
    root, out = Path(input_dir), Path(output_dir)
    if not root.is_dir():
        raise ValueError(f"Prediction directory does not exist: {root}")
    groups, duplicates = _rows(root)
    metadata = _metadata(root)
    summaries = defaultdict(dict)
    for (split, sender), rows in sorted(groups.items()):
        item = metadata.get(sender, {})
        summaries[split][sender] = _group_summary(rows, duplicates[split, sender], item.get("validation", {}),
                                                   split, item.get("lock_file_verified", False))
    cost = _json(root / "cost.json")
    if not (_number(cost.get("actual_usd"), low=0) and isinstance(cost.get("source"), str) and cost["source"]):
        cost = None
    diagnostics = {}
    for sender, item in metadata.items():
        diagnostics[sender] = {}
        for source, keys in (("factual", ("gate_passed", "accuracy", "test_clusters", "sender_layer", "receiver_layer")),
                             ("validation", ("complete", "controls_valid", "identity_max_abs", "worlds", "quick")),
                             ("diagnostic", ("stage", "reason")),
                             ("interface-diagnostic", ("status", "worlds", "source_split", "heldout_opened", "arms"))):
            if source in item:
                diagnostics[sender][source] = {k: item[source][k] for k in keys if k in item[source]}
    result = {"schema": 1, "groups": dict(summaries), "cost": cost, "diagnostics": diagnostics,
              "bootstrap": {"unit": "world", "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES}}
    out.mkdir(parents=True, exist_ok=True)
    _write(out / "summary.json", json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    _write(out / "report.md", _report_md(result))
    _write(out / "release-note.md", _release_note(result))
    with (out / "predictions.jsonl.gz").open("wb") as file, gzip.GzipFile(filename="", mode="wb", fileobj=file, mtime=0) as raw:
        for group_key in sorted(groups):
            for _, row in sorted(groups[group_key].items()):
                raw.write((_dump(row) + "\n").encode("utf-8"))
    manifest = {"worlds": _replay(groups, out), "conditions": CONDITIONS, "result": result}
    _write(out / "manifest.js", "window.platoManifest = " + _dump(manifest) + ";\n")
    for name in ("index.html", "replay.css", "replay.js"):
        shutil.copyfile(Path(__file__).with_name("assets") / name, out / name)
    _figure(summaries.get("test", {}), out)
    _accuracy_figure(summaries.get("test", {}), out, "condition-accuracy", "Held-out")
    _accuracy_figure(summaries.get("validation", {}), out, "validation-accuracy", "Validation")
    return result
