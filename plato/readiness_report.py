"""Offline reporting for the CPU validation milestone. No model or GPU is loaded here."""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import shutil

from .config import MODELS
from .readiness import ARMS, THRESHOLD, VARIANTS, WORLDS, canonical, read_shard
from .report import _summary

LABELS = {"sender": "Sender retrieves the category", "full_0": "Receiver reads the evidence · rulebook 1",
          "full_1": "Receiver reads the evidence · rulebook 2"}
ASSETS = Path(__file__).with_name("assets")


def load(root):
    configs = [p for p in Path(root).rglob("*-of-*.config.json")
               if p.parent.name == "readiness" and not p.name.startswith("._")]
    if not configs:
        raise ValueError("No readiness shard configurations found")
    world_rows, manifests = {}, []
    signature, layout = None, None
    for path in sorted(configs):
        manifest = json.loads(path.read_text())
        conf = manifest["config"]
        digest = hashlib.sha256(canonical(conf)).hexdigest()
        shard, shards = manifest["shard"], manifest["shards"]
        if (digest != manifest["config_sha256"] or (signature and signature != digest)
                or (layout and layout != shards) or not (0 <= shard < shards <= WORLDS)
                or conf.get("split") != "validation" or conf.get("device") != "cpu"
                or conf.get("precision") != "float32" or conf.get("greedy") is not True
                or conf.get("sender") != list(MODELS["qwen"]) or conf.get("receiver") != list(MODELS["smol"])
                or conf.get("worlds") != WORLDS or conf.get("threshold") != THRESHOLD
                or conf.get("variants") != VARIANTS or conf.get("max_new_tokens") != 12):
            raise ValueError(f"Readiness configuration mismatch in {path}")
        signature, layout = digest, shards
        stem = f"{shard:04d}-of-{shards:04d}"
        if path.name != stem + ".config.json":
            raise ValueError(f"Unexpected readiness shard path {path}")
        dest = path.with_name(stem + ".jsonl")
        receipt_path = path.with_name(stem + ".receipt.json")
        worlds = read_shard(dest, list(range(shard, WORLDS, shards)), digest)
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if (not dest.exists() or receipt.get("rows_sha256") != hashlib.sha256(dest.read_bytes()).hexdigest()
                    or receipt.get("config_sha256") != digest or receipt.get("complete") is not True
                    or receipt.get("worlds") != len(worlds) or len(worlds) != len(range(shard, WORLDS, shards))):
                raise ValueError(f"Invalid readiness completion receipt {receipt_path}")
        for rows in worlds:
            world_id = rows[0]["world"]
            if world_id in world_rows:
                raise ValueError(f"Duplicate readiness world {world_id}; select a single run directory")
            world_rows[world_id] = rows
        manifests.append({"config": manifest, "completed_worlds": len(worlds), "final_receipt": receipt_path.is_file()})
    if not world_rows:
        raise ValueError("No completed readiness world has been recorded")
    return [r for w in sorted(world_rows) for r in world_rows[w]], manifests


def stats(rows):
    worlds = sorted({r["world"] for r in rows})
    lanes = {}
    for arm, book in ARMS:
        selected = [r for r in rows if (r["condition"],r["rulebook"]) == (arm,book)]
        key = "sender" if book is None else f"full_{book}"
        s = _summary((r["world"], int(r["correct"])) for r in selected)
        s["correct"] = sum(r["correct"] for r in selected)
        s["max_final_accuracy"] = (s["correct"] + VARIANTS * (WORLDS-len(worlds))) / (VARIANTS * WORLDS)
        s["point_threshold_met"] = s["mean"] >= THRESHOLD
        mistakes = Counter(r["first_answer"] for r in selected if not r["correct"])
        s["common_incorrect_first_lines"] = [{"text": t, "count": n} for t,n in mistakes.most_common(3)]
        lanes[key] = s
    complete = worlds == list(range(WORLDS))
    if complete:
        decision = "prerequisites_met" if all(s["point_threshold_met"] for s in lanes.values()) else "prerequisites_failed"
    elif any(s["max_final_accuracy"] < THRESHOLD for s in lanes.values()):
        decision = "partial_threshold_out_of_reach"
    else:
        decision = "partial"
    return {"schema": 1, "milestone": "cpu_validation_readiness", "device": "cpu", "precision": "float32",
            "worlds": len(worlds), "expected_worlds": WORLDS, "world_ids": worlds, "rows": len(rows),
            "complete": complete, "threshold": THRESHOLD, "decision": decision, "heldout_opened": False,
            "lanes": lanes, "interval": "95% percentile bootstrap; resample whole worlds, 10,000 replicates, seed 73639"}


def decision_text(result):
    if result["decision"] == "prerequisites_met":
        return "Both prerequisites met the 95% accuracy threshold on all 64 CPU validation worlds."
    if result["decision"] == "prerequisites_failed":
        return "The CPU validation run finished. At least one prerequisite fell below the 95% accuracy threshold."
    if result["decision"] == "partial_threshold_out_of_reach":
        return ("This is a partial run. Even perfect accuracy on every remaining world would leave "
                "at least one prerequisite below the fixed 95% threshold.")
    return "This is a partial CPU run. The full-cohort decision remains open."


def percentage(value):
    return "unavailable" if value is None else f"{100*value:.1f}%"


def _figure(s, out):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import PercentFormatter
    figure = Figure(figsize=(8.6,3.5), layout="constrained")
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    for y,(key,label) in enumerate(LABELS.items()):
        m = s["lanes"][key]
        x = m["mean"]
        error = None if m["low"] is None else [[max(0,x-m["low"])],[max(0,m["high"]-x)]]
        ax.errorbar(x,y,xerr=error,color="#225f53" if y == 0 else "#9b502d",
                    fmt="o" if y==0 else "s",capsize=4,ms=8)
    ax.axvline(THRESHOLD,ls="--",lw=1.3,color="#546058", label="95% prerequisite")
    ax.set_title(f'CPU baseline accuracy on {s["worlds"]} of {WORLDS} validation worlds', loc="left", fontsize=15)
    ax.set_yticks(range(3),list(LABELS.values()))
    ax.set_ylim(2.65,-.6)
    ax.set_xlim(-.03,1.03); ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Exact first-answer accuracy · 95% whole-world bootstrap intervals")
    ax.spines[["top","left","right"]].set_visible(False); ax.tick_params(axis="y",length=0)
    ax.grid(axis="x",alpha=.16); ax.legend(frameon=False,loc="lower left")
    for ext in ("svg","png"):
        figure.savefig(out / ("readiness."+ext), dpi=200,bbox_inches="tight")


def generate(root, output_dir):
    """Export a measured readiness milestone from complete worlds in final or in-progress shards."""
    rows, manifests = load(root)
    result = stats(rows)
    result["statement"] = decision_text(result)
    result["config_sha256"] = manifests[0]["config"]["config_sha256"]
    out = Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    (out/"summary.json").write_text(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2)+"\n")
    (out/"provenance.json").write_text(json.dumps(manifests,ensure_ascii=False,allow_nan=False,indent=2)+"\n")
    with (out/"observations.jsonl.gz").open("wb") as file, gzip.GzipFile(filename="",fileobj=file,mode="wb",mtime=0) as raw:
        for r in rows:
            raw.write(canonical(r)+b"\n")
    payload = {"summary":result,"labels":LABELS,"rows":rows}
    (out/"readiness-data.js").write_bytes(b"window.readinessData = "+canonical(payload)+b";\n")
    for src,dst in (("readiness.html","index.html"),("readiness.js","readiness.js"),("replay.css","replay.css")):
        shutil.copyfile(ASSETS/src,out/dst)
    _figure(result,out)
    lines=["# Can these two frozen models do their part of a private-fact handoff?", "",
           result["statement"], "",
           f'This milestone records {len(rows):,} outputs from {result["worlds"]} of {WORLDS} validation worlds. '
           "Qwen2.5-1.5B reads the private evidence and names a category. SmolLM2-1.7B separately receives that evidence "
           "and one of two routing tables, and names a destination. Eight category variants share each world.", "",
           "The two CPU checks use frozen FP32 weights and the original prompts, greedy decoder, and exact first-answer scorer. "
           "The receiver receives the full evidence directly. This milestone leaves the cross-model held-out test sealed.", "",
           "| Measured condition | Correct / scored | Accuracy | 95% world-bootstrap interval |",
           "|---|---:|---:|---:|"]
    for key,label in LABELS.items():
        m=result["lanes"][key]
        ci="Unavailable with one world" if m["low"] is None else f'{percentage(m["low"])} to {percentage(m["high"])}'
        lines.append(f'| {label} | {m["correct"]} / {m["observations"]} | {percentage(m["mean"])} | {ci} |')
    lines += ["", "![CPU readiness exact-match rates](readiness.svg)", "",
              "Each sender answer is scored once. The two receiver rulebooks are reported separately. "
              "The planned threshold is 95% in each row. Confidence intervals resample entire worlds and preserve "
              "the eight related variants in each draw.", "",
              "The milestone establishes the behavior of these two prompts and checkpoints on CPU FP32. "
              "GPU BF16 performance and cross-model activation transfer remain unmeasured in this audit.", "",
              "[Explore every case](index.html) · [Download all predictions](observations.jsonl.gz) · "
              "[Run configuration and checkpoint revisions](provenance.json)."]
    (out/"report.md").write_text("\n".join(lines)+"\n")
    (out/"release-note.md").write_text("\n\n".join([
        "# CPU readiness results",result["statement"],
        f'Across {result["worlds"]} of {WORLDS} validation worlds, Qwen retrieved the private category on '
        f'{percentage(result["lanes"]["sender"]["mean"])} of variants. With the complete evidence, '
        f'SmolLM2 selected the right destination on {percentage(result["lanes"]["full_0"]["mean"])} and '
        f'{percentage(result["lanes"]["full_1"]["mean"])} of variants under the two rulebooks.',
        "Replay every model answer and inspect both rulebooks. The cross-model activation experiment remains sealed."])+"\n")
    return result
