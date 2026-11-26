"""Independently verify GPU positive-control recordings and build an offline replay."""
from collections import defaultdict
from functools import lru_cache
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import tarfile

from .config import MODELS
from .report import _summary
from .worlds import CATEGORIES, exact_answer, first_answer, make_world

BUNDLE = "plato-gpu-controls.tar.gz"
ASSETS = Path(__file__).with_name("assets")
MODEL_LABEL = {"qwen": "Qwen2.5-1.5B", "pythia": "Pythia-1.4B-deduped", "receiver": "SmolLM2-1.7B"}
GPU = {"qwen": "NVIDIA L40S", "pythia": "NVIDIA L4"}
FILE_NAMES = {"receipt.json", "observations.jsonl.gz"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _code(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json(path):
    return json.loads(path.read_text(encoding="utf8"))


def _save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _public(name):
    path = PurePosixPath(name)
    if path.as_posix() != name or path.is_absolute() or ".." in path.parts:
        return False
    parts = path.parts
    return (parts in {("gpu-controls","billing.json"), ("gpu-benchmarks","billing.json"),
                      ("gpu-benchmarks","timings.json"), ("gpu-development","receipt.json"),
                      ("gpu-development","observations.jsonl.gz")}
            or (len(parts) == 3 and parts[0] in {"gpu-controls", "gpu-amended"}
                and parts[1] in {"qwen","pythia"} and parts[2] in FILE_NAMES))


def archive(root, out):
    paths = sorted(p for p in root.rglob("*") if not p.is_symlink() and p.is_file()
                   and _public(p.relative_to(root).as_posix()))
    target = out / BUNDLE
    with target.open("wb") as data, gzip.GzipFile(filename="", fileobj=data, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w|") as tar:
            for p in paths:
                info = tarfile.TarInfo(p.relative_to(root).as_posix())
                info.size = p.stat().st_size
                with p.open("rb") as stream:
                    tar.addfile(info, stream)
    target.with_suffix(".gz.sha256").write_text(f"{sha(target)}  {BUNDLE}\n")


def unpack(source, dest):
    total = 0
    with tarfile.open(source, "r|gz") as archive_file:
        for item in archive_file:
            _require(item.isfile() and _public(item.name), f"Unexpected release entry {item.name}")
            total += item.size
            _require(total < 2_000_000_000, "Control archive exceeds 2 GB")
            target = dest / item.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as file, archive_file.extractfile(item) as reader:
                shutil.copyfileobj(reader, file)


def is_control_archive(source):
    with tarfile.open(source, "r|gz") as tar:
        first = next(iter(tar), None)
    return first is not None and _public(first.name)


@lru_cache(maxsize=512)
def _world(split, index):
    return make_world(split, index)


def _lane(condition, book, layer):
    return f"{condition}_{book}_{layer}"


def _label(condition, book, layer):
    kind = {"sender":"Sender retrieval", "full":"Full evidence", "self":"Receiver's own vector"}[condition]
    return kind + (f" · layer {layer}" if layer is not None else "") + (f" · rulebook {book+1}" if book is not None else "")


def _verify_rows(rows, raw_world, prompt_world, world_ids, layout):
    keys, by_lane, packets = set(), defaultdict(list), {}
    for row in rows:
        i, v, arm, book, layer = [row[k] for k in ("world","variant","condition","rulebook","layer")]
        spec = (arm, book, layer)
        _require(type(i) is int and i in world_ids and type(v) is int and v in range(8)
                 and spec in layout and (i,v,spec) not in keys, f"Unexpected or repeated control {(i,v,spec)}")
        keys.add((i,v,spec))
        world = raw_world(i)
        w = prompt_world(i)
        target = world.category(v) if arm == "sender" else str(world.books[book][v])
        prompt = w.retrieval_text(v) if arm == "sender" else (
            w.receiver_text(book, evidence=w.evidence(v))[0] if arm == "full" else w.receiver_text(book)[0])
        answer = row["answer"]
        _require(isinstance(answer,str) and row["target"] == target and row["prompt"] == prompt
                 and row["category"] == CATEGORIES[v] and type(row["correct"]) is bool
                 and row["correct"] == exact_answer(answer,target), f"Prompt, target, or score mismatch at {(i,v,spec)}")
        if "rulebook_table" in row:
            expected_table = None if book is None else list(world.books[book])
            _require(row["rulebook_table"] == expected_table, "Recorded rulebook conflicts with generator")
        if "first_answer" in row:
            _require(first_answer(answer) == row["first_answer"], "First line mismatch")
        if arm == "self":
            source = w.sender_text(v)
            _require(row.get("source_prompt", source) == source, "Self vector has incorrect source prompt")
            digest = row["packet_sha256"]
            _require(isinstance(digest,str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
                     and row.get("packet_bytes",4096) == 4096, "Invalid receiver vector metadata")
            packet = packets.setdefault((i,v,layer), digest)
            _require(packet == digest, "The receiver's vector differs across rulebooks")
        by_lane[spec].append(row)
    _require(len(keys) == len(world_ids)*8*len(layout), "Incomplete world/variant/rulebook grid")
    result = {}
    for arm,book,layer in layout:
        values = by_lane[arm,book,layer]
        metric = _summary((r["world"],r["correct"]) for r in values)
        metric.update(label=_label(arm,book,layer), correct=sum(r["correct"] for r in values))
        if metric["correct"] == 0:
            metric["any_success_world_upper_95"] = -math.expm1(math.log(.05)/metric["worlds"])
        result[_lane(arm,book,layer)] = metric
    return result


def _read(folder):
    path = folder / "observations.jsonl.gz"
    receipt = _json(folder / "receipt.json")
    recorded = receipt.get("rows_sha256", receipt.get("observations_sha256"))
    if recorded is not None:
        _require(recorded == sha(path), f"Rows checksum mismatch in {folder}")
    with gzip.open(path, "rt", encoding="utf8") as file:
        return [json.loads(line) for line in file], receipt


def _direct(metrics):
    return all(m["mean"] >= .95 for k,m in metrics.items() if k.startswith(("full_","sender_")))


def _validation(folder, sender, fmt=None):
    from .interface_development import adapt
    rows, receipt = _read(folder)
    stage = "fresh amended validation" if fmt else "original validation"
    _require((receipt["protocol"] == ("fresh-amendment-bf16-v1" if fmt else "original-bf16-v1")) and receipt["sender"] == sender
             and receipt["sender_revision"] == list(MODELS[sender]) and receipt["receiver_revision"] == list(MODELS["smol"])
             and receipt["split"] == "validation" and receipt["worlds"] == 64 and receipt["rows"] == len(rows)
             and receipt["rows_sha256"] and receipt["actual_gpu"] == GPU[sender] and receipt["dtype"] == "bfloat16",
             f"Unexpected original GPU model/configuration for {sender}")
    errors = receipt["identity_by_layer"]
    expected_checks = {"sender","receiver_book0","receiver_book1"} if fmt else {"sender","receiver"}
    _require(set(errors) == expected_checks and all(len(group) == 3 for group in errors.values()),
             "Incomplete identity-replacement checks")
    values = [e for group in errors.values() for e in group.values()]
    _require(all(math.isfinite(e) and e >= 0 for e in values) and max(values) == receipt["identity_max_abs"],
             "Identity error mismatch")
    receiver_key = "receiver_book0" if fmt else "receiver"
    layers = sorted(map(int, errors[receiver_key]))
    if fmt:
        _require(set(errors["receiver_book1"]) == set(errors["receiver_book0"]) and receipt["selected_format"] == fmt
                 and receipt["validation_ids"] == [10000,10063] and receipt["precision"] == "bfloat16"
                 and all(r["interface"] == fmt and r["protocol"] == receipt["protocol"] for r in rows),
                 "Amended control lock differs from development selection")
    layout = [("sender",None,None), ("full",0,None), ("full",1,None)]
    layout += [("self",book,l) for l in layers for book in (0,1)]
    worlds = list(range(10000,10064)) if fmt else list(range(64))
    _require(all(r["sender"]==sender and r["split"]=="validation" for r in rows), "Unexpected original split or sender")
    def original(i):
        return _world("validation",i)
    metrics = _verify_rows(rows,original,lambda i: adapt(original(i),fmt) if fmt else original(i),worlds,layout)
    _require({k:[m["correct"],m["observations"]] for k,m in metrics.items()} == receipt["counts"],
             "Aggregate disagrees with provider receipt")
    passes = [l for l in layers if all(metrics[_lane("self",b,l)]["mean"] >= .8 for b in (0,1))]
    _require(_direct(metrics) == receipt["direct_pass"] and {str(l):l in passes for l in layers} == receipt["self_pass_by_layer"],
             "Gate computation disagrees with receipt")
    if max(values) > 1e-4 or not _direct(metrics) or not passes:
        decision = "failed"
    else:
        decision = "requires factual layer selection"
    info = dict(label=MODEL_LABEL[sender] + " · " + stage, stage=stage, interface=fmt or "original",
                world_ids=worlds, metrics=metrics, identity_by_layer=errors, identity_max_abs=max(values),
                self_candidate_layers=layers, both_books_self_passing_layers=passes, decision=decision,
                note=f"64 {stage} worlds, eight categories each. {GPU[sender]} in BF16. "
                     "Receiver-vector controls are shown for every candidate layer.")
    return info, rows


def _pct(v):
    return "—" if v is None else f"{100*v:.1f}%"


def _chart(summary,out):
    # Keep reproducible figures independent of the local clock and matplotlib cache.
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    with plt.rc_context({"svg.hashsalt":"plato-gpu-controls", "font.size":9}):
        fig,axes=plt.subplots(1,2,figsize=(10.5,4.3),layout="constrained",sharey=True)
        handles = {}
        for ax,donor in zip(axes,GPU):
            for key,offset,color in (("original",-.09,"#8f4929"),("amended",.09,"#166450")):
                group = summary["groups"].get(key+"-"+donor)
                if group is None:
                    continue
                layer = next(r["layer"] for r in summary["key_results"]
                             if r["sender"] == donor and r["stage"] == group["stage"])
                keys = [_lane("sender",None,None), _lane("full",0,None), _lane("full",1,None),
                        _lane("self",0,layer),_lane("self",1,layer)]
                ms=[group["metrics"][k] for k in keys]
                label = "Original" if key=="original" else "Fresh validation · " + group["interface"]
                handles[label] = ax.errorbar([m["mean"]*100 for m in ms], [i+offset for i in range(5)], xerr=[
                    [100*max(0.,m["mean"]-m["low"]) for m in ms], [100*max(0.,m["high"]-m["mean"]) for m in ms]],
                    fmt="o",capsize=3,color=color)
            for threshold,y0,y1 in ((95,-.4,2.4),(80,2.6,4.4)):
                handles["Required accuracy"] = ax.plot([threshold,threshold],[y0,y1],color="#53645d",ls="--",lw=1)[0]
            ax.set(xlim=(-3,103),ylim=(4.55,-.55),xlabel="Exact-answer accuracy (%)",
                   title=MODEL_LABEL[donor],yticks=range(5),
                   yticklabels=["Sender retrieval","Full evidence · book 1","Full evidence · book 2",
                                "Own vector · book 1","Own vector · book 2"])
            ax.spines[["top","right","left"]].set_visible(False)
            ax.grid(axis="x",alpha=.16);ax.tick_params(axis="y",length=0)
        fig.legend(handles.values(),handles.keys(),loc="outside upper center",ncols=3,frameon=False)
        fig.savefig(out/"controls.svg",metadata={"Date":None});plt.close(fig)


def _development(folder):
    from .interface_development import adapt, FORMATS, WORLD_IDS
    rows, receipt = _read(folder)
    _require(receipt["gpu"] == "NVIDIA L40S" and receipt["dtype"] == "bfloat16" and receipt["world_ids"] == list(WORLD_IDS)
             and receipt["rows"] == len(rows) and receipt["model_revisions"] == {k:list(v) for k,v in MODELS.items()}
             and receipt["observations_sha256"], "Unexpected development configuration")
    grouped = defaultdict(list)
    for r in rows:
        _require(r["world"] in WORLD_IDS and r["split"] == "bench" and r["format"] in FORMATS
                 and r["model"] in MODEL_LABEL and r["dtype"] == "bfloat16",
                 "Unexpected development format or sample")
        grouped[r["format"],r["model"]].append(r)
    for fmt in FORMATS:
        for model in ("qwen","pythia","receiver"):
            record = grouped[fmt,model]
            _require(record, f"Missing development condition for {fmt}, {model}")
            if model == "receiver":
                layers = sorted({r["layer"] for r in record if r["condition"]=="self"})
                _require(len(layers)==3, "Missing development candidate layer")
                layout = [("full",b,None) for b in (0,1)] + [("self",b,l) for l in layers for b in (0,1)]
            else:
                layout = [("sender",None,None)]
            def original(i):
                return _world("bench",i)
            metrics = _verify_rows(record,original,lambda i: adapt(original(i),fmt),WORLD_IDS,layout)
            label = f"{MODEL_LABEL[model]} · development · {fmt}"
            info = dict(label=label,stage="development",interface=fmt,world_ids=list(WORLD_IDS),metrics=metrics,
                        decision="development only", note="Eight development worlds for prompt comparison. "
                        "Receiver-only observations are shared by both donor comparisons.")
            yield f"dev-{fmt}-{model}", info, record


def _verify_selection(groups, receipt):
    from .interface_development import FORMATS
    ranks = {}
    for order,fmt in enumerate(FORMATS):
        measured = groups["dev-"+fmt+"-receiver"]["metrics"]
        rate = lambda arm, b=None, l=None: measured[_lane(arm,b,l)]["mean"]
        full = min(rate("full",b) for b in (0,1))
        layers = receipt["layers"]
        patch = max(min(rate("self",b,l) for b in (0,1)) for l in layers)
        scores = [min(groups["dev-"+fmt+"-"+m]["metrics"][_lane("sender",None,None)]["mean"]/.95,
                      full/.95,patch/.8) for m in GPU]
        ranks[fmt] = (sum(x >= 1 for x in scores),max(scores),sum(scores),full,-order)
    _require(max(FORMATS[1:],key=ranks.get) == receipt["selected_format"], "Amendment nomination cannot be reproduced")


def _add(out, groups, slug, group, rows):
    name = slug+".js"
    group["file"] = name
    public = []
    for r in rows:
        public.append({**{k:r.get(k) for k in ("world","variant","target","correct","prompt","source_prompt",
                                                 "packet_sha256","answer")},
                       "lane":_lane(r["condition"],r["rulebook"],r["layer"])})
    (out/name).write_text("window.receiveControlRows("+_code(slug)+","+_code(public)+");\n",encoding="utf8")
    groups[slug] = group


def _report(summary, out):
    lines = ["# Can each model do its part of the private-fact handoff?", "", summary["statement"], "",
             "Both senders and the shared SmolLM2 receiver are frozen. Each sender reads private records and names the "
             "queried category. The receiver either gets all records in text or gets its own evidence-derived BF16 "
             "vector at a candidate layer plus the current rulebook. Each secret has a different correct destination "
             "under the second rulebook.", "",
             "![GPU positive controls by pair and interface](controls.svg)", "",
             "Score the first nonempty answer line after trimming whitespace, removing a terminal period or exclamation "
             "mark, and folding case. Each interval uses 10,000 percentile bootstrap samples of independent worlds "
             "with seed 73639. A zero-width interval means the observed worlds all had the same accuracy. "
             "After zero successful worlds the one-sided 95% exact upper bound on the chance of any correct variant in a fresh "
             "world is 4.6% for 64 validation worlds. The required rates are 95% for exact sender and full-evidence answers and 80% for the same-model "
             "patch on both books at the same layer. Identity replacement must preserve logits within 0.0001.", ""]
    if summary.get("selected_development_format"):
        lines += [f"The eight-world development grid nominated **{summary['selected_development_format']}** "
                  "using the frozen ranking rule. Development scores were excluded from validation.", ""]
    if summary["key_results"]:
        lines += ["## Control results by interface","","| Sender | Interface | Sender retrieval | Full evidence, books 1 / 2 | "
                  "Same-model vector, books 1 / 2 | Layer |","|---|---|---:|---:|---:|---:|"]
        for row in summary["key_results"]:
            fmt = row["interface"]
            lines.append(f"| {MODEL_LABEL[row['sender']]} | {fmt} | {_pct(row['retrieval'])} | "
                         f"{_pct(row['full'][0])} / {_pct(row['full'][1])} | "
                         f"{_pct(row['self'][0])} / {_pct(row['self'][1])} | {row['layer']} |")
        lines += ["","This compact table shows the candidate patch layer with the highest accuracy on its weaker book, "
                  "with ties resolved toward the lower layer. All layers and intervals appear below.", ""]
    if "cost" in summary:
        cost = summary["cost"]
        where = cost.get("scope", "the experiment apps selected for this billing export")
        if cost.get("actual_usd") is None:
            amount = f"${cost['reported_so_far_usd']:.6f} posted so far"
        else:
            amount = f"${cost['actual_usd']:.6f} gross usage"
        lines += [f"Modal reported {amount} for {where}. Billing timestamp {cost['as_of_utc']}. "
                  "The downloadable archive includes the provider export.", ""]
    for group in summary["groups"].values():
        lines += ["## "+group["label"], "", group["note"], ""]
        if "identity_max_abs" in group:
            lines += [f"Gate outcome **{group['decision']}**. Maximum identity-logit error "
                      f"{group['identity_max_abs']:.8g}. Receiver layer indices are zero-based.", "",
                      "| Identity check | Layer | Max absolute logit change |", "|---|---:|---:|"]
            for model,layers in group["identity_by_layer"].items():
                for layer,error in sorted(layers.items(),key=lambda a:int(a[0])):
                    lines.append(f"| {model.title()} | {layer} | {error:.8g} |")
            lines.append("")
        lines += ["| Condition | Correct / scored | Accuracy | World bootstrap 95% interval |","|---|---:|---:|---:|"]
        for m in group["metrics"].values():
            lines.append(f"| {m['label']} | {m['correct']} / {m['observations']} | {_pct(m['mean'])} | "
                         f"{_pct(m['low'])} to {_pct(m['high'])} |")
        lines.append("")
    lines += ["## Reproduce", "",
              "Install the package from the release tag, then run the following command with the release asset.", "",
              "```sh",f"plato reproduce --quick --archive {BUNDLE} --out release","```","",
              "The verifier regenerates the expected category, rulebook destination and prompt for every observation. "
              "It independently scores each raw answer to a binary score and checks that the BF16 packet hashes "
              "agree across the two books. The archive includes the original raw observations and worker receipts.",
              "", "[Replay every case](index.html) · [Machine-readable results](summary.json) · "
              "[Download raw evidence](plato-gpu-controls.tar.gz)"]
    (out/"report.md").write_text("\n".join(lines)+"\n", encoding="utf8")


def generate(root, out):
    root, out = Path(root), Path(out)
    out.mkdir(parents=True,exist_ok=True)
    groups = {}
    original = root / "gpu-controls"
    for donor in GPU:
        info, rows = _validation(original/donor,donor)
        _add(out,groups,"original-"+donor,info,rows)
    dev = root/"gpu-development"
    amended = root/"gpu-amended"
    chosen = None
    if dev.is_dir():
        dev_receipt = _json(dev/"receipt.json")
        chosen = dev_receipt["selected_format"]
        for slug,info,rows in _development(dev):
            _add(out,groups,slug,info,rows)
        _verify_selection(groups,dev_receipt)
    if amended.is_dir():
        _require(bool(chosen), "Fresh amended validation needs the frozen development selection receipt")
        for donor in GPU:
            if (amended/donor).is_dir():
                amended_receipt = _json(amended/donor/"receipt.json")
                _require(amended_receipt["selection_sha256"] == sha(dev/"receipt.json")
                         and amended_receipt["original_rows_sha256"] == sha(original/donor/"observations.jsonl.gz"),
                         "Fresh validation failed provenance checks against the raw original and selection")
                info, rows = _validation(amended/donor,donor,chosen)
                _add(out,groups,"amended-"+donor,info,rows)
    original_failed = [k for k in GPU if groups["original-"+k]["decision"]=="failed"]
    amended_failed = [k for k in GPU if "amended-"+k in groups and groups["amended-"+k]["decision"]=="failed"]
    if len(original_failed) == len(amended_failed) == 2:
        statement = ("Both model pairs failed the positive controls on the original and fresh amended validation "
                     "worlds. Neither pair qualified for the 250-world test or the 2,500-world confirmation. "
                     "Cross-model transfer accuracy remains unmeasured.")
    elif len(original_failed) == 2:
        statement = ("Both model pairs failed the original GPU positive controls. The recorded measurements "
                     "show where each interface broke before a cross-model transfer test.")
    else:
        statement = ("GPU validation measured sender retrieval, full-evidence lookup and the receiver's own "
                     "evidence-derived vector. Qualifying a layer for the transfer study also requires selection on familiar facts.")
    key_results = []
    for donor in GPU:
        for phase in ("original","amended"):
            group = groups.get(phase+"-"+donor)
            if not group:
                continue
            metrics = group["metrics"]
            def rates(arm,layer=None):
                return [metrics[_lane(arm,b,layer)]["mean"] for b in (0,1)]
            layer = max(group["self_candidate_layers"],key=lambda l:min(rates("self",l)))
            key_results.append(dict(sender=donor,stage=group["stage"],interface=group["interface"],
                                    retrieval=metrics[_lane("sender",None,None)]["mean"],full=rates("full"),
                                    self=rates("self",layer),layer=layer))
    summary = {"schema":1,"key_results":key_results,"categories":list(CATEGORIES),"groups":groups,"statement":statement,
               "selected_development_format":chosen,
               "interval":"Percentile bootstrap over whole worlds, 10,000 draws, seed 73639"}
    bill = original/"billing.json"
    if bill.is_file():
        cost = _json(bill)
        amount = cost.get("actual_usd") if cost.get("actual_usd") is not None else cost.get("reported_so_far_usd")
        _require(isinstance(amount,(int,float)) and math.isfinite(amount) and amount >= 0 and cost.get("currency")=="USD"
                 and isinstance(cost.get("as_of_utc"),str) and cost.get("source"), "Malformed Modal cost report")
        summary["cost"] = cost
    _save(out/"summary.json",summary)
    # Preserve registration order in navigation and the aggregate tables.
    (out/"controls-index.js").write_text("window.controlIndex = "+json.dumps(summary,ensure_ascii=False,allow_nan=False)+";\n",encoding="utf8")
    for name,target in [("gpu-controls.html","index.html"),("gpu-controls.js","gpu-controls.js"),("replay.css","replay.css")]:
        shutil.copyfile(ASSETS/name,out/target)
    _chart(summary,out); _report(summary,out); archive(root,out)
    return summary
