#!/usr/bin/env python3
"""Render a 25-second captioned silent clip from a generated Plato report."""
import argparse
import gzip
import json
import math
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1280, 720
INK, MUTED, PAPER, GREEN = "#182c26", "#50655c", "#f8f8f1", "#145c49"


def _font(size):
    for name in ("DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _lines(draw, text, font, width):
    lines = []
    for paragraph in text.splitlines():
        current = ""
        for word in paragraph.split():
            candidate = (current + " " + word).strip()
            if current and draw.textlength(candidate, font=font) > width:
                lines.append(current)
                current = ""
            current = (current + " " + word).strip()
            while len(current) > 1 and draw.textlength(current, font=font) > width:
                stop = len(current) - 1
                while stop > 1 and draw.textlength(current[:stop], font=font) > width:
                    stop -= 1
                lines.append(current[:stop])
                current = current[stop:]
        lines.append(current)
    return lines


def _card(title, body, footer):
    frame = Image.new("RGB", (WIDTH, HEIGHT), PAPER)
    d = ImageDraw.Draw(frame)
    d.rectangle((70, 57, 92, 79), fill=GREEN)
    d.text((112, 51), "PLATO   /   ACTIVATION REPLAY", fill=MUTED, font=_font(26))
    d.line((70, 108, 1208, 108), fill="#b9c7bc", width=2)
    y = 145
    for line in _lines(d, title, _font(55), 1110):
        d.text((70, y), line, font=_font(55), fill=INK); y += 68
    y += 26
    body_font = 38
    while body_font > 18:
        wrapped = _lines(d, body, _font(body_font), 1120)
        if y + len(wrapped) * math.ceil(body_font * 1.35) < 625:
            break
        body_font -= 1
    for line in wrapped:
        d.text((72, y), line, font=_font(body_font), fill=INK)
        y += math.ceil(body_font * 1.35)
    d.text((70, 647), footer, fill=MUTED, font=_font(25))
    return frame


def _load_script(path, prefix, suffix):
    raw = path.read_text()
    if not raw.startswith(prefix) or not raw.strip().endswith(suffix):
        raise ValueError(f"Invalid replay asset: {path}")
    return json.loads(raw.strip()[len(prefix):-len(suffix)])


def _pct(n):
    return "unavailable" if n is None else f"{n * 100:.1f}%"


def _diagnostic_cards(report, sender, result):
    group = result["groups"].get("validation", {}).get(sender)
    validation = result["diagnostics"].get(sender, {}).get("validation", {})
    if not group or validation.get("complete") is not True:
        raise ValueError("A clip requires a complete held-out or validation cohort")
    reference = None
    with gzip.open(report / "predictions.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if (r["split"], r["sender"], r["world"], r["rulebook"], r["variant"], r["condition"]) == (
                    "validation", sender, 0, 0, 0, "factual"):
                reference = r
                break
    if reference is None:
        raise ValueError("A validation recording is required to illustrate the diagnostic")
    footer = f"{sender.title()} sender   /   Validation only   /   No held-out claim"
    def rates(lane):
        return " and ".join(_pct(group["books"][b][lane]["mean"]) for b in ("0", "1"))
    status = "met" if validation.get("controls_valid") else "was not met"
    return [
        _card("Can one frozen model tell another a fact only it saw?",
              "We tested whether the receiver could read one mapped activation before releasing held-out results.", footer),
        _card("One recorded validation case",
              f"The sender saw category {reference['category']}. The expected destination was {reference['target']}. "
              f"The receiver answered {repr(reference['answer'])[:100]}.", footer),
        _card("Check the two models separately.",
              f"Sender retrieval accuracy: {rates('sender')} across the two rulebooks.\n"
              f"Receiver with the full evidence: {rates('full')}.", footer),
        _card("Can the receiver use its own state?",
              f"Receiver-oracle accuracy: {rates('self')}.\n"
              f"Factual-map accuracy on the same validation cases: {rates('factual')}.", footer),
        _card("What the controls support",
              f"The validation gate {status}. The report contains {group['coverage']['worlds']} validation worlds "
              "and no held-out results for this sender. Explore the raw recordings and full control breakdown.", footer),
    ]


def cards(report, sender):
    manifest = _load_script(report / "manifest.js", "window.platoManifest = ", ";")
    worlds = sorted((r for r in manifest["worlds"] if r["sender"] == sender), key=lambda x: x["world"])
    if not worlds:
        return _diagnostic_cards(report, sender, manifest["result"])
    group = manifest["result"]["groups"]["test"][sender]
    checks = group["bridges"]["factual"]["checks"]
    evidence = ("complete_heldout", "lock_verified", "receiver_input_fixed", "same_packet_verified",
                "wrong_packet_verified", "targets_always_change")
    if not all(checks.get(name) for name in evidence):
        raise ValueError("A release clip requires a complete held-out cohort with verified packets and lock")
    world = worlds[0]
    data = _load_script(report / "worlds" / (world["key"] + ".js"), "window.platoWorld(", ");")
    rows = {(r["rulebook"], r["variant"], r["condition"]): r for r in data["rows"]}
    first, second = (rows[0, v, "factual"] for v in (0, 1))
    wrong, other_book = rows[0, 0, "factual_wrong"], rows[1, 0, "factual"]
    fmt = lambda row: repr(row["answer"])[:110]
    if first["receiver_prompt"] != second["receiver_prompt"]:
        raise ValueError("The first two variants have different receiver prompts")
    count = group["coverage"]["worlds"]
    fact = group["books"]["0"]["factual_lift"]
    footer = f"{sender.title()} sender   /   World {world['world']}   /   Recorded held-out predictions"
    return [
        _card("Can one frozen model tell another a fact only it saw?",
              "One model reads the category. Another has the destination rule. We pass one mapped activation between them.",
              f"{sender.title()} sender   /   Frozen factual map"),
        _card("Change only the sender’s fact.",
              f"Category {first['category']} → expected {first['target']}. The receiver answered {fmt(first)}.\n"
              f"Category {second['category']} → expected {second['target']}. It answered {fmt(second)}.", footer),
        _card("Swap in the wrong message.",
              f"The expected destination stays {first['target']}.\nMatched packet → {fmt(first)}\n"
              f"Packet for {wrong['wrong_category']} → {fmt(wrong)}", footer),
        _card("Change the receiver’s rulebook.",
              f"Category {first['category']} stays fixed.\nFirst rulebook → expected {first['target']}, answered {fmt(first)}.\n"
              f"Second rulebook → expected {other_book['target']}, answered {fmt(other_book)}.", footer),
        _card("Across the full held-out cohort",
              f"Factual map accuracy: {_pct(group['books']['0']['factual']['mean'])} in rulebook 1 and "
              f"{_pct(group['books']['1']['factual']['mean'])} in rulebook 2.\n"
              f"Rulebook 1 gain over wrong packets: {100 * fact['mean']:+.1f} pp "
              f"[{100 * fact['low']:+.1f}, {100 * fact['high']:+.1f}].\n"
              f"Correct on both rulebooks: {_pct(group['bridges']['factual']['joint']['mean'])}.",
              f"{count} worlds   /   95% world-bootstrap interval   /   See the report for gate checks"),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sender", choices=["qwen", "pythia"], default="qwen")
    args = parser.parse_args()
    frames = cards(args.report, args.sender)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-video_size", f"{WIDTH}x{HEIGHT}", "-framerate", "1/5", "-i", "-", "-vf", "fps=24",
               "-t", "25", "-an", "-c:v", "libx264", "-crf", "22", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(args.output)]
    subprocess.run(command, input=b"".join(frame.tobytes() for frame in frames), check=True)


if __name__ == "__main__":
    main()
