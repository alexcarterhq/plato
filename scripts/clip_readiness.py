#!/usr/bin/env python3
"""Render a 25-second H.264 milestone clip from actual exported CPU observations."""
import argparse
import gzip
import json
import math
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw
from clip import _font, _lines, HEIGHT, WIDTH, INK, MUTED, GREEN, PAPER


def _card(title, body, footer):
    img=Image.new("RGB",(WIDTH,HEIGHT),PAPER)
    d=ImageDraw.Draw(img)
    d.rectangle((70,57,92,79),fill=GREEN)
    d.text((112,51),"PLATO   /   CPU READINESS AUDIT",font=_font(26),fill=MUTED)
    d.line((70,108,1208,108),fill="#b9c7bc",width=2)
    y=141
    for line in _lines(d,title,_font(55),1110):
        d.text((70,y),line,font=_font(55),fill=INK); y+=67
    y+=22
    size=38
    while size>18:
        lines=_lines(d,body,_font(size),1110)
        if y+len(lines)*math.ceil(size*1.33)<625: break
        size-=1
    for line in lines:
        d.text((70,y),line,font=_font(size),fill=INK); y+=math.ceil(size*1.33)
    d.text((70,647),footer,font=_font(24),fill=MUTED)
    return img


def cards(report):
    summary=json.loads((report/"summary.json").read_text())
    if summary.get("milestone")!="cpu_validation_readiness" or not summary.get("world_ids"):
        raise ValueError("Record actual CPU readiness worlds first")
    world=min(summary["world_ids"])
    with gzip.open(report/"observations.jsonl.gz","rt",encoding="utf-8") as f:
        rows=[r for r in map(json.loads,f) if r["world"]==world and r["variant"]==0]
    example={(r["condition"],r["rulebook"]):r for r in rows}
    a,b,c=(example[k] for k in (("sender",None),("full",0),("full",1)))
    def answer(r):
        s=r["first_answer"] or "⟨empty⟩"
        return s if len(s)<=110 else s[:109]+"…"
    def result(key):
        m=summary["lanes"][key]
        return f'{m["correct"]}/{m["observations"]} ({100*m["mean"]:.1f}%)'
    foot=f"Recorded CPU FP32 validation   /   World {world}, category {a['category']}"
    return [
        _card("Can each model do its part of the handoff?",
              "Qwen reads a private category. SmolLM2 receives the complete evidence and a routing table. We test each prerequisite separately.",
              "Frozen Qwen2.5-1.5B   /   Frozen SmolLM2-1.7B"),
        _card("Can Qwen retrieve the category?",f"Expected: {a['target']}\nFirst answer line: {answer(a)}",foot),
        _card("Can SmolLM2 apply the rule?",f"First rulebook → expected {b['target']}, answered {answer(b)}.\n"
              f"Second rulebook → expected {c['target']}, answered {answer(c)}.",foot),
        _card("The full set of observed cases",
              f"Sender retrieval: {result('sender')}.\nReceiver, rulebook 1: {result('full_0')}.\n"
              f"Receiver, rulebook 2: {result('full_1')}.",
              f'{summary["worlds"]} of 64 validation worlds   /   Eight category variants per world'),
        _card("What the CPU audit establishes",summary["statement"]+" The cross-model held-out test remains sealed.",
              "Replay every original prompt and answer in the accompanying report."),
    ]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report",type=Path);parser.add_argument("output",type=Path)
    args=parser.parse_args()
    images=cards(args.report);args.output.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-n","-f","rawvideo","-pix_fmt","rgb24",
                    "-video_size",f"{WIDTH}x{HEIGHT}","-framerate","1/5","-i","-","-vf","fps=24","-t","25",
                    "-an","-c:v","libx264","-crf","22","-pix_fmt","yuv420p","-movflags","+faststart",str(args.output)],
                   input=b"".join(i.tobytes() for i in images),check=True)


if __name__=="__main__":
    main()
