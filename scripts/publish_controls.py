#!/usr/bin/env python3
"""Build an offline GPU control replay and deterministic, reproducible evidence archive."""
import argparse
import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from plato.control_release import generate  # noqa: E402


def clip(source, summary, path):
    import subprocess
    from clip import HEIGHT, WIDTH, _card
    with gzip.open(source/"gpu-controls/qwen/observations.jsonl.gz", "rt",encoding="utf8") as file:
        rows = [r for r in map(json.loads,file) if r["world"]==0 and r["variant"] in (0,1)]
    index = {(r["variant"],r["condition"],r["rulebook"]):r for r in rows if r["condition"]=="full"}
    def show(row):
        line = row["first_answer"] or "⟨empty⟩"
        short = line[:120] + ("…" if len(line)>120 else "")
        return f"Expected {row['target']}. SmolLM2 answered {short!r}."
    groups=summary["groups"]
    pair="original-qwen"
    m=groups[pair]["metrics"]
    first, changed, other = index[0,"full",0], index[1,"full",0], index[0,"full",1]
    frames = [
        _card("Can one frozen model tell another a fact only it saw?",
              "First, can SmolLM2 read a complete record and use its private table to choose a destination?",
              "Plato   /   Original GPU validation controls"),
        _card("Change the secret.",f"{first['category'].title()}. {show(first)}\n{changed['category'].title()}. {show(changed)}",
              "Original validation world 0   /   Same rulebook, full evidence supplied"),
        _card("Change the rulebook.",f"The category is {other['category']}. {show(other)}",
              "Original validation world 0   /   Second rulebook, full evidence supplied"),
        _card("Across 64 original worlds",
              "Qwen retrieval " + f"{m['sender_None_None']['correct']}/512.\n"
              + "SmolLM2 with all records " + f"{m['full_0_None']['correct']}/512 on the first book and "
              + f"{m['full_1_None']['correct']}/512 on the second.",
              "Eight private categories per world   /   Exact first-line answers"),
        _card("What the controls establish",summary["statement"],
              "Open the replay to inspect every prompt, answer and score."),
    ]
    subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y","-f","rawvideo","-pix_fmt","rgb24",
                    "-video_size",f"{WIDTH}x{HEIGHT}","-framerate","1/5","-i","-","-vf","fps=24","-t","25",
                    "-an","-c:v","libx264","-crf","22","-pix_fmt","yuv420p","-movflags","+faststart",str(path)],
                   input=b"".join(f.tobytes() for f in frames),check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--clip",action="store_true",help="Also render the 25-second MP4 using Pillow and ffmpeg")
    args=parser.parse_args()
    summary=generate(args.input,args.out)
    if args.clip:
        clip(args.input,summary,args.out/"gpu-controls.mp4")
    print(summary["statement"])
    print((args.out/"index.html").resolve())


if __name__ == "__main__":
    main()
