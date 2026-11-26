"""Capture GitHub's rolling 14-day traffic snapshot at a release milestone."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("milestone", choices=("baseline", "48h", "7d"))
    parser.add_argument("--repo", default="alexcarterhq/plato")
    parser.add_argument("--out", type=Path, default=Path("results/reach"))
    args = parser.parse_args()
    data = {"repo": args.repo, "captured_utc": datetime.now(timezone.utc).isoformat(),
            "milestone": args.milestone, "source_window": "GitHub rolling 14-day traffic"}
    for name, endpoint in (("views","views"), ("clones","clones"), ("referrers","popular/referrers")):
        result = subprocess.run(["gh","api",f"repos/{args.repo}/traffic/{endpoint}"],
                                check=True, capture_output=True, text=True)
        data[name] = json.loads(result.stdout)
    args.out.mkdir(exist_ok=True, parents=True)
    path = args.out / f"{args.milestone}.json"
    with path.open("x") as file:
        json.dump(data, file, indent=2)
        file.write("\n")
    print(path)


if __name__ == "__main__":
    main()
