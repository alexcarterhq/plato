"""Export aggregate Modal gross usage for an explicit set of experiment app IDs."""
import argparse
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path


def main():
    from modal.billing import workspace_billing_report

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", required=True, type=datetime.fromisoformat)
    parser.add_argument("--app", action="append", required=True, dest="apps")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.since.tzinfo is None or any(not value.startswith("ap-") for value in args.apps):
        parser.error("Provide a timezone-aware start and explicit Modal app IDs")
    now = datetime.now(timezone.utc)
    rows = workspace_billing_report(start=args.since.astimezone(timezone.utc), end=now + timedelta(hours=1),
                                    resolution="h", tag_names=["project"])
    selected = [r for r in rows if r["object_id"] in set(args.apps)
                and r.get("tags", {}).get("project") == "plato-transfer"]
    found = {r["object_id"] for r in selected}
    missing = set(args.apps) - found
    gross = sum((Decimal(str(r["cost"])) for r in selected), Decimal(0))
    report = {"actual_usd": float(gross) if not missing else None,
              "reported_so_far_usd": float(gross), "currency": "USD",
              "source": "Modal workspace billing report, hourly gross usage for named experiment apps",
              "as_of_utc": now.isoformat(), "apps_requested": len(set(args.apps)),
              "apps_reported": len(found), "provider_reporting_can_lag": True}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".partial")
    tmp.write_text(json.dumps(report, indent=2) + "\n")
    tmp.replace(args.out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
