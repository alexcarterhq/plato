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
    parser.add_argument("--scope", default="the named Modal apps")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.since.tzinfo is None or any(not value.startswith("ap-") for value in args.apps):
        parser.error("Provide a timezone-aware start and explicit Modal app IDs")
    now = datetime.now(timezone.utc)
    rows = workspace_billing_report(start=args.since.astimezone(timezone.utc), end=now + timedelta(hours=1),
                                    resolution="h", tag_names=["project"])
    tagged = [r for r in rows if r.get("tags", {}).get("project") == "plato-transfer"]
    requested = set(args.apps)
    selected = [r for r in tagged if r["object_id"] in requested]
    found = {r["object_id"] for r in selected}
    missing = requested - found
    gross = sum((Decimal(str(r["cost"])) for r in selected), Decimal(0))
    report = {"actual_usd": float(gross) if not missing else None,
              "reported_so_far_usd": float(gross), "currency": "USD",
              "source": "Modal workspace billing report, hourly gross usage for named experiment apps",
              "as_of_utc": now.isoformat(), "apps_requested": len(set(args.apps)),
              "apps_reported": len(found), "provider_reporting_can_lag": True,
              "since_utc": args.since.astimezone(timezone.utc).isoformat(),
              "scope": args.scope,
              "by_app_usd": {
                  app: float(sum((Decimal(str(r["cost"])) for r in selected if r["object_id"] == app), Decimal(0)))
                  if app in found else None for app in sorted(requested)
              },
              "all_project_tagged_gross_usd": float(sum((Decimal(str(r["cost"])) for r in tagged), Decimal(0))),
              "project_tag": "plato-transfer"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".partial")
    tmp.write_text(json.dumps(report, indent=2) + "\n")
    tmp.replace(args.out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
