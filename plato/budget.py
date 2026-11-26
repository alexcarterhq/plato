"""Local transactional admission for a single experiment's Modal controller."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
import json
import math
from pathlib import Path
import sqlite3
import uuid


CAPS = {"pilot": 10, "facts": 30, "primary": 50, "replication": 50, "robustness": 20, "recovery": 10}
MAX_CREDIT_USD = Decimal("176.11")
PREPAID_BUFFER_USD = Decimal("6.11")
GPU_RATES = {"CPU": Decimal("0"), "CPU-readiness": Decimal("0"), "CPU-science": Decimal("0"),
             "L4": Decimal("0.000222"), "L40S": Decimal("0.000542")}
CPU_RATE = Decimal("0.0000131")
RAM_RATE = Decimal("0.00000222")
CORES = 2
RAM_GIB = 16
STARTUP_SECONDS = 180
SHUTDOWN_SECONDS = 30
MICRO = 1_000_000


class BudgetError(ValueError):
    pass


def micros(dollars):
    amount = Decimal(str(dollars))
    if not amount.is_finite() or amount < 0:
        raise BudgetError("Amount must be finite and nonnegative")
    return int((amount * MICRO).to_integral_value(rounding=ROUND_CEILING))


def resources(hardware):
    return {"CPU-science": (8, 32), "CPU-readiness": (4, 32)}.get(hardware, (CORES, RAM_GIB))


def rate(hardware):
    cores, ram_gib = resources(hardware)
    return GPU_RATES[hardware] + cores * CPU_RATE + ram_gib * RAM_RATE


def envelope(hardware, seconds):
    return micros(rate(hardware) * math.ceil(seconds))


class Ledger:
    def __init__(self, path):
        self.path = Path(path)

    @contextmanager
    def transaction(self, create=False):
        if not create and not self.path.is_file():
            raise BudgetError("Initialize a ledger from verified Modal billing settings first")
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        finally:
            db.close()

    def initialize(self, credit, gross_limit, gross_used, net_limit, expires, workspace):
        """With a verified $0 net limit, unknown grant expiry can stop work without charging the card."""
        now = datetime.now(timezone.utc)
        next_cycle = now.replace(year=now.year + (now.month == 12), month=now.month % 12 + 1,
                                 day=1, hour=0, minute=0, second=0, microsecond=0)
        if expires is not None and (expires.tzinfo is None or expires <= now):
            raise BudgetError("Enter a future date and time including the UTC offset")
        cutoff = next_cycle if expires is None else min(next_cycle, expires)
        if not workspace.strip():
            raise BudgetError("A verified workspace name is required")
        zero_net = net_limit is not None and micros(net_limit) == 0
        if not zero_net and (gross_limit is None or gross_used is None or expires is None):
            raise BudgetError("Prepaid-only admission requires verified credit expiry, gross limit and gross usage")
        buffer = 0 if zero_net else micros(PREPAID_BUFFER_USD)
        available = min(micros(credit), micros(MAX_CREDIT_USD)) - buffer
        if (gross_limit is None) != (gross_used is None):
            raise BudgetError("Gross limit and usage must be provided together")
        if gross_limit is not None:
            available = min(available, micros(gross_limit) - micros(gross_used) - buffer)
        if available <= 0:
            raise BudgetError("No remaining credit or gross spending headroom")
        config = {"cap": available, "expires": cutoff.isoformat(), "workspace": workspace,
                  "funding_mode": "zero_net_limit" if zero_net else "prepaid_credit",
                  "unallocated_buffer_microusd": buffer,
                  "provided_credit_expiry": expires.isoformat() if expires else None,
                  "run": uuid.uuid4().hex, "credit": str(credit),
                  "gross_limit": str(gross_limit) if gross_limit is not None else None,
                  "gross_used": str(gross_used) if gross_used is not None else None, "net_limit": str(net_limit) if net_limit is not None else None}
        with self.transaction(create=True) as db:
            db.execute("CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY, data TEXT NOT NULL)")
            if db.execute("SELECT 1 FROM settings").fetchone():
                raise BudgetError("This ledger is already initialized")
            db.execute("INSERT INTO settings VALUES (1, ?)", (json.dumps(config),))
            db.execute("""CREATE TABLE attempts (
                id TEXT PRIMARY KEY, job TEXT NOT NULL, pool TEXT NOT NULL, status TEXT NOT NULL,
                reserved INTEGER NOT NULL, charged INTEGER NOT NULL, result TEXT, opened TEXT NOT NULL
            )""")

    def config(self):
        with self.transaction() as db:
            return json.loads(db.execute("SELECT data FROM settings").fetchone()[0])

    def freeze(self, key, value):
        with self.transaction() as db:
            db.execute("CREATE TABLE IF NOT EXISTS manifest (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            previous = db.execute("SELECT value FROM manifest WHERE key=?", (key,)).fetchone()
            if previous is not None and previous["value"] != value:
                raise BudgetError(f"Frozen setting changed for {key}")
            db.execute("INSERT OR IGNORE INTO manifest VALUES (?,?)", (key, value))

    def setting(self, key):
        with self.transaction() as db:
            db.execute("CREATE TABLE IF NOT EXISTS manifest (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            row = db.execute("SELECT value FROM manifest WHERE key=?", (key,)).fetchone()
            return row["value"] if row else None

    def reserve(self, job, pool, hardware, seconds):
        if pool not in CAPS or hardware not in GPU_RATES or not 0 < seconds <= 86400:
            raise BudgetError("Invalid budget pool, hardware, or runtime")
        reserve_seconds = STARTUP_SECONDS + seconds + SHUTDOWN_SECONDS
        amount = envelope(hardware, reserve_seconds)
        with self.transaction() as db:
            config = json.loads(db.execute("SELECT data FROM settings").fetchone()[0])
            now = datetime.now(timezone.utc)
            if (datetime.fromisoformat(config["expires"]) - now).total_seconds() <= reserve_seconds:
                raise BudgetError("Reservation would extend past verified credit or billing-cycle expiry")
            parts = job.split("/")
            if len(parts) == 6 and parts[1] == "cpu" and parts[3] == "prescreen":
                # Bounded pilots and full continuations update the same shard checkpoint.
                slot = "/".join([*parts[:4], "*", parts[5]])
                if db.execute("SELECT 1 FROM attempts WHERE job GLOB ? AND status='running'", (slot,)).fetchone():
                    raise BudgetError("This prescreen shard already has a live attempt")
            rows = db.execute("SELECT status FROM attempts WHERE job=?", (job,)).fetchall()
            if any(row["status"] in {"running", "done"} for row in rows):
                raise BudgetError(f"Job already running or complete: {job}")
            # Failed provider calls and new attempts both consume credit.
            if rows:
                pool = "recovery"
            total, phase = db.execute("""SELECT COALESCE(SUM(charged),0),
                COALESCE(SUM(CASE WHEN pool=? THEN charged ELSE 0 END),0) FROM attempts""",
                (pool,)).fetchone()
            if total + amount > config["cap"] or phase + amount > micros(CAPS[pool]):
                raise BudgetError(f"Reservation exceeds remaining budget for {pool} or project")
            token = uuid.uuid4().hex
            db.execute("INSERT INTO attempts VALUES (?,?,?,'running',?,?,NULL,?)",
                       (token, job, pool, amount, amount, now.isoformat()))
        return token

    def finish(self, token, result=None, amount=None, error=None):
        """Missing result or unknown bill leaves the full reservation encumbered."""
        with self.transaction() as db:
            row = db.execute("SELECT * FROM attempts WHERE id=?", (token,)).fetchone()
            if not row or row["status"] != "running":
                raise BudgetError("Attempt is absent or already settled")
            charged = row["reserved"] if amount is None else amount
            if not isinstance(charged, int) or charged < 0:
                raise BudgetError("Expected a nonnegative microdollar charge")
            db.execute("UPDATE attempts SET status=?,charged=?,result=? WHERE id=?",
                       ("done" if result is not None else "failed", charged,
                        json.dumps(result if error is None else {"error": error}), token))

    def completed(self, job):
        with self.transaction() as db:
            row = db.execute("SELECT result FROM attempts WHERE job=? AND status='done'", (job,)).fetchone()
            return json.loads(row[0]) if row else None

    def attempts(self):
        with self.transaction() as db:
            return [dict(row) for row in db.execute("SELECT * FROM attempts ORDER BY rowid")]

    def reconcile(self, token, dollars):
        """Replace a terminal estimate with provider gross usage attributable to this attempt."""
        charge = micros(dollars)
        with self.transaction() as db:
            result = db.execute("UPDATE attempts SET charged=? WHERE id=? AND status!='running'", (charge, token))
            if result.rowcount != 1:
                raise BudgetError("Reconciliation needs a terminal attempt")
