from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import tempfile
import unittest
from pathlib import Path

from modal_app import ProviderRejectedConfiguration, attempt
from plato.budget import BudgetError, Ledger, envelope, rate, STARTUP_SECONDS, SHUTDOWN_SECONDS


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        self.ledger = Ledger(Path(self.work.name) / "ledger.sqlite")
        self.deadline = datetime.now(timezone.utc) + timedelta(days=1)

    def init(self, limit=170):
        self.ledger.initialize(limit, limit + 20, 20, 0, self.deadline, "test")

    def test_preflight(self):
        for credit, gross, spent, net in ((170, None, None, 1), (170, 50, 50, 0), (170, None, 0, 0)):
            with self.assertRaises(BudgetError):
                self.ledger.initialize(credit, gross, spent, net, self.deadline, "test")
        self.init()
        with self.assertRaises(BudgetError):
            self.init()
        self.assertEqual(self.ledger.config()["cap"], 170_000_000)

    def test_rejects_budget_crossing_with_concurrent_launches(self):
        self.init(1)

        def reserve(i):
            try:
                return self.ledger.reserve(str(i), "pilot", "L4", 1800)
            except BudgetError:
                return None

        with ThreadPoolExecutor(max_workers=8) as workers:
            tokens = list(workers.map(reserve, range(8)))
        self.assertEqual(sum(i is not None for i in tokens), 1)
        self.assertLessEqual(sum(r["charged"] for r in self.ledger.attempts()), 1_000_000)

    def test_duplicate_and_interrupted_attempt(self):
        self.init()
        first = self.ledger.reserve("worlds", "primary", "L4", 300)
        with self.assertRaises(BudgetError):
            self.ledger.reserve("worlds", "primary", "L4", 300)
        self.ledger.finish(first)
        second = self.ledger.reserve("worlds", "primary", "L4", 300)
        self.ledger.finish(second, {"score": 0.8}, 134)
        failed, done = self.ledger.attempts()
        self.assertEqual(failed["charged"], envelope("L4", STARTUP_SECONDS + 300 + SHUTDOWN_SECONDS))
        self.assertEqual(done["pool"], "recovery")
        self.assertEqual(self.ledger.completed("worlds"), {"score": 0.8})
        self.ledger.reconcile(first, 0.03)
        self.assertEqual(self.ledger.attempts()[0]["charged"], 30_000)
        with self.assertRaises(BudgetError):
            self.ledger.reserve("worlds", "primary", "L4", 300)

    def test_period_cutoff_and_phase_cap(self):
        self.init()
        with self.assertRaises(BudgetError):
            self.ledger.reserve("whole-day", "primary", "L4", 86400)
        with self.assertRaises(BudgetError):
            self.ledger.reserve("expensive", "pilot", "L40S", 36000)

    def test_wrapper_never_submits_before_admission_and_reuses_results(self):
        self.init(1)
        called = []

        def remote(token):
            called.append(token)
            return {"worlds": 8}

        with self.assertRaises(BudgetError):
            attempt(self.ledger, "too costly", "pilot", "L40S", 3600, remote)
        one = attempt(self.ledger, "can afford", "pilot", "L4", 60, remote)
        two = attempt(self.ledger, "can afford", "pilot", "L4", 60, remote)
        self.assertEqual(one, two)
        self.assertEqual(len(called), 1)

    def test_freezes_shards_and_preserves_ambiguous_remote_reservations(self):
        self.init()
        self.ledger.freeze("test/shards", "4")
        self.ledger.freeze("test/shards", "4")
        with self.assertRaises(BudgetError):
            self.ledger.freeze("test/shards", "1")

        def disconnect(token):
            raise ConnectionError("RPC acknowledgement lost")

        with self.assertRaises(ConnectionError):
            attempt(self.ledger, "pending", "primary", "L4", 30, disconnect)
        with self.assertRaises(BudgetError):
            self.ledger.reserve("pending", "primary", "L4", 30)
        self.assertEqual(self.ledger.attempts()[0]["status"], "running")

    def test_zero_net_with_credit_below_gross_ceiling(self):
        self.ledger.initialize(160, 1000, 3, 0, self.deadline, "test")
        self.assertEqual(self.ledger.config()["cap"], 160_000_000)

    def test_verified_credit_and_zero_net_need_no_gross_report_or_known_grant_expiry(self):
        self.ledger.initialize(500, None, None, 0, None, "test")
        state = self.ledger.config()
        self.assertEqual(state["cap"], 176_110_000)
        self.assertIsNone(state["gross_used"])
        self.assertIsNone(state["provided_credit_expiry"])
        self.assertGreater(datetime.fromisoformat(state["expires"]), datetime.now(timezone.utc))

    def test_prepaid_balance_with_unknown_net_limit(self):
        self.ledger.initialize("176.11", 250, 0, None, self.deadline, "test")
        config = self.ledger.config()
        self.assertEqual(config["cap"], 170_000_000)
        self.assertIsNone(config["net_limit"])
        self.assertEqual(config["funding_mode"], "prepaid_credit")
        self.assertEqual(config["unallocated_buffer_microusd"], 6_110_000)

    def test_positive_net_without_expiry_does_not_admit_prepaid_work(self):
        with self.assertRaises(BudgetError):
            self.ledger.initialize("176.11", 250, 0, 30, None, "test")

    def test_registration_denial_is_terminal_and_conservatively_charged(self):
        self.init()

        def denied(token):
            raise ProviderRejectedConfiguration("Please add a payment method to use L4 GPU functions.")

        with self.assertRaises(ProviderRejectedConfiguration):
            attempt(self.ledger, "bench-denied", "pilot", "L4", 90, denied)
        row = self.ledger.attempts()[0]
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["charged"], row["reserved"])
        self.assertIn("payment method", row["result"])

    def test_larger_readiness_cpu_uses_its_actual_resource_rate(self):
        self.init()
        self.assertEqual(rate("CPU-readiness"), 2 * rate("CPU"))
        token = self.ledger.reserve("readiness", "pilot", "CPU-readiness", 900)
        row = self.ledger.attempts()[0]
        self.assertEqual(row["id"], token)
        self.assertEqual(row["reserved"], envelope("CPU-readiness", STARTUP_SECONDS + 900 + SHUTDOWN_SECONDS))
