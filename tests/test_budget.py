import json
import tempfile
import unittest
from pathlib import Path

from switchboard import BALANCED, COST_FIRST, Policy, QUALITY_FIRST
from switchboard.budget import BudgetPosition, apply_budget_pressure, budget_position


def write_trace(path, records):
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


class PolicyBudgetFieldTests(unittest.TestCase):
    def test_budget_usd_defaults_to_none(self):
        self.assertIsNone(BALANCED.budget_usd)

    def test_budget_usd_rejects_zero_and_negative(self):
        for bad in (0, -5.0):
            with self.assertRaises(ValueError):
                Policy("x", 0.5, 0.3, 0.2, budget_usd=bad)

    def test_budget_usd_rejects_bool(self):
        with self.assertRaises(ValueError):
            Policy("x", 0.5, 0.3, 0.2, budget_usd=True)

    def test_budget_usd_accepts_positive_number(self):
        p = Policy("x", 0.5, 0.3, 0.2, budget_usd=100.0)
        self.assertEqual(p.budget_usd, 100.0)

    def test_budget_period_days_rejects_non_positive(self):
        with self.assertRaises(ValueError):
            Policy("x", 0.5, 0.3, 0.2, budget_period_days=0)

    def test_budget_period_days_rejects_bool(self):
        with self.assertRaises(ValueError):
            Policy("x", 0.5, 0.3, 0.2, budget_period_days=True)

    def test_budget_max_cost_shift_must_be_in_unit_range(self):
        with self.assertRaises(ValueError):
            Policy("x", 0.5, 0.3, 0.2, budget_max_cost_shift=1.5)
        with self.assertRaises(ValueError):
            Policy("x", 0.5, 0.3, 0.2, budget_max_cost_shift=-0.1)


class BudgetPositionTests(unittest.TestCase):
    def test_raises_if_policy_has_no_budget(self):
        with self.assertRaises(ValueError):
            budget_position(BALANCED, None)

    def test_missing_trace_file_is_cold_start(self):
        policy = Policy("p", 0.5, 0.3, 0.2, budget_usd=10.0)
        pos = budget_position(policy, "/nonexistent/path/traces.jsonl")
        self.assertEqual(pos.sample_count, 0)
        self.assertEqual(pos.pressure, 0.0)
        self.assertIn("cold start", pos.describe())

    def test_none_trace_path_is_cold_start(self):
        policy = Policy("p", 0.5, 0.3, 0.2, budget_usd=10.0)
        pos = budget_position(policy, None)
        self.assertEqual(pos.sample_count, 0)
        self.assertEqual(pos.pressure, 0.0)

    def test_empty_trace_file_is_cold_start(self):
        policy = Policy("p", 0.5, 0.3, 0.2, budget_usd=10.0)
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "traces.jsonl"
            trace.write_text("")
            pos = budget_position(policy, trace)
            self.assertEqual(pos.sample_count, 0)
            self.assertEqual(pos.pressure, 0.0)

    def test_sums_spend_within_window(self):
        policy = Policy("p", 0.5, 0.3, 0.2, budget_usd=1.0, budget_period_days=30)
        now = 1_700_000_000.0
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "traces.jsonl"
            write_trace(trace, [
                {"ts": now - 86_400, "total_cost_usd": 0.20},
                {"ts": now - 2 * 86_400, "total_cost_usd": 0.30},
            ])
            pos = budget_position(policy, trace, now=now)
            self.assertAlmostEqual(pos.spend_usd, 0.50)
            self.assertEqual(pos.sample_count, 2)
            self.assertAlmostEqual(pos.pressure, 0.50)

    def test_excludes_records_outside_the_window(self):
        policy = Policy("p", 0.5, 0.3, 0.2, budget_usd=1.0, budget_period_days=7)
        now = 1_700_000_000.0
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "traces.jsonl"
            write_trace(trace, [
                {"ts": now - 86_400, "total_cost_usd": 0.40},  # inside 7d window
                {"ts": now - 30 * 86_400, "total_cost_usd": 5.00},  # long expired
            ])
            pos = budget_position(policy, trace, now=now)
            self.assertAlmostEqual(pos.spend_usd, 0.40)
            self.assertEqual(pos.sample_count, 1)

    def test_ignores_plan_event_records(self):
        # run_plan's trace_event records have no total_cost_usd/ts shape a
        # per-task summary does (they use "event" as the discriminator) --
        # task_records() must filter them out before summing.
        policy = Policy("p", 0.5, 0.3, 0.2, budget_usd=1.0)
        now = 1_700_000_000.0
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "traces.jsonl"
            write_trace(trace, [
                {"event": "plan_proposed", "ts": now, "request": "do a thing"},
                {"ts": now - 3600, "total_cost_usd": 0.10},
            ])
            pos = budget_position(policy, trace, now=now)
            self.assertAlmostEqual(pos.spend_usd, 0.10)
            self.assertEqual(pos.sample_count, 1)

    def test_pressure_clamped_to_one_when_over_budget(self):
        policy = Policy("p", 0.5, 0.3, 0.2, budget_usd=1.0)
        now = 1_700_000_000.0
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "traces.jsonl"
            write_trace(trace, [{"ts": now, "total_cost_usd": 5.0}])
            pos = budget_position(policy, trace, now=now)
            self.assertEqual(pos.pressure, 1.0)

    def test_describe_names_spend_budget_and_window(self):
        pos = BudgetPosition(spend_usd=0.5, budget_usd=1.0, period_days=30, pressure=0.5, sample_count=3)
        text = pos.describe()
        self.assertIn("0.5000", text)
        self.assertIn("1.00", text)
        self.assertIn("30d", text)
        self.assertIn("50%", text)


class ApplyBudgetPressureTests(unittest.TestCase):
    def test_zero_pressure_returns_the_same_policy(self):
        pos = BudgetPosition(0.0, 10.0, 30, 0.0, sample_count=0)
        self.assertIs(apply_budget_pressure(BALANCED, pos), BALANCED)

    def test_full_pressure_moves_the_configured_max_shift_into_cost(self):
        policy = Policy("p", 0.5, 0.3, 0.2, budget_usd=10.0, budget_max_cost_shift=0.3)
        pos = BudgetPosition(10.0, 10.0, 30, pressure=1.0, sample_count=5)
        adjusted = apply_budget_pressure(policy, pos)
        self.assertAlmostEqual(adjusted.cost_weight, 0.6)

    def test_half_pressure_moves_half_the_max_shift(self):
        policy = Policy("p", 0.5, 0.3, 0.2, budget_usd=10.0, budget_max_cost_shift=0.3)
        pos = BudgetPosition(5.0, 10.0, 30, pressure=0.5, sample_count=5)
        adjusted = apply_budget_pressure(policy, pos)
        self.assertAlmostEqual(adjusted.cost_weight, 0.45)

    def test_weights_still_sum_to_one(self):
        policy = Policy("p", 0.6, 0.1, 0.3, budget_usd=10.0)
        for pressure in (0.1, 0.4, 0.75, 1.0):
            pos = BudgetPosition(pressure * 10, 10.0, 30, pressure, sample_count=1)
            adjusted = apply_budget_pressure(policy, pos)
            total = adjusted.quality_weight + adjusted.cost_weight + adjusted.latency_weight
            self.assertAlmostEqual(total, 1.0, places=6)

    def test_shift_is_proportional_to_quality_and_latency_share(self):
        # quality:latency starts at 3:1 (0.6:0.2); the shift should come out
        # of each in that same ratio, not split evenly or all from one side.
        policy = Policy("p", 0.6, 0.2, 0.2, budget_usd=10.0, budget_max_cost_shift=0.4)
        pos = BudgetPosition(10.0, 10.0, 30, pressure=1.0, sample_count=1)
        adjusted = apply_budget_pressure(policy, pos)
        quality_lost = policy.quality_weight - adjusted.quality_weight
        latency_lost = policy.latency_weight - adjusted.latency_weight
        self.assertAlmostEqual(quality_lost / latency_lost, 3.0, places=6)

    def test_cost_first_style_policy_with_no_quality_or_latency_weight_is_untouched(self):
        # Nothing to take a shift from; must not push a weight negative.
        policy = Policy("all-cost", 0.0, 1.0, 0.0, budget_usd=10.0)
        pos = BudgetPosition(10.0, 10.0, 30, pressure=1.0, sample_count=1)
        adjusted = apply_budget_pressure(policy, pos)
        self.assertEqual(adjusted.cost_weight, 1.0)
        self.assertEqual(adjusted.quality_weight, 0.0)
        self.assertEqual(adjusted.latency_weight, 0.0)

    def test_never_pushes_a_weight_below_zero(self):
        policy = QUALITY_FIRST  # 0.85 / 0.05 / 0.10
        pos = BudgetPosition(10.0, 10.0, 30, pressure=1.0, sample_count=1)
        adjusted = apply_budget_pressure(policy, pos)
        self.assertGreaterEqual(adjusted.quality_weight, 0.0)
        self.assertGreaterEqual(adjusted.latency_weight, 0.0)
        self.assertGreaterEqual(adjusted.cost_weight, 0.0)

    def test_max_shift_of_zero_disables_the_effect(self):
        policy = Policy("p", 0.5, 0.3, 0.2, budget_usd=10.0, budget_max_cost_shift=0.0)
        pos = BudgetPosition(10.0, 10.0, 30, pressure=1.0, sample_count=1)
        adjusted = apply_budget_pressure(policy, pos)
        self.assertAlmostEqual(adjusted.cost_weight, policy.cost_weight)

    def test_adjusted_policy_keeps_the_original_name_and_thresholds(self):
        policy = Policy(
            "custom", 0.5, 0.3, 0.2, budget_usd=10.0,
            audit_pass_threshold=0.8, max_escalations=3,
        )
        pos = BudgetPosition(10.0, 10.0, 30, pressure=1.0, sample_count=1)
        adjusted = apply_budget_pressure(policy, pos)
        self.assertEqual(adjusted.name, "custom")
        self.assertEqual(adjusted.audit_pass_threshold, 0.8)
        self.assertEqual(adjusted.max_escalations, 3)


if __name__ == "__main__":
    unittest.main()
