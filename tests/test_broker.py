import json
import tempfile
import unittest
from pathlib import Path

from switchboard import (
    BALANCED,
    COST_FIRST,
    Broker,
    ModelSpec,
    MockProvider,
    Policy,
    ProviderPool,
    ProviderUnavailable,
    Registry,
    ScriptedProvider,
    Task,
    build_retry_prompt,
    demo_registry,
    pick_auditor,
)

PASS_VERDICT = '{"pass": true, "score": 0.9, "issues": []}'
FAIL_VERDICT = '{"pass": false, "score": 0.2, "issues": ["wrong"]}'


def make_broker(policy=BALANCED, trace_path=None):
    return Broker(demo_registry(), ProviderPool([MockProvider()]), policy, trace_path)


class AuditorTests(unittest.TestCase):
    def test_auditor_is_never_the_producer(self):
        registry = demo_registry()
        for producer in registry.all():
            auditor = pick_auditor(registry, producer)
            self.assertNotEqual(auditor.model_id, producer.model_id)

    def test_passing_output_is_verified(self):
        result = make_broker().run(Task(prompt="summarize this memo", task_type="summarization"))
        self.assertTrue(result.verified)
        self.assertFalse(result.escalated)


class EscalationTests(unittest.TestCase):
    def test_failed_audit_escalates_one_tier(self):
        # FORCE_AUDIT_FAIL makes the mock auditor fail every attempt, so the
        # broker should escalate once (policy default) and then stop.
        task = Task(prompt="FORCE_AUDIT_FAIL do something easy", task_type="extraction", complexity=0.2)
        result = make_broker().run(task)
        self.assertTrue(result.escalated)
        self.assertEqual(len(result.attempts), 2)
        tiers = [a.tier for a in result.attempts]
        self.assertLess(("small", "mid", "frontier").index(tiers[0]),
                        ("small", "mid", "frontier").index(tiers[1]))
        # Every audit failed, so the final output must be flagged unverified.
        self.assertFalse(result.verified)

    def test_escalation_budget_is_respected(self):
        policy = Policy("strict", 0.5, 0.3, 0.2, max_escalations=2)
        task = Task(prompt="FORCE_AUDIT_FAIL easy task", task_type="extraction", complexity=0.2)
        result = make_broker(policy).run(task)
        self.assertLessEqual(len(result.attempts), 3)  # initial + 2 escalations

    def test_audit_disabled_skips_verification(self):
        policy = Policy("no_audit", 0.5, 0.3, 0.2, audit_enabled=False)
        result = make_broker(policy).run(Task(prompt="hello", task_type="reasoning"))
        self.assertFalse(result.verified)  # unaudited output is never "verified"
        self.assertIsNone(result.attempts[0].audit_passed)

    def test_escalation_recovers_when_the_stronger_model_passes(self):
        # MockProvider can only pass or fail wholesale; a scripted provider
        # can make the first attempt fail and the escalated one succeed, which
        # is the case escalation actually exists for.
        broker = Broker(
            demo_registry(),
            ProviderPool([ScriptedProvider({
                "atlas-small": ["weak draft"],
                "atlas-mid": ["stronger draft"],
                "atlas-frontier": [FAIL_VERDICT, PASS_VERDICT],
            }, name="mock")]),
            COST_FIRST,
        )
        result = broker.run(Task(prompt="pull the dates", task_type="extraction", complexity=0.2))
        self.assertTrue(result.escalated)
        self.assertTrue(result.verified)
        self.assertEqual(result.final_model, "atlas-mid")
        self.assertEqual(result.final_text, "stronger draft")

    def test_escalation_targets_the_best_model_for_this_task_type(self):
        # Regression: the escalation target was hardcoded to "reasoning", so a
        # failed coding task escalated to whichever model reasoned best.
        registry = Registry([
            ModelSpec("cheap", "mock", "small", 0.1, 0.5, latency="fast",
                      capabilities={"coding": 0.5, "reasoning": 0.5, "audit": 0.4}),
            ModelSpec("thinker", "mock", "frontier", 3.0, 15.0,
                      capabilities={"coding": 0.60, "reasoning": 0.99, "audit": 0.9}),
            ModelSpec("coder", "mock", "frontier", 3.0, 15.0,
                      capabilities={"coding": 0.99, "reasoning": 0.60, "audit": 0.9}),
        ])
        broker = Broker(
            registry,
            ProviderPool([ScriptedProvider(name="mock", default=FAIL_VERDICT)]),
            BALANCED,
        )
        result = broker.run(Task(prompt="FORCE fail", task_type="coding", complexity=0.3))
        self.assertEqual(result.attempts[0].model_id, "cheap")
        self.assertEqual(result.attempts[1].model_id, "coder")

    def test_attempt_roles_are_recorded(self):
        task = Task(prompt="FORCE_AUDIT_FAIL easy", task_type="extraction", complexity=0.2)
        result = make_broker().run(task)
        self.assertEqual([a.role for a in result.attempts], ["initial", "escalation"])

    def test_auditor_model_is_recorded_per_attempt(self):
        result = make_broker().run(Task(prompt="summarize", task_type="summarization"))
        self.assertEqual(result.attempts[0].auditor_model, "atlas-frontier")


class EscalationFeedbackTests(unittest.TestCase):
    """Escalation is a repair, not a blind re-roll. Re-sending the bare prompt
    to a bigger model discards the one diagnostic the first attempt produced."""

    ISSUE = "missing the third clause"

    def run_with_recorder(self, prompt="pull the dates", complexity=0.2):
        fail = json.dumps({"pass": False, "score": 0.2, "issues": [self.ISSUE]})
        provider = ScriptedProvider({
            "atlas-small": ["draft one"],
            "atlas-mid": ["draft two"],
            "atlas-frontier": [fail],  # auditor fails every attempt
        }, name="mock")
        broker = Broker(demo_registry(), ProviderPool([provider]), BALANCED)
        result = broker.run(Task(prompt=prompt, task_type="extraction", complexity=complexity))
        return result, provider.calls

    def test_first_attempt_gets_the_prompt_verbatim(self):
        _, calls = self.run_with_recorder()
        self.assertEqual(calls[0], ("atlas-small", "pull the dates"))

    def test_retry_prompt_carries_the_auditor_findings(self):
        result, calls = self.run_with_recorder()
        self.assertTrue(result.escalated)
        retry = next(p for m, p in calls if m == "atlas-mid")
        self.assertIn(self.ISSUE, retry)
        self.assertIn("was audited and failed", retry)
        self.assertIn("pull the dates", retry)  # the original ask survives

    def test_audit_grades_against_the_original_prompt_not_the_repair_briefing(self):
        # The auditor must judge the work against what was asked for, not
        # against the instructions we gave the retry. Leaking the briefing in
        # would let the second audit grade the repair process instead.
        _, calls = self.run_with_recorder()
        audit_prompts = [p for m, p in calls if m == "atlas-frontier"]
        self.assertEqual(len(audit_prompts), 2)
        for prompt in audit_prompts:
            self.assertIn("pull the dates", prompt)
            self.assertNotIn("was audited and failed", prompt)

    def test_feedback_is_flagged_per_attempt(self):
        result, _ = self.run_with_recorder()
        self.assertFalse(result.attempts[0].had_audit_feedback)
        self.assertTrue(result.attempts[1].had_audit_feedback)

    def test_no_feedback_when_the_first_attempt_passes(self):
        result = make_broker().run(Task(prompt="summarize", task_type="summarization"))
        self.assertFalse(result.attempts[0].had_audit_feedback)

    def test_feedback_survives_a_failover(self):
        # The findings describe what is wrong with the work, not why we
        # changed models, so an outage mid-escalation must not discard them.
        fail = json.dumps({"pass": False, "score": 0.2, "issues": [self.ISSUE]})
        provider = ScriptedProvider({
            "atlas-small": ["draft one"],
            "atlas-mid": [ProviderUnavailable("mid is down")],
            "atlas-frontier": [fail, "frontier draft"],
        }, name="mock")
        result = Broker(demo_registry(), ProviderPool([provider]), BALANCED).run(
            Task(prompt="pull the dates", task_type="extraction", complexity=0.2)
        )
        self.assertEqual([a.role for a in result.attempts], ["initial", "escalation", "failover"])
        self.assertFalse(result.attempts[0].had_audit_feedback)
        self.assertTrue(result.attempts[1].had_audit_feedback)
        self.assertTrue(result.attempts[2].had_audit_feedback)
        retry = next(p for m, p in provider.calls if m == "atlas-frontier" and self.ISSUE in p)
        self.assertIn("was audited and failed", retry)

    def test_a_verbose_auditor_cannot_inflate_the_retry_prompt(self):
        issues = [f"issue number {i}" for i in range(25)]
        prompt = build_retry_prompt("do the thing", issues)
        self.assertIn("issue number 0", prompt)
        self.assertNotIn("issue number 20", prompt)
        self.assertIn("(+15 further issue(s) omitted)", prompt)

    def test_no_issues_leaves_the_prompt_untouched(self):
        self.assertEqual(build_retry_prompt("do the thing", []), "do the thing")


class CrossLabTraceTests(unittest.TestCase):
    def test_same_lab_audit_is_recorded_on_the_attempt(self):
        result = make_broker().run(Task(prompt="summarize", task_type="summarization"))
        self.assertFalse(result.attempts[0].cross_lab_audit)

    def test_unaudited_attempt_has_no_cross_lab_verdict(self):
        policy = Policy("no_audit", 0.5, 0.3, 0.2, audit_enabled=False)
        result = make_broker(policy).run(Task(prompt="hello", task_type="reasoning"))
        self.assertIsNone(result.attempts[0].cross_lab_audit)

    def test_trace_surfaces_audit_independence(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "traces.jsonl"
            make_broker(trace_path=trace).run(Task(prompt="x", task_type="summarization"))
            record = json.loads(trace.read_text().strip())
            self.assertIn("final_audit_cross_lab", record)
            self.assertFalse(record["final_audit_cross_lab"])
            self.assertIn("cross_lab_audit", record["attempts"][0])


class CostAccountingTests(unittest.TestCase):
    def test_costs_are_recorded_for_generation_and_audit(self):
        result = make_broker().run(Task(prompt="summarize this memo", task_type="summarization"))
        self.assertGreater(result.generation_cost_usd, 0.0)
        self.assertGreater(result.audit_cost_usd, 0.0)
        self.assertAlmostEqual(
            result.total_cost_usd, result.generation_cost_usd + result.audit_cost_usd
        )

    def test_attempt_costs_sum_to_the_total(self):
        task = Task(prompt="FORCE_AUDIT_FAIL easy", task_type="extraction", complexity=0.2)
        result = make_broker().run(task)
        self.assertAlmostEqual(
            result.total_cost_usd, sum(a.total_cost_usd for a in result.attempts)
        )

    def test_escalation_costs_more_than_a_single_attempt(self):
        easy = make_broker().run(Task(prompt="easy", task_type="extraction", complexity=0.2))
        hard = make_broker().run(
            Task(prompt="FORCE_AUDIT_FAIL easy", task_type="extraction", complexity=0.2)
        )
        self.assertGreater(hard.total_cost_usd, easy.total_cost_usd)

    def test_baseline_is_the_strongest_model_for_the_task_type(self):
        result = make_broker().run(Task(prompt="summarize", task_type="summarization"))
        self.assertEqual(result.baseline_model, "atlas-frontier")
        self.assertGreater(result.baseline_cost_usd, 0.0)

    def test_savings_are_reported_against_that_baseline(self):
        result = make_broker().run(Task(prompt="summarize", task_type="summarization"))
        self.assertAlmostEqual(
            result.savings_vs_baseline_usd,
            result.baseline_cost_usd - result.total_cost_usd,
        )

    def test_audit_cost_is_zero_when_auditing_is_off(self):
        policy = Policy("no_audit", 0.5, 0.3, 0.2, audit_enabled=False)
        result = make_broker(policy).run(Task(prompt="hello", task_type="reasoning"))
        self.assertEqual(result.audit_cost_usd, 0.0)
        self.assertGreater(result.generation_cost_usd, 0.0)

    def test_cheaper_auditor_lowers_audit_spend(self):
        expensive = make_broker().run(Task(prompt="summarize this", task_type="summarization"))
        thrifty_policy = Policy(
            "thrifty", 0.5, 0.3, 0.2, auditor_selection="cheapest_qualified"
        )
        thrifty = make_broker(thrifty_policy).run(
            Task(prompt="summarize this", task_type="summarization")
        )
        self.assertLess(thrifty.audit_cost_usd, expensive.audit_cost_usd)


class RoutingFlagTests(unittest.TestCase):
    def test_underqualified_routing_is_surfaced_on_the_result(self):
        result = make_broker().run(
            Task(prompt="review this", task_type="legal_analysis", complexity=0.75)
        )
        self.assertTrue(result.underqualified)
        self.assertTrue(result.warnings)
        self.assertEqual(result.final_model, "atlas-frontier")


class TracingTests(unittest.TestCase):
    def test_trace_written_as_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "traces.jsonl"
            broker = make_broker(trace_path=trace)
            broker.run(Task(prompt="task one", task_type="summarization"))
            broker.run(Task(prompt="task two", task_type="extraction"))
            lines = trace.read_text().strip().splitlines()
            self.assertEqual(len(lines), 2)
            record = json.loads(lines[0])
            for key in ("chosen_model", "verified", "escalated", "rationale"):
                self.assertIn(key, record)

    def test_trace_carries_cost_accounting(self):
        # The traces are the dataset a learned router would train on, and the
        # question they need to answer is "was this route worth it".
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "traces.jsonl"
            make_broker(trace_path=trace).run(
                Task(prompt="task one", task_type="summarization")
            )
            record = json.loads(trace.read_text().strip())
            for key in (
                "generation_cost_usd",
                "audit_cost_usd",
                "total_cost_usd",
                "baseline_cost_usd",
                "baseline_model",
                "savings_vs_baseline_usd",
                "failed_over",
                "underqualified",
                "gates",
                "warnings",
            ):
                self.assertIn(key, record)
            self.assertGreater(record["total_cost_usd"], 0.0)

    def test_trace_records_per_attempt_costs(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "traces.jsonl"
            make_broker(trace_path=trace).run(
                Task(prompt="FORCE_AUDIT_FAIL easy", task_type="extraction", complexity=0.2)
            )
            record = json.loads(trace.read_text().strip())
            self.assertEqual(len(record["attempts"]), 2)
            for attempt in record["attempts"]:
                self.assertIn("cost_usd", attempt)
                self.assertIn("role", attempt)


if __name__ == "__main__":
    unittest.main()
