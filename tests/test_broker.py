import json
import tempfile
import unittest
from pathlib import Path

from switchboard import (
    BALANCED,
    COST_FIRST,
    Broker,
    Completion,
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


class BareProvider:
    """A Provider double with no `synthetic` attribute at all.

    Stands in for a real vendor adapter written before this flag existed, to
    exercise the getattr(..., False) default rather than assuming every
    Provider implementation opts in.
    """

    def __init__(self, name: str, text: str) -> None:
        self.name = name
        self._text = text

    def complete(self, model_id: str, prompt: str, max_tokens: int = 1024) -> Completion:
        return Completion(text=self._text, model_id=model_id, input_tokens=10, output_tokens=10)


def two_provider_registry() -> Registry:
    """Two models on distinct provider names, for exercising cross-lab paths
    with providers that are not the built-in `MockProvider`."""
    return Registry(
        [
            ModelSpec(
                model_id="real-a",
                provider="lab-a",
                tier="frontier",
                input_cost=1.0,
                output_cost=1.0,
                latency="fast",
                capabilities={"summarization": 0.9, "audit": 0.9},
            ),
            ModelSpec(
                model_id="real-b",
                provider="lab-b",
                tier="frontier",
                input_cost=1.0,
                output_cost=1.0,
                latency="fast",
                capabilities={"summarization": 0.9, "audit": 0.9},
            ),
        ]
    )


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
            # coding=0.72 clears the qualification bar comfortably, not just
            # barely — a model scored at the bar is only as good as "unknown"
            # (see UNKNOWN_CAPABILITY_PRIOR) and legitimately loses the
            # initial routing pick to a near-perfect frontier model even
            # under a cost-conscious policy. This fixture is testing
            # escalation's target choice, not the routing tradeoff itself.
            ModelSpec("cheap", "mock", "small", 0.1, 0.5, latency="fast",
                      capabilities={"coding": 0.72, "reasoning": 0.72, "audit": 0.4}),
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

    def test_escalation_respects_the_policy_not_just_capability(self):
        # Regression: escalation maximised capability and ignored the policy,
        # so a cost-first run could fail an audit and jump to the priciest
        # model in the catalog. Moving up a tier is the quality step; which
        # model within that tier is still a cost/quality tradeoff.
        registry = Registry([
            ModelSpec("cheap", "labA", "small", 0.1, 0.5, latency="fast",
                      capabilities={"reasoning": 0.5, "audit": 0.4}),
            ModelSpec("frontier-value", "labB", "frontier", 2.0, 12.0,
                      capabilities={"reasoning": 0.92, "audit": 0.9}),
            ModelSpec("frontier-premium", "labC", "frontier", 10.0, 50.0,
                      capabilities={"reasoning": 0.95, "audit": 0.9}),
        ])
        task = Task(prompt="x", task_type="reasoning", complexity=0.3)
        pool = ProviderPool([ScriptedProvider(name=n, default=FAIL_VERDICT)
                             for n in ("labA", "labB", "labC")])

        def escalation_target(policy):
            # Exercised directly rather than through a full run: under a
            # quality-first policy the *initial* route already lands on
            # frontier, so a full run has no escalation step to observe. The
            # choice of target is the unit under test.
            broker = Broker(registry, pool, policy)
            return broker._escalation_target(
                registry.get("cheap"), task, tried=set()
            ).model_id

        self.assertEqual(escalation_target(Policy("cf", 0.30, 0.60, 0.10)), "frontier-value")
        self.assertEqual(escalation_target(Policy("qf", 0.85, 0.05, 0.10)), "frontier-premium")

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


class SyntheticProvenanceTests(unittest.TestCase):
    """evals/catalog_feedback.py must be able to tell a canned demo run from a
    measured one without guessing from model ids or suspiciously round scores.
    """

    def test_mock_provider_attempts_are_marked_synthetic(self):
        result = make_broker().run(Task(prompt="summarize", task_type="summarization"))
        self.assertTrue(result.attempts[0].synthetic)

    def test_scripted_provider_attempts_are_marked_synthetic(self):
        registry = two_provider_registry()
        pool = ProviderPool([
            ScriptedProvider({"real-a": [PASS_VERDICT]}, name="lab-a"),
            ScriptedProvider({"real-b": [PASS_VERDICT]}, name="lab-b"),
        ])
        result = Broker(registry, pool, BALANCED).run(
            Task(prompt="summarize", task_type="summarization")
        )
        self.assertTrue(result.attempts[0].synthetic)

    def test_provider_error_attempt_still_reports_synthetic(self):
        registry = two_provider_registry()
        pool = ProviderPool([
            ScriptedProvider({"real-a": [ProviderUnavailable("503")]}, name="lab-a"),
            ScriptedProvider({"real-b": [PASS_VERDICT]}, name="lab-b"),
        ])
        result = Broker(registry, pool, BALANCED).run(
            Task(prompt="summarize", task_type="summarization")
        )
        failed_attempt = next(a for a in result.attempts if a.error)
        self.assertTrue(failed_attempt.synthetic)

    def test_a_provider_without_the_attribute_defaults_to_not_synthetic(self):
        # A third-party Provider written before this flag existed must not be
        # silently mistaken for a mock — that would make real traces
        # invisible to trace-driven catalog feedback, not just mocks visible.
        registry = two_provider_registry()
        pool = ProviderPool([
            BareProvider("lab-a", "a genuine-looking answer"),
            BareProvider("lab-b", PASS_VERDICT),
        ])
        result = Broker(registry, pool, BALANCED).run(
            Task(prompt="summarize", task_type="summarization")
        )
        self.assertFalse(result.attempts[0].synthetic)

    def test_a_synthetic_auditor_taints_a_real_producer_attempt(self):
        # The producer might be real while the auditor that graded it is a
        # canned stand-in; the pass/fail on this attempt is still not
        # evidence about the producer, so it must be excluded from scoring.
        registry = two_provider_registry()
        pool = ProviderPool([
            BareProvider("lab-a", "a genuine-looking answer"),
            ScriptedProvider({"real-b": [PASS_VERDICT]}, name="lab-b"),
        ])
        result = Broker(registry, pool, BALANCED).run(
            Task(prompt="summarize", task_type="summarization")
        )
        self.assertTrue(result.attempts[0].synthetic)

    def test_trace_carries_synthetic_flag_on_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "traces.jsonl"
            make_broker(trace_path=trace).run(Task(prompt="x", task_type="summarization"))
            record = json.loads(trace.read_text().strip())
            self.assertIn("synthetic", record["attempts"][0])
            self.assertTrue(record["attempts"][0]["synthetic"])


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


def three_lab_registry_with_small() -> Registry:
    """A small-tier producer plus three frontier-tier audit-qualified models
    on three distinct labs, so a panel has enough independent seats."""
    caps = {"reasoning": 0.9, "extraction": 0.9, "audit": 0.9}
    return Registry([
        ModelSpec("small-producer", "lab-p", "small", 0.1, 0.5, latency="fast",
                  capabilities=dict(caps)),
        ModelSpec("auditor-a", "lab-a", "frontier", 1.0, 2.0, capabilities=dict(caps)),
        ModelSpec("auditor-b", "lab-b", "frontier", 1.0, 2.0, capabilities=dict(caps)),
        ModelSpec("auditor-c", "lab-c", "frontier", 1.0, 2.0, capabilities=dict(caps)),
    ])


def three_lab_pool_with_producer(auditor_scripts: dict[str, str]) -> ProviderPool:
    labs = {"auditor-a": "lab-a", "auditor-b": "lab-b", "auditor-c": "lab-c"}
    providers = [ScriptedProvider({"small-producer": ["draft"]}, name="lab-p")]
    providers += [
        ScriptedProvider({model_id: [text]}, name=labs[model_id])
        for model_id, text in auditor_scripts.items()
    ]
    return ProviderPool(providers)


class MultiAuditorPolicyValidationTests(unittest.TestCase):
    def test_multi_auditor_count_defaults_to_one(self):
        self.assertEqual(BALANCED.multi_auditor_count, 1)

    def test_multi_auditor_complexity_gate_defaults_to_none(self):
        self.assertIsNone(BALANCED.multi_auditor_complexity_gate)

    def test_rejects_a_zero_count(self):
        with self.assertRaises(ValueError):
            Policy("bad", 0.5, 0.3, 0.2, multi_auditor_count=0)

    def test_rejects_a_negative_count(self):
        with self.assertRaises(ValueError):
            Policy("bad", 0.5, 0.3, 0.2, multi_auditor_count=-1)

    def test_rejects_a_boolean_count(self):
        with self.assertRaises(ValueError):
            Policy("bad", 0.5, 0.3, 0.2, multi_auditor_count=True)

    def test_rejects_a_gate_outside_zero_one(self):
        with self.assertRaises(ValueError):
            Policy("bad", 0.5, 0.3, 0.2, multi_auditor_complexity_gate=1.5)

    def test_accepts_a_gate_of_none_explicitly(self):
        Policy("ok", 0.5, 0.3, 0.2, multi_auditor_complexity_gate=None)  # no raise

    def test_task_high_stakes_defaults_to_false(self):
        self.assertFalse(Task(prompt="x").high_stakes)


class MultiAuditorBrokerTests(unittest.TestCase):
    """Policy knob + Task flag together decide whether a task gets a panel
    instead of one auditor — the trap is letting a 2-1 split read as pass."""

    def test_default_policy_never_uses_a_panel(self):
        # multi_auditor_count defaults to 1; high_stakes alone must not be
        # enough on its own to spend a panel's worth of audit calls.
        result = make_broker().run(
            Task(prompt="summarize", task_type="summarization", high_stakes=True)
        )
        self.assertIsNone(result.attempts[0].consensus_verdicts)

    def test_high_stakes_task_triggers_a_panel(self):
        policy = Policy("panel", 0.5, 0.3, 0.2, multi_auditor_count=3)
        registry = three_lab_registry_with_small()
        pool = three_lab_pool_with_producer({
            "auditor-a": PASS_VERDICT, "auditor-b": PASS_VERDICT, "auditor-c": PASS_VERDICT,
        })
        result = Broker(registry, pool, policy).run(
            Task(prompt="x", task_type="reasoning", complexity=0.2, high_stakes=True)
        )
        self.assertIsNotNone(result.attempts[0].consensus_verdicts)
        self.assertEqual(len(result.attempts[0].consensus_verdicts), 3)
        self.assertTrue(result.attempts[0].audit_unanimous)

    def test_complexity_gate_triggers_a_panel_without_high_stakes(self):
        policy = Policy(
            "panel", 0.5, 0.3, 0.2, multi_auditor_count=3,
            multi_auditor_complexity_gate=0.7,
        )
        registry = three_lab_registry_with_small()
        pool = three_lab_pool_with_producer({
            "auditor-a": PASS_VERDICT, "auditor-b": PASS_VERDICT, "auditor-c": PASS_VERDICT,
        })
        result = Broker(registry, pool, policy).run(
            Task(prompt="x", task_type="reasoning", complexity=0.8)
        )
        self.assertIsNotNone(result.attempts[0].consensus_verdicts)

    def test_below_the_complexity_gate_and_not_high_stakes_uses_one_auditor(self):
        policy = Policy(
            "panel", 0.5, 0.3, 0.2, multi_auditor_count=3,
            multi_auditor_complexity_gate=0.7,
        )
        registry = three_lab_registry_with_small()
        pool = three_lab_pool_with_producer({
            "auditor-a": PASS_VERDICT, "auditor-b": PASS_VERDICT, "auditor-c": PASS_VERDICT,
        })
        result = Broker(registry, pool, policy).run(
            Task(prompt="x", task_type="reasoning", complexity=0.2)
        )
        self.assertIsNone(result.attempts[0].consensus_verdicts)
        self.assertIsNone(result.attempts[0].audit_unanimous)

    def test_a_split_panel_still_passes_on_majority(self):
        policy = Policy("panel", 0.5, 0.3, 0.2, multi_auditor_count=3)
        registry = three_lab_registry_with_small()
        pool = three_lab_pool_with_producer({
            "auditor-a": PASS_VERDICT, "auditor-b": PASS_VERDICT, "auditor-c": FAIL_VERDICT,
        })
        result = Broker(registry, pool, policy).run(
            Task(prompt="x", task_type="reasoning", complexity=0.2, high_stakes=True)
        )
        self.assertTrue(result.verified)
        self.assertFalse(result.attempts[0].audit_unanimous)

    def test_a_split_panel_is_not_silently_a_clean_pass(self):
        # The item's trap, checked at the Broker boundary: the disagreement
        # must be readable from the result, not just from internal objects.
        policy = Policy("panel", 0.5, 0.3, 0.2, multi_auditor_count=3)
        registry = three_lab_registry_with_small()
        pool = three_lab_pool_with_producer({
            "auditor-a": PASS_VERDICT, "auditor-b": PASS_VERDICT, "auditor-c": FAIL_VERDICT,
        })
        result = Broker(registry, pool, policy).run(
            Task(prompt="x", task_type="reasoning", complexity=0.2, high_stakes=True)
        )
        detail = result.attempts[0].consensus_verdicts
        self.assertEqual(sum(1 for d in detail if d["passed"]), 2)
        self.assertEqual(sum(1 for d in detail if not d["passed"]), 1)
        self.assertTrue(any("split" in i for i in result.attempts[0].audit_issues))

    def test_a_majority_failing_panel_escalates(self):
        policy = Policy("panel", 0.5, 0.3, 0.2, multi_auditor_count=3)
        registry = three_lab_registry_with_small()
        pool = three_lab_pool_with_producer({
            "auditor-a": FAIL_VERDICT, "auditor-b": FAIL_VERDICT, "auditor-c": PASS_VERDICT,
        })
        result = Broker(registry, pool, policy).run(
            Task(prompt="x", task_type="reasoning", complexity=0.2, high_stakes=True)
        )
        self.assertTrue(result.escalated)

    def test_panel_cost_is_larger_than_a_single_auditor_and_lands_in_audit_cost(self):
        # "the extra cost reported in the existing cost accounting" (item 7):
        # no new cost field, the panel's spend must show up in the same
        # audit_cost_usd a single auditor already reports through.
        registry = three_lab_registry_with_small()
        panel_policy = Policy("panel", 0.5, 0.3, 0.2, multi_auditor_count=3)
        panel_pool = three_lab_pool_with_producer({
            "auditor-a": PASS_VERDICT, "auditor-b": PASS_VERDICT, "auditor-c": PASS_VERDICT,
        })
        panel_result = Broker(registry, panel_pool, panel_policy).run(
            Task(prompt="x", task_type="reasoning", complexity=0.2, high_stakes=True)
        )

        single_policy = Policy("single", 0.5, 0.3, 0.2)
        single_pool = three_lab_pool_with_producer({"auditor-a": PASS_VERDICT})
        single_result = Broker(registry, single_pool, single_policy).run(
            Task(prompt="x", task_type="reasoning", complexity=0.2)
        )

        self.assertGreater(panel_result.audit_cost_usd, single_result.audit_cost_usd)
        # One attempt, no escalation (unanimous pass) — the whole panel's
        # spend is this one attempt's audit_cost_usd, and nothing else.
        self.assertEqual(len(panel_result.attempts), 1)
        self.assertAlmostEqual(
            panel_result.attempts[0].audit_cost_usd, panel_result.audit_cost_usd
        )

    def test_trace_records_final_audit_unanimous(self):
        policy = Policy("panel", 0.5, 0.3, 0.2, multi_auditor_count=3)
        registry = three_lab_registry_with_small()
        pool = three_lab_pool_with_producer({
            "auditor-a": PASS_VERDICT, "auditor-b": PASS_VERDICT, "auditor-c": FAIL_VERDICT,
        })
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "traces.jsonl"
            Broker(registry, pool, policy, trace_path=trace).run(
                Task(prompt="x", task_type="reasoning", complexity=0.2, high_stakes=True)
            )
            record = json.loads(trace.read_text().strip())
            self.assertIn("final_audit_unanimous", record)
            self.assertFalse(record["final_audit_unanimous"])
            self.assertIn("consensus_verdicts", record["attempts"][0])
            self.assertEqual(len(record["attempts"][0]["consensus_verdicts"]), 3)

    def test_trace_final_audit_unanimous_is_none_without_a_panel(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "traces.jsonl"
            make_broker(trace_path=trace).run(Task(prompt="x", task_type="summarization"))
            record = json.loads(trace.read_text().strip())
            self.assertIsNone(record["final_audit_unanimous"])


class OutputCeilingTests(unittest.TestCase):
    """Three separate truncation incidents in one session say the protocol's
    1024 default is too low once thinking tokens come out of the same budget."""

    class Recorder:
        name = "mock"

        def __init__(self):
            self.ceilings = []

        def complete(self, model_id, prompt, max_tokens=1024):
            from switchboard import Completion

            self.ceilings.append(max_tokens)
            if "You are auditing" in prompt:
                return Completion(PASS_VERDICT, model_id, 10, 10)
            return Completion("output", model_id, 10, 10)

    def test_the_policy_ceiling_reaches_generation_and_audit(self):
        provider = self.Recorder()
        policy = Policy("wide", 0.5, 0.3, 0.2, max_output_tokens=4321)
        Broker(demo_registry(), ProviderPool([provider]), policy).run(
            Task(prompt="x", task_type="summarization"))
        self.assertTrue(provider.ceilings)
        self.assertTrue(all(c == 4321 for c in provider.ceilings), provider.ceilings)

    def test_the_default_clears_the_observed_truncation_points(self):
        # deepseek-v4-flash emitted zero characters at 1024 and valid JSON at
        # 3000; claude-opus-5 spent 340 of 400 output tokens thinking.
        self.assertGreaterEqual(BALANCED.max_output_tokens, 2_000)

    def test_a_zero_ceiling_is_refused(self):
        with self.assertRaises(ValueError):
            Policy("bad", 0.5, 0.3, 0.2, max_output_tokens=0)
