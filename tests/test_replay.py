"""Replay fidelity: run a plan, throw the objects away, rebuild from the file.

The side-by-side test is the load-bearing one. A trace that merely *looks*
complete is easy to write; one that actually reconstructs the run is the only
kind that supports offline analysis or lets someone else check this repo's
claims. So the comparison is field by field, against the live result, and it
is deliberately picky.
"""

import json
import tempfile
import unittest
from pathlib import Path

from switchboard import (
    BALANCED,
    Broker,
    MockProvider,
    ProviderPool,
    ScriptedProvider,
    Task,
    demo_registry,
)
from switchboard.replay import read_trace, replay_plans, task_records

COMPOUND = (
    "Extract the pricing from these five pages, then summarize the findings, "
    "then recommend how we should respond."
)


class ReplayHarness(unittest.TestCase):
    def run_plan_traced(self, request=COMPOUND, provider=None, policy=BALANCED):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        trace = Path(tmp.name) / "t.jsonl"
        broker = Broker(
            demo_registry(),
            ProviderPool([provider or MockProvider()]),
            policy,
            trace_path=trace,
        )
        live = broker.run_plan(request)
        return live, replay_plans(read_trace(trace)), trace


class SideBySideTests(ReplayHarness):
    """The definition-of-done test: replayed == live, field by field."""

    def test_one_plan_is_reconstructed(self):
        live, replayed, _ = self.run_plan_traced()
        self.assertEqual(len(replayed), 1)
        self.assertTrue(replayed[0].completed)

    def test_plan_level_fields_match(self):
        live, (rep,), _ = self.run_plan_traced()
        self.assertEqual(rep.request, live.plan.request)
        self.assertEqual(rep.planned_by, live.plan.planned_by)
        self.assertAlmostEqual(rep.confidence, live.plan.confidence)
        self.assertEqual(rep.rationale, live.plan.rationale)
        self.assertEqual(rep.verified, live.verified)
        self.assertEqual(rep.final_text, live.final_text)
        self.assertEqual(rep.is_split, live.is_split)

    def test_the_plan_object_itself_is_rebuilt(self):
        live, (rep,), _ = self.run_plan_traced()
        self.assertIsNotNone(rep.plan)
        self.assertEqual(rep.plan.steps, live.plan.steps)
        self.assertEqual(rep.plan.signals, live.plan.signals)

    def test_dispatch_order_is_preserved(self):
        # The order steps ran is the single thing a summary cannot tell you.
        live, (rep,), _ = self.run_plan_traced()
        self.assertEqual(rep.dispatch_order, tuple(s.step_id for s in live.steps))

    def test_every_step_matches_field_by_field(self):
        live, (rep,), _ = self.run_plan_traced()
        self.assertEqual(len(rep.steps), len(live.steps))
        for replayed_step, live_step in zip(rep.steps, live.steps):
            with self.subTest(step=live_step.step_id):
                self.assertEqual(replayed_step.step_id, live_step.step_id)
                self.assertEqual(replayed_step.task_type, live_step.step.task_type)
                self.assertAlmostEqual(replayed_step.complexity, live_step.step.complexity)
                self.assertEqual(replayed_step.final_model, live_step.result.final_model)
                self.assertEqual(replayed_step.verified, live_step.result.verified)
                self.assertEqual(replayed_step.escalated, live_step.result.escalated)
                self.assertEqual(replayed_step.output_text, live_step.result.final_text)
                self.assertEqual(replayed_step.attempts, len(live_step.result.attempts))
                self.assertTrue(replayed_step.completed)

    def test_context_threading_is_reconstructable(self):
        # Which upstream output went into which step, and whether it was cut.
        live, (rep,), _ = self.run_plan_traced()
        for replayed_step, live_step in zip(rep.steps, live.steps):
            with self.subTest(step=live_step.step_id):
                self.assertEqual(replayed_step.injected_chars, live_step.injected_chars)
                self.assertEqual(replayed_step.injected_truncated, live_step.injected_truncated)
                self.assertEqual(replayed_step.injected_from, live_step.injected_from)

    def test_costs_match_to_the_traced_precision(self):
        live, (rep,), _ = self.run_plan_traced()
        self.assertAlmostEqual(rep.routed_cost_usd, live.routed_cost_usd, places=7)
        self.assertAlmostEqual(
            rep.baseline_best_model_usd, live.baseline_best_model_usd, places=7
        )
        self.assertAlmostEqual(
            rep.baseline_single_call_usd, live.baseline_single_call_usd, places=7
        )
        self.assertEqual(rep.baseline_single_call_model, live.baseline_single_call_model)

    def test_the_modelled_baseline_stays_labelled_as_modelled(self):
        # It is the "what you would have done without Switchboard" number and
        # it was never run. A replay that forgets that is quoting a fiction.
        _, (rep,), _ = self.run_plan_traced()
        self.assertTrue(rep.baseline_single_call_is_modelled)


class MixedStreamTests(ReplayHarness):
    """Plan events and per-task summaries share one file."""

    def test_task_records_are_the_ones_without_an_event_key(self):
        live, _, trace = self.run_plan_traced()
        records = read_trace(trace)
        tasks = task_records(records)
        self.assertEqual(len(tasks), len(live.steps))
        for record in tasks:
            self.assertNotIn("event", record)
            self.assertIn("chosen_model", record)  # unchanged legacy shape

    def test_existing_task_record_shape_is_untouched(self):
        # Load-bearing: traces written before plan events must stay readable.
        _, _, trace = self.run_plan_traced()
        record = task_records(read_trace(trace))[0]
        for key in ("task_type", "policy", "chosen_model", "verified", "rationale",
                    "total_cost_usd", "attempts"):
            self.assertIn(key, record)

    def test_plain_task_runs_produce_no_plan_events(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        trace = Path(tmp.name) / "t.jsonl"
        broker = Broker(demo_registry(), ProviderPool([MockProvider()]), BALANCED,
                        trace_path=trace)
        broker.run(Task(prompt="summarize this", task_type="summarization"))
        self.assertEqual(replay_plans(read_trace(trace)), [])

    def test_several_plans_in_one_file_are_kept_apart(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        trace = Path(tmp.name) / "t.jsonl"
        broker = Broker(demo_registry(), ProviderPool([MockProvider()]), BALANCED,
                        trace_path=trace)
        broker.run_plan(COMPOUND)
        broker.run_plan("Summarize this memo.")
        replayed = replay_plans(read_trace(trace))
        self.assertEqual(len(replayed), 2)
        self.assertTrue(replayed[0].is_split)
        self.assertFalse(replayed[1].is_split)


class DegradedAndPartialTests(ReplayHarness):
    def test_a_truncated_stream_replays_as_incomplete_rather_than_vanishing(self):
        # A run that died halfway is exactly what you want to inspect.
        _, _, trace = self.run_plan_traced()
        lines = trace.read_text().splitlines()
        trace.write_text("\n".join(lines[:-1]) + "\n")  # drop plan_completed
        (rep,) = replay_plans(read_trace(trace))
        self.assertFalse(rep.completed)
        self.assertTrue(rep.steps)

    def test_a_malformed_line_raises_rather_than_being_skipped(self):
        # Silently dropping an unparseable record is how a replay quietly
        # stops matching the run it claims to reproduce.
        _, _, trace = self.run_plan_traced()
        trace.write_text(trace.read_text() + "{not json\n")
        with self.assertRaises(ValueError) as ctx:
            read_trace(trace)
        self.assertIn("not valid JSON", str(ctx.exception))

    def test_no_split_plans_replay_too(self):
        live, (rep,), _ = self.run_plan_traced(request="Summarize this memo.")
        self.assertFalse(rep.is_split)
        self.assertEqual(len(rep.steps), 1)
        self.assertEqual(rep.final_text, live.final_text)


class ExecutionSemanticsTests(ReplayHarness):
    """run_plan must not bypass anything the broker already does."""

    def test_each_step_goes_through_the_full_broker_path(self):
        live, _, _ = self.run_plan_traced()
        for step in live.steps:
            with self.subTest(step=step.step_id):
                self.assertTrue(step.result.attempts)
                self.assertTrue(step.result.routing_rationale)
                self.assertIsNotNone(step.result.attempts[-1].auditor_model)

    def test_downstream_steps_receive_upstream_output(self):
        live, _, _ = self.run_plan_traced()
        self.assertEqual(live.steps[0].injected_chars, 0)
        self.assertGreater(live.steps[1].injected_chars, 0)
        self.assertEqual(live.steps[1].injected_from, ("s1",))

    def test_context_cap_truncates_and_says_so(self):
        from switchboard import Policy

        # A tiny cap forces truncation; the point is that it is *named*.
        policy = Policy("tiny-ctx", 0.5, 0.3, 0.2, plan_context_cap_chars=10)
        live, (rep,), _ = self.run_plan_traced(policy=policy)
        downstream = live.steps[1]
        self.assertTrue(downstream.injected_truncated)
        self.assertEqual(downstream.injected_chars, 10)
        self.assertTrue(rep.steps[1].injected_truncated)

    def test_plan_verified_requires_every_step_to_pass(self):
        fail = '{"pass": false, "score": 0.1, "issues": ["no"]}'
        provider = ScriptedProvider(
            {"atlas-frontier": [fail]}, name="mock", default="draft output"
        )
        live, (rep,), _ = self.run_plan_traced(provider=provider)
        self.assertFalse(live.verified)
        self.assertFalse(rep.verified)


if __name__ == "__main__":
    unittest.main()


class PlanLevelAuditTests(ReplayHarness):
    """Per-step audits cannot see whether the assembled answer is coherent."""

    def policy(self, **kw):
        from switchboard import Policy

        return Policy("audited", 0.5, 0.3, 0.2, plan_final_audit=True, **kw)

    def test_off_by_default(self):
        live, _, _ = self.run_plan_traced()
        self.assertIsNone(live.final_audit)

    def test_when_on_it_audits_the_assembled_answer(self):
        live, _, _ = self.run_plan_traced(policy=self.policy())
        self.assertIsNotNone(live.final_audit)
        self.assertTrue(live.final_audit.auditor_model)

    def test_its_cost_lands_in_the_plan_total(self):
        plain, _, _ = self.run_plan_traced()
        audited, _, _ = self.run_plan_traced(policy=self.policy())
        self.assertGreater(audited.routed_cost_usd, plain.routed_cost_usd)

    def test_a_failed_plan_audit_makes_the_plan_unverified(self):
        # Every step can pass on its own and the assembled answer still not
        # address what was asked. That is the whole point of the extra audit.
        # Keyed on WHICH audit rather than on call ordering: the plan-level
        # audit is the only one whose prompt carries the whole original
        # request, so the test does not silently break when the step count
        # changes.
        from switchboard import Completion

        class PassStepsFailPlan:
            name = "mock"

            def complete(self, model_id, prompt, max_tokens=1024):
                if "You are auditing" in prompt:
                    whole = COMPOUND in prompt
                    verdict = (
                        '{"pass": false, "score": 0.1, "issues": ["does not answer it"]}'
                        if whole else '{"pass": true, "score": 0.9, "issues": []}'
                    )
                    return Completion(verdict, model_id, 20, 20)
                return Completion("step output", model_id, 20, 20)

        live, _, _ = self.run_plan_traced(
            provider=PassStepsFailPlan(), policy=self.policy())
        self.assertTrue(all(s.result.verified for s in live.steps))
        self.assertFalse(live.final_audit.passed)
        self.assertFalse(live.verified)
