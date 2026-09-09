"""Triage: prompt -> (task_type, complexity), and the honesty rules around it.

The eval suite measures whether the classifier is *good* (evals/triage_eval.py,
with a held-out set). These tests pin the behaviours that must hold regardless
of accuracy: determinism, conservative defaults, and — most importantly — that
a result never claims a model classified it when the heuristic did.
"""

import json
import tempfile
import unittest
from pathlib import Path

from switchboard import (
    BALANCED,
    TASK_TYPES,
    Broker,
    MockProvider,
    ProviderPool,
    ProviderUnavailable,
    Policy,
    ScriptedProvider,
    Task,
    classify_heuristic,
    classify_with_model,
    demo_registry,
    route,
    triage_task,
)

# The cheapest model in the demo catalog — what model triage reaches for.
CHEAPEST = "atlas-small"

LABELED = [
    ("Extract all email addresses from this contact sheet.", "extraction"),
    ("Parse these invoices and return the totals as JSON.", "extraction"),
    ("Pull the plan names out of these competitor pages.", "extraction"),
    ("Summarize this board memo for a non-executive reader.", "summarization"),
    ("tl;dr this thread.", "summarization"),
    ("Condense the research notes into key takeaways.", "summarization"),
    ("Refactor this module to remove the duplicated retry logic.", "coding"),
    ("Write a webhook handler that updates the pricing table.", "coding"),
    ("Debug this stack trace and tell me what is failing.", "coding"),
    ("Write landing page copy for our new pricing tier.", "creative"),
    ("Brainstorm ten taglines for a developer tools launch.", "creative"),
    ("Write a short story about a lighthouse keeper.", "creative"),
    ("Should we migrate off the monolith? Walk through the tradeoffs.", "reasoning"),
    ("Recommend a pricing response to the competitor's free tier.", "reasoning"),
    ("Analyze why our conversion dropped after the redesign.", "reasoning"),
]


class HeuristicTests(unittest.TestCase):
    def test_labeled_prompts_classify_correctly(self):
        for prompt, expected in LABELED:
            with self.subTest(prompt=prompt):
                self.assertEqual(classify_heuristic(prompt).task_type, expected)

    def test_always_emits_a_known_task_type(self):
        # A label the catalog has no capability data for is worse than a wrong
        # one: the router silently falls back to a flat prior for every model.
        for prompt, _ in LABELED:
            self.assertIn(classify_heuristic(prompt).task_type, TASK_TYPES)

    def test_is_deterministic(self):
        # Same prompt, same route, every run — otherwise traces stop being
        # comparable over time and the evals stop meaning anything.
        for prompt, _ in LABELED:
            results = {classify_heuristic(prompt).task_type for _ in range(5)}
            self.assertEqual(len(results), 1)

    def test_unmatched_prompt_defaults_conservatively(self):
        # Nothing matched: default to the most demanding label so an unknown
        # request routes up rather than landing on the cheapest model.
        verdict = classify_heuristic("zzzz qqqq wwww")
        self.assertEqual(verdict.task_type, "reasoning")
        self.assertEqual(verdict.confidence, 0.0)

    def test_low_confidence_is_reported_not_hidden(self):
        # A default is a guess. Callers filtering on confidence need to be able
        # to tell "matched nothing" from "matched decisively".
        self.assertEqual(classify_heuristic("zzzz qqqq").confidence, 0.0)
        self.assertGreater(classify_heuristic("Summarize this memo.").confidence, 0.5)

    def test_signals_explain_the_classification(self):
        verdict = classify_heuristic("Refactor this module.")
        self.assertTrue(verdict.signals)
        self.assertTrue(any("coding" in s for s in verdict.signals))

    def test_difficulty_cues_move_complexity(self):
        easy = classify_heuristic("Write a simple, quick script.")
        hard = classify_heuristic(
            "Write a production script from scratch handling distributed edge cases."
        )
        self.assertLess(easy.complexity, hard.complexity)

    def test_complexity_stays_in_range(self):
        extremes = [
            "simple quick basic trivial short briefly just one-line straightforward",
            "novel from scratch production distributed architecture security "
            "concurrency edge cases prove optimize migrate rollback tradeoffs "
            "ambiguous rigorous end-to-end scale compliance",
        ]
        for prompt in extremes + [p for p, _ in LABELED]:
            with self.subTest(prompt=prompt[:40]):
                self.assertGreaterEqual(classify_heuristic(prompt).complexity, 0.0)
                self.assertLessEqual(classify_heuristic(prompt).complexity, 1.0)

    def test_source_is_labeled_heuristic(self):
        verdict = classify_heuristic("Summarize this.")
        self.assertEqual(verdict.source, "heuristic")
        self.assertTrue(verdict.is_heuristic)
        self.assertIn("heuristic", verdict.describe())

    def test_code_fence_beats_prose(self):
        # A fenced block is not a hint; it is the strongest structural signal
        # available and should beat a stray verb from another category.
        self.assertEqual(
            classify_heuristic("Have a look at this:\n```\ndef f(x):\n  return x\n```").task_type,
            "coding",
        )


class ModelTriageTests(unittest.TestCase):
    """The model layer must be strictly optional and must never lie about
    having run."""

    def registry_and(self, reply):
        registry = demo_registry()
        providers = ProviderPool([ScriptedProvider({CHEAPEST: [reply]}, name="mock")])
        return registry, providers

    def test_valid_model_reply_is_used(self):
        registry, providers = self.registry_and('{"task_type": "creative", "complexity": 0.42}')
        verdict = classify_with_model("ambiguous prompt", registry, providers)
        self.assertEqual(verdict.task_type, "creative")
        self.assertAlmostEqual(verdict.complexity, 0.42)
        self.assertEqual(verdict.source, f"model:{CHEAPEST}")

    def test_prose_wrapped_json_is_recovered(self):
        registry, providers = self.registry_and(
            'Sure!\n{"task_type": "coding", "complexity": 0.6}\nHope that helps.'
        )
        self.assertEqual(
            classify_with_model("x", registry, providers).task_type, "coding"
        )

    def test_uses_the_cheapest_model(self):
        # Triage is a classification, not the work. Paying frontier rates to
        # decide where to send a cheap task defeats the point.
        registry = demo_registry()
        provider = ScriptedProvider(
            {CHEAPEST: ['{"task_type": "coding", "complexity": 0.5}']}, name="mock"
        )
        classify_with_model("x", registry, ProviderPool([provider]))
        self.assertEqual(provider.calls[0][0], CHEAPEST)

    def _assert_falls_back(self, reply_or_error, prompt="Summarize this memo."):
        registry = demo_registry()
        providers = ProviderPool([ScriptedProvider({CHEAPEST: [reply_or_error]}, name="mock")])
        verdict = classify_with_model(prompt, registry, providers)
        # The critical assertion: the fallback must NOT claim a model produced
        # it. A trace saying "model:atlas-small" when the model failed is a lie
        # that survives into every downstream analysis of the traces.
        self.assertEqual(verdict.source, "heuristic")
        self.assertEqual(verdict.task_type, classify_heuristic(prompt).task_type)
        self.assertTrue(any("model triage failed" in s for s in verdict.signals))
        return verdict

    def test_provider_outage_falls_back(self):
        self._assert_falls_back(ProviderUnavailable("classifier down"))

    def test_unparseable_reply_falls_back(self):
        self._assert_falls_back("I think this is probably a coding task?")

    def test_empty_reply_falls_back(self):
        self._assert_falls_back("")

    def test_unknown_label_falls_back(self):
        # A model inventing its own taxonomy is exactly the failure the
        # catalog cannot absorb.
        self._assert_falls_back('{"task_type": "legal_analysis", "complexity": 0.5}')

    def test_out_of_range_complexity_falls_back(self):
        self._assert_falls_back('{"task_type": "coding", "complexity": 7}')

    def test_missing_field_falls_back(self):
        self._assert_falls_back('{"task_type": "coding"}')

    def test_fallback_never_raises(self):
        # Triage is a routing input. It must not be able to take the run down.
        registry = demo_registry()
        providers = ProviderPool([ScriptedProvider({}, name="mock")])
        self.assertTrue(classify_with_model("x", registry, providers).task_type)


class TriageTaskTests(unittest.TestCase):
    def test_resolves_type_and_complexity(self):
        resolved, verdict = triage_task(Task(prompt="tl;dr this thread.", task_type="auto"))
        self.assertEqual(resolved.task_type, "summarization")
        self.assertEqual(resolved.complexity, verdict.complexity)

    def test_replaces_the_placeholder_complexity(self):
        # A caller who did not know the task type cannot have known the
        # complexity either; leaving the 0.5 default would blend a real
        # estimate with a placeholder and hide which was which.
        resolved, _ = triage_task(Task(prompt="tl;dr this.", task_type="auto"))
        self.assertNotEqual(resolved.complexity, 0.5)

    def test_preserves_the_caller_fields_it_cannot_infer(self):
        original = Task(
            prompt="Summarize this.", task_type="auto",
            est_input_tokens=4321, est_output_tokens=765, needs_fast_response=True,
            assumed_cache_hit_rate=0.6,
        )
        resolved, _ = triage_task(original)
        self.assertEqual(resolved.est_input_tokens, 4321)
        self.assertEqual(resolved.est_output_tokens, 765)
        self.assertTrue(resolved.needs_fast_response)
        self.assertEqual(resolved.prompt, original.prompt)
        self.assertEqual(resolved.assumed_cache_hit_rate, 0.6)

    def test_heuristic_is_used_unless_the_model_is_requested(self):
        registry = demo_registry()
        provider = ScriptedProvider({CHEAPEST: ['{"task_type": "creative", "complexity": 0.9}']},
                                    name="mock")
        _, verdict = triage_task(
            Task(prompt="Refactor this module.", task_type="auto"),
            registry, ProviderPool([provider]),
        )
        self.assertEqual(verdict.source, "heuristic")
        self.assertEqual(provider.calls, [])


class RouterGuardTests(unittest.TestCase):
    def test_routing_an_unresolved_task_is_refused(self):
        # Silently routing "auto" would score every model on a flat prior and
        # hand the decision to cost weight, with a confident-looking rationale.
        with self.assertRaises(ValueError) as ctx:
            route(Task(prompt="x", task_type="auto"), demo_registry(), BALANCED)
        self.assertIn("auto", str(ctx.exception))


class BrokerIntegrationTests(unittest.TestCase):
    def broker(self, **kw):
        return Broker(demo_registry(), ProviderPool([MockProvider()]), BALANCED, **kw)

    def test_auto_task_is_classified_and_routed(self):
        result = self.broker().run(
            Task(prompt="Refactor this module to remove duplicated retry logic.",
                 task_type="auto")
        )
        self.assertIsNotNone(result.triage)
        self.assertEqual(result.triage.task_type, "coding")
        self.assertTrue(result.final_model)

    def test_rationale_leads_with_the_triage_and_names_the_layer(self):
        result = self.broker().run(Task(prompt="tl;dr this thread.", task_type="auto"))
        self.assertTrue(result.routing_rationale.startswith("triage: classified as summarization"))
        self.assertIn("(heuristic)", result.routing_rationale)

    def test_explicit_task_type_skips_triage_entirely(self):
        result = self.broker().run(Task(prompt="anything", task_type="summarization"))
        self.assertIsNone(result.triage)
        self.assertNotIn("triage:", result.routing_rationale)

    def test_triage_changes_where_work_lands(self):
        # The point of the feature: two bare prompts, no caller labels, and
        # they route to different tiers.
        broker = self.broker()
        easy = broker.run(Task(prompt="tl;dr this thread.", task_type="auto"))
        hard = broker.run(Task(
            prompt="Design a distributed consensus protocol from scratch with proofs.",
            task_type="auto"))
        self.assertEqual(easy.attempts[0].tier, "small")
        self.assertEqual(hard.attempts[0].tier, "frontier")

    def test_trace_records_which_layer_classified(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "t.jsonl"
            self.broker(trace_path=trace).run(
                Task(prompt="Summarize this memo.", task_type="auto")
            )
            record = json.loads(trace.read_text().strip())
            self.assertEqual(record["triage_source"], "heuristic")
            self.assertEqual(record["task_type"], "summarization")
            self.assertIsNotNone(record["triage_confidence"])

    def test_trace_triage_fields_are_null_when_not_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "t.jsonl"
            self.broker(trace_path=trace).run(
                Task(prompt="x", task_type="summarization")
            )
            self.assertIsNone(json.loads(trace.read_text().strip())["triage_source"])

    def test_model_triage_is_opt_in_at_the_broker(self):
        # A prompt matching no keyword scores 0.00 confidence, which is what
        # sends it to the model.
        registry = demo_registry()
        provider = ScriptedProvider(name="mock", default='{"task_type": "creative", "complexity": 0.8}')
        broker = Broker(registry, ProviderPool([provider]), BALANCED, triage_use_model=True)
        result = broker.run(Task(prompt="zzzz qqqq wwww", task_type="auto"))
        self.assertEqual(result.triage.source, f"model:{CHEAPEST}")
        self.assertIn(f"model:{CHEAPEST}", result.routing_rationale)

    def test_confident_heuristic_does_not_pay_for_a_model_call(self):
        # Measured: the heuristic is right 90% of the time and its confidence
        # perfectly separates its errors. Asking a model anyway buys latency
        # and, on the tuned set, slightly *worse* accuracy.
        registry = demo_registry()
        provider = ScriptedProvider(name="mock", default='{"task_type": "creative", "complexity": 0.8}')
        broker = Broker(registry, ProviderPool([provider]), BALANCED, triage_use_model=True)
        result = broker.run(Task(prompt="Refactor this module.", task_type="auto"))
        self.assertEqual(result.triage.source, "heuristic")
        self.assertEqual(result.triage.task_type, "coding")
        # calls still contains the generation and audit; what must be absent
        # is a *triage* call, identified by the classifier prompt.
        self.assertEqual([p for _, p in provider.calls if "Classify this task" in p], [])

    def test_threshold_above_one_always_asks_the_model(self):
        registry = demo_registry()
        policy = Policy("always", 0.5, 0.3, 0.2, triage_confidence_threshold=1.01)
        provider = ScriptedProvider(name="mock", default='{"task_type": "creative", "complexity": 0.8}')
        _, verdict = triage_task(
            Task(prompt="Refactor this module.", task_type="auto"),
            registry, ProviderPool([provider]), policy, use_model=True,
        )
        self.assertEqual(verdict.source, f"model:{CHEAPEST}")

    def test_threshold_zero_never_asks_the_model(self):
        registry = demo_registry()
        policy = Policy("never", 0.5, 0.3, 0.2, triage_confidence_threshold=0.0)
        provider = ScriptedProvider(name="mock", default='{"task_type": "creative", "complexity": 0.8}')
        _, verdict = triage_task(
            Task(prompt="zzzz qqqq", task_type="auto"),
            registry, ProviderPool([provider]), policy, use_model=True,
        )
        self.assertEqual(verdict.source, "heuristic")
        self.assertEqual(provider.calls, [])


if __name__ == "__main__":
    unittest.main()
