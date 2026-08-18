"""Auditor tests that exercise judgement, not just plumbing.

MockProvider can only pass or fail wholesale, so tests written against it
confirm the audit loop is wired up and nothing more. These drive the auditor
with ScriptedProvider, putting exact verdict text on the wire — including the
malformed responses real models actually return.
"""

import unittest

from switchboard import (
    AUDIT_PROMPT_HEADER,
    AUDIT_PROMPT_TEMPLATE,
    BALANCED,
    Completion,
    ModelSpec,
    MockProvider,
    Policy,
    ProviderPool,
    ProviderUnavailable,
    Registry,
    ScriptedProvider,
    Task,
    audit,
    audit_consensus,
    demo_registry,
    pick_auditor,
    pick_auditors,
)

PRODUCER = "atlas-small"
AUDITOR = "atlas-frontier"  # highest "audit" capability in the demo catalog


def run_audit(verdict_text, policy=BALANCED, registry=None):
    """Audit a fixed output with the auditor returning exactly `verdict_text`."""
    registry = registry or demo_registry()
    producer = registry.get(PRODUCER)
    provider = ScriptedProvider({AUDITOR: [verdict_text]}, name="mock")
    from switchboard import Completion

    return audit(
        Task(prompt="summarize this", task_type="summarization"),
        Completion(text="a summary", model_id=PRODUCER),
        producer,
        registry,
        ProviderPool([provider]),
        policy,
    )


class VerdictParsingTests(unittest.TestCase):
    def test_plain_json_verdict(self):
        v = run_audit('{"pass": true, "score": 0.91, "issues": []}')
        self.assertTrue(v.passed)
        self.assertAlmostEqual(v.score, 0.91)
        self.assertEqual(v.auditor_model, AUDITOR)

    def test_fenced_json_is_tolerated(self):
        v = run_audit('```json\n{"pass": true, "score": 0.88, "issues": []}\n```')
        self.assertTrue(v.passed)
        self.assertAlmostEqual(v.score, 0.88)

    def test_bare_fence_without_language_tag(self):
        v = run_audit('```\n{"pass": true, "score": 0.8, "issues": []}\n```')
        self.assertTrue(v.passed)

    def test_json_embedded_in_prose_is_recovered(self):
        # Rejecting this would fail closed on formatting, not on quality, and
        # bill an escalation for the privilege.
        v = run_audit(
            'Sure! Here is my assessment:\n'
            '{"pass": true, "score": 0.82, "issues": ["minor nit"]}\n'
            'Let me know if you need more.'
        )
        self.assertTrue(v.passed)
        self.assertEqual(v.issues, ["minor nit"])

    def test_braces_inside_strings_do_not_break_extraction(self):
        v = run_audit('{"pass": false, "score": 0.2, "issues": ["bad {json} in output"]}')
        self.assertFalse(v.passed)
        self.assertEqual(v.issues, ["bad {json} in output"])

    def test_unparseable_verdict_fails_closed(self):
        v = run_audit("I think it looks pretty good honestly")
        self.assertFalse(v.passed)
        self.assertEqual(v.score, 0.0)
        self.assertTrue(any("unparseable" in i for i in v.issues))

    def test_empty_verdict_fails_closed(self):
        v = run_audit("   ")
        self.assertFalse(v.passed)
        self.assertTrue(any("empty" in i for i in v.issues))

    def test_non_object_json_fails_closed(self):
        v = run_audit("[1, 2, 3]")
        self.assertFalse(v.passed)

    def test_score_outside_range_fails_closed(self):
        # Clamping would launder a schema-violating response into a pass.
        v = run_audit('{"pass": true, "score": 9.5, "issues": []}')
        self.assertFalse(v.passed)
        self.assertTrue(any("outside [0, 1]" in i for i in v.issues))

    def test_non_numeric_score_fails_closed(self):
        v = run_audit('{"pass": true, "score": "excellent", "issues": []}')
        self.assertFalse(v.passed)
        self.assertTrue(any("non-numeric" in i for i in v.issues))

    def test_nan_score_fails_closed(self):
        v = run_audit('{"pass": true, "score": NaN, "issues": []}')
        self.assertFalse(v.passed)

    def test_stringified_boolean_is_tolerated(self):
        v = run_audit('{"pass": "true", "score": 0.9, "issues": []}')
        self.assertTrue(v.passed)

    def test_issues_as_bare_string_is_coerced(self):
        v = run_audit('{"pass": false, "score": 0.3, "issues": "missing the third clause"}')
        self.assertEqual(v.issues, ["missing the third clause"])


class ThresholdTests(unittest.TestCase):
    def test_pass_below_policy_threshold_is_a_failure(self):
        v = run_audit('{"pass": true, "score": 0.5, "issues": []}')
        self.assertFalse(v.passed)

    def test_threshold_failure_explains_itself(self):
        # A trace reader must be able to tell "the auditor failed it" from
        # "the auditor passed it and the policy overruled".
        v = run_audit('{"pass": true, "score": 0.5, "issues": []}')
        self.assertTrue(any("below policy threshold" in i for i in v.issues))

    def test_lenient_policy_accepts_the_same_score(self):
        lenient = Policy("lenient", 0.5, 0.3, 0.2, audit_pass_threshold=0.4)
        v = run_audit('{"pass": true, "score": 0.5, "issues": []}', policy=lenient)
        self.assertTrue(v.passed)


class AuditorSelectionTests(unittest.TestCase):
    def test_auditor_is_never_the_producer(self):
        registry = demo_registry()
        for producer in registry.all():
            self.assertNotEqual(pick_auditor(registry, producer).model_id, producer.model_id)

    def test_independence_holds_under_every_selection_strategy(self):
        registry = demo_registry()
        for strategy in ("most_capable", "cheapest_qualified"):
            policy = Policy("p", 0.5, 0.3, 0.2, auditor_selection=strategy)
            for producer in registry.all():
                chosen = pick_auditor(registry, producer, policy)
                self.assertNotEqual(chosen.model_id, producer.model_id, strategy)

    def test_most_capable_is_the_default(self):
        registry = demo_registry()
        chosen = pick_auditor(registry, registry.get("atlas-small"), BALANCED)
        self.assertEqual(chosen.model_id, "atlas-frontier")

    def test_cheapest_qualified_avoids_paying_frontier_rates(self):
        registry = demo_registry()
        policy = Policy("thrifty", 0.5, 0.3, 0.2, auditor_selection="cheapest_qualified")
        chosen = pick_auditor(registry, registry.get("atlas-small"), policy)
        # atlas-mid clears the 0.7 audit floor at a fifth of frontier's price.
        self.assertEqual(chosen.model_id, "atlas-mid")

    def test_cheapest_qualified_falls_back_when_nothing_clears_the_floor(self):
        registry = demo_registry()
        policy = Policy(
            "impossible", 0.5, 0.3, 0.2,
            auditor_selection="cheapest_qualified",
            min_auditor_capability=0.99,
        )
        chosen = pick_auditor(registry, registry.get("atlas-small"), policy)
        self.assertEqual(chosen.model_id, "atlas-frontier")

    def test_audit_cost_is_recorded(self):
        v = run_audit('{"pass": true, "score": 0.9, "issues": []}')
        self.assertGreater(v.cost_usd, 0.0)
        self.assertGreater(v.input_tokens, 0)


class OtherLabMock(MockProvider):
    """A second lab, for cross-lab audit tests."""

    name = "otherlab"


def two_lab_registry():
    """The same capable model offered by two different providers."""
    caps = {"reasoning": 0.9, "extraction": 0.8, "summarization": 0.85, "audit": 0.9}
    return Registry([
        ModelSpec("lab-a-model", "mock", "frontier", 1.0, 2.0, capabilities=dict(caps)),
        ModelSpec("lab-b-model", "otherlab", "frontier", 1.0, 2.0, capabilities=dict(caps)),
    ])


class CrossLabTests(unittest.TestCase):
    """Two models from one lab share training data and alignment, so their
    blind spots correlate. A same-lab pass is weaker evidence than the number
    suggests, and the verdict should say which kind of audit it was."""

    def test_auditor_prefers_a_different_lab(self):
        registry = two_lab_registry()
        for producer in registry.all():
            self.assertNotEqual(
                pick_auditor(registry, producer, BALANCED).provider, producer.provider
            )

    def test_single_lab_catalog_still_audits(self):
        # A same-lab audit beats no audit; it just gets flagged.
        registry = demo_registry()
        chosen = pick_auditor(registry, registry.get("atlas-small"), BALANCED)
        self.assertNotEqual(chosen.model_id, "atlas-small")

    def test_same_lab_audit_is_flagged_on_the_verdict(self):
        v = run_audit('{"pass": true, "score": 0.9, "issues": []}')  # demo = one lab
        self.assertTrue(v.passed)
        self.assertFalse(v.cross_lab)

    def test_cross_lab_audit_is_flagged_on_the_verdict(self):
        from switchboard import Completion

        registry = two_lab_registry()
        producer = registry.get("lab-a-model")
        provider = ScriptedProvider(
            {"lab-b-model": ['{"pass": true, "score": 0.9, "issues": []}']}, name="otherlab"
        )
        verdict = audit(
            Task(prompt="x", task_type="reasoning"),
            Completion(text="y", model_id="lab-a-model"),
            producer,
            registry,
            ProviderPool([provider]),
            BALANCED,
        )
        self.assertTrue(verdict.cross_lab)
        self.assertEqual(verdict.auditor_model, "lab-b-model")

    def test_independence_outranks_price(self):
        # A cheap same-lab auditor must not beat a pricier cross-lab one:
        # the cross-lab filter runs before the selection strategy.
        registry = Registry([
            ModelSpec("producer", "labA", "mid", 1.0, 5.0, capabilities={"audit": 0.9}),
            ModelSpec("cheap-same-lab", "labA", "small", 0.01, 0.05, capabilities={"audit": 0.9}),
            ModelSpec("pricey-other-lab", "labB", "frontier", 9.0, 40.0, capabilities={"audit": 0.9}),
        ])
        policy = Policy("thrifty", 0.5, 0.3, 0.2, auditor_selection="cheapest_qualified")
        chosen = pick_auditor(registry, registry.get("producer"), policy)
        self.assertEqual(chosen.model_id, "pricey-other-lab")

    def test_cheapest_qualified_still_applies_within_the_cross_lab_pool(self):
        registry = Registry([
            ModelSpec("producer", "labA", "mid", 1.0, 5.0, capabilities={"audit": 0.9}),
            ModelSpec("otherlab-cheap", "labB", "small", 0.10, 0.50, capabilities={"audit": 0.8}),
            ModelSpec("otherlab-dear", "labB", "frontier", 3.0, 15.0, capabilities={"audit": 0.95}),
        ])
        policy = Policy("thrifty", 0.5, 0.3, 0.2, auditor_selection="cheapest_qualified")
        self.assertEqual(
            pick_auditor(registry, registry.get("producer"), policy).model_id, "otherlab-cheap"
        )
        default = Policy("default", 0.5, 0.3, 0.2)
        self.assertEqual(
            pick_auditor(registry, registry.get("producer"), default).model_id, "otherlab-dear"
        )

    def test_cross_lab_preference_can_be_disabled(self):
        registry = Registry([
            ModelSpec("producer", "labA", "mid", 1.0, 5.0, capabilities={"audit": 0.5}),
            ModelSpec("same-lab-strong", "labA", "frontier", 3.0, 15.0, capabilities={"audit": 0.99}),
            ModelSpec("other-lab-weak", "labB", "small", 0.1, 0.5, capabilities={"audit": 0.6}),
        ])
        policy = Policy("no_cross", 0.5, 0.3, 0.2, prefer_cross_lab_auditor=False)
        self.assertEqual(
            pick_auditor(registry, registry.get("producer"), policy).model_id, "same-lab-strong"
        )

    def test_producer_is_still_excluded_in_a_single_lab_catalog(self):
        registry = Registry([
            ModelSpec("only-a", "labA", "mid", 1.0, 5.0, capabilities={"audit": 0.9}),
            ModelSpec("only-b", "labA", "frontier", 3.0, 15.0, capabilities={"audit": 0.95}),
        ])
        for producer in registry.all():
            self.assertNotEqual(
                pick_auditor(registry, producer, BALANCED).model_id, producer.model_id
            )


class PromptHygieneTests(unittest.TestCase):
    def test_no_test_only_marker_reaches_the_prompt(self):
        # Regression: the audit prompt used to carry a "[SWITCHBOARD_AUDIT]"
        # marker that existed only so the offline mock could recognise it.
        self.assertNotIn("SWITCHBOARD_AUDIT", AUDIT_PROMPT_TEMPLATE)

    def test_mock_detects_audits_via_the_real_prompt(self):
        # The mock and the auditor share one constant, so they cannot drift.
        self.assertIn(MockProvider.AUDIT_SENTINEL, AUDIT_PROMPT_TEMPLATE)
        self.assertEqual(MockProvider.AUDIT_SENTINEL, AUDIT_PROMPT_HEADER)

    def test_audit_prompt_carries_task_and_output(self):
        registry = demo_registry()
        provider = ScriptedProvider({AUDITOR: ['{"pass": true, "score": 0.9}']}, name="mock")
        from switchboard import Completion

        audit(
            Task(prompt="THE ORIGINAL ASK", task_type="coding"),
            Completion(text="THE PRODUCED OUTPUT", model_id=PRODUCER),
            registry.get(PRODUCER),
            registry,
            ProviderPool([provider]),
            BALANCED,
        )
        _, prompt = provider.calls[0]
        self.assertIn("THE ORIGINAL ASK", prompt)
        self.assertIn("THE PRODUCED OUTPUT", prompt)
        self.assertIn("coding", prompt)


class AuditorAvailabilityTests(unittest.TestCase):
    def test_auditor_outage_propagates_as_provider_error(self):
        registry = demo_registry()
        provider = ScriptedProvider({AUDITOR: [ProviderUnavailable("down")]}, name="mock")
        from switchboard import Completion

        with self.assertRaises(ProviderUnavailable):
            audit(
                Task(prompt="x", task_type="summarization"),
                Completion(text="y", model_id=PRODUCER),
                registry.get(PRODUCER),
                registry,
                ProviderPool([provider]),
                BALANCED,
            )


def three_lab_registry() -> Registry:
    """A producer plus three audit-qualified models on three distinct labs."""
    caps = {"reasoning": 0.9, "audit": 0.9}
    return Registry([
        ModelSpec("producer", "lab-p", "frontier", 1.0, 2.0, capabilities=dict(caps)),
        ModelSpec("auditor-a", "lab-a", "frontier", 1.0, 2.0, capabilities=dict(caps)),
        ModelSpec("auditor-b", "lab-b", "frontier", 1.0, 2.0, capabilities=dict(caps)),
        ModelSpec("auditor-c", "lab-c", "frontier", 1.0, 2.0, capabilities=dict(caps)),
    ])


def three_lab_pool(scripts: dict[str, str]) -> ProviderPool:
    """One ScriptedProvider per lab, so each auditor can be told to answer
    differently — MockProvider can only pass or fail wholesale."""
    labs = {"auditor-a": "lab-a", "auditor-b": "lab-b", "auditor-c": "lab-c"}
    return ProviderPool([
        ScriptedProvider({model_id: [text]}, name=labs[model_id])
        for model_id, text in scripts.items()
    ])


PASS = '{"pass": true, "score": 0.9, "issues": []}'
FAIL = '{"pass": false, "score": 0.2, "issues": ["bad"]}'


class PickAuditorsTests(unittest.TestCase):
    def test_returns_the_requested_count(self):
        registry = three_lab_registry()
        chosen = pick_auditors(registry, registry.get("producer"), BALANCED, 3)
        self.assertEqual(len(chosen), 3)

    def test_never_includes_the_producer(self):
        registry = three_lab_registry()
        chosen = pick_auditors(registry, registry.get("producer"), BALANCED, 3)
        self.assertNotIn("producer", [m.model_id for m in chosen])

    def test_all_seats_are_distinct_models(self):
        registry = three_lab_registry()
        chosen = pick_auditors(registry, registry.get("producer"), BALANCED, 3)
        self.assertEqual(len({m.model_id for m in chosen}), 3)

    def test_spreads_across_distinct_labs_before_repeating_one(self):
        registry = three_lab_registry()
        chosen = pick_auditors(registry, registry.get("producer"), BALANCED, 3)
        self.assertEqual({m.provider for m in chosen}, {"lab-a", "lab-b", "lab-c"})

    def test_shrinks_rather_than_raises_when_the_catalog_is_too_small(self):
        registry = two_lab_registry()  # only one non-producer model per side
        chosen = pick_auditors(registry, registry.get("lab-a-model"), BALANCED, 5)
        self.assertEqual(len(chosen), 1)  # lab-b-model is the only candidate

    def test_a_fourth_seat_repeats_a_lab_rather_than_going_empty(self):
        registry = three_lab_registry()
        chosen = pick_auditors(registry, registry.get("producer"), BALANCED, 4)
        self.assertEqual(len(chosen), 3)  # only 3 non-producer models exist total

    def test_rejects_a_non_positive_count(self):
        registry = three_lab_registry()
        with self.assertRaises(ValueError):
            pick_auditors(registry, registry.get("producer"), BALANCED, 0)


class AuditConsensusTests(unittest.TestCase):
    """The trap this item names: a 2-1 split must not read as a clean pass."""

    def test_unanimous_pass(self):
        registry = three_lab_registry()
        producer = registry.get("producer")
        pool = three_lab_pool({"auditor-a": PASS, "auditor-b": PASS, "auditor-c": PASS})
        v = audit_consensus(
            Task(prompt="x", task_type="reasoning"),
            Completion(text="y", model_id="producer"),
            producer, registry, pool, BALANCED, 3,
        )
        self.assertTrue(v.passed)
        self.assertTrue(v.unanimous)
        self.assertEqual(v.pass_count, 3)
        self.assertEqual(v.fail_count, 0)

    def test_unanimous_fail(self):
        registry = three_lab_registry()
        producer = registry.get("producer")
        pool = three_lab_pool({"auditor-a": FAIL, "auditor-b": FAIL, "auditor-c": FAIL})
        v = audit_consensus(
            Task(prompt="x", task_type="reasoning"),
            Completion(text="y", model_id="producer"),
            producer, registry, pool, BALANCED, 3,
        )
        self.assertFalse(v.passed)
        self.assertTrue(v.unanimous)

    def test_two_one_split_passes_on_majority_but_is_not_unanimous(self):
        registry = three_lab_registry()
        producer = registry.get("producer")
        pool = three_lab_pool({"auditor-a": PASS, "auditor-b": PASS, "auditor-c": FAIL})
        v = audit_consensus(
            Task(prompt="x", task_type="reasoning"),
            Completion(text="y", model_id="producer"),
            producer, registry, pool, BALANCED, 3,
        )
        self.assertTrue(v.passed)  # 2 of 3 is a majority
        self.assertFalse(v.unanimous)  # but it must not read as a clean pass
        self.assertEqual(v.pass_count, 2)
        self.assertEqual(v.fail_count, 1)

    def test_split_is_not_averaged_away(self):
        # The trap, checked directly: the disagreement must survive as
        # visible signal, not be collapsed into a single smooth score.
        registry = three_lab_registry()
        producer = registry.get("producer")
        pool = three_lab_pool({"auditor-a": PASS, "auditor-b": PASS, "auditor-c": FAIL})
        v = audit_consensus(
            Task(prompt="x", task_type="reasoning"),
            Completion(text="y", model_id="producer"),
            producer, registry, pool, BALANCED, 3,
        )
        self.assertEqual(len(v.verdicts), 3)
        self.assertTrue(any(not vv.passed for vv in v.verdicts))
        self.assertTrue(any("split" in i for i in v.issues))
        self.assertIn("SPLIT", v.describe())

    def test_even_panel_tie_fails_closed(self):
        registry = three_lab_registry()
        producer = registry.get("producer")
        pool = three_lab_pool({"auditor-a": PASS, "auditor-b": FAIL})
        v = audit_consensus(
            Task(prompt="x", task_type="reasoning"),
            Completion(text="y", model_id="producer"),
            producer, registry, pool, BALANCED, 2,
        )
        self.assertFalse(v.passed)  # 1-1 is not "more than half"
        self.assertFalse(v.unanimous)

    def test_cost_is_the_sum_of_every_seat(self):
        registry = three_lab_registry()
        producer = registry.get("producer")
        pool = three_lab_pool({"auditor-a": PASS, "auditor-b": PASS, "auditor-c": PASS})
        v = audit_consensus(
            Task(prompt="x", task_type="reasoning"),
            Completion(text="y", model_id="producer"),
            producer, registry, pool, BALANCED, 3,
        )
        self.assertAlmostEqual(v.cost_usd, sum(vv.cost_usd for vv in v.verdicts))
        self.assertGreater(v.cost_usd, max(vv.cost_usd for vv in v.verdicts))

    def test_issues_attribute_each_finding_to_its_auditor(self):
        registry = three_lab_registry()
        producer = registry.get("producer")
        pool = three_lab_pool({"auditor-a": PASS, "auditor-b": PASS, "auditor-c": FAIL})
        v = audit_consensus(
            Task(prompt="x", task_type="reasoning"),
            Completion(text="y", model_id="producer"),
            producer, registry, pool, BALANCED, 3,
        )
        self.assertTrue(any(i.startswith("[auditor-c]") for i in v.issues))

    def test_shortfall_is_recorded_not_silently_eaten(self):
        registry = two_lab_registry()
        producer = registry.get("lab-a-model")
        pool = ProviderPool([
            ScriptedProvider({"lab-b-model": [PASS]}, name="otherlab"),
        ])
        v = audit_consensus(
            Task(prompt="x", task_type="reasoning"), Completion(text="y", model_id="lab-a-model"),
            producer, registry, pool, BALANCED, 3,
        )
        self.assertEqual(v.requested_count, 3)
        self.assertEqual(len(v.verdicts), 1)
        self.assertTrue(any("requested 3" in i for i in v.issues))

    def test_mean_score_is_reported_alongside_not_instead_of_the_vote(self):
        registry = three_lab_registry()
        producer = registry.get("producer")
        pool = three_lab_pool({
            "auditor-a": '{"pass": true, "score": 1.0, "issues": []}',
            "auditor-b": '{"pass": true, "score": 1.0, "issues": []}',
            "auditor-c": '{"pass": false, "score": 0.0, "issues": ["bad"]}',
        })
        v = audit_consensus(
            Task(prompt="x", task_type="reasoning"),
            Completion(text="y", model_id="producer"),
            producer, registry, pool, BALANCED, 3,
        )
        self.assertAlmostEqual(v.score, 2 / 3)
        self.assertTrue(v.passed)  # the vote, not the mean score, decides
        self.assertFalse(v.unanimous)


if __name__ == "__main__":
    unittest.main()
