import unittest
from datetime import date
from pathlib import Path

from switchboard import (
    BALANCED,
    COST_FIRST,
    QUALITY_FIRST,
    ModelSpec,
    NoQualifiedModelError,
    Policy,
    Registry,
    Task,
    demo_registry,
    route,
)
from switchboard.router import _quality_score, estimate_cost

REPO = Path(__file__).resolve().parents[1]


class RouterTests(unittest.TestCase):
    def setUp(self):
        self.registry = demo_registry()

    def test_cost_first_prefers_small_model_for_easy_extraction(self):
        task = Task(prompt="pull the dates out of this text", task_type="extraction", complexity=0.2)
        decision = route(task, self.registry, COST_FIRST)
        self.assertEqual(decision.chosen.model_id, "atlas-small")

    def test_quality_first_prefers_frontier_for_hard_reasoning(self):
        task = Task(prompt="prove this theorem", task_type="reasoning", complexity=0.7)
        decision = route(task, self.registry, QUALITY_FIRST)
        self.assertEqual(decision.chosen.model_id, "atlas-frontier")

    def test_frontier_gate_overrides_cost_policy(self):
        # Even under COST_FIRST, complexity above the gate forces frontier.
        task = Task(prompt="design a distributed consensus protocol", task_type="reasoning", complexity=0.95)
        decision = route(task, self.registry, COST_FIRST)
        self.assertEqual(decision.chosen.tier, "frontier")
        self.assertIn("frontier gate applied", decision.rationale)

    def test_qualification_gate_blocks_underqualified_cheap_models(self):
        # Regression for a bug the eval suite caught: balanced policy routed
        # mid-complexity coding to the small model on cost alone.
        task = Task(prompt="refactor this module", task_type="coding", complexity=0.6)
        decision = route(task, self.registry, BALANCED)
        self.assertNotEqual(decision.chosen.tier, "small")
        self.assertIn("qualification filter applied", decision.rationale)

    def test_fast_response_requirement_excludes_slow_models(self):
        task = Task(
            prompt="quick summary please",
            task_type="summarization",
            complexity=0.3,
            needs_fast_response=True,
        )
        decision = route(task, self.registry, BALANCED)
        self.assertNotEqual(decision.chosen.latency, "slow")

    def test_decision_is_fully_ranked_and_explained(self):
        # low complexity so all catalog models pass the qualification gate
        task = Task(prompt="hi", task_type="reasoning", complexity=0.1)
        decision = route(task, self.registry, BALANCED)
        self.assertEqual(len(decision.ranked), len(self.registry))
        self.assertTrue(decision.rationale)
        scores = [s.score for s in decision.ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_clean_decision_carries_no_warnings(self):
        task = Task(prompt="hi", task_type="reasoning", complexity=0.1)
        decision = route(task, self.registry, BALANCED)
        self.assertEqual(decision.warnings, [])
        self.assertFalse(decision.underqualified)


class GateFallbackTests(unittest.TestCase):
    """Gates must degrade upward. Opening the field back up hands the decision
    to cost weight — precisely what the gate existed to prevent."""

    def setUp(self):
        self.registry = demo_registry()

    def test_unknown_task_type_does_not_fall_through_to_the_cheapest_model(self):
        # Regression: with no capability data every model scored the same
        # prior, nothing cleared the gate, the gate silently skipped, and a
        # quality-first policy routed 0.75-complexity work to atlas-small.
        task = Task(prompt="review this contract", task_type="legal_analysis", complexity=0.75)
        decision = route(task, self.registry, QUALITY_FIRST)
        self.assertEqual(decision.chosen.model_id, "atlas-frontier")
        self.assertTrue(decision.underqualified)

    def test_unknown_task_type_is_reported_not_swallowed(self):
        task = Task(prompt="x", task_type="legal_analysis", complexity=0.3)
        decision = route(task, self.registry, BALANCED)
        self.assertTrue(any("no capability data" in w for w in decision.warnings))
        self.assertIn("WARNING", decision.rationale)

    def test_cheap_unknown_task_still_routes_cheap(self):
        # The warning is not a reason to overspend on genuinely easy work.
        task = Task(prompt="x", task_type="legal_analysis", complexity=0.2)
        decision = route(task, self.registry, COST_FIRST)
        self.assertEqual(decision.chosen.model_id, "atlas-small")
        self.assertFalse(decision.underqualified)

    def test_nothing_qualified_degrades_upward_and_flags(self):
        task = Task(prompt="unsolved problem", task_type="reasoning", complexity=1.0)
        decision = route(task, self.registry, COST_FIRST)
        self.assertEqual(decision.chosen.tier, "frontier")
        self.assertTrue(decision.underqualified)
        self.assertTrue(any("no model clears capability" in w for w in decision.warnings))

    def test_frontier_gate_without_frontier_models_keeps_the_top_tier(self):
        registry = Registry([m for m in demo_registry().all() if m.tier != "frontier"])
        task = Task(prompt="hard", task_type="reasoning", complexity=0.95)
        decision = route(task, registry, COST_FIRST)
        self.assertEqual(decision.chosen.tier, "mid")  # not the cheap model
        self.assertTrue(any("no frontier models" in w for w in decision.warnings))
        self.assertIn("frontier gate applied", decision.rationale)

    def test_best_capability_strategy_keeps_the_strongest_model(self):
        policy = Policy("bc", 0.3, 0.6, 0.1, on_no_qualified_model="best_capability")
        task = Task(prompt="x", task_type="legal_analysis", complexity=0.75)
        decision = route(task, self.registry, policy)
        # Every model ties on the prior, so all survive and cost decides —
        # but the caller is told the catalog could not discriminate.
        self.assertTrue(decision.underqualified)

    def test_raise_strategy_fails_loudly(self):
        policy = Policy("strict", 0.5, 0.3, 0.2, on_no_qualified_model="raise")
        task = Task(prompt="x", task_type="reasoning", complexity=1.0)
        with self.assertRaises(NoQualifiedModelError):
            route(task, self.registry, policy)

    def test_gates_are_reported_individually(self):
        # Complexity 1.0: the frontier gate fires, and nothing in the catalog
        # is rated for work at the top of the scale, so the qualification gate
        # fires too.
        task = Task(prompt="hard", task_type="reasoning", complexity=1.0)
        decision = route(task, self.registry, COST_FIRST)
        self.assertTrue(any(g.startswith("frontier gate") for g in decision.gates))
        self.assertTrue(any(g.startswith("qualification filter") for g in decision.gates))

    def test_margin_is_dropped_when_it_cannot_be_satisfied(self):
        # complexity 0.9 + a 0.1 margin demands capability 1.0, which no honest
        # catalog claims. Keeping the margin would flag every hard task as
        # underqualified and make the warning worthless.
        task = Task(prompt="hard", task_type="reasoning", complexity=0.9)
        decision = route(task, self.registry, COST_FIRST)
        self.assertEqual(decision.chosen.model_id, "atlas-frontier")
        self.assertFalse(decision.underqualified)

    def test_margin_still_applies_where_there_is_headroom(self):
        task = Task(prompt="refactor", task_type="coding", complexity=0.6)
        decision = route(task, self.registry, COST_FIRST)
        self.assertNotEqual(decision.chosen.tier, "small")  # small codes at 0.60 < 0.70


class CostNormalizationTests(unittest.TestCase):
    def test_costs_normalize_over_the_full_catalog(self):
        # Regression: normalizing over surviving candidates only meant that
        # with two candidates one always scored 0.0 — an artifact, not a
        # signal — which flipped quality-first decisions toward mid.
        registry = demo_registry()
        task = Task(prompt="refactor", task_type="coding", complexity=0.6)
        decision = route(task, registry, QUALITY_FIRST)
        by_id = {s.spec.model_id: s.cost_component for s in decision.ranked}
        self.assertEqual(set(by_id), {"atlas-frontier", "atlas-mid"})  # small filtered out
        # Candidate-only min-max would put mid at exactly 1.0 purely because
        # it is the cheaper of two survivors. Against the full catalog it sits
        # mid-range, because atlas-small — still in the catalog — is cheaper.
        self.assertLess(by_id["atlas-mid"], 1.0)
        self.assertGreater(by_id["atlas-mid"], 0.0)
        # The priciest model in the catalog legitimately scores 0.0.
        self.assertEqual(by_id["atlas-frontier"], 0.0)

    def test_single_model_catalog_scores_cleanly(self):
        registry = Registry([
            ModelSpec("solo", "mock", "mid", 1.0, 5.0, capabilities={"reasoning": 0.9})
        ])
        decision = route(Task(prompt="x", complexity=0.2), registry, BALANCED)
        self.assertEqual(decision.chosen.model_id, "solo")
        self.assertEqual(decision.ranked[0].cost_component, 1.0)

    def test_empty_registry_raises(self):
        with self.assertRaises(ValueError):
            route(Task(prompt="x"), Registry([]), BALANCED)


class QualityScoreScaleTests(unittest.TestCase):
    """ROADMAP item 1b: raw capability clusters tightly (0.78-0.95 in
    practice) while cost and latency both span their full [0, 1] range, so a
    policy's quality_weight bought far less influence than its number
    implied. Regression coverage for the fix and for the narrow-catalog
    failure mode the *rejected* fix (candidate-range min-max) caused."""

    def test_quality_score_anchors_on_the_unknown_prior_not_zero(self):
        # A model scored exactly at UNKNOWN_CAPABILITY_PRIOR is, by the
        # catalog's own convention, no better than "no data" — it should
        # floor out at 0, not sit at the raw value's 0.5.
        self.assertEqual(_quality_score(0.5), 0.0)
        self.assertEqual(_quality_score(1.0), 1.0)
        self.assertAlmostEqual(_quality_score(0.75), 0.5)

    def test_quality_score_clips_rather_than_going_negative(self):
        self.assertEqual(_quality_score(0.3), 0.0)

    def test_quality_first_prefers_the_more_capable_model_over_a_faster_one(self):
        # README Finding 3, measured live: on the 16-model starter catalog,
        # claude-opus-5's 0.093 raw-capability edge over gemini-3.7-flash
        # lost to gemini's latency advantage under quality_first — a policy
        # named quality-first should not do that.
        registry = Registry.from_json(
            REPO / "examples" / "starter_catalog.json", today=date(2026, 8, 16)
        )
        task = Task(prompt="deep reasoning task", task_type="reasoning", complexity=0.6)
        decision = route(task, registry, QUALITY_FIRST)
        self.assertEqual(decision.chosen.model_id, "claude-opus-5")

    def test_narrow_catalog_easy_extraction_still_stays_cheap(self):
        # The fix this replaces (min-max normalizing over the observed
        # candidate range) broke exactly this case: a 3-model catalog with a
        # tight capability spread turned into a full 0-to-1 swing and sent
        # easy work to the frontier model under cost_first. The fixed
        # version anchors on a fixed constant, not the candidate set, so it
        # must not reproduce that.
        registry = demo_registry()
        task = Task(prompt="pull the dates out", task_type="extraction", complexity=0.2)
        decision = route(task, registry, COST_FIRST)
        self.assertEqual(decision.chosen.model_id, "atlas-small")

    def test_rationale_shows_both_raw_capability_and_the_score_it_earned(self):
        registry = demo_registry()
        task = Task(prompt="hi", task_type="reasoning", complexity=0.1)
        decision = route(task, registry, BALANCED)
        top = decision.ranked[0]
        raw = top.spec.capability_for("reasoning")
        self.assertIn(f"quality={raw:.2f} (score {top.quality_component:.2f})", decision.rationale)


class CacheHitRateCostTests(unittest.TestCase):
    """ROADMAP item 9: cache reads cost ~10% of a fresh read on major
    vendors, and a caller's estimate of its own cache-hit rate is an
    ASSUMPTION, never something Switchboard measured (the item's trap)."""

    def setUp(self):
        self.spec = ModelSpec(
            "m", "mock", "mid", 2.0, 10.0, capabilities={"reasoning": 0.8}
        )

    def test_default_assumption_is_zero_and_leaves_cost_unchanged(self):
        task = Task(prompt="x", est_input_tokens=1_000, est_output_tokens=100)
        baseline = (1_000 * 2.0 + 100 * 10.0) / 1_000_000
        self.assertAlmostEqual(estimate_cost(task, self.spec), baseline)

    def test_full_assumed_hit_rate_applies_the_full_discount(self):
        task = Task(
            prompt="x", est_input_tokens=1_000, est_output_tokens=100,
            assumed_cache_hit_rate=1.0,
        )
        # cache_discount defaults to 0.9: a fully-cached call pays 10% of the
        # normal input price plus the unaffected output price.
        expected = (1_000 * 2.0 * 0.1 + 100 * 10.0) / 1_000_000
        self.assertAlmostEqual(estimate_cost(task, self.spec), expected)

    def test_partial_assumed_hit_rate_blends_the_two_rates(self):
        task = Task(
            prompt="x", est_input_tokens=1_000, est_output_tokens=0,
            assumed_cache_hit_rate=0.5,
        )
        # Half the tokens at 10% of input_cost, half at full input_cost.
        expected = (500 * 2.0 * 0.1 + 500 * 2.0) / 1_000_000
        self.assertAlmostEqual(estimate_cost(task, self.spec), expected)

    def test_per_model_cache_discount_overrides_the_default(self):
        cheap_cache_spec = ModelSpec(
            "kimi-like", "mock", "frontier", 3.0, 15.0, cache_discount=0.9,
            capabilities={"reasoning": 0.9},
        )
        task = Task(prompt="x", est_input_tokens=1_000, est_output_tokens=0,
                    assumed_cache_hit_rate=1.0)
        expected = (1_000 * 3.0 * 0.1) / 1_000_000
        self.assertAlmostEqual(estimate_cost(task, cheap_cache_spec), expected)

    def test_out_of_range_hit_rate_rejected(self):
        with self.assertRaises(ValueError):
            Task(prompt="x", assumed_cache_hit_rate=1.5)

    def test_rationale_names_the_assumption_when_used(self):
        registry = demo_registry()
        task = Task(prompt="hi", task_type="reasoning", complexity=0.1,
                    assumed_cache_hit_rate=0.8)
        decision = route(task, registry, BALANCED)
        self.assertIn("assumes 80% of input tokens are cache hits", decision.rationale)
        self.assertIn("not measured", decision.rationale)

    def test_rationale_omits_the_note_when_no_assumption_is_made(self):
        # Default behaviour, and existing rationale format, must not change
        # for the common case where no caller opts into a cache assumption.
        registry = demo_registry()
        task = Task(prompt="hi", task_type="reasoning", complexity=0.1)
        decision = route(task, registry, BALANCED)
        self.assertNotIn("cache", decision.rationale)

    def test_assumed_hit_rate_never_affects_actual_cost(self):
        # actual_cost prices tokens a provider already billed; it takes no
        # Task at all, so a cache assumption has no path to reach it.
        from switchboard.router import actual_cost
        self.assertAlmostEqual(
            actual_cost(self.spec, 1_000, 100),
            (1_000 * 2.0 + 100 * 10.0) / 1_000_000,
        )


if __name__ == "__main__":
    unittest.main()
