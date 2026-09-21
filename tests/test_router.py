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
    UnsupportedFeatureError,
    demo_registry,
    fits_features,
    route,
)
from switchboard.router import _quality_score

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


class FeatureGateTests(unittest.TestCase):
    """Feature flags are a hard gate, not a score (ROADMAP item 12): a
    cheaper or higher-capability model that lacks a required feature must
    never win over one that has it."""

    def setUp(self):
        # atlas-small is cheapest and would win most policies on cost or
        # capability alone; only atlas-mid supports vision. A gate that
        # folded features into the weighted score, instead of filtering
        # candidates before scoring, would still let atlas-small win here.
        self.registry = Registry([
            ModelSpec(
                model_id="atlas-small", provider="mock", tier="small",
                input_cost=0.10, output_cost=0.50, latency="fast",
                capabilities={"extraction": 0.90},
            ),
            ModelSpec(
                model_id="atlas-mid", provider="mock", tier="mid",
                input_cost=0.80, output_cost=4.00, latency="medium",
                capabilities={"extraction": 0.84},
                features=frozenset({"vision", "json_mode"}),
            ),
            ModelSpec(
                model_id="atlas-frontier", provider="mock", tier="frontier",
                input_cost=3.00, output_cost=15.00, latency="slow",
                capabilities={"extraction": 0.88},
                features=frozenset({"tool_use"}),
            ),
        ])

    def test_required_feature_excludes_models_that_lack_it(self):
        task = Task(
            prompt="describe this image", task_type="extraction", complexity=0.2,
            required_features=frozenset({"vision"}),
        )
        decision = route(task, self.registry, COST_FIRST)
        self.assertEqual(decision.chosen.model_id, "atlas-mid")
        self.assertIn("feature flag gate applied", decision.rationale)

    def test_feature_gate_is_never_folded_into_score(self):
        # Without the gate, cost_first would pick atlas-small: cheapest, and
        # capability alone doesn't disqualify it. The gate must still win.
        task = Task(
            prompt="describe this image", task_type="extraction", complexity=0.2,
            required_features=frozenset({"vision"}),
        )
        decision = route(task, self.registry, COST_FIRST)
        self.assertNotEqual(decision.chosen.model_id, "atlas-small")

    def test_multiple_required_features_need_every_one(self):
        task = Task(
            prompt="x", task_type="extraction", complexity=0.2,
            required_features=frozenset({"vision", "json_mode"}),
        )
        decision = route(task, self.registry, BALANCED)
        self.assertEqual(decision.chosen.model_id, "atlas-mid")

    def test_no_model_supports_the_required_feature_raises(self):
        task = Task(
            prompt="x", task_type="extraction", complexity=0.2,
            required_features=frozenset({"nonexistent_feature"}),
        )
        with self.assertRaises(UnsupportedFeatureError):
            route(task, self.registry, BALANCED)

    def test_no_required_features_is_a_no_op(self):
        task = Task(prompt="x", task_type="extraction", complexity=0.2)
        decision = route(task, self.registry, COST_FIRST)
        self.assertEqual(decision.chosen.model_id, "atlas-small")
        self.assertNotIn("feature flag gate applied", decision.rationale)

    def test_fits_features_helper(self):
        mid = self.registry.get("atlas-mid")
        small = self.registry.get("atlas-small")
        needs_vision = Task(prompt="x", required_features=frozenset({"vision"}))
        self.assertTrue(fits_features(needs_vision, mid))
        self.assertFalse(fits_features(needs_vision, small))

    def test_task_accepts_a_list_and_stores_a_frozenset(self):
        task = Task(prompt="x", required_features=["vision", "json_mode"])
        self.assertEqual(task.required_features, frozenset({"vision", "json_mode"}))

    def test_task_empty_feature_name_rejected(self):
        with self.assertRaises(ValueError):
            Task(prompt="x", required_features=["vision", ""])


if __name__ == "__main__":
    unittest.main()
