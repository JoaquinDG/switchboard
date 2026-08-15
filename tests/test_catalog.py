"""Validation for the JSON catalogs shipped in examples/.

These files are the part of the repo most likely to rot: prices change, models
retire, and a hand-edited JSON file has no compiler. A catalog that silently
stops loading, or quietly loses a tier so escalation has no ladder, breaks
routing everywhere at once — so both shipped catalogs are checked in CI.
"""

import datetime
import json
import tempfile
import unittest
import warnings
from pathlib import Path

from switchboard import (
    LATENCY_CLASSES,
    TIERS,
    CatalogStaleWarning,
    Registry,
)

REPO = Path(__file__).resolve().parents[1]
STARTER = REPO / "examples" / "starter_catalog.json"
EXAMPLE = REPO / "examples" / "catalog.example.json"
CATALOGS = {"starter_catalog.json": STARTER, "catalog.example.json": EXAMPLE}


class CatalogLoadTests(unittest.TestCase):
    def test_every_shipped_catalog_loads(self):
        for name, path in CATALOGS.items():
            with self.subTest(catalog=name):
                self.assertGreater(len(Registry.from_json(path)), 0)

    def test_every_model_is_fully_specified(self):
        for name, path in CATALOGS.items():
            for spec in Registry.from_json(path).all():
                with self.subTest(catalog=name, model=spec.model_id):
                    self.assertTrue(spec.model_id)
                    self.assertTrue(spec.provider)
                    self.assertIn(spec.tier, TIERS)
                    self.assertIn(spec.latency, LATENCY_CLASSES)
                    self.assertGreater(spec.input_cost, 0.0)
                    self.assertGreater(spec.output_cost, 0.0)
                    self.assertGreater(spec.context_window, 0)

    def test_capability_scores_are_in_range(self):
        for name, path in CATALOGS.items():
            for spec in Registry.from_json(path).all():
                self.assertTrue(spec.capabilities, f"{spec.model_id} has no scores")
                for task_type, score in spec.capabilities.items():
                    with self.subTest(catalog=name, model=spec.model_id, task=task_type):
                        self.assertGreaterEqual(score, 0.0)
                        self.assertLessEqual(score, 1.0)

    def test_every_tier_is_represented(self):
        # Escalation walks small -> mid -> frontier. A missing rung does not
        # error; it silently shortens the ladder.
        for name, path in CATALOGS.items():
            registry = Registry.from_json(path)
            for tier in TIERS:
                with self.subTest(catalog=name, tier=tier):
                    self.assertTrue(registry.by_tier(tier), f"{name} has no {tier} model")

    def test_output_costs_at_least_match_input_costs(self):
        # Every vendor charges at least as much for output as for input. A
        # flipped pair is the classic catalog typo and would systematically
        # misprice every long-generation task.
        for name, path in CATALOGS.items():
            for spec in Registry.from_json(path).all():
                with self.subTest(catalog=name, model=spec.model_id):
                    self.assertGreaterEqual(spec.output_cost, spec.input_cost)

    def test_costs_are_ordered_by_tier(self):
        # Not a law of nature, but if a "small" model costs more than a
        # "frontier" one in the same catalog, a tier label is wrong.
        for name, path in CATALOGS.items():
            registry = Registry.from_json(path)
            cheapest_frontier = min(m.output_cost for m in registry.by_tier("frontier"))
            dearest_small = max(m.output_cost for m in registry.by_tier("small"))
            with self.subTest(catalog=name):
                self.assertLess(dearest_small, cheapest_frontier)


class StarterCatalogTests(unittest.TestCase):
    """The starter catalog makes stronger promises than the example template:
    its prices are real, so its provenance has to be checkable."""

    def setUp(self):
        self.raw = json.loads(STARTER.read_text())
        self.registry = Registry.from_json(STARTER)

    def test_spans_enough_models_and_providers_to_be_useful(self):
        # Lower bounds, not exact counts: the catalog is meant to grow, and a
        # hardcoded total turns every addition into a test failure that says
        # nothing about whether the addition was correct.
        self.assertGreaterEqual(len(self.registry), 12)
        providers = {m.provider for m in self.registry.all()}
        self.assertGreaterEqual(len(providers), 4, f"only {providers}")

    def test_more_than_one_provider_per_tier(self):
        # Cross-lab auditing needs somewhere to cross to, and provider failover
        # needs an alternative that is not the vendor that just went down.
        for tier in TIERS:
            providers = {m.provider for m in self.registry.by_tier(tier)}
            with self.subTest(tier=tier):
                self.assertGreater(len(providers), 1, f"{tier} has only {providers}")

    def test_every_price_cites_its_source(self):
        for entry in self.raw["models"]:
            with self.subTest(model=entry["model_id"]):
                self.assertTrue(entry.get("_source", "").startswith("http"))

    def test_declares_when_it_was_verified(self):
        self.assertIsNotNone(self.registry.last_verified)

    def test_source_urls_cover_every_provider_used(self):
        cited = set(self.raw["_sources"])
        used = {m.provider for m in self.registry.all()}
        self.assertEqual(used - cited, set())

    def test_disclaimer_marks_capability_scores_as_estimates(self):
        # The single most important honesty claim in the repo: prices were
        # looked up, capability scores were guessed. If this text ever
        # disappears the file starts implying a benchmark that never ran.
        disclaimer = " ".join(self.raw["_disclaimer"]).upper()
        self.assertIn("CAPABILITY SCORES ARE ESTIMATES", disclaimer)

    def test_underscore_keys_are_metadata_not_fields(self):
        # _source and _note sit beside the numbers they justify; the loader
        # must ignore them rather than reject the entry.
        self.assertIn("_sources", self.registry.metadata)
        self.assertNotIn("_source", vars(self.registry.get("claude-opus-5")))


class FreshnessWarningTests(unittest.TestCase):
    """Prices go stale. A router confidently using last quarter's price list is
    exactly the failure this project exists to prevent, so say so on load."""

    def write(self, last_verified):
        payload = {
            "_last_verified": last_verified,
            "models": [
                {
                    "model_id": "m", "provider": "p", "tier": "mid",
                    "input_cost": 1.0, "output_cost": 2.0,
                    "capabilities": {"reasoning": 0.8},
                }
            ],
        }
        path = Path(self.tmp) / "catalog.json"
        path.write_text(json.dumps(payload))
        return path

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def test_stale_catalog_warns(self):
        path = self.write("2026-01-01")
        with self.assertWarns(CatalogStaleWarning):
            Registry.from_json(path, today=datetime.date(2026, 8, 15))

    def test_warning_names_the_age(self):
        path = self.write("2026-06-01")
        with self.assertWarns(CatalogStaleWarning) as ctx:
            Registry.from_json(path, today=datetime.date(2026, 8, 15))
        self.assertIn("75 days ago", str(ctx.warning))

    def test_fresh_catalog_is_silent(self):
        path = self.write("2026-08-01")
        with warnings.catch_warnings():
            warnings.simplefilter("error", CatalogStaleWarning)
            Registry.from_json(path, today=datetime.date(2026, 8, 15))

    def test_boundary_is_not_off_by_one(self):
        path = self.write("2026-06-16")  # exactly 60 days before
        with warnings.catch_warnings():
            warnings.simplefilter("error", CatalogStaleWarning)
            Registry.from_json(path, today=datetime.date(2026, 8, 15))

    def test_undated_catalog_does_not_warn(self):
        # Silence is the right default: the example template is deliberately
        # undated because its prices are placeholders, not observations.
        payload = {"models": [{
            "model_id": "m", "provider": "p", "tier": "mid",
            "input_cost": 1.0, "output_cost": 2.0, "capabilities": {"reasoning": 0.8},
        }]}
        path = Path(self.tmp) / "undated.json"
        path.write_text(json.dumps(payload))
        with warnings.catch_warnings():
            warnings.simplefilter("error", CatalogStaleWarning)
            registry = Registry.from_json(path)
        self.assertIsNone(registry.age_in_days())

    def test_unparseable_date_is_ignored_rather_than_crashing(self):
        path = self.write("last tuesday")
        with warnings.catch_warnings():
            warnings.simplefilter("error", CatalogStaleWarning)
            registry = Registry.from_json(path)
        self.assertIsNone(registry.last_verified)


class CatalogRoutingTests(unittest.TestCase):
    """A catalog that loads but cannot route is still broken."""

    def test_starter_catalog_routes_every_known_task_type(self):
        from switchboard import BALANCED, Task, route

        registry = Registry.from_json(STARTER)
        for task_type in sorted(registry.known_task_types()):
            for complexity in (0.2, 0.6, 0.9):
                with self.subTest(task=task_type, complexity=complexity):
                    decision = route(
                        Task(prompt="x", task_type=task_type, complexity=complexity),
                        registry,
                        BALANCED,
                    )
                    self.assertTrue(decision.chosen.model_id)
                    self.assertFalse(decision.warnings, decision.warnings)

    def test_starter_catalog_runs_end_to_end_offline(self):
        # A real catalog names real vendors, so this only works because
        # mock_pool stands in for each one. Without it the first non-"mock"
        # provider raises and the catalog is undemoable without API keys.
        from switchboard import BALANCED, Broker, Task, mock_pool

        registry = Registry.from_json(STARTER)
        broker = Broker(registry, mock_pool(registry), BALANCED)
        result = broker.run(Task(prompt="pull the dates", task_type="extraction", complexity=0.3))
        self.assertTrue(result.verified)
        self.assertGreater(result.total_cost_usd, 0.0)

    def test_multi_provider_catalog_produces_cross_lab_audits(self):
        # The practical payoff of a four-provider catalog: audits stop being
        # same-family by default.
        from switchboard import BALANCED, Broker, Task, mock_pool

        registry = Registry.from_json(STARTER)
        broker = Broker(registry, mock_pool(registry), BALANCED)
        result = broker.run(Task(prompt="x", task_type="coding", complexity=0.5))
        self.assertTrue(result.attempts[-1].cross_lab_audit)

    def test_mock_pool_covers_exactly_the_catalog_providers(self):
        from switchboard import mock_pool

        registry = Registry.from_json(STARTER)
        pool = mock_pool(registry)
        self.assertEqual(set(pool.names()), {m.provider for m in registry.all()})

    def test_mock_pool_accepts_a_plain_list_of_names(self):
        from switchboard import mock_pool

        self.assertEqual(mock_pool(["b", "a", "a"]).names(), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
