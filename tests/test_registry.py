"""Catalog validation.

The catalog is hand-maintained and the router trusts it completely, so a typo
in a capability score is a routing bug with no stack trace. Validation happens
at construction, and load errors name the entry that caused them.
"""

import json
import tempfile
import unittest
from pathlib import Path

from switchboard import ModelSpec, Registry, demo_registry

VALID = {
    "model_id": "m1",
    "provider": "anthropic",
    "tier": "mid",
    "input_cost": 1.0,
    "output_cost": 5.0,
}


class ModelSpecValidationTests(unittest.TestCase):
    def spec(self, **overrides):
        return ModelSpec(**{**VALID, **overrides})

    def test_valid_spec_constructs(self):
        self.assertEqual(self.spec().model_id, "m1")

    def test_bad_tier_rejected(self):
        with self.assertRaises(ValueError):
            self.spec(tier="enormous")

    def test_bad_latency_rejected(self):
        with self.assertRaises(ValueError):
            self.spec(latency="instant")

    def test_negative_cost_rejected(self):
        with self.assertRaises(ValueError):
            self.spec(input_cost=-1.0)

    def test_capability_above_one_rejected(self):
        # A 95 meant as "95%" would otherwise dominate every routing decision.
        with self.assertRaises(ValueError) as ctx:
            self.spec(capabilities={"reasoning": 95})
        self.assertIn("reasoning", str(ctx.exception))

    def test_capability_below_zero_rejected(self):
        with self.assertRaises(ValueError):
            self.spec(capabilities={"coding": -0.2})

    def test_non_numeric_capability_rejected(self):
        with self.assertRaises(ValueError):
            self.spec(capabilities={"coding": "high"})

    def test_boolean_capability_rejected(self):
        with self.assertRaises(ValueError):
            self.spec(capabilities={"coding": True})

    def test_zero_context_window_rejected(self):
        with self.assertRaises(ValueError):
            self.spec(context_window=0)

    def test_empty_model_id_rejected(self):
        with self.assertRaises(ValueError):
            self.spec(model_id="")

    def test_features_default_to_empty(self):
        self.assertEqual(self.spec().features, frozenset())

    def test_features_accepts_a_list_and_stores_a_frozenset(self):
        # JSON has no set type, so the loader hands this a list.
        spec = self.spec(features=["vision", "json_mode"])
        self.assertEqual(spec.features, frozenset({"vision", "json_mode"}))

    def test_empty_feature_name_rejected(self):
        with self.assertRaises(ValueError):
            self.spec(features=["vision", ""])

    def test_supports_requires_every_feature(self):
        spec = self.spec(features={"vision", "json_mode"})
        self.assertTrue(spec.supports({"vision"}))
        self.assertTrue(spec.supports({"vision", "json_mode"}))
        self.assertFalse(spec.supports({"vision", "tool_use"}))
        self.assertTrue(spec.supports(set()))


class RegistryTests(unittest.TestCase):
    def test_duplicate_model_id_rejected(self):
        registry = Registry([ModelSpec(**VALID)])
        with self.assertRaises(ValueError):
            registry.add(ModelSpec(**VALID))

    def test_known_task_types_unions_the_catalog(self):
        types = demo_registry().known_task_types()
        self.assertIn("coding", types)
        self.assertIn("audit", types)
        self.assertNotIn("legal_analysis", types)

    def test_has_capability_data(self):
        registry = demo_registry()
        self.assertTrue(registry.has_capability_data("reasoning"))
        self.assertFalse(registry.has_capability_data("legal_analysis"))

    def test_unknown_task_type_uses_the_prior(self):
        spec = ModelSpec(**VALID, capabilities={"coding": 0.8})
        self.assertEqual(spec.capability_for("coding"), 0.8)
        self.assertEqual(spec.capability_for("nonesuch"), 0.5)


class FromJSONTests(unittest.TestCase):
    def load(self, payload):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "catalog.json"
            path.write_text(payload if isinstance(payload, str) else json.dumps(payload))
            return Registry.from_json(path)

    def test_valid_catalog_loads(self):
        registry = self.load({"models": [VALID]})
        self.assertEqual(len(registry), 1)

    def test_example_catalog_in_repo_is_valid(self):
        example = Path(__file__).resolve().parents[1] / "examples" / "catalog.example.json"
        self.assertEqual(len(Registry.from_json(example)), 3)

    def test_invalid_json_names_the_file(self):
        with self.assertRaises(ValueError) as ctx:
            self.load("{not json")
        self.assertIn("not valid JSON", str(ctx.exception))

    def test_missing_models_key_rejected(self):
        with self.assertRaises(ValueError):
            self.load({"model": [VALID]})

    def test_empty_catalog_rejected(self):
        with self.assertRaises(ValueError):
            self.load({"models": []})

    def test_missing_required_field_names_the_entry(self):
        broken = {k: v for k, v in VALID.items() if k != "output_cost"}
        with self.assertRaises(ValueError) as ctx:
            self.load({"models": [broken]})
        self.assertIn("m1", str(ctx.exception))
        self.assertIn("output_cost", str(ctx.exception))

    def test_unknown_field_names_the_entry(self):
        # A typo'd key would otherwise raise an opaque TypeError from the
        # dataclass constructor, forty models into a hand-edited file.
        with self.assertRaises(ValueError) as ctx:
            self.load({"models": [{**VALID, "latenty": "fast"}]})
        self.assertIn("m1", str(ctx.exception))
        self.assertIn("latenty", str(ctx.exception))

    def test_bad_capability_names_the_entry(self):
        with self.assertRaises(ValueError) as ctx:
            self.load({"models": [{**VALID, "capabilities": {"coding": 8.0}}]})
        self.assertIn("m1", str(ctx.exception))

    def test_features_load_as_a_list_and_become_a_frozenset(self):
        registry = self.load({"models": [{**VALID, "features": ["json_mode", "vision"]}]})
        self.assertEqual(registry.get("m1").features, frozenset({"json_mode", "vision"}))


if __name__ == "__main__":
    unittest.main()
