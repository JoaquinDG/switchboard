"""Live-provider wiring, and opt-in integration tests against real APIs.

Two kinds of test live here.

The **wiring** tests always run. They check that a catalog maps to the right
adapters, endpoints, and environment variables without making a single network
call — the mapping is the part most likely to be silently wrong (a DeepSeek
model quietly billed to an OpenAI key would work right up until it didn't).

The **integration** tests only run when `SWITCHBOARD_LIVE_TESTS=1` *and* the
relevant key is set. They cost real money, so they are opt-in twice over and
never run in CI. Requiring an explicit env var on top of key presence means a
developer with keys exported for ordinary work does not start paying for test
runs by accident.

    SWITCHBOARD_LIVE_TESTS=1 PYTHONPATH=src python3 -m unittest tests.test_live
"""

import os
import unittest

from switchboard import (
    BALANCED,
    AnthropicProvider,
    Broker,
    ModelSpec,
    OpenAICompatibleProvider,
    Registry,
    Task,
)
from switchboard.providers.live import (
    KNOWN_PROVIDERS,
    build_provider,
    key_status,
    live_pool,
    usable_registry,
)

LIVE = os.environ.get("SWITCHBOARD_LIVE_TESTS") == "1"


def has_key(provider: str) -> bool:
    spec = KNOWN_PROVIDERS.get(provider)
    return bool(spec and os.environ.get(spec.env_var))


def live_reason(provider: str) -> str:
    spec = KNOWN_PROVIDERS.get(provider)
    env = spec.env_var if spec else "?"
    return f"set SWITCHBOARD_LIVE_TESTS=1 and {env} to run live tests"


def catalog(*entries) -> Registry:
    return Registry([
        ModelSpec(model_id=mid, provider=prov, tier=tier, input_cost=1.0, output_cost=2.0,
                  capabilities={"reasoning": 0.9, "audit": 0.9, "extraction": 0.8})
        for mid, prov, tier in entries
    ])


class ProviderMappingTests(unittest.TestCase):
    """No network. Checks each vendor gets its own key and its own endpoint."""

    def test_every_known_provider_has_a_distinct_env_var(self):
        # Reusing one vendor's key against another's endpoint would either
        # fail confusingly or, worse, bill the wrong account.
        env_vars = [s.env_var for s in KNOWN_PROVIDERS.values()]
        self.assertEqual(len(env_vars), len(set(env_vars)))

    def test_anthropic_maps_to_the_messages_adapter(self):
        self.assertIsInstance(build_provider("anthropic"), AnthropicProvider)

    def test_deepseek_uses_its_own_endpoint_and_key(self):
        provider = build_provider("deepseek")
        self.assertIsInstance(provider, OpenAICompatibleProvider)
        self.assertEqual(provider.name, "deepseek")
        self.assertIn("deepseek.com", provider.base_url)
        self.assertEqual(provider.env_var, "DEEPSEEK_API_KEY")

    def test_google_uses_the_openai_compatible_endpoint(self):
        provider = build_provider("google")
        self.assertIn("generativelanguage.googleapis.com", provider.base_url)
        self.assertEqual(provider.env_var, "GEMINI_API_KEY")

    def test_non_openai_vendors_keep_max_tokens(self):
        # Only OpenAI's own newer models require max_completion_tokens;
        # sending it to a compatible vendor is a hard 400.
        self.assertEqual(build_provider("deepseek").max_tokens_param, "max_tokens")
        self.assertEqual(build_provider("google").max_tokens_param, "max_tokens")

    def test_openai_gets_max_completion_tokens(self):
        self.assertEqual(build_provider("openai").max_tokens_param, "max_completion_tokens")

    def test_unknown_provider_is_refused_with_guidance(self):
        with self.assertRaises(KeyError) as ctx:
            build_provider("some-vendor")
        self.assertIn("KNOWN_PROVIDERS", str(ctx.exception))

    def test_starter_catalog_providers_are_all_known(self):
        from pathlib import Path

        starter = Path(__file__).resolve().parents[1] / "examples" / "starter_catalog.json"
        registry = Registry.from_json(starter)
        for provider in {m.provider for m in registry.all()}:
            with self.subTest(provider=provider):
                self.assertIn(provider, KNOWN_PROVIDERS)


class KeyStatusTests(unittest.TestCase):
    def test_reports_presence_only(self):
        # The function must never hand back a key value, only a boolean.
        for value in key_status().values():
            self.assertIsInstance(value, bool)

    def test_covers_every_known_provider_by_default(self):
        self.assertEqual(set(key_status()), set(KNOWN_PROVIDERS))


class LivePoolTests(unittest.TestCase):
    """No network — exercises the skip logic with the ambient environment."""

    def test_missing_keys_are_skipped_not_fatal(self):
        registry = catalog(("m", "anthropic", "mid"), ("n", "nonexistent-vendor", "mid"))
        pool, skipped = live_pool(registry)
        self.assertIn("nonexistent-vendor", skipped)
        self.assertNotIn("nonexistent-vendor", pool.names())

    def test_strict_mode_raises_instead(self):
        registry = catalog(("n", "nonexistent-vendor", "mid"))
        with self.assertRaises(KeyError):
            live_pool(registry, skip_missing_keys=False)

    def test_usable_registry_drops_unkeyed_models(self):
        # Routing to a model whose provider has no key would surface as an
        # outage and a failover, which works but measures the wrong thing.
        registry = catalog(("keyed", "anthropic", "mid"), ("unkeyed", "nonexistent", "mid"))
        pool, _ = live_pool(registry)
        usable = usable_registry(registry, pool)
        for spec in usable.all():
            self.assertTrue(pool.has(spec.provider))

    def test_usable_registry_preserves_catalog_metadata(self):
        registry = catalog(("m", "anthropic", "mid"))
        registry.metadata["_last_verified"] = "2026-08-15"
        pool, _ = live_pool(registry)
        self.assertEqual(
            usable_registry(registry, pool).metadata.get("_last_verified"), "2026-08-15"
        )


@unittest.skipUnless(LIVE and has_key("anthropic"), live_reason("anthropic"))
class LiveAnthropicTests(unittest.TestCase):
    def test_lists_models(self):
        models = AnthropicProvider().list_models()
        self.assertTrue(models)
        self.assertTrue(all(isinstance(m, str) for m in models))

    def test_completes_a_tiny_prompt(self):
        result = AnthropicProvider().complete(
            "claude-haiku-4-5-20251001", "Reply with exactly: OK", max_tokens=8
        )
        self.assertTrue(result.text.strip())
        self.assertGreater(result.input_tokens, 0)
        self.assertGreater(result.output_tokens, 0)


@unittest.skipUnless(LIVE and has_key("openai"), live_reason("openai"))
class LiveOpenAITests(unittest.TestCase):
    def test_lists_models(self):
        self.assertTrue(build_provider("openai").list_models())


@unittest.skipUnless(LIVE and has_key("deepseek"), live_reason("deepseek"))
class LiveDeepSeekTests(unittest.TestCase):
    def test_lists_models(self):
        self.assertTrue(build_provider("deepseek").list_models())


@unittest.skipUnless(
    LIVE and sum(has_key(p) for p in ("anthropic", "openai", "deepseek")) >= 2,
    "set SWITCHBOARD_LIVE_TESTS=1 and at least two provider keys",
)
class LiveCrossLabAuditTests(unittest.TestCase):
    """The claim the offline suite cannot test: a real model from one lab
    grading a real model from another."""

    def test_real_cross_lab_audit_produces_a_real_verdict(self):
        from pathlib import Path

        starter = Path(__file__).resolve().parents[1] / "examples" / "starter_catalog.json"
        registry = Registry.from_json(starter)
        pool, _ = live_pool(registry)
        usable = usable_registry(registry, pool)

        result = Broker(usable, pool, BALANCED).run(Task(
            prompt="Reply with exactly the word: OK",
            task_type="extraction", complexity=0.1,
            est_input_tokens=20, est_output_tokens=10,
        ))
        last = result.attempts[-1]
        self.assertIsNotNone(last.auditor_model)
        self.assertTrue(last.cross_lab_audit, "expected a cross-lab audit with 2+ providers")
        self.assertGreater(result.total_cost_usd, 0.0)
        self.assertGreater(result.audit_cost_usd, 0.0)


if __name__ == "__main__":
    unittest.main()
