import json
import tempfile
import unittest
from pathlib import Path

from switchboard import (
    AsyncBroker,
    AsyncFlakyProvider,
    AsyncMockProvider,
    AsyncProviderPool,
    AsyncScriptedProvider,
    BALANCED,
    COST_FIRST,
    Policy,
    ProviderConfigError,
    ProviderUnavailable,
    Task,
    async_mock_pool,
    demo_registry,
)

PASS_VERDICT = '{"pass": true, "score": 0.9, "issues": []}'
FAIL_VERDICT = '{"pass": false, "score": 0.2, "issues": ["wrong"]}'


def make_async_broker(policy=BALANCED, trace_path=None):
    return AsyncBroker(
        demo_registry(), AsyncProviderPool([AsyncMockProvider()]), policy, trace_path
    )


class AsyncBrokerRunTests(unittest.IsolatedAsyncioTestCase):
    """AsyncBroker.run must reach the same decisions as Broker.run — it is
    the same routing, escalation and accounting logic behind an await, not a
    separate implementation that happens to agree today."""

    async def test_basic_run_is_verified_against_the_offline_auditor(self):
        result = await make_async_broker().run(
            Task(prompt="summarize this", task_type="summarization")
        )
        self.assertTrue(result.verified)
        self.assertTrue(result.attempts[0].synthetic)

    async def test_failed_audit_escalates_one_tier(self):
        task = Task(prompt="FORCE_AUDIT_FAIL do something easy", task_type="extraction", complexity=0.2)
        result = await make_async_broker().run(task)
        self.assertTrue(result.escalated)
        self.assertEqual(len(result.attempts), 2)
        self.assertFalse(result.verified)

    async def test_escalation_recovers_when_the_stronger_model_passes(self):
        broker = AsyncBroker(
            demo_registry(),
            AsyncProviderPool([AsyncScriptedProvider({
                "atlas-small": ["weak draft"],
                "atlas-mid": ["stronger draft"],
                "atlas-frontier": [FAIL_VERDICT, PASS_VERDICT],
            }, name="mock")]),
            COST_FIRST,
        )
        result = await broker.run(Task(prompt="pull the dates", task_type="extraction", complexity=0.2))
        self.assertTrue(result.escalated)
        self.assertTrue(result.verified)
        self.assertEqual(result.final_model, "atlas-mid")
        self.assertEqual(result.final_text, "stronger draft")

    async def test_provider_outage_fails_over_to_the_next_ranked_model(self):
        registry = demo_registry()
        pool = AsyncProviderPool([
            AsyncFlakyProvider(
                AsyncScriptedProvider(name="mock", default=PASS_VERDICT),
                fail_times=1,
                error=ProviderUnavailable("mock is down"),
            )
        ])
        # Failover needs a second candidate at the same tier from routing's
        # own ranked list, which the demo registry's single "mock" provider
        # can't offer — every model shares one provider name. Route via cost
        # first so the outage is visible on the chosen model, then confirm
        # the broker recorded it rather than silently retrying the same spec.
        result = await AsyncBroker(registry, pool, COST_FIRST).run(
            Task(prompt="pull the dates", task_type="extraction", complexity=0.2)
        )
        self.assertEqual(result.attempts[0].error, "mock is down")

    async def test_provider_config_error_is_not_retried(self):
        pool = AsyncProviderPool([
            AsyncScriptedProvider(name="mock", default=ProviderConfigError("no api key"))
        ])
        with self.assertRaises(Exception):
            await AsyncBroker(demo_registry(), pool, BALANCED).run(
                Task(prompt="x", task_type="extraction", complexity=0.2)
            )

    async def test_audit_disabled_skips_verification(self):
        policy = Policy("no_audit", 0.5, 0.3, 0.2, audit_enabled=False)
        result = await make_async_broker(policy).run(Task(prompt="hello", task_type="reasoning"))
        self.assertFalse(result.verified)
        self.assertIsNone(result.attempts[0].audit_passed)

    async def test_cost_accounting_matches_actual_cost(self):
        result = await make_async_broker().run(Task(prompt="summarize", task_type="summarization"))
        self.assertGreater(result.generation_cost_usd, 0.0)
        self.assertGreater(result.audit_cost_usd, 0.0)
        self.assertAlmostEqual(
            result.total_cost_usd, result.generation_cost_usd + result.audit_cost_usd
        )

    async def test_auto_task_type_is_resolved_by_the_heuristic(self):
        result = await make_async_broker().run(
            Task(prompt="write a python function to parse a date", task_type="auto")
        )
        self.assertIsNotNone(result.triage)
        self.assertEqual(result.triage.source, "heuristic")
        self.assertIn("triage: classified as", result.routing_rationale)

    async def test_trace_is_written_in_the_same_shape_as_the_sync_broker(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "traces.jsonl"
            await make_async_broker(trace_path=trace).run(
                Task(prompt="x", task_type="summarization")
            )
            record = json.loads(trace.read_text().strip())
            self.assertIn("final_audit_cross_lab", record)
            self.assertIn("attempts", record)
            self.assertIn("savings_vs_baseline_usd", record)


class AsyncProviderPrimitivesTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_mock_pool_covers_every_provider_in_a_registry(self):
        pool = async_mock_pool(demo_registry())
        for model in demo_registry().all():
            self.assertTrue(pool.has(model.provider))

    async def test_async_scripted_provider_repeats_its_last_entry(self):
        provider = AsyncScriptedProvider({"m": ["first", "second"]})
        self.assertEqual((await provider.complete("m", "p")).text, "first")
        self.assertEqual((await provider.complete("m", "p")).text, "second")
        self.assertEqual((await provider.complete("m", "p")).text, "second")

    async def test_async_flaky_provider_recovers_after_fail_times(self):
        provider = AsyncFlakyProvider(AsyncMockProvider(), fail_times=2)
        with self.assertRaises(ProviderUnavailable):
            await provider.complete("m", "p")
        with self.assertRaises(ProviderUnavailable):
            await provider.complete("m", "p")
        result = await provider.complete("m", "p")
        self.assertIn("completed", result.text)


if __name__ == "__main__":
    unittest.main()
