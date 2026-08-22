"""Auditor failover: one grader's outage must not disable verification.

Measured context (live run, 2026-08-21): `cheapest_qualified` concentrated 38
of 60 audits onto a single model, whose provider spent the day returning
503s — so 23% of the run's audits established nothing. These tests pin the
fix: a failed audit *call* retries on a different qualified auditor within
`policy.max_auditor_failovers`, and still fails closed when the budget or the
catalog is exhausted. Fail-closed semantics are unchanged; only the number of
graders consulted before giving up is new.
"""

from __future__ import annotations

import unittest
from dataclasses import replace

from switchboard import COST_FIRST, Task, demo_registry
from switchboard.auditor import pick_auditor
from switchboard.broker import Broker
from switchboard.providers.base import (
    ProviderConfigError,
    ProviderPool,
    ProviderUnavailable,
    ScriptedProvider,
)

PASS_VERDICT = '{"pass": true, "score": 0.9, "issues": []}'

# Routes to atlas-small under cost_first (same case routing_eval pins).
# Escalation is disabled so these tests observe the audit path in isolation.
TASK = Task(prompt="extract the emails", task_type="extraction", complexity=0.2)
POLICY = replace(COST_FIRST, max_escalations=0)


def outage(model_id: str) -> ProviderUnavailable:
    return ProviderUnavailable(
        f"injected 503 on {model_id}", provider="mock", model_id=model_id
    )


def audit_calls(provider: ScriptedProvider, model_id: str) -> int:
    """How many audit prompts reached the given model."""
    from switchboard.prompts import AUDIT_PROMPT_HEADER

    return sum(
        1
        for called_model, prompt in provider.calls
        if called_model == model_id and AUDIT_PROMPT_HEADER in prompt
    )


class AuditorFailoverTest(unittest.TestCase):
    def test_outage_fails_over_to_next_auditor_and_verdict_stands(self):
        provider = ScriptedProvider({
            "atlas-small": ["the extracted emails"],
            "atlas-frontier": [outage("atlas-frontier")],  # first-choice auditor
            "atlas-mid": [PASS_VERDICT],                   # second choice
        })
        result = Broker(demo_registry(), ProviderPool([provider]), POLICY).run(TASK)

        self.assertTrue(result.verified)
        last = result.attempts[-1]
        self.assertEqual(last.auditor_model, "atlas-mid")
        # The trace says how the verdict was obtained: second-choice grader,
        # and which outage made it so.
        self.assertTrue(any("auditor failover" in i for i in last.audit_issues))
        self.assertTrue(any("atlas-frontier" in i for i in last.audit_issues))

    def test_zero_budget_fails_closed_without_trying_another_auditor(self):
        provider = ScriptedProvider({
            "atlas-small": ["the extracted emails"],
            "atlas-frontier": [outage("atlas-frontier")],
            "atlas-mid": [PASS_VERDICT],
        })
        policy = replace(POLICY, max_auditor_failovers=0)
        result = Broker(demo_registry(), ProviderPool([provider]), policy).run(TASK)

        self.assertFalse(result.verified)
        last = result.attempts[-1]
        self.assertFalse(last.audit_passed)
        self.assertTrue(any("failing closed" in i for i in last.audit_issues))
        self.assertEqual(audit_calls(provider, "atlas-mid"), 0)

    def test_every_auditor_down_fails_closed_and_names_each_outage(self):
        provider = ScriptedProvider({
            "atlas-small": ["the extracted emails"],
            "atlas-frontier": [outage("atlas-frontier")],
            "atlas-mid": [outage("atlas-mid")],
        })
        result = Broker(demo_registry(), ProviderPool([provider]), POLICY).run(TASK)

        self.assertFalse(result.verified)
        issues = result.attempts[-1].audit_issues
        self.assertTrue(any("failing closed" in i for i in issues))
        self.assertTrue(any("atlas-frontier" in i for i in issues))
        self.assertTrue(any("atlas-mid" in i for i in issues))

    def test_config_error_is_not_failed_over(self):
        # A missing key is a deployment bug. Routing around it would hide the
        # misconfiguration behind a quieter bill — same rule as the producer
        # side, so it fails closed loudly instead.
        provider = ScriptedProvider({
            "atlas-small": ["the extracted emails"],
            "atlas-frontier": [
                ProviderConfigError(
                    "no API key", provider="mock", model_id="atlas-frontier"
                )
            ],
            "atlas-mid": [PASS_VERDICT],
        })
        result = Broker(demo_registry(), ProviderPool([provider]), POLICY).run(TASK)

        self.assertFalse(result.verified)
        self.assertEqual(audit_calls(provider, "atlas-mid"), 0)

    def test_unattributable_outage_fails_closed(self):
        # An error that cannot name its model cannot be excluded, so the loop
        # must not spin retrying into the same failure.
        provider = ScriptedProvider({
            "atlas-small": ["the extracted emails"],
            "atlas-frontier": [ProviderUnavailable("injected 503, anonymous")],
            "atlas-mid": [PASS_VERDICT],
        })
        result = Broker(demo_registry(), ProviderPool([provider]), POLICY).run(TASK)

        self.assertFalse(result.verified)
        self.assertEqual(audit_calls(provider, "atlas-mid"), 0)


class PickAuditorExcludeTest(unittest.TestCase):
    def test_exclusion_moves_to_next_candidate(self):
        registry = demo_registry()
        producer = registry.get("atlas-small")
        first = pick_auditor(registry, producer)
        self.assertEqual(first.model_id, "atlas-frontier")
        second = pick_auditor(registry, producer, exclude={"atlas-frontier"})
        self.assertEqual(second.model_id, "atlas-mid")

    def test_exhausted_pool_raises_and_names_the_excluded(self):
        registry = demo_registry()
        producer = registry.get("atlas-small")
        with self.assertRaises(ValueError) as ctx:
            pick_auditor(
                registry, producer, exclude={"atlas-frontier", "atlas-mid"}
            )
        self.assertIn("atlas-frontier", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
