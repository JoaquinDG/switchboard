"""Routing evals: does the router make the decisions we intended?

This is a scenario suite, not a unit test: each case encodes a product
expectation ("cheap extraction should go to the small model") and the harness
reports pass/fail per policy. Run it after any change to the catalog or
scoring — routing regressions are silent in production but loud here.

Usage:
    PYTHONPATH=src python3 evals/routing_eval.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from switchboard import BALANCED, COST_FIRST, QUALITY_FIRST, Registry, Task, demo_registry, route

STARTER_CATALOG = Path(__file__).resolve().parent.parent / "examples" / "starter_catalog.json"


@dataclass(frozen=True)
class Scenario:
    name: str
    task: Task
    policy_name: str
    expect_tier: str | None = None
    expect_model: str | None = None
    forbid_tier: str | None = None
    forbid_latency: str | None = None
    expect_underqualified: bool | None = None
    expect_warning: str | None = None  # substring match
    # Defaults to the 3-model demo registry; pass "starter" to run against
    # the 16-model examples/starter_catalog.json instead. ROADMAP item 1b's
    # scale bug only showed up on the wider catalog, so its regression
    # scenario needs to run there specifically.
    registry_name: str = "demo"

    def check(self, decision) -> list[str]:
        """Return the reasons this scenario failed, empty if it passed."""
        chosen = decision.chosen
        problems = []
        if self.expect_model and chosen.model_id != self.expect_model:
            problems.append(f"expected model {self.expect_model}")
        if self.expect_tier and chosen.tier != self.expect_tier:
            problems.append(f"expected tier {self.expect_tier}")
        if self.forbid_tier and chosen.tier == self.forbid_tier:
            problems.append(f"must not use tier {self.forbid_tier}")
        if self.forbid_latency and chosen.latency == self.forbid_latency:
            problems.append(f"must not use {self.forbid_latency} models")
        if (
            self.expect_underqualified is not None
            and decision.underqualified != self.expect_underqualified
        ):
            problems.append(f"expected underqualified={self.expect_underqualified}")
        if self.expect_warning and not any(
            self.expect_warning in w for w in decision.warnings
        ):
            problems.append(f"expected a warning containing {self.expect_warning!r}")
        return problems


POLICIES = {p.name: p for p in (QUALITY_FIRST, BALANCED, COST_FIRST)}

SCENARIOS = [
    Scenario(
        "easy extraction stays cheap under cost_first",
        Task(prompt="extract emails", task_type="extraction", complexity=0.2),
        "cost_first",
        expect_model="atlas-small",
    ),
    Scenario(
        "hard reasoning hits the frontier gate even under cost_first",
        Task(prompt="novel proof", task_type="reasoning", complexity=0.95),
        "cost_first",
        expect_tier="frontier",
    ),
    Scenario(
        "quality_first sends creative work to frontier",
        Task(prompt="write a launch narrative", task_type="creative", complexity=0.6),
        "quality_first",
        expect_tier="frontier",
    ),
    Scenario(
        "latency-sensitive summarization avoids slow models",
        Task(prompt="tl;dr now", task_type="summarization", complexity=0.3, needs_fast_response=True),
        "balanced",
        forbid_latency="slow",
    ),
    Scenario(
        "balanced routes mid-complexity coding away from the small model",
        Task(prompt="refactor this module", task_type="coding", complexity=0.6),
        "balanced",
        forbid_tier="small",
    ),
    # --- Gate fallback expectations -------------------------------------
    # Regression cases for gates that used to fail *open*: when a gate left
    # nothing standing it silently restored the full candidate list, handing
    # the decision to cost weight — exactly what the gate existed to prevent.
    Scenario(
        "an unscored task type does not fall through to the cheapest model",
        Task(prompt="review this contract", task_type="legal_analysis", complexity=0.75),
        "quality_first",
        forbid_tier="small",
        expect_underqualified=True,
        expect_warning="no capability data",
    ),
    Scenario(
        "an unscored task type is still cheap when the work is genuinely easy",
        Task(prompt="tag this ticket", task_type="legal_analysis", complexity=0.2),
        "cost_first",
        expect_model="atlas-small",
        expect_underqualified=False,
    ),
    Scenario(
        "work beyond every model's rating routes to the best and says so",
        Task(prompt="unsolved problem", task_type="reasoning", complexity=1.0),
        "cost_first",
        expect_tier="frontier",
        expect_underqualified=True,
        expect_warning="no model clears capability",
    ),
    # --- ROADMAP 1b: quality's weight has to outweigh what its number says --
    # README Finding 3 measured this exact case going the wrong way: a 0.093
    # raw-capability advantage for claude-opus-5 lost to gemini-3.7-flash's
    # 0.080 latency advantage, because raw capability clusters near the top
    # of [0, 1] while cost/latency use their full range. Only showed up on
    # the 16-model starter catalog — the 3-model demo catalog's capability
    # gaps happened to be wide enough to already win.
    Scenario(
        "quality_first is not swayed by a latency win the way it used to be",
        Task(prompt="deep reasoning task", task_type="reasoning", complexity=0.6),
        "quality_first",
        expect_model="claude-opus-5",
        registry_name="starter",
    ),
]


def run() -> int:
    registries = {"demo": demo_registry(), "starter": Registry.from_json(STARTER_CATALOG)}
    failures = 0
    print(f"{'scenario':<62} {'policy':<14} {'chose':<16} result")
    print("-" * 104)
    for sc in SCENARIOS:
        registry = registries[sc.registry_name]
        decision = route(sc.task, registry, POLICIES[sc.policy_name])
        problems = sc.check(decision)
        failures += 1 if problems else 0
        status = "FAIL" if problems else "PASS"
        print(f"{sc.name:<62} {sc.policy_name:<14} {decision.chosen.model_id:<16} {status}")
        for problem in problems:
            print(f"{'':<62} {'':<14} {'':<16}   -> {problem}")
    print("-" * 104)
    print(f"{len(SCENARIOS) - failures}/{len(SCENARIOS)} scenarios passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run())
