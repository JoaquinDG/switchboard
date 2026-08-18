"""Budget-aware policy: shift routing weight toward cost as spend nears a cap.

Policies are static weights, chosen once by a person as a product decision
(see ``policies.py``'s module docstring). Spend is not static — a team three
weeks into a monthly cap wants a different tradeoff from one on day one, and
until now that meant editing the policy in code mid-month.

This module reads what a Broker has actually spent from its own trace file
and derives an *effective* policy for the call in progress. The caller's
``Policy`` object never mutates — it stays exactly what the team configured —
and the shift is a separate, visible step: ``budget_position`` measures where
spend stands, and ``apply_budget_pressure`` is the only thing that touches
weights, moving them out of ``quality_weight``/``latency_weight`` and into
``cost_weight`` in proportion to how close spend is to the cap.

Cold start is the trap this item names explicitly: no trace file, or a trace
file with nothing in the budget window, must report zero pressure, not an
exhausted budget. ``BudgetPosition.sample_count == 0`` is what makes that
distinction impossible to lose on the way to the rationale string.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path

from .policies import Policy
from .replay import read_trace, task_records

__all__ = ["BudgetPosition", "apply_budget_pressure", "budget_position"]


@dataclass(frozen=True)
class BudgetPosition:
    """Where a policy's budget window currently stands."""

    spend_usd: float
    budget_usd: float
    period_days: int
    # Fraction of budget_usd spent in the window, clamped to [0, 1]. 0.0 both
    # for "nothing spent yet" and "no trace history at all" — sample_count is
    # what tells the two apart.
    pressure: float
    sample_count: int

    def describe(self) -> str:
        if self.sample_count == 0:
            return "budget: no trace history in window, pressure=0% (cold start, unshifted)"
        return (
            f"budget: ${self.spend_usd:.4f}/${self.budget_usd:.2f} spent over "
            f"the last {self.period_days}d ({self.pressure:.0%} of cap, "
            f"n={self.sample_count})"
        )


def budget_position(
    policy: Policy, trace_path: str | Path | None, *, now: float | None = None
) -> BudgetPosition:
    """Cumulative spend recorded in ``trace_path`` inside the policy's window.

    A missing trace file, or one with nothing inside ``budget_period_days``,
    is a cold start (``sample_count == 0``, ``pressure == 0.0``) rather than
    an error — a fresh deployment must not read as an exhausted budget.
    ``now`` exists only so tests can pin the clock; production callers always
    take the real one.
    """
    if policy.budget_usd is None:
        raise ValueError(
            "policy.budget_usd is not set — this policy has no budget to position against"
        )
    now = time.time() if now is None else now
    since = now - policy.budget_period_days * 86_400

    spend = 0.0
    count = 0
    if trace_path is not None and Path(trace_path).exists():
        for record in task_records(read_trace(trace_path)):
            ts = record.get("ts")
            cost = record.get("total_cost_usd")
            if not isinstance(ts, (int, float)) or not isinstance(cost, (int, float)):
                continue
            if ts < since:
                continue
            spend += cost
            count += 1

    pressure = 0.0 if count == 0 else max(0.0, min(1.0, spend / policy.budget_usd))
    return BudgetPosition(spend, policy.budget_usd, policy.budget_period_days, pressure, count)


def apply_budget_pressure(policy: Policy, position: BudgetPosition) -> Policy:
    """Shift weight from quality/latency into cost, proportional to pressure.

    The shift is taken out of ``quality_weight`` and ``latency_weight`` in
    proportion to their own share of the two, so a quality-heavy policy stays
    quality-heavy relative to latency even as pressure rises — only the mix
    between the scoring terms and cost moves. Capped at
    ``policy.budget_max_cost_shift``, reached once spend meets or exceeds the
    cap; spend overshooting the budget does not push the shift any further.
    """
    if position.pressure <= 0.0:
        return policy
    other = policy.quality_weight + policy.latency_weight
    if other <= 0.0:
        # Nothing to take the shift from (a policy that already spends its
        # entire weight on cost). Leave it as is rather than push a weight
        # negative.
        return policy
    shift = min(policy.budget_max_cost_shift * position.pressure, other)
    quality_frac = policy.quality_weight / other
    latency_frac = policy.latency_weight / other
    return replace(
        policy,
        quality_weight=policy.quality_weight - shift * quality_frac,
        latency_weight=policy.latency_weight - shift * latency_frac,
        cost_weight=policy.cost_weight + shift,
    )
