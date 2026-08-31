"""Reconstruct what happened from a trace alone.

A decision log that cannot be replayed is a diary, not an audit trail. If the
only way to know what a plan did is to have been watching, then the traces
cannot support the two things they exist for: offline analysis of routing
quality, and someone else checking the claims in this repository.

So the rule is: **everything needed to reconstruct a plan run is in the
trace**. Not a summary of it, not a pointer to it — the plan, the dispatch
order, what context was threaded into which step, what each step cost, and
what the totals were compared against. `tests/test_replay.py` proves it by
running a plan, throwing the live objects away, and rebuilding from the file.

## Reading a mixed stream

The trace holds two kinds of record and they are told apart by one key:

* records **with** an `"event"` key are plan events, added by `run_plan`;
* records **without** one are the per-task summaries `run()` has always
  written.

That asymmetry is deliberate. Adding an `"event"` discriminator to the
existing records would have been tidier, and would have changed the shape of
every trace already on disk. Additive beats tidy when someone else's tooling
may already be reading the old shape.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .planner import Plan, PlanStep

__all__ = [
    "ReplayedPlan",
    "ReplayedStep",
    "read_trace",
    "replay_plans",
    "task_records",
]

# Plan events, in the order run_plan emits them.
PLAN_EVENTS = (
    "plan_proposed",
    "plan_degraded",
    "step_dispatched",
    "step_completed",
    "plan_completed",
    "attempt_discarded",
    "step_skipped",
)


@dataclass
class ReplayedStep:
    """One step of a plan, rebuilt from its dispatch and completion events."""

    step_id: str
    step: PlanStep | None = None
    task_type: str = ""
    complexity: float = 0.0
    est_input_tokens: int = 0
    injected_chars: int = 0
    injected_truncated: bool = False
    injected_from: tuple[str, ...] = ()
    final_model: str = ""
    verified: bool = False
    escalated: bool = False
    truncated: bool = False
    adds_value: bool | None = None
    generation_cost_usd: float = 0.0
    audit_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    output_text: str = ""
    attempts: int = 0
    # False when the stream ended before this step reported completion — a
    # crashed or still-running plan, which is a real thing to be able to see.
    completed: bool = False


@dataclass
class ReplayedPlan:
    """A plan run rebuilt from the trace, with nothing inferred."""

    request: str = ""
    planned_by: str = ""
    confidence: float = 0.0
    rationale: str = ""
    signals: tuple[str, ...] = ()
    plan: Plan | None = None
    steps: list[ReplayedStep] = field(default_factory=list)
    verified: bool = False
    routed_cost_usd: float = 0.0
    baseline_best_model_usd: float = 0.0
    baseline_single_call_usd: float = 0.0
    baseline_single_call_model: str = ""
    baseline_single_call_is_modelled: bool = True
    final_text: str = ""
    assembled_text: str = ""
    skipped_steps: list[tuple[str, str]] = field(default_factory=list)
    # Set when the planner failed and the request was routed as one task.
    degraded_reason: str | None = None
    # Model plans that failed validation and were paid for anyway.
    discarded_attempts: list[dict] = field(default_factory=list)
    completed: bool = False

    @property
    def is_split(self) -> bool:
        return len(self.steps) > 1

    @property
    def dispatch_order(self) -> tuple[str, ...]:
        """The order steps actually ran, which is the point of replaying."""
        return tuple(s.step_id for s in self.steps)


def read_trace(path: str | Path) -> list[dict]:
    """Load a JSONL trace, skipping blank lines.

    A malformed line raises rather than being dropped: silently skipping a
    record you cannot parse is how a replay quietly stops matching the run.
    """
    records = []
    for number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}: line {number} is not valid JSON ({e})") from None
    return records


def task_records(records: list[dict]) -> list[dict]:
    """The per-task summaries: records with no `event` key.

    These predate plan events and their shape is unchanged, which is why the
    discriminator is an absence rather than a value.
    """
    return [r for r in records if "event" not in r]


def _plan_from_event(payload: dict) -> Plan | None:
    """Rebuild the Plan object from the plan_proposed event."""
    raw_steps = payload.get("steps") or []
    try:
        steps = tuple(
            PlanStep(
                step_id=s["step_id"],
                prompt=s["prompt"],
                task_type=s["task_type"],
                complexity=s["complexity"],
                est_input_tokens=s["est_input_tokens"],
                est_output_tokens=s["est_output_tokens"],
                depends_on=tuple(s.get("depends_on") or ()),
            )
            for s in raw_steps
        )
    except (KeyError, TypeError):
        return None
    if not steps:
        return None
    return Plan(
        request=payload.get("request", ""),
        steps=steps,
        planned_by=payload.get("planned_by", ""),
        confidence=payload.get("confidence", 0.0),
        rationale=payload.get("rationale", ""),
        signals=tuple(payload.get("signals") or ()),
    )


def replay_plans(records: list[dict]) -> list[ReplayedPlan]:
    """Rebuild every plan run present in a trace, in order.

    Tolerates a truncated stream: a plan whose `plan_completed` never arrived
    comes back with `completed=False` rather than being dropped, because a run
    that died halfway is exactly the thing you want to inspect.
    """
    plans: list[ReplayedPlan] = []
    current: ReplayedPlan | None = None
    pending: dict[str, ReplayedStep] = {}

    for record in records:
        event = record.get("event")
        if event is None:
            continue  # a per-task summary; detail, not structure

        if event == "plan_proposed":
            current = ReplayedPlan(
                request=record.get("request", ""),
                planned_by=record.get("planned_by", ""),
                confidence=record.get("confidence", 0.0),
                rationale=record.get("rationale", ""),
                signals=tuple(record.get("signals") or ()),
                plan=_plan_from_event(record),
            )
            pending = {}
            plans.append(current)

        elif current is None:
            continue  # events before any plan started; nothing to attach to

        elif event == "plan_degraded":
            current.degraded_reason = record.get("reason")

        elif event == "attempt_discarded":
            current.discarded_attempts.append(dict(record))

        elif event == "step_dispatched":
            step = ReplayedStep(
                step_id=record.get("step_id", ""),
                task_type=record.get("task_type", ""),
                complexity=record.get("complexity", 0.0),
                est_input_tokens=record.get("est_input_tokens", 0),
                injected_chars=record.get("injected_chars", 0),
                injected_truncated=record.get("injected_truncated", False),
                injected_from=tuple(record.get("injected_from") or ()),
            )
            if current.plan is not None:
                step.step = next(
                    (s for s in current.plan.steps if s.step_id == step.step_id), None
                )
            pending[step.step_id] = step
            current.steps.append(step)

        elif event == "step_completed":
            step = pending.get(record.get("step_id", ""))
            if step is None:
                continue
            step.final_model = record.get("final_model", "")
            step.verified = record.get("verified", False)
            step.escalated = record.get("escalated", False)
            step.truncated = record.get("truncated", False)
            step.adds_value = record.get("adds_value")
            step.generation_cost_usd = record.get("generation_cost_usd", 0.0)
            step.audit_cost_usd = record.get("audit_cost_usd", 0.0)
            step.total_cost_usd = record.get("total_cost_usd", 0.0)
            step.output_text = record.get("output_text", "")
            step.attempts = record.get("attempts", 0)
            step.completed = True

        elif event == "plan_completed":
            current.verified = record.get("verified", False)
            current.routed_cost_usd = record.get("routed_cost_usd", 0.0)
            current.baseline_best_model_usd = record.get("baseline_best_model_usd", 0.0)
            current.baseline_single_call_usd = record.get("baseline_single_call_usd", 0.0)
            current.baseline_single_call_model = record.get("baseline_single_call_model", "")
            current.baseline_single_call_is_modelled = record.get(
                "baseline_single_call_is_modelled", True
            )
            current.final_text = record.get("final_text", "")
            current.assembled_text = record.get("assembled_text", "")
            current.skipped_steps = [
                tuple(pair) for pair in record.get("skipped_steps") or []
            ]
            current.completed = True
            current = None

    return plans
