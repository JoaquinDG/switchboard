"""Tasks and routing policies.

A Task describes *what needs doing* (type, complexity, expected token volume,
latency need). A Policy describes *what the business cares about* (quality,
cost, latency) as explicit weights. Keeping these separate is the core design
decision: engineers define tasks, but the quality/cost tradeoff is a product
decision — so it lives in one visible, versionable place instead of being
scattered through code.
"""

from __future__ import annotations

from dataclasses import dataclass

# What to do when no model in the catalog clears the qualification gate.
# The one option deliberately *not* offered is "score everything anyway":
# that was the original fail-open behaviour, and it silently routed
# unqualified work to whatever was cheapest.
NO_QUALIFIED_MODEL_STRATEGIES = ("escalate_tier", "best_capability", "raise")

AUDITOR_SELECTION_STRATEGIES = ("most_capable", "cheapest_qualified")


class NoQualifiedModelError(RuntimeError):
    """Raised when nothing qualifies and the policy says to fail loudly."""


@dataclass(frozen=True)
class Task:
    """A unit of work to route."""

    prompt: str
    task_type: str = "reasoning"  # matches capability keys in the registry
    complexity: float = 0.5  # 0-1; >= frontier_gate forces frontier tier
    est_input_tokens: int = 1_000
    est_output_tokens: int = 500
    needs_fast_response: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.complexity <= 1.0:
            raise ValueError("complexity must be in [0, 1]")
        if not self.task_type:
            raise ValueError("task_type must be a non-empty string")
        for label, value in (
            ("est_input_tokens", self.est_input_tokens),
            ("est_output_tokens", self.est_output_tokens),
        ):
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{label} must be a non-negative int, got {value!r}")


@dataclass(frozen=True)
class Policy:
    """Explicit weights for the routing tradeoff. Weights should sum to ~1."""

    name: str
    quality_weight: float
    cost_weight: float
    latency_weight: float
    # Tasks at or above this complexity are only eligible for frontier models.
    frontier_gate: float = 0.8
    # Qualification gate: a model's capability for the task type must exceed
    # task complexity by this margin to be eligible. Prevents cheap models
    # from winning tasks they aren't qualified for just by being cheap.
    capability_margin: float = 0.1
    # What happens when the gate leaves nothing standing. Default degrades
    # *upward* (restrict to the highest tier available) rather than opening
    # the field back up to the whole catalog.
    on_no_qualified_model: str = "escalate_tier"
    # Audit settings
    audit_enabled: bool = True
    audit_pass_threshold: float = 0.7
    max_escalations: int = 1
    # How the auditor is chosen. "most_capable" always picks the strongest
    # audit model, which on a cheap task can cost more than the work being
    # audited — visible now that costs are tracked. "cheapest_qualified"
    # picks the cheapest model clearing min_auditor_capability instead.
    auditor_selection: str = "most_capable"
    min_auditor_capability: float = 0.7
    # Prefer an auditor from a different provider when the catalog has one.
    # Two models from the same lab share training data and alignment, so their
    # blind spots correlate — a same-lab audit is weaker evidence than the
    # pass rate suggests. Independence outranks both capability and price, so
    # this filter runs before auditor_selection.
    prefer_cross_lab_auditor: bool = True
    # Triage: ask a model only when the heuristic's confidence falls below
    # this. Measured on 40 labeled prompts, the heuristic's confidence is a
    # clean separator of its own errors — every wrong classification scored
    # exactly 0.00 while correct ones averaged 0.93 — so gating captures the
    # model layer's accuracy on ~12% of the calls. Set to 0.0 to never ask a
    # model, or above 1.0 to always ask.
    triage_confidence_threshold: float = 0.25
    # How many times the broker may reroute to the next-ranked model when a
    # provider call fails outright (outage, rate limit, timeout). Distinct
    # from escalation, which is about quality, not availability.
    max_provider_failovers: int = 2

    def __post_init__(self) -> None:
        total = self.quality_weight + self.cost_weight + self.latency_weight
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"policy weights must sum to 1.0, got {total}")
        for label, value in (
            ("quality_weight", self.quality_weight),
            ("cost_weight", self.cost_weight),
            ("latency_weight", self.latency_weight),
            ("frontier_gate", self.frontier_gate),
            ("capability_margin", self.capability_margin),
            ("audit_pass_threshold", self.audit_pass_threshold),
            ("min_auditor_capability", self.min_auditor_capability),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must be in [0, 1], got {value}")
        if self.auditor_selection not in AUDITOR_SELECTION_STRATEGIES:
            raise ValueError(
                f"auditor_selection must be one of {AUDITOR_SELECTION_STRATEGIES}, "
                f"got {self.auditor_selection!r}"
            )
        for label, value in (
            ("max_escalations", self.max_escalations),
            ("max_provider_failovers", self.max_provider_failovers),
        ):
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{label} must be a non-negative int, got {value!r}")
        if self.on_no_qualified_model not in NO_QUALIFIED_MODEL_STRATEGIES:
            raise ValueError(
                f"on_no_qualified_model must be one of {NO_QUALIFIED_MODEL_STRATEGIES}, "
                f"got {self.on_no_qualified_model!r}"
            )


# Three opinionated presets. Most teams should start with BALANCED and only
# move after looking at trace data.
QUALITY_FIRST = Policy("quality_first", 0.85, 0.05, 0.10)
BALANCED = Policy("balanced", 0.50, 0.30, 0.20)
COST_FIRST = Policy("cost_first", 0.30, 0.60, 0.10, frontier_gate=0.9)

PRESETS: dict[str, Policy] = {
    p.name: p for p in (QUALITY_FIRST, BALANCED, COST_FIRST)
}
