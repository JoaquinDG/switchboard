"""The Broker: route -> run -> audit -> escalate, with a full trace.

Escalation logic: if the audit fails and the policy allows escalation, re-run
the task on the next tier up (small -> mid -> frontier) and re-audit. Already
at frontier, or out of escalation budget? Return the output flagged as
unverified rather than silently retrying forever — surfacing "we could not
verify this" is a feature, not a failure.

Two things are tracked alongside the output, because a brokerage that cannot
report either one is not doing its job:

*Money.* Every attempt records what it actually cost at observed token counts,
audits included. Verification is not free, and the honest comparison for
"routing saved us money" has to charge for the audits that made the routing
trustworthy. `baseline_cost_usd` prices the same work on the model you would
have used if you always reached for the best one.

*Availability.* A provider outage is not a quality problem, so it does not
consume escalation budget. It moves the task to the next-ranked model from the
same routing decision, within a separate failover budget.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from .auditor import AuditVerdict, audit
from .policies import Policy, Task
from .prompts import build_retry_prompt
from .providers.base import Completion, ProviderConfigError, ProviderError, ProviderPool
from .registry import TIER_RANK, ModelSpec, Registry
from .router import RoutingDecision, actual_cost, route
from .triage import AUTO, Triage, triage_task

_ESCALATION_ORDER = {"small": "mid", "mid": "frontier", "frontier": None}


@dataclass
class Attempt:
    """One model's go at the task, and what it cost to find out.

    An attempt exists even when the provider call failed outright, so the
    trace shows the outage rather than silently skipping to whatever worked.
    """

    model_id: str
    tier: str
    output_text: str
    audit_passed: bool | None
    audit_score: float | None
    audit_issues: list[str] = field(default_factory=list)
    # Why this model ran: "initial", "escalation" (audit failed), or
    # "failover" (the previous model's provider was unavailable).
    role: str = "initial"
    auditor_model: str | None = None
    # False when the audit was same-lab (see pick_auditor). None if unaudited.
    cross_lab_audit: bool | None = None
    # True when this attempt was given the previous audit's findings to fix.
    had_audit_feedback: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    audit_cost_usd: float = 0.0
    # Set when the provider call itself failed; output_text is empty.
    error: str | None = None

    @property
    def total_cost_usd(self) -> float:
        """Generation plus the audit that graded it."""
        return self.cost_usd + self.audit_cost_usd


@dataclass
class BrokerResult:
    """The output, plus everything needed to decide whether to trust it.

    Deliberately more than the text: `verified` and `underqualified` say how
    much confidence is warranted, and the cost fields say what that confidence
    cost. A caller that only reads `final_text` is using this like a plain API
    client, which is the equilibrium the project exists to break.
    """

    final_text: str
    final_model: str
    verified: bool  # True only if the final output passed audit
    routing_rationale: str
    attempts: list[Attempt] = field(default_factory=list)
    # Routing flags worth acting on: no model was rated for this task, or the
    # catalog had no data for the task type at all.
    underqualified: bool = False
    warnings: list[str] = field(default_factory=list)
    # Set when the task arrived as task_type="auto" and triage resolved it.
    # Carries which layer decided, so a reader never mistakes a keyword guess
    # for a classification someone paid a model to make.
    triage: Triage | None = None
    # Cost accounting, in USD, from observed token counts.
    generation_cost_usd: float = 0.0
    audit_cost_usd: float = 0.0
    baseline_cost_usd: float = 0.0
    baseline_model: str = ""

    @property
    def escalated(self) -> bool:
        """True if a failed audit pushed the work up a tier."""
        return sum(1 for a in self.attempts if a.role == "escalation") > 0

    @property
    def failed_over(self) -> bool:
        """True if a provider outage moved the work to another model."""
        return sum(1 for a in self.attempts if a.role == "failover") > 0

    @property
    def total_cost_usd(self) -> float:
        """Everything this task cost, verification included."""
        return self.generation_cost_usd + self.audit_cost_usd

    @property
    def savings_vs_baseline_usd(self) -> float:
        """Money not spent versus always using the strongest model.

        Negative when verification cost more than it saved — which is real
        information, not an error, and the number a cost-first policy should
        be judged on.
        """
        return self.baseline_cost_usd - self.total_cost_usd


class Broker:
    """Runs a task end to end and reports what happened.

    The router decides *where* work goes; the Broker owns what happens when
    that decision meets reality — audits that fail, providers that fall over,
    and the bill for both.
    """

    def __init__(
        self,
        registry: Registry,
        providers: ProviderPool,
        policy: Policy,
        trace_path: str | Path | None = None,
        *,
        triage_use_model: bool = False,
    ) -> None:
        self.registry = registry
        self.providers = providers
        self.policy = policy
        self.trace_path = Path(trace_path) if trace_path else None
        # Opt-in: spend a cheap model call to classify task_type="auto" tasks
        # instead of using the offline heuristic. Off by default so the
        # library keeps working with no providers and no budget.
        self.triage_use_model = triage_use_model

    def run(self, task: Task) -> BrokerResult:
        """Route, execute, verify, and escalate or reroute as needed.

        Pass ``task_type="auto"`` to have triage infer the type and complexity
        first. Raises ProviderError only when every routing option has been
        exhausted; a failed *audit* is reported as ``verified=False`` rather
        than raised, because unverified output is a result, not an error.
        """
        # Named distinctly from the audit verdict below: they are both
        # "verdicts" and one silently shadowing the other put the auditor's
        # object on BrokerResult.triage.
        triage_verdict: Triage | None = None
        if task.task_type == AUTO:
            task, triage_verdict = triage_task(
                task,
                self.registry,
                self.providers,
                self.policy,
                use_model=self.triage_use_model,
            )

        decision = route(task, self.registry, self.policy)
        if triage_verdict is not None:
            # Prepend rather than append: how the task was classified is the
            # first thing that has to be true for the rest of the rationale to
            # mean anything, and it names the layer that decided.
            decision = replace(
                decision, rationale=f"{triage_verdict.describe()}; {decision.rationale}"
            )
        spec = decision.chosen
        attempts: list[Attempt] = []
        escalations_used = 0
        failovers_used = 0
        tried: set[str] = set()
        role = "initial"
        first_success: Completion | None = None
        # Findings from the last failed audit, handed to the next attempt.
        # Survives failover as well as escalation: the findings describe what
        # is wrong with the work, not why we changed models.
        feedback: list[str] = []

        while True:
            tried.add(spec.model_id)
            try:
                output = self._complete(spec, task, feedback)
            except ProviderError as e:
                attempts.append(
                    Attempt(
                        model_id=spec.model_id,
                        tier=spec.tier,
                        output_text="",
                        audit_passed=None,
                        audit_score=None,
                        role=role,
                        had_audit_feedback=bool(feedback),
                        error=str(e),
                    )
                )
                # A missing key or bad credentials is a deployment bug. Routing
                # around it would quietly move traffic to a pricier provider
                # and hide the misconfiguration until the invoice arrives.
                fallback = (
                    None
                    if isinstance(e, ProviderConfigError)
                    else self._failover_target(decision, tried)
                )
                if fallback is None or failovers_used >= self.policy.max_provider_failovers:
                    result = self._finalize(task, decision, attempts, "", spec, first_success, triage_verdict)
                    self._trace(task, decision, result)
                    raise ProviderError(
                        f"all routing options exhausted after {len(attempts)} provider "
                        f"failure(s); last error: {e}",
                        provider=spec.provider,
                        model_id=spec.model_id,
                    ) from e
                failovers_used += 1
                spec, role = fallback, "failover"
                continue

            if first_success is None:
                first_success = output

            verdict = self._maybe_audit(task, output, spec)
            attempts.append(
                Attempt(
                    model_id=spec.model_id,
                    tier=spec.tier,
                    output_text=output.text,
                    audit_passed=None if verdict is None else verdict.passed,
                    audit_score=None if verdict is None else verdict.score,
                    audit_issues=[] if verdict is None else list(verdict.issues),
                    role=role,
                    auditor_model=None if verdict is None else verdict.auditor_model,
                    cross_lab_audit=None if verdict is None else verdict.cross_lab,
                    had_audit_feedback=bool(feedback),
                    input_tokens=output.input_tokens,
                    output_tokens=output.output_tokens,
                    cost_usd=actual_cost(spec, output.input_tokens, output.output_tokens),
                    audit_cost_usd=0.0 if verdict is None else verdict.cost_usd,
                )
            )

            passed = verdict is None or verdict.passed
            next_spec = self._escalation_target(spec, task, tried)
            can_escalate = (
                not passed
                and next_spec is not None
                and escalations_used < self.policy.max_escalations
            )
            if not can_escalate:
                result = self._finalize(
                    task, decision, attempts, output.text, spec, first_success, triage_verdict
                )
                self._trace(task, decision, result)
                return result

            # Escalation is a repair, not a blind re-roll: hand the stronger
            # model what the auditor actually objected to.
            feedback = list(verdict.issues) if verdict else []
            escalations_used += 1
            spec, role = next_spec, "escalation"

    # -- internals ---------------------------------------------------------

    def _complete(
        self, spec: ModelSpec, task: Task, feedback: list[str] | None = None
    ) -> Completion:
        provider = self.providers.get(spec.provider)
        return provider.complete(spec.model_id, build_retry_prompt(task.prompt, feedback or []))

    def _maybe_audit(
        self, task: Task, output: Completion, spec: ModelSpec
    ) -> AuditVerdict | None:
        if not self.policy.audit_enabled or len(self.registry) < 2:
            return None
        try:
            return audit(task, output, spec, self.registry, self.providers, self.policy)
        except ProviderError as e:
            # An audit we could not run is an audit that did not pass. Treating
            # an auditor outage as a pass would make verification evaporate
            # exactly when the platform is least healthy.
            return AuditVerdict(
                passed=False,
                score=0.0,
                issues=[f"auditor unavailable ({e}); failing closed"],
            )

    def _failover_target(
        self, decision: RoutingDecision, tried: set[str]
    ) -> ModelSpec | None:
        """Next-best model from the same routing decision, not yet attempted."""
        for scored in decision.ranked:
            if scored.spec.model_id not in tried:
                return scored.spec
        return None

    def _escalation_target(
        self, spec: ModelSpec, task: Task, tried: set[str]
    ) -> ModelSpec | None:
        """Best model in the next tier up, judged on *this task's* type.

        Previously hardcoded to "reasoning", so a failed coding task escalated
        to whichever model reasoned best rather than the one that codes best.
        """
        tier = _ESCALATION_ORDER.get(spec.tier)
        while tier is not None:
            candidates = self.registry.by_tier(tier)
            untried = [m for m in candidates if m.model_id not in tried]
            pool = untried or candidates
            if pool:
                return max(pool, key=lambda m: m.capability_for(task.task_type))
            tier = _ESCALATION_ORDER.get(tier)
        return None

    def _baseline_model(self, task: Task) -> ModelSpec:
        """What "always use the best model" would have meant for this task."""
        return max(
            self.registry.all(),
            key=lambda m: (m.capability_for(task.task_type), TIER_RANK.get(m.tier, 0)),
        )

    def _finalize(
        self,
        task: Task,
        decision: RoutingDecision,
        attempts: list[Attempt],
        final_text: str,
        final_spec: ModelSpec,
        first_success: Completion | None,
        triage_verdict: Triage | None = None,
    ) -> BrokerResult:
        last = attempts[-1] if attempts else None
        baseline_spec = self._baseline_model(task)
        # Price the baseline on tokens we actually observed where possible;
        # fall back to the task's estimates when nothing completed. Token
        # counts differ per model, so this is an approximation — but a far
        # closer one than estimates alone, and it never flatters the router.
        if first_success is not None:
            baseline_cost = actual_cost(
                baseline_spec, first_success.input_tokens, first_success.output_tokens
            )
        else:
            baseline_cost = actual_cost(
                baseline_spec, task.est_input_tokens, task.est_output_tokens
            )
        return BrokerResult(
            final_text=final_text,
            final_model=final_spec.model_id,
            verified=bool(last and last.audit_passed),
            routing_rationale=decision.rationale,
            attempts=attempts,
            underqualified=decision.underqualified,
            warnings=list(decision.warnings),
            triage=triage_verdict,
            generation_cost_usd=sum(a.cost_usd for a in attempts),
            audit_cost_usd=sum(a.audit_cost_usd for a in attempts),
            baseline_cost_usd=baseline_cost,
            baseline_model=baseline_spec.model_id,
        )

    def _trace(self, task: Task, decision: RoutingDecision, result: BrokerResult) -> None:
        """Append a JSONL trace record — the raw material for offline evals."""
        if not self.trace_path:
            return
        record = {
            "ts": time.time(),
            "task_type": task.task_type,
            "complexity": task.complexity,
            # Null unless the caller sent task_type="auto". "source" is the
            # honesty field: heuristic guess vs a model that was paid to look.
            "triage_source": result.triage.source if result.triage else None,
            "triage_confidence": result.triage.confidence if result.triage else None,
            "policy": decision.policy_name,
            "chosen_model": decision.chosen.model_id,
            "final_model": result.final_model,
            "verified": result.verified,
            "escalated": result.escalated,
            "failed_over": result.failed_over,
            "underqualified": result.underqualified,
            # Top-level so a trace query can ask "what share of our verified
            # results were signed off by a model from the same lab?"
            "final_audit_cross_lab": (
                result.attempts[-1].cross_lab_audit if result.attempts else None
            ),
            "gates": list(decision.gates),
            "warnings": list(result.warnings),
            "generation_cost_usd": round(result.generation_cost_usd, 8),
            "audit_cost_usd": round(result.audit_cost_usd, 8),
            "total_cost_usd": round(result.total_cost_usd, 8),
            "baseline_model": result.baseline_model,
            "baseline_cost_usd": round(result.baseline_cost_usd, 8),
            "savings_vs_baseline_usd": round(result.savings_vs_baseline_usd, 8),
            "attempts": [asdict(a) for a in result.attempts],
            "rationale": result.routing_rationale,
        }
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
