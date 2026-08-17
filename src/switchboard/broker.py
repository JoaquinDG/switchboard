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
from .router import RoutingDecision, actual_cost, fits_context, route, score_models
from .planner import (
    Plan,
    plan_request,
    validate_plan,
    PlanStep,
    PlanValidationError,
    no_split_plan,
    plan_heuristic,
    topological_order,
)
from .triage import AUTO, Triage, triage_task

_ESCALATION_ORDER = {"small": "mid", "mid": "frontier", "frontier": None}

# How a dependent step is handed its predecessor's output. Delimited rather
# than concatenated so the receiving model can tell the brief from the input.
_CONTEXT_TEMPLATE = """{prompt}

--- OUTPUT OF STEP {step_id} (use this as your input) ---
{context}
--- END OF STEP {step_id} OUTPUT ---"""


@dataclass
class StepResult:
    """One plan step, its routing outcome, and what was threaded into it."""

    step_id: str
    step: PlanStep
    result: BrokerResult
    # Characters of upstream output injected, and whether the cap bit. A
    # truncated context is named rather than swallowed: it is the most likely
    # cause of a confidently wrong later step.
    injected_chars: int = 0
    injected_truncated: bool = False
    injected_from: tuple[str, ...] = ()


@dataclass
class PlanResult:
    """An executed plan: the output, the bill, and what it is compared against."""

    plan: Plan
    steps: list[StepResult] = field(default_factory=list)
    # The LAST step's output. Correct for a transform chain, where each step
    # rewrites its predecessor and the final rewrite is the answer.
    final_text: str = ""
    # Every step's output, labelled. Correct for a request that asks for
    # several deliverables — "extract X, summarise Y, recommend Z" is answered
    # by all three, and the last step alone is a third of the answer.
    #
    # Found live: the plan-level audit read final_text against the original
    # request and reported "skips the first requested step entirely". It was
    # right. The extraction had happened; it just was not in what got audited.
    assembled_text: str = ""
    # Cost of what actually ran, audits and escalations included.
    routed_cost_usd: float = 0.0
    # Every step priced on the strongest model qualified for its own type.
    baseline_best_model_usd: float = 0.0
    # The whole request as ONE call to the strongest model, at the request's
    # own token estimate. This is the "what you would have done without
    # Switchboard" number and it is MODELLED, never run — see
    # `baseline_single_call_is_modelled`.
    baseline_single_call_usd: float = 0.0
    baseline_single_call_model: str = ""
    baseline_single_call_is_modelled: bool = True
    # Model plans that failed validation. They were generated and billed, so
    # they are reported rather than quietly absorbed.
    discarded_attempts: list[dict] = field(default_factory=list)
    # Set only when policy.plan_final_audit is on: one audit of the assembled
    # answer against the original request.
    final_audit: AuditVerdict | None = None
    # (step_id, reason) for steps never dispatched because something they
    # depend on failed its audit. Reported rather than silently missing.
    skipped_steps: list[tuple[str, str]] = field(default_factory=list)

    @property
    def verified(self) -> bool:
        """True only if every step passed its own audit.

        Strict and step-wise. Without `plan_final_audit` this says only that
        no individual step failed — every step can pass on its own and the
        assembled answer still not address the original request. With it on,
        the coherence check must pass too.
        """
        if self.skipped_steps:
            return False  # part of the plan never ran
        if not self.steps or not all(s.result.verified for s in self.steps):
            return False
        if self.final_audit is not None:
            return self.final_audit.passed
        return True

    @property
    def is_split(self) -> bool:
        return self.plan.is_split

    @property
    def saved_vs_single_call_usd(self) -> float:
        """Against the modelled single-call baseline. Modelled, not measured."""
        return self.baseline_single_call_usd - self.routed_cost_usd


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
    # Vendor stop reason, and whether it means the output was cut off at the
    # token ceiling. Truncation is a mechanical failure: escalating to a
    # stronger model does not fix it, and reasoning models truncate *more*
    # because thinking tokens come out of the same budget.
    stop_reason: str = ""
    truncated: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    audit_cost_usd: float = 0.0
    # Set when the provider call itself failed; output_text is empty.
    error: str | None = None
    # True if the producer or the auditor (whichever is relevant) was a canned
    # stand-in rather than a real vendor call. evals/catalog_feedback.py must
    # skip these when measuring capability from traces — a MockProvider audit
    # is not evidence about the model it "graded".
    synthetic: bool = False

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
    def truncated(self) -> bool:
        """True if the final attempt was cut off at the token ceiling.

        Distinct from `verified`: a truncated answer usually fails its audit,
        but the fix is a bigger `max_tokens`, not a bigger model.
        """
        return bool(self.attempts and self.attempts[-1].truncated)

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
        plan_use_model: bool = False,
    ) -> None:
        self.registry = registry
        self.providers = providers
        self.policy = policy
        self.trace_path = Path(trace_path) if trace_path else None
        # Opt-in: spend a cheap model call to classify task_type="auto" tasks
        # instead of using the offline heuristic. Off by default so the
        # library keeps working with no providers and no budget.
        self.triage_use_model = triage_use_model
        # Opt-in: let a cheap model propose the decomposition when the
        # heuristic reports it cannot judge. Off by default, like triage's.
        self.plan_use_model = plan_use_model

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
                        synthetic=self._provider_synthetic(spec.provider),
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
                    stop_reason=output.stop_reason,
                    truncated=output.truncated,
                    input_tokens=output.input_tokens,
                    output_tokens=output.output_tokens,
                    cost_usd=actual_cost(spec, output.input_tokens, output.output_tokens),
                    audit_cost_usd=0.0 if verdict is None else verdict.cost_usd,
                    synthetic=self._provider_synthetic(spec.provider)
                    or bool(verdict is not None and verdict.synthetic),
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

    def run_plan(self, request: str | Plan) -> PlanResult:
        """Decompose a compound request, run each step, and report the bill.

        A caller-supplied Plan is validated identically to a generated one and
        then treated as authoritative — the same courtesy the broker extends to
        a caller-supplied `task_type`.

        Every step is dispatched through `run()` unchanged. Nothing here
        bypasses triage, the gates, auditing, escalation or failover; the
        planner only decides what the units of work are.
        """
        plan, discarded = self._resolve_plan(request)
        for attempt in discarded:
            # A rejected plan still cost money. Trace it before anything else
            # so the bill and the reason are both recoverable.
            self._trace_event("attempt_discarded", dict(attempt))
        self._trace_event("plan_proposed", {
            "request": plan.request,
            "planned_by": plan.planned_by,
            "confidence": plan.confidence,
            "rationale": plan.rationale,
            "signals": list(plan.signals),
            "is_split": plan.is_split,
            "steps": [asdict(step) for step in plan.steps],
        })

        outputs: dict[str, str] = {}
        results: list[StepResult] = []
        skipped: list[tuple[str, str]] = []
        poisoned: set[str] = set()
        cap = self.policy.plan_context_cap_chars

        for step in topological_order(plan):
            upstream_failed = sorted(poisoned.intersection(step.depends_on))
            if upstream_failed and self.policy.plan_halt_dependents_on_failure:
                # Its input is output already judged wrong, and it would
                # consume that as ground truth. Skipping is not giving up on
                # the plan — independent steps still run.
                reason = (f"depends on {', '.join(upstream_failed)}, which did not "
                          f"verify; its output would be built on rejected input")
                skipped.append((step.step_id, reason))
                poisoned.add(step.step_id)
                self._trace_event("step_skipped", {
                    "step_id": step.step_id, "reason": reason,
                    "depends_on_failed": upstream_failed,
                })
                continue

            prompt, injected, truncated, sources = self._thread_context(step, outputs, cap)
            # Injected context is real input the model pays for, so the routing
            # estimate is re-derived at dispatch instead of trusting the
            # planner's guess from before anything had run.
            task = Task(
                prompt=prompt,
                task_type=step.task_type,
                complexity=step.complexity,
                est_input_tokens=step.est_input_tokens + injected // 4,
                est_output_tokens=step.est_output_tokens,
            )
            self._trace_event("step_dispatched", {
                "step_id": step.step_id,
                "task_type": step.task_type,
                "complexity": step.complexity,
                "est_input_tokens": task.est_input_tokens,
                "injected_chars": injected,
                "injected_truncated": truncated,
                "injected_from": list(sources),
            })

            result = self.run(task)
            outputs[step.step_id] = result.final_text
            if not result.verified:
                poisoned.add(step.step_id)
            results.append(StepResult(
                step_id=step.step_id, step=step, result=result,
                injected_chars=injected, injected_truncated=truncated,
                injected_from=sources,
            ))
            self._trace_event("step_completed", {
                "step_id": step.step_id,
                "final_model": result.final_model,
                "verified": result.verified,
                "escalated": result.escalated,
                "truncated": result.truncated,
                "generation_cost_usd": round(result.generation_cost_usd, 8),
                "audit_cost_usd": round(result.audit_cost_usd, 8),
                "total_cost_usd": round(result.total_cost_usd, 8),
                "output_text": result.final_text,
                "attempts": len(result.attempts),
            })

        plan_result = self._finalize_plan(plan, results)
        plan_result.discarded_attempts = list(discarded)
        plan_result.skipped_steps = skipped
        if self.policy.plan_final_audit and results and not skipped:
            plan_result.final_audit = self._audit_plan(plan, plan_result)
            plan_result.routed_cost_usd += plan_result.final_audit.cost_usd
            self._trace_event("plan_audited", {
                "passed": plan_result.final_audit.passed,
                "score": plan_result.final_audit.score,
                "issues": list(plan_result.final_audit.issues),
                "auditor_model": plan_result.final_audit.auditor_model,
                "cross_lab": plan_result.final_audit.cross_lab,
                "cost_usd": round(plan_result.final_audit.cost_usd, 8),
            })
        self._trace_event("plan_completed", {
            "verified": plan_result.verified,
            "is_split": plan_result.is_split,
            "steps": len(plan_result.steps),
            "skipped_steps": [list(pair) for pair in plan_result.skipped_steps],
            "routed_cost_usd": round(plan_result.routed_cost_usd, 8),
            "baseline_best_model_usd": round(plan_result.baseline_best_model_usd, 8),
            "baseline_single_call_usd": round(plan_result.baseline_single_call_usd, 8),
            "baseline_single_call_model": plan_result.baseline_single_call_model,
            "baseline_single_call_is_modelled": True,
            "final_text": plan_result.final_text,
            "assembled_text": plan_result.assembled_text,
        })
        return plan_result

    def _resolve_plan(self, request: str | Plan) -> tuple[Plan, list[dict]]:
        """Validate a supplied plan, or build one; fail closed to single-task."""
        if isinstance(request, Plan):
            validate_plan(request)  # supplied plans get no special treatment
            return request, []
        try:
            return plan_request(
                request, self.registry, self.providers, self.policy,
                use_model=self.plan_use_model,
            )
        except PlanValidationError as e:
            # A planner that cannot produce a usable plan must not take the
            # request down with it. Degrade to routing the whole thing as one
            # task and say so, rather than failing the caller's request.
            self._trace_event("plan_degraded", {"reason": str(e), "request": request})
            return no_split_plan(request, f"planner failed ({e}); routed as one task"), []

    def _thread_context(
        self, step: PlanStep, outputs: dict[str, str], cap: int
    ) -> tuple[str, int, bool, tuple[str, ...]]:
        """Append upstream outputs to a step's prompt, under a labelled fence."""
        if not step.depends_on:
            return step.prompt, 0, False, ()
        prompt = step.prompt
        injected = 0
        truncated = False
        used: list[str] = []
        for dep in step.depends_on:
            context = outputs.get(dep, "")
            if not context:
                continue
            if cap and len(context) > cap:
                context = context[:cap]
                truncated = True
            prompt = _CONTEXT_TEMPLATE.format(
                prompt=prompt, step_id=dep, context=context
            )
            injected += len(context)
            used.append(dep)
        return prompt, injected, truncated, tuple(used)

    def _finalize_plan(self, plan: Plan, results: list[StepResult]) -> PlanResult:
        """Assemble the result and price the two baselines."""
        routed = sum(r.result.total_cost_usd for r in results)
        best_model = sum(r.result.baseline_cost_usd for r in results)

        # The single-call baseline: the untouched request on the strongest
        # model for its own inferred type, at its own token estimate. Modelled
        # rather than run — nobody is charged to produce a comparison number.
        whole = no_split_plan(plan.request, "baseline").steps[0]
        strongest = max(
            self.registry.all(),
            key=lambda m: (m.capability_for(whole.task_type), TIER_RANK.get(m.tier, 0)),
        )
        single = actual_cost(
            strongest,
            whole.est_input_tokens + sum(s.step.est_input_tokens for s in results),
            whole.est_output_tokens + sum(s.step.est_output_tokens for s in results),
        )
        assembled = "\n\n".join(
            f"[{r.step_id} · {r.step.task_type}]\n{r.result.final_text.strip()}"
            for r in results
        )
        return PlanResult(
            plan=plan,
            steps=results,
            final_text=results[-1].result.final_text if results else "",
            assembled_text=assembled,
            routed_cost_usd=routed,
            baseline_best_model_usd=best_model,
            baseline_single_call_usd=single,
            baseline_single_call_model=strongest.model_id,
        )

    def _audit_plan(self, plan: Plan, result: PlanResult) -> AuditVerdict:
        """One audit of the assembled answer against the ORIGINAL request.

        Per-step audits each judge a fragment against its own fragment of the
        brief. None of them can see whether the assembled answer actually
        addresses what was asked — that is a different question, and it needs
        the original request in front of it.
        """
        producer = self.registry.get(result.steps[-1].result.final_model)
        # The ASSEMBLED answer, not just the last step: the original request
        # may have asked for several things, and auditing one third of the
        # answer against all of it reports failures that did not happen.
        assembled = Completion(
            text=result.assembled_text or result.final_text,
            model_id=producer.model_id,
            input_tokens=0,
            output_tokens=0,
        )
        try:
            return audit(
                Task(prompt=plan.request, task_type=plan.steps[-1].task_type,
                     complexity=max(s.complexity for s in plan.steps)),
                assembled, producer, self.registry, self.providers, self.policy,
            )
        except ProviderError as e:
            # Same rule as a per-step audit: verification that cannot run has
            # not passed.
            return AuditVerdict(
                passed=False, score=0.0,
                issues=[f"plan-level auditor unavailable ({e}); failing closed"],
            )

    def _trace_event(self, event: str, payload: dict) -> None:
        """Append a named plan event to the same JSONL stream.

        New records carry an "event" key; the existing per-task summary records
        do not, and are not modified. A reader treats a missing "event" as the
        legacy task record. Purely additive, so committed traces stay readable.
        """
        if not self.trace_path:
            return
        record = {"ts": time.time(), "event": event, **payload}
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    # -- internals ---------------------------------------------------------

    def _complete(
        self, spec: ModelSpec, task: Task, feedback: list[str] | None = None
    ) -> Completion:
        provider = self.providers.get(spec.provider)
        return provider.complete(
            spec.model_id,
            build_retry_prompt(task.prompt, feedback or []),
            max_tokens=self.policy.max_output_tokens,
        )

    def _provider_synthetic(self, provider_name: str) -> bool:
        """Whether the named provider is a canned stand-in, not a real vendor.

        Looked up defensively: a provider that failed to resolve tells us
        nothing about syntheticity, and this must never raise on top of an
        already-failed attempt.
        """
        try:
            provider = self.providers.get(provider_name)
        except KeyError:
            return False
        return getattr(provider, "synthetic", False)

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
        """Best model in the next tier up, scored under the same policy.

        Three corrections live here. It was once hardcoded to "reasoning", so
        a failed coding task escalated to whichever model reasoned best. It
        then maximised capability for the task type — which quietly ignored
        the policy, so a cost-first run could fail an audit and jump to the
        priciest model in the catalog. Moving up a tier is already the
        quality step; *which* model in that tier is still a cost/quality
        tradeoff, and the policy is what decides tradeoffs. Third: a tier can
        contain models that do not fit this task's context — escalating to
        one would trade an audit failure for a hard provider error, so the
        context gate applies here exactly as it does in `route()`.
        """
        tier = _ESCALATION_ORDER.get(spec.tier)
        while tier is not None:
            candidates = [m for m in self.registry.by_tier(tier) if fits_context(task, m)]
            untried = [m for m in candidates if m.model_id not in tried]
            pool = untried or candidates
            if pool:
                return score_models(task, pool, self.registry, self.policy)[0].spec
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
            "truncated": result.truncated,
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
