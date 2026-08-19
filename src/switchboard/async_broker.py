"""AsyncBroker: the same route -> run -> audit -> escalate pipeline as
Broker, driven by asyncio so a caller running several tasks at once (an
agentic loop dispatching N independent requests, say) can have their
provider and auditor calls actually overlap instead of blocking one after
another on a single thread.

Nothing about *what* gets decided differs from the sync Broker. Routing
(`route`), escalation targeting, failover targeting, baseline pricing, and
result/trace construction are the exact same functions Broker itself calls
— imported from `broker.py`, not reimplemented here. The only thing that
changes is how the provider and auditor calls are made: `await`ed against an
`AsyncProviderPool` instead of blocking a thread against a `ProviderPool`.
That is the one part that actually benefits from being async, so it is the
only part with two versions.

Scope: `run()` only, not `run_plan()`. A plan's payoff from async is running
its *independent* steps concurrently, which needs a dependency-aware
scheduler of its own — layering that on top of `run()` is the natural next
step, not this one. Half-porting `run_plan`'s ~120 lines of context-threading
and skip-on-failure logic here, still serially, would be exactly the forked
and half-maintained second path this item exists to avoid; better to ship a
working `run()` than a partial `run_plan()`.

Model-based triage (`Broker(triage_use_model=True)`) is also out of scope
for the same reason: `classify_with_model` makes a blocking `Provider.complete`
call, which needs an async counterpart of its own before it can be offered
here honestly. `AsyncBroker.run()` always resolves `task_type="auto"` with
the offline heuristic.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .auditor import AuditVerdict, audit_async
from .broker import (
    Attempt,
    BrokerResult,
    _build_broker_result,
    _is_provider_synthetic,
    _pick_escalation_target,
    _pick_failover_target,
    _should_audit,
    _write_trace_record,
)
from .policies import Policy, Task
from .prompts import build_retry_prompt
from .providers.base import (
    AsyncProviderPool,
    Completion,
    ProviderConfigError,
    ProviderError,
)
from .registry import ModelSpec, Registry
from .router import actual_cost, route
from .triage import AUTO, Triage, triage_task


class AsyncBroker:
    """Async counterpart to Broker. See the module docstring for scope."""

    def __init__(
        self,
        registry: Registry,
        providers: AsyncProviderPool,
        policy: Policy,
        trace_path: str | Path | None = None,
    ) -> None:
        self.registry = registry
        self.providers = providers
        self.policy = policy
        self.trace_path = Path(trace_path) if trace_path else None

    async def run(self, task: Task) -> BrokerResult:
        """Route, execute, verify, and escalate or reroute as needed.

        Same contract as `Broker.run`: raises ProviderError only once every
        routing option is exhausted, and a failed audit is a result
        (`verified=False`), not an exception.
        """
        triage_verdict: Triage | None = None
        if task.task_type == AUTO:
            task, triage_verdict = triage_task(task, use_model=False)

        decision = route(task, self.registry, self.policy)
        if triage_verdict is not None:
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
        feedback: list[str] = []

        while True:
            tried.add(spec.model_id)
            try:
                output = await self._complete(spec, task, feedback)
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
                        synthetic=_is_provider_synthetic(self.providers, spec.provider),
                    )
                )
                fallback = (
                    None
                    if isinstance(e, ProviderConfigError)
                    else _pick_failover_target(decision, tried)
                )
                if fallback is None or failovers_used >= self.policy.max_provider_failovers:
                    result = _build_broker_result(
                        self.registry, task, decision, attempts, "", spec,
                        first_success, triage_verdict,
                    )
                    _write_trace_record(self.trace_path, task, decision, result)
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

            verdict = await self._maybe_audit(task, output, spec)
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
                    synthetic=_is_provider_synthetic(self.providers, spec.provider)
                    or bool(verdict is not None and verdict.synthetic),
                )
            )

            passed = verdict is None or verdict.passed
            next_spec = _pick_escalation_target(self.registry, self.policy, spec, task, tried)
            can_escalate = (
                not passed
                and next_spec is not None
                and escalations_used < self.policy.max_escalations
            )
            if not can_escalate:
                result = _build_broker_result(
                    self.registry, task, decision, attempts, output.text, spec,
                    first_success, triage_verdict,
                )
                _write_trace_record(self.trace_path, task, decision, result)
                return result

            feedback = list(verdict.issues) if verdict else []
            escalations_used += 1
            spec, role = next_spec, "escalation"

    async def _complete(
        self, spec: ModelSpec, task: Task, feedback: list[str] | None = None
    ) -> Completion:
        provider = self.providers.get(spec.provider)
        return await provider.complete(
            spec.model_id,
            build_retry_prompt(task.prompt, feedback or []),
            max_tokens=self.policy.max_output_tokens,
        )

    async def _maybe_audit(
        self, task: Task, output: Completion, spec: ModelSpec
    ) -> AuditVerdict | None:
        if not _should_audit(self.policy, self.registry):
            return None
        try:
            return await audit_async(
                task, output, spec, self.registry, self.providers, self.policy
            )
        except ProviderError as e:
            return AuditVerdict(
                passed=False,
                score=0.0,
                issues=[f"auditor unavailable ({e}); failing closed"],
            )
