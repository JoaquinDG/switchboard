"""Cross-model auditing.

The core idea: a second model, chosen for audit capability and *never the
model that produced the output*, grades the output against the task and
returns a structured verdict. Self-grading is disallowed by construction
because models are systematically charitable to their own outputs.

The verdict is deliberately simple (pass/score/issues). Rich rubrics belong
in the audit prompt, not the schema — schemas ossify, prompts iterate.

Parsing is tolerant of *format* (fences, surrounding prose, stray keys) and
strict about *meaning*: anything it cannot read as a verdict is a failure.
Verification that defaults to "pass" on errors is not verification.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .policies import Policy, Task
from .prompts import AUDIT_PROMPT_TEMPLATE
from .providers.base import AsyncProviderPool, Completion, ProviderPool
from .registry import ModelSpec, Registry
from .router import actual_cost

__all__ = [
    "AUDIT_PROMPT_TEMPLATE",
    "AuditVerdict",
    "audit",
    "audit_async",
    "pick_auditor",
]

_FENCE_RE = re.compile(r"^```[A-Za-z0-9_-]*\s*\n?(?P<body>.*?)\n?\s*```$", re.DOTALL)


@dataclass(frozen=True)
class AuditVerdict:
    """One model's graded judgement of another model's output."""

    passed: bool
    score: float
    issues: list[str] = field(default_factory=list)
    auditor_model: str = ""
    # False when producer and auditor come from the same provider. The audit
    # still happened; it is just weaker evidence, and a pass rate built from
    # same-lab audits should be read with that in mind.
    cross_lab: bool = True
    raw: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    # True when the auditor that produced this verdict was a canned stand-in
    # (MockProvider, ScriptedProvider) rather than a real vendor call. A
    # verdict from a mock auditor is not evidence about the producer model;
    # traces carry this so offline demo runs can never be scored as measured.
    synthetic: bool = False


def pick_auditor(
    registry: Registry, producer: ModelSpec, policy: Policy | None = None
) -> ModelSpec:
    """Audit-capable model that is not the producer.

    Independence is non-negotiable and enforced here by construction, in two
    degrees. A different *model* is required: self-grading inflates pass rates.
    A different *lab* is preferred: two models from the same provider share
    training data and alignment, so their blind spots correlate and a pass
    means less than the number suggests. Where the catalog has only one lab
    the audit still runs, and the verdict admits `cross_lab=False`.

    Which of the surviving models to use is a policy question: `most_capable`
    (default) buys the strictest available grader, while `cheapest_qualified`
    buys the cheapest one clearing `min_auditor_capability` — which matters
    because on a cheap task the audit can otherwise cost more than the work it
    grades. Both run *within* the cross-lab pool: independence is worth more
    than either capability or price.
    """
    candidates = [m for m in registry.all() if m.model_id != producer.model_id]
    if not candidates:
        raise ValueError("need at least two models in the registry to audit")

    if policy is None or policy.prefer_cross_lab_auditor:
        cross_lab = [m for m in candidates if m.provider != producer.provider]
        if cross_lab:
            candidates = cross_lab

    if policy is not None and policy.auditor_selection == "cheapest_qualified":
        qualified = [
            m
            for m in candidates
            if m.capability_for("audit") >= policy.min_auditor_capability
        ]
        if qualified:
            # Rank on output price: audit responses are short prompts in, a
            # small JSON verdict out, but output tokens still dominate rates.
            return min(qualified, key=lambda m: (m.output_cost, m.input_cost))
        # Nothing clears the floor — fall through to the strongest available
        # rather than auditing with a model the policy calls unqualified.

    return max(candidates, key=lambda m: m.capability_for("audit"))


def _strip_fences(text: str) -> str:
    """Remove a surrounding markdown code fence, language tag included."""
    stripped = text.strip()
    match = _FENCE_RE.match(stripped)
    return match.group("body").strip() if match else stripped


def _first_json_object(text: str) -> str | None:
    """Extract the first balanced {...} block, ignoring braces inside strings.

    Models routinely wrap the verdict in a sentence of preamble. Rejecting
    those outright fails closed on a formatting quirk rather than on quality,
    which throws away good work and inflates escalation spend.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _parse_verdict(text: str) -> tuple[bool, float, list[str]]:
    """Parse a verdict, tolerating fences and prose; unparseable -> fail closed."""
    if not isinstance(text, str) or not text.strip():
        return False, 0.0, ["auditor returned an empty verdict; failing closed"]

    candidate = _strip_fences(text)
    data: object = None
    for attempt in (candidate, _first_json_object(candidate)):
        if not attempt:
            continue
        try:
            data = json.loads(attempt)
            break
        except json.JSONDecodeError:
            data = None

    if not isinstance(data, dict):
        return False, 0.0, ["auditor returned unparseable verdict; failing closed"]

    raw_pass = data.get("pass", False)
    if isinstance(raw_pass, str):  # tolerate "true"/"false"
        raw_pass = raw_pass.strip().lower() == "true"
    passed = bool(raw_pass)

    try:
        score = float(data.get("score", 0.0))
    except (TypeError, ValueError):
        return False, 0.0, ["auditor returned a non-numeric score; failing closed"]
    if score != score or score in (float("inf"), float("-inf")):  # NaN / inf
        return False, 0.0, ["auditor returned a non-finite score; failing closed"]
    if not 0.0 <= score <= 1.0:
        # Out of range means the auditor ignored the schema. Clamping and
        # carrying on would launder a broken response into a pass.
        return False, 0.0, [f"auditor score {score} outside [0, 1]; failing closed"]

    raw_issues = data.get("issues", [])
    if isinstance(raw_issues, str):
        issues = [raw_issues]
    elif isinstance(raw_issues, (list, tuple)):
        issues = [str(i) for i in raw_issues]
    else:
        issues = [f"auditor returned malformed issues field: {raw_issues!r}"]

    return passed, score, issues


def _audit_prompt(task: Task, output: Completion) -> str:
    return AUDIT_PROMPT_TEMPLATE.format(
        task_type=task.task_type,
        prompt=task.prompt,
        output=output.text,
    )


def _build_verdict(
    output: Completion,
    producer: ModelSpec,
    auditor_spec: ModelSpec,
    provider: object,
    result: Completion,
    policy: Policy,
) -> AuditVerdict:
    """Turn a raw auditor completion into a verdict.

    Shared by `audit` and `audit_async` so the parsing, truncation note, and
    threshold check — the actual judgement logic — can't drift between the
    sync and async call paths. Only *getting* `result` differs between them.
    """
    passed, score, issues = _parse_verdict(result.text)

    if output.truncated:
        # The auditor sees a cut-off answer and quite reasonably calls it
        # incomplete. Without this line the trace reads as a quality failure
        # and the obvious response is "escalate", which buys a stronger model
        # that will truncate at the same ceiling — and reasoning models
        # truncate sooner, because thinking spends the same budget.
        issues = [
            f"output was TRUNCATED by the provider (stop_reason="
            f"{output.stop_reason!r}); raise max_tokens rather than escalating"
        ] + list(issues)

    if passed and score < policy.audit_pass_threshold:
        # The auditor said pass, the policy disagrees. Record why, so a trace
        # reader is not left wondering where the failure came from.
        issues = list(issues) + [
            f"auditor passed with score {score:.2f}, below policy threshold "
            f"{policy.audit_pass_threshold:.2f}"
        ]
        passed = False

    return AuditVerdict(
        passed=passed,
        score=score,
        issues=issues,
        auditor_model=auditor_spec.model_id,
        cross_lab=auditor_spec.provider != producer.provider,
        raw=result.text,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=actual_cost(auditor_spec, result.input_tokens, result.output_tokens),
        synthetic=getattr(provider, "synthetic", False),
    )


def audit(
    task: Task,
    output: Completion,
    producer: ModelSpec,
    registry: Registry,
    providers: ProviderPool,
    policy: Policy,
) -> AuditVerdict:
    """Have a different model grade the producer's output."""
    auditor_spec = pick_auditor(registry, producer, policy)
    provider = providers.get(auditor_spec.provider)
    result = provider.complete(
        auditor_spec.model_id, _audit_prompt(task, output), max_tokens=policy.max_output_tokens
    )
    return _build_verdict(output, producer, auditor_spec, provider, result, policy)


async def audit_async(
    task: Task,
    output: Completion,
    producer: ModelSpec,
    registry: Registry,
    providers: AsyncProviderPool,
    policy: Policy,
) -> AuditVerdict:
    """Async counterpart to `audit`: same selection and verdict logic, awaited."""
    auditor_spec = pick_auditor(registry, producer, policy)
    provider = providers.get(auditor_spec.provider)
    result = await provider.complete(
        auditor_spec.model_id, _audit_prompt(task, output), max_tokens=policy.max_output_tokens
    )
    return _build_verdict(output, producer, auditor_spec, provider, result, policy)
