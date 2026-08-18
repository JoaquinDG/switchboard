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

For high-stakes work, one auditor is one opinion. `audit_consensus` runs the
identical verdict logic across N independent auditors and aggregates by
strict majority rather than averaging — a `ConsensusVerdict` keeps every
seat's verdict, because a 2-1 split is signal a reviewer needs to see, not
noise to smooth into a single score.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .policies import Policy, Task
from .prompts import AUDIT_PROMPT_TEMPLATE
from .providers.base import Completion, ProviderPool
from .registry import ModelSpec, Registry
from .router import actual_cost

__all__ = [
    "AUDIT_PROMPT_TEMPLATE",
    "AuditVerdict",
    "ConsensusVerdict",
    "audit",
    "audit_consensus",
    "pick_auditor",
    "pick_auditors",
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


@dataclass(frozen=True)
class ConsensusVerdict:
    """N independent auditors' verdicts on the same output, aggregated by
    strict majority — deliberately not an average.

    Averaging a 2-1 split into a single 0.67-ish number reads as a smooth
    pass and throws away exactly the thing a panel exists to surface: that
    the graders disagreed. `unanimous` says whether they agreed at all, and
    `verdicts` keeps every individual verdict rather than collapsing them, so
    a trace reader (or `Broker`, via `Attempt.consensus_verdicts`) can always
    see who passed, who failed, and why.

    A tie (even panel size, half and half) fails closed: `passed` requires
    more than half the panel, not "at least half".
    """

    passed: bool
    verdicts: list[AuditVerdict] = field(default_factory=list)
    unanimous: bool = True
    # How many auditors the policy asked for. May exceed len(verdicts) when
    # the catalog does not have that many independent models — see
    # pick_auditors. Recorded so a shortfall is visible, not silently eaten.
    requested_count: int = 0

    @property
    def pass_count(self) -> int:
        return sum(1 for v in self.verdicts if v.passed)

    @property
    def fail_count(self) -> int:
        return len(self.verdicts) - self.pass_count

    @property
    def score(self) -> float:
        """Mean score, reported alongside — never instead of — the vote."""
        if not self.verdicts:
            return 0.0
        return sum(v.score for v in self.verdicts) / len(self.verdicts)

    @property
    def cost_usd(self) -> float:
        return sum(v.cost_usd for v in self.verdicts)

    @property
    def auditor_model(self) -> str:
        return "+".join(v.auditor_model for v in self.verdicts)

    @property
    def cross_lab(self) -> bool:
        return all(v.cross_lab for v in self.verdicts) if self.verdicts else False

    @property
    def synthetic(self) -> bool:
        return any(v.synthetic for v in self.verdicts)

    @property
    def issues(self) -> list[str]:
        """The disagreement itself, then every auditor's own issues, attributed."""
        merged: list[str] = []
        if len(self.verdicts) < self.requested_count:
            merged.append(
                f"panel requested {self.requested_count} auditors but the "
                f"catalog only supports {len(self.verdicts)} independent model(s)"
            )
        if not self.unanimous:
            votes = ", ".join(
                f"{v.auditor_model}={'pass' if v.passed else 'fail'}" for v in self.verdicts
            )
            merged.append(
                f"audit panel split {self.pass_count}-{self.fail_count} ({votes})"
            )
        for v in self.verdicts:
            for issue in v.issues:
                merged.append(f"[{v.auditor_model}] {issue}")
        return merged

    def describe(self) -> str:
        vote = "unanimous" if self.unanimous else "SPLIT"
        return (
            f"{self.pass_count}/{len(self.verdicts)} auditors passed ({vote}): "
            + "; ".join(
                f"{v.auditor_model}={'pass' if v.passed else 'fail'} ({v.score:.2f})"
                for v in self.verdicts
            )
        )


def _best_auditor(candidates: list[ModelSpec], policy: Policy | None) -> ModelSpec:
    """Rank a candidate pool by the policy's `auditor_selection` strategy.

    Shared by `pick_auditor` and `pick_auditors` so a panel is chosen by
    exactly the same rule as a single auditor — a panel differs from one
    auditor only in count, never in how each seat is picked.
    """
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

    return _best_auditor(candidates, policy)


def pick_auditors(
    registry: Registry, producer: ModelSpec, policy: Policy | None, count: int
) -> list[ModelSpec]:
    """Up to `count` distinct, independent auditors for a consensus panel.

    Independence compounds here in the same two degrees as `pick_auditor`,
    plus a third: a panel of auditors is only stronger evidence than one
    auditor if its seats are not the same model — or the same lab — asked
    twice under different names. Distinct providers are exhausted before any
    provider repeats, so a 3-seat panel on a 2-lab catalog spreads 1+1 across
    labs and only fills the third seat from a lab already used, rather than
    silently shrinking to 2.

    Returns fewer than `count` models when the registry does not have that
    many candidates excluding the producer — never raises for a shortfall,
    since a smaller-but-real panel is more honest than refusing to audit at
    all. `audit_consensus` records the shortfall on the verdict.
    """
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    candidates = [m for m in registry.all() if m.model_id != producer.model_id]
    if not candidates:
        raise ValueError("need at least two models in the registry to audit")

    if policy is None or policy.prefer_cross_lab_auditor:
        cross_lab = [m for m in candidates if m.provider != producer.provider]
        if cross_lab:
            candidates = cross_lab

    chosen: list[ModelSpec] = []
    used_ids: set[str] = set()
    used_providers: set[str] = set()
    for _ in range(count):
        remaining = [m for m in candidates if m.model_id not in used_ids]
        if not remaining:
            break
        fresh_provider = [m for m in remaining if m.provider not in used_providers]
        pick = _best_auditor(fresh_provider or remaining, policy)
        chosen.append(pick)
        used_ids.add(pick.model_id)
        used_providers.add(pick.provider)
    return chosen


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


def _run_audit(
    task: Task,
    output: Completion,
    producer: ModelSpec,
    auditor_spec: ModelSpec,
    providers: ProviderPool,
    policy: Policy,
) -> AuditVerdict:
    """Have one specific model grade the producer's output.

    The half of `audit()` that does not include *choosing* the auditor,
    factored out so a consensus panel judges each seat by the identical rule
    a single audit uses — the only difference between one auditor and N is
    how many times this runs.
    """
    provider = providers.get(auditor_spec.provider)
    prompt = AUDIT_PROMPT_TEMPLATE.format(
        task_type=task.task_type,
        prompt=task.prompt,
        output=output.text,
    )
    result = provider.complete(
        auditor_spec.model_id, prompt, max_tokens=policy.max_output_tokens
    )
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
    return _run_audit(task, output, producer, auditor_spec, providers, policy)


def audit_consensus(
    task: Task,
    output: Completion,
    producer: ModelSpec,
    registry: Registry,
    providers: ProviderPool,
    policy: Policy,
    count: int,
) -> ConsensusVerdict:
    """Have `count` independent auditors grade the same output; majority wins.

    Each seat is judged by the exact same `_run_audit` a single-auditor call
    uses — a panel is not a different kind of verification, just more of it.
    The result is a `ConsensusVerdict`, which keeps every individual verdict
    rather than averaging them: see its docstring for why that matters.
    """
    auditors = pick_auditors(registry, producer, policy, count)
    if not auditors:
        raise ValueError("need at least two models in the registry to audit")
    verdicts = [
        _run_audit(task, output, producer, a, providers, policy) for a in auditors
    ]
    pass_count = sum(1 for v in verdicts if v.passed)
    return ConsensusVerdict(
        passed=pass_count * 2 > len(verdicts),  # strict majority; a tie fails closed
        verdicts=verdicts,
        unanimous=pass_count in (0, len(verdicts)),
        requested_count=count,
    )
