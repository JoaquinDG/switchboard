"""Compound-request decomposition: one messy sentence into routable steps.

Switchboard routes one task to one model. Real requests are compound — "pull
the competitor pricing, summarise it, then draft the new page" — and a
compound request routed whole must route at its *hardest* sub-task, so the
extraction runs at frontier prices along with everything else. Measured on the
starter catalog that is 2.9x ($0.1600 as one task, $0.0545 as four). The
savings the router exists to produce therefore depend on a decomposition the
library did not perform; `examples/agentic_workflow.py` hand-writes its steps.

**The planner proposes, the engine disposes.** Decomposition is a routing
input, not a showcase. A planner that shreds a simple request into expensive
confetti destroys the economics the router protects, so the bias runs hard
*against* splitting: a request must show explicit multi-step structure to be
split, and ambiguity resolves to no-split.

Architecture mirrors `triage.py`, deliberately:

1. **Heuristic** (default, offline, deterministic). Structural signals only —
   enumeration and sequence connectives. Never length, never comma chains:
   "extract names, emails and phone numbers" is one extraction task, and a
   long prompt is not a compound one.
2. **Model** (optional, gated). Consulted only when the heuristic's confidence
   falls below the policy threshold, on the cheapest qualified model. Its
   output passes the same strict validation, gets exactly one repair attempt,
   and on any failure falls back to the heuristic's answer.

The honesty rule carries over verbatim: `planned_by` names the layer that
actually decided. A heuristic plan is never dressed up as a model's, and a
model call that failed is credited to the heuristic that rescued it.

The confidence signal is doing real work here. A request with several distinct
imperative verbs but no explicit structure — "read this contract, extract the
payment terms, and tell me if we should sign it" — is exactly where the
heuristic should decline to split *and say it is unsure*, handing the decision
to the model layer. That is the whole reason confidence is reported rather
than a boolean.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace

from .triage import TASK_TYPES, classify_heuristic

__all__ = [
    "MAX_STEPS",
    "plan_request",
    "plan_with_model",
    "Plan",
    "PlanStep",
    "PlanValidationError",
    "no_split_plan",
    "parse_plan",
    "plan_heuristic",
    "topological_order",
    "validate_plan",
]

# A plan longer than this is a planner malfunction, not an ambitious request.
# Bounding it keeps a runaway model from proposing a hundred billable steps.
MAX_STEPS = 12

# How planned_by reports a request that was deliberately not split.
NOT_SPLIT = "none"


class PlanValidationError(ValueError):
    """A plan is structurally unusable. Callers fail closed to single-task."""


@dataclass(frozen=True)
class PlanStep:
    """One routable unit of a decomposed request."""

    step_id: str
    prompt: str
    task_type: str
    complexity: float
    est_input_tokens: int
    est_output_tokens: int
    # step_ids whose OUTPUT this step needs. Threaded in at dispatch time.
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class Plan:
    """A decomposition, and an account of who produced it and why."""

    request: str
    steps: tuple[PlanStep, ...]
    # "heuristic" | "model:<model_id>" | "none". Never a layer that did not
    # actually decide — a failed model call is credited to the heuristic.
    planned_by: str
    confidence: float
    rationale: str
    signals: tuple[str, ...] = ()

    @property
    def is_split(self) -> bool:
        """True when the request was actually decomposed."""
        return len(self.steps) > 1

    def describe(self) -> str:
        """One line for a rationale string."""
        if not self.is_split:
            return f"plan: not split ({self.planned_by}), {self.rationale}"
        return (
            f"plan: {len(self.steps)} steps ({self.planned_by}, "
            f"confidence {self.confidence:.2f})"
        )


# --------------------------------------------------------------------------
# Validation — strict about meaning, tolerant about transport.
# --------------------------------------------------------------------------


def validate_plan(plan: Plan) -> None:
    """Raise PlanValidationError unless the plan is structurally executable.

    Deliberately unforgiving. A plan that half-parses is worse than none: it
    spends money on steps derived from a shape nobody checked.
    """
    if not plan.request.strip():
        raise PlanValidationError("plan has an empty request")
    if not plan.steps:
        raise PlanValidationError("plan has no steps")
    if len(plan.steps) > MAX_STEPS:
        raise PlanValidationError(
            f"plan has {len(plan.steps)} steps, more than the {MAX_STEPS} cap; "
            f"treating as a planner malfunction rather than an ambitious request"
        )

    seen: set[str] = set()
    for index, step in enumerate(plan.steps, start=1):
        expected = f"s{index}"
        if step.step_id != expected:
            raise PlanValidationError(
                f"step ids must be contiguous s1..sN in order; "
                f"position {index} is {step.step_id!r}, expected {expected!r}"
            )
        if step.step_id in seen:
            raise PlanValidationError(f"duplicate step id {step.step_id!r}")
        seen.add(step.step_id)

        if not step.prompt.strip():
            raise PlanValidationError(f"{step.step_id}: empty prompt")
        if step.task_type not in TASK_TYPES:
            raise PlanValidationError(
                f"{step.step_id}: unknown task_type {step.task_type!r}; "
                f"known: {sorted(TASK_TYPES)}"
            )
        if not isinstance(step.complexity, (int, float)) or isinstance(step.complexity, bool):
            raise PlanValidationError(f"{step.step_id}: complexity must be a number")
        if not 0.0 <= step.complexity <= 1.0:
            raise PlanValidationError(
                f"{step.step_id}: complexity {step.complexity} outside [0, 1]"
            )
        for label, value in (
            ("est_input_tokens", step.est_input_tokens),
            ("est_output_tokens", step.est_output_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise PlanValidationError(
                    f"{step.step_id}: {label} must be a non-negative int, got {value!r}"
                )

    ids = {s.step_id for s in plan.steps}
    for step in plan.steps:
        for dep in step.depends_on:
            if dep == step.step_id:
                raise PlanValidationError(f"{step.step_id} depends on itself")
            if dep not in ids:
                raise PlanValidationError(
                    f"{step.step_id} depends on {dep!r}, which is not a step in this plan"
                )
        if len(set(step.depends_on)) != len(step.depends_on):
            raise PlanValidationError(f"{step.step_id} lists a dependency twice")

    topological_order(plan)  # raises on a cycle


def topological_order(plan: Plan) -> tuple[PlanStep, ...]:
    """Steps in dependency order, raising PlanValidationError on a cycle.

    Ties break on step_id so the order is deterministic: two runs of the same
    plan must dispatch in the same sequence or the traces stop comparing.
    """
    by_id = {s.step_id: s for s in plan.steps}
    unresolved = {s.step_id: set(s.depends_on) for s in plan.steps}
    ordered: list[PlanStep] = []

    while unresolved:
        ready = sorted(sid for sid, deps in unresolved.items() if not deps)
        if not ready:
            raise PlanValidationError(
                f"dependency cycle among steps {sorted(unresolved)}"
            )
        for sid in ready:
            ordered.append(by_id[sid])
            del unresolved[sid]
        for deps in unresolved.values():
            deps.difference_update(ready)

    return tuple(ordered)


# --------------------------------------------------------------------------
# Transport tolerance — the same parsing philosophy as the auditor.
# --------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^```[A-Za-z0-9_-]*\s*\n?(?P<body>.*?)\n?\s*```$", re.DOTALL)


def _strip_fences(text: str) -> str:
    match = _FENCE_RE.match(text.strip())
    return match.group("body").strip() if match else text.strip()


def _first_json_object(text: str) -> str | None:
    """First balanced {...}, ignoring braces inside strings."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = escaped = False
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


def parse_plan(text: str, request: str, planned_by: str) -> Plan:
    """Build a validated Plan from a model's reply.

    Tolerant about how the JSON arrives — fenced, prose-wrapped, whatever —
    and unforgiving about what it says. Raises PlanValidationError, which the
    caller turns into exactly one repair attempt and then a fail-closed.
    """
    candidate = _strip_fences(text or "")
    data = None
    for attempt in (candidate, _first_json_object(candidate)):
        if not attempt:
            continue
        try:
            data = json.loads(attempt)
            break
        except json.JSONDecodeError:
            data = None
    if not isinstance(data, dict):
        raise PlanValidationError("planner reply was not a JSON object")

    raw_steps = data.get("steps")
    if not isinstance(raw_steps, list):
        raise PlanValidationError("planner reply has no 'steps' list")

    steps: list[PlanStep] = []
    for i, raw in enumerate(raw_steps, start=1):
        if not isinstance(raw, dict):
            raise PlanValidationError(f"steps[{i - 1}] is not an object")
        depends = raw.get("depends_on", ())
        if isinstance(depends, str):
            depends = (depends,)
        if not isinstance(depends, (list, tuple)):
            raise PlanValidationError(f"step {i}: depends_on must be a list")
        try:
            steps.append(
                PlanStep(
                    step_id=str(raw.get("step_id") or f"s{i}"),
                    prompt=str(raw.get("prompt", "")),
                    task_type=str(raw.get("task_type", "")).strip().lower(),
                    complexity=float(raw.get("complexity", 0.5)),
                    est_input_tokens=int(raw.get("est_input_tokens", 0)),
                    est_output_tokens=int(raw.get("est_output_tokens", 0)),
                    depends_on=tuple(str(d) for d in depends),
                )
            )
        except (TypeError, ValueError) as e:
            raise PlanValidationError(f"step {i}: {e}") from None

    plan = Plan(
        request=request,
        steps=tuple(steps),
        planned_by=planned_by,
        confidence=float(data.get("confidence", 0.5) or 0.5),
        rationale=str(data.get("rationale", "")) or "model-proposed decomposition",
    )
    validate_plan(plan)
    return plan


# --------------------------------------------------------------------------
# The heuristic splitter.
# --------------------------------------------------------------------------

# Explicit sequencing. These are the ONLY prose markers that authorise a
# split: they state that one thing happens after another.
_SEQUENCE = [
    (3, r"\bthen\b"),
    (3, r"\bafter (?:that|which)\b"),
    (3, r"\bonce (?:that|you|it)(?:'s| is| have| has)? (?:done|finished|ready)\b"),
    (3, r"\bfinally\b"),
    (2, r"\bnext,"),
    (2, r"\blastly\b"),
    (2, r"\bafterwards?\b"),
    (2, r"\bfollowed by\b"),
    (2, r"\band then\b"),
]

# Enumeration: the writer already did the decomposition for us.
_NUMBERED = re.compile(r"(?:^|\n)\s*(?:\(?\d+[.)]|[-*•])\s+", re.MULTILINE)
_ORDINAL = re.compile(
    r"\b(?:first|second|third|fourth|fifth)(?:ly)?\s*[,:]", re.IGNORECASE
)

# Verbs that name a *kind* of work. Several distinct kinds with no explicit
# structure is the signature of a compound request the heuristic cannot safely
# cut — it lowers confidence instead, which is what opens the model gate.
_WORK_VERBS = {
    "extraction": r"\b(?:extract|parse|pull|scrape|list all|find all)\b",
    "summarization": r"\b(?:summari[sz]e|condense|recap|boil down|tl;?dr)\b",
    "coding": r"\b(?:refactor|debug|implement|write (?:a |the )?(?:function|script|handler|test|query))\b",
    "creative": r"\b(?:draft|write (?:the |a |me |us )?(?:copy|post|story|email|page|something)|brainstorm)\b",
    # "work out" / "figure out" are ordinary phrasings for the same judgement
    # work as "recommend" or "assess" — measured missing on a case that scored
    # confidently single-task (0.95) while actually being two jobs (ROADMAP 1c).
    "reasoning": (
        r"\b(?:recommend|evaluate|decide|assess|analy[sz]e|compare|"
        r"tell me (?:if|whether)|should we|work(?:ed)? out|figure out)\b"
    ),
}

_MIN_FRAGMENT_WORDS = 3


def _absorb_short(parts: list[str], signals: list[str]) -> list[str]:
    """Fold sub-minimum fragments into a neighbour instead of discarding them.

    Dropping them silently omits part of the user's request: "extract the
    pricing, then summarize it, then recommend a response" split on `then`
    yields a two-word middle fragment, and filtering it out produced a plan
    that never summarised anything. A fragment too small to stand alone is
    still work someone asked for, so it joins its predecessor.
    """
    kept: list[str] = []
    for part in (p for p in parts if p.strip()):
        if len(part.split()) < _MIN_FRAGMENT_WORDS and kept:
            kept[-1] = f"{kept[-1]}, {part}".strip()
            signals.append(f"absorbed short fragment {part!r} into the previous step")
        else:
            kept.append(part)
    # A short leading fragment has no predecessor; give it its successor.
    if len(kept) > 1 and len(kept[0].split()) < _MIN_FRAGMENT_WORDS:
        signals.append(f"absorbed short leading fragment {kept[0]!r}")
        kept = [f"{kept[0]}, {kept[1]}".strip()] + kept[2:]
    return kept


def _segment(request: str) -> tuple[list[str], list[str], str]:
    """Cut a request on explicit structure only. Returns (fragments, signals, kind)."""
    signals: list[str] = []

    if _NUMBERED.search(request):
        parts = [p.strip() for p in _NUMBERED.split(request) if p.strip()]
        if len(parts) > 1:
            signals.append(f"enumeration: {len(parts)} listed items")
            return parts, signals, "enumeration"

    if _ORDINAL.search(request):
        parts = [p.strip() for p in _ORDINAL.split(request) if p and p.strip()]
        if len(parts) > 1:
            signals.append(f"ordinal words: {len(parts)} segments")
            return parts, signals, "ordinal"

    pattern = "|".join(p for _, p in _SEQUENCE)
    if re.search(pattern, request, re.IGNORECASE):
        for weight, p in _SEQUENCE:
            if re.search(p, request, re.IGNORECASE):
                signals.append(f"sequence+{weight}:{p}")
        parts = [p.strip(" ,;.") for p in re.split(pattern, request, flags=re.IGNORECASE)]
        parts = _absorb_short(parts, signals)
        if len(parts) > 1:
            return parts, signals, "sequence"

    return [request.strip()], signals, "none"


def _distinct_work_kinds(request: str) -> set[str]:
    """Which kinds of work the request names, regardless of structure."""
    return {
        kind
        for kind, pattern in _WORK_VERBS.items()
        if re.search(pattern, request, re.IGNORECASE)
    }


def _estimate_tokens(fragment: str, complexity: float) -> tuple[int, int]:
    """Rough per-step token estimates. Estimates, and labelled as such.

    Re-estimated at dispatch with real counts from prior steps, so being
    approximate here costs accuracy in the plan preview, not in the bill.
    """
    est_in = max(120, len(fragment) // 3)
    est_out = int(200 + 900 * complexity)
    return est_in, est_out


def no_split_plan(
    request: str,
    reason: str,
    *,
    planned_by: str = NOT_SPLIT,
    confidence: float = 1.0,
    signals: tuple[str, ...] = (),
    est_input_tokens: int | None = None,
    est_output_tokens: int | None = None,
) -> Plan:
    """A one-step plan carrying the whole request.

    Single-step rather than zero-step so execution is uniform: `run_plan`
    always walks a list, and "not split" needs no special case anywhere
    downstream.
    """
    verdict = classify_heuristic(request)
    default_in, default_out = _estimate_tokens(request, verdict.complexity)
    plan = Plan(
        request=request,
        steps=(
            PlanStep(
                step_id="s1",
                prompt=request,
                task_type=verdict.task_type,
                complexity=verdict.complexity,
                est_input_tokens=(
                    default_in if est_input_tokens is None else est_input_tokens
                ),
                est_output_tokens=(
                    default_out if est_output_tokens is None else est_output_tokens
                ),
            ),
        ),
        planned_by=planned_by,
        confidence=confidence,
        rationale=reason,
        signals=signals,
    )
    validate_plan(plan)
    return plan


def plan_heuristic(
    request: str,
    *,
    est_input_tokens: int | None = None,
    est_output_tokens: int | None = None,
) -> Plan:
    """Decompose a request using explicit structure only.

    Biased hard against splitting. Only enumeration, ordinals, and sequence
    connectives authorise a cut. Comma chains do not ("extract names, emails
    and phone numbers" is one extraction), and neither does length — a long
    prompt is not a compound one, and treating it as such is the expensive
    failure this planner exists to avoid.

    When several kinds of work are named but no structure marks them out, the
    heuristic declines to split *and reports low confidence*, which is the
    signal that opens the model gate.
    """
    text = (request or "").strip()
    if not text:
        raise PlanValidationError("cannot plan an empty request")

    fragments, signals, kind = _segment(text)
    kinds = _distinct_work_kinds(text)

    if kind == "none":
        if len(kinds) > 1:
            # Compound-looking, structurally unmarked. Do not guess where the
            # seams are; say so and let the gate decide whether to pay a model.
            return no_split_plan(
                text,
                reason=(
                    f"no explicit structure, but {len(kinds)} kinds of work named "
                    f"({', '.join(sorted(kinds))}); declining to guess the split"
                ),
                planned_by="heuristic",
                confidence=0.20,
                signals=tuple(signals + [f"work-kinds:{','.join(sorted(kinds))}"]),
                est_input_tokens=est_input_tokens,
                est_output_tokens=est_output_tokens,
            )
        return no_split_plan(
            text,
            reason="single task: no enumeration, ordinals, or sequence connectives",
            planned_by="heuristic",
            confidence=0.95,
            signals=tuple(signals),
            est_input_tokens=est_input_tokens,
            est_output_tokens=est_output_tokens,
        )

    # Classify each fragment with the existing triage classifier, then merge
    # adjacent fragments of the same kind: "extract A. extract B." is two
    # sentences describing one job, and splitting it buys two audits for
    # nothing.
    classified = [(f, classify_heuristic(f)) for f in fragments]
    merged: list[tuple[str, object]] = []
    for fragment, verdict in classified:
        if merged and merged[-1][1].task_type == verdict.task_type:
            prev_fragment, prev_verdict = merged[-1]
            keep = prev_verdict if prev_verdict.complexity >= verdict.complexity else verdict
            merged[-1] = (f"{prev_fragment} {fragment}".strip(), keep)
            signals.append(f"merged adjacent {verdict.task_type} fragments")
        else:
            merged.append((fragment, verdict))

    if len(merged) < 2:
        return no_split_plan(
            text,
            reason=(
                f"{kind} structure found, but all fragments are the same kind of "
                f"work ({merged[0][1].task_type}); one step is cheaper than two"
            ),
            planned_by="heuristic",
            confidence=0.80,
            signals=tuple(signals),
            est_input_tokens=est_input_tokens,
            est_output_tokens=est_output_tokens,
        )

    # Distribute any caller-supplied totals across the steps instead of
    # discarding them. Silently ignoring them made every split plan invent its
    # own scale: a request declared at 12k input tokens planned as three steps
    # of ~120, so the router priced a large job as a trivial one and the eval's
    # cost comparison measured this estimator rather than the economics.
    kept = merged[:MAX_STEPS]
    weights = [max(v.complexity, 0.05) for _, v in kept]
    total_weight = sum(weights)

    steps: list[PlanStep] = []
    for i, (fragment, verdict) in enumerate(kept, start=1):
        est_in, est_out = _estimate_tokens(fragment, verdict.complexity)
        if est_input_tokens is not None:
            # The first step consumes the source material; later steps read
            # their predecessor's output, which is added at dispatch from the
            # real thing rather than guessed at here.
            est_in = est_input_tokens if i == 1 else est_in
        if est_output_tokens is not None:
            # Output is shared out by complexity: splitting divides the work,
            # it does not multiply it.
            est_out = max(1, int(est_output_tokens * weights[i - 1] / total_weight))
        steps.append(
            PlanStep(
                step_id=f"s{i}",
                prompt=fragment,
                task_type=verdict.task_type,
                complexity=verdict.complexity,
                est_input_tokens=est_in,
                est_output_tokens=est_out,
                # Linear chain: the markers that authorise a split are
                # sequential ones, so each step is assumed to want the last
                # one's output. A planner that inferred finer dependencies
                # would be guessing.
                depends_on=() if i == 1 else (f"s{i - 1}",),
            )
        )

    # Confidence tracks how explicit the structure was, not how good the split
    # looks. Enumeration is the writer's own decomposition; prose connectives
    # are an inference.
    confidence = {"enumeration": 0.95, "ordinal": 0.85, "sequence": 0.75}[kind]
    plan = Plan(
        request=text,
        steps=tuple(steps),
        planned_by="heuristic",
        confidence=confidence,
        rationale=(
            f"split into {len(steps)} steps on {kind} structure "
            f"({', '.join(s.task_type for s in steps)})"
        ),
        signals=tuple(signals),
    )
    validate_plan(plan)
    return plan


_PLANNER_PROMPT = """Break this request into the fewest steps that each need a
different KIND of work. If it is really one job, return exactly one step.

Splitting costs money: every step is a separate model call with its own audit.
Only split where the kind of work genuinely changes.

task_type must be one of: {types}
complexity is 0-1: 0.2 a lookup, 0.5 ordinary professional work, 0.9 genuinely hard.
depends_on lists the step_ids whose OUTPUT this step needs.

Respond with ONLY this JSON object, no prose, no markdown fences:
{{"confidence": 0.0-1.0, "rationale": "why you split it this way (or did not)",
  "steps": [{{"step_id": "s1", "prompt": "...", "task_type": "...",
              "complexity": 0.0-1.0, "est_input_tokens": 0,
              "est_output_tokens": 0, "depends_on": []}}]}}

Step ids must be s1, s2, ... in order. At most {max_steps} steps.

REQUEST:
{request}
"""

_REPAIR_SUFFIX = """

Your previous reply was rejected: {reason}

Return corrected JSON in exactly the schema above, and nothing else."""


# Planning is a JSON-emitting task, and reasoning models spend thinking tokens
# from the same budget before they emit anything. Measured on deepseek-v4-flash:
# at the 1024 default it burned the entire budget thinking and returned ZERO
# visible characters, which arrived here looking like malformed JSON. A simple
# plan needs ~650 output tokens once thinking is paid for; this leaves headroom.
PLANNER_MAX_TOKENS = 4000


def _cheapest(registry):
    """Planning is a classification, not the work. Never frontier by default."""
    return min(registry.all(), key=lambda m: (m.output_cost, m.input_cost))


def plan_with_model(
    request: str,
    registry,
    providers,
    policy=None,
    *,
    model_id: str | None = None,
    fallback: Plan | None = None,
) -> tuple[Plan, list[dict]]:
    """Ask a cheap model to decompose, with exactly one repair attempt.

    Returns the plan and any discarded attempts — a rejected reply was still
    generated and still billed, so it is reported rather than quietly absorbed.

    On any failure the fallback plan stands, and `planned_by` says
    ``heuristic``. A model that did not produce a usable plan gets no credit
    for the one that ran instead; that is the same honesty rule triage follows,
    and it is what keeps trace analysis meaningful.
    """
    from .router import actual_cost  # local: avoids a cycle at import time

    heuristic = fallback if fallback is not None else plan_heuristic(request)
    discarded: list[dict] = []
    try:
        spec = registry.get(model_id) if model_id else _cheapest(registry)
        provider = providers.get(spec.provider)
    except Exception as e:  # noqa: BLE001 - planning must never take a run down
        return with_planned_by(heuristic, "heuristic"), discarded

    prompt = _PLANNER_PROMPT.format(
        types=list(TASK_TYPES), max_steps=MAX_STEPS, request=request
    )
    for attempt in range(2):  # first try, then exactly one repair
        try:
            reply = provider.complete(spec.model_id, prompt, max_tokens=PLANNER_MAX_TOKENS)
        except Exception as e:  # noqa: BLE001
            discarded.append({
                "model_id": spec.model_id, "reason": f"{type(e).__name__}: {e}",
                "cost_usd": 0.0, "repair": attempt == 1,
            })
            break

        if reply.truncated:
            # Say truncated, not malformed. And do NOT spend a repair attempt:
            # a retry at the same ceiling truncates identically, which is the
            # same waste as escalating a truncated answer to a bigger model.
            discarded.append({
                "model_id": spec.model_id,
                "reason": (f"reply TRUNCATED at max_tokens ({PLANNER_MAX_TOKENS}); "
                           f"raise the ceiling rather than retrying"),
                "input_tokens": reply.input_tokens,
                "output_tokens": reply.output_tokens,
                "cost_usd": actual_cost(spec, reply.input_tokens, reply.output_tokens),
                "repair": attempt == 1,
                "truncated": True,
            })
            break

        try:
            plan = parse_plan(reply.text, request, f"model:{spec.model_id}")
            return plan, discarded
        except PlanValidationError as e:
            discarded.append({
                "model_id": spec.model_id,
                "reason": str(e),
                "input_tokens": reply.input_tokens,
                "output_tokens": reply.output_tokens,
                "cost_usd": actual_cost(spec, reply.input_tokens, reply.output_tokens),
                "repair": attempt == 1,
                "raw": reply.text[:400],
            })
            if attempt == 1:
                break
            prompt = prompt + _REPAIR_SUFFIX.format(reason=e)

    return with_planned_by(heuristic, "heuristic"), discarded


def plan_request(
    request: str,
    registry=None,
    providers=None,
    policy=None,
    *,
    use_model: bool = False,
    est_input_tokens: int | None = None,
    est_output_tokens: int | None = None,
) -> tuple[Plan, list[dict]]:
    """Plan a request, consulting a model only where the heuristic is unsure.

    Confidence-gated for the same reason triage is: the heuristic is free and
    right most of the time, and paying a model to re-decide a call it already
    made confidently buys latency and nothing else. The gate opens exactly
    where the heuristic says it cannot judge — several kinds of work named
    with no structure marking the seams.
    """
    heuristic = plan_heuristic(
        request,
        est_input_tokens=est_input_tokens,
        est_output_tokens=est_output_tokens,
    )
    threshold = getattr(policy, "plan_confidence_threshold", 0.25)
    if (
        use_model
        and registry is not None
        and providers is not None
        and heuristic.confidence < threshold
    ):
        return plan_with_model(
            request, registry, providers, policy, fallback=heuristic
        )
    return heuristic, []


def with_planned_by(plan: Plan, planned_by: str) -> Plan:
    """Relabel a plan's author. Used when a model plan is rejected and the
    heuristic's answer stands in — the credit must follow the work."""
    return replace(plan, planned_by=planned_by)
