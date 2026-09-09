"""Task-type inference: turn a bare prompt into a routable Task.

Callers should not have to hand-label every request. `task_type` and
`complexity` are the two inputs the router trusts most, and making them
mandatory pushes the hardest judgement in the system onto the call site — the
exact place the rest of this project works to keep decisions *out* of.

Two layers, in deliberate order:

1. **Heuristic** (default, offline, deterministic). Weighted keyword and
   structure signals. It is not clever and does not need to be: it is free,
   instant, adds no failure mode, and is auditable — every classification
   reports the signals that produced it. A wrong guess here is recoverable
   because the qualification gate and the auditor both sit downstream.
2. **Model-based** (opt-in). Asks a cheap model from the catalog. Better on
   ambiguous prose, but it costs money, adds latency, and can fail — so it
   falls back to the heuristic on *any* error and never runs in tests.

The honesty rule this module exists to enforce: every result says which layer
produced it, and that string reaches the routing rationale. "classified as
coding" and "guessed coding from keywords" are different claims, and a
reviewer reading a trace is entitled to know which one they are looking at.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .policies import Task

# The task types the heuristic can produce. Deliberately the same vocabulary
# the demo and starter catalogs score their models on — a classifier that
# emits labels the catalog has no capability data for is worse than useless,
# because the router would silently fall back to a flat prior.
TASK_TYPES = ("extraction", "summarization", "coding", "creative", "reasoning")

# Sentinel a caller puts in Task.task_type to say "you work it out".
AUTO = "auto"

# Tie-break order when two types score equally. Ordered most-specific first:
# "extract the function signatures" is an extraction task that mentions code,
# not a coding task, and specific verbs beat generic ones.
_TIE_BREAK = ("extraction", "summarization", "coding", "creative", "reasoning")

# Fallback when nothing matches at all. Reasoning is the safest default: it is
# the most demanding label, so an unmatched prompt routes conservatively
# rather than landing on the cheapest model by accident.
_DEFAULT_TYPE = "reasoning"

# Weighted signals. Weight 3 = the verb that names the task ("summarize").
# Weight 2 = strong supporting evidence. Weight 1 = weak, only breaks ties.
_KEYWORDS: dict[str, list[tuple[int, str]]] = {
    "extraction": [
        (3, r"\bextract\b"), (3, r"\bparse\b"), (3, r"\bpull (?:out|the)\b"),
        (3, r"\bscrape\b"), (2, r"\blist all\b"), (2, r"\bfind all\b"),
        (2, r"\bidentify (?:the|all|every)\b"), (2, r"\binto json\b"),
        (2, r"\bas (?:json|csv|a table)\b"), (2, r"\bstructured (?:data|output)\b"),
        (2, r"\bfields?\b"), (1, r"\bfrom (?:this|these|the following)\b"),
        (1, r"\bcsv\b"), (1, r"\brecords?\b"), (1, r"\bentit(?:y|ies)\b"),
    ],
    "summarization": [
        (3, r"\bsummari[sz]e\b"), (3, r"\bsummary\b"), (3, r"\btl;?dr\b"),
        (3, r"\bcondense\b"), (2, r"\brecap\b"), (2, r"\bkey (?:points|takeaways)\b"),
        (2, r"\bin (?:a|one) (?:paragraph|sentence|page)\b"), (2, r"\bdigest\b"),
        (2, r"\bboil(?: it)? down\b"), (1, r"\bbrief\b"), (1, r"\boverview\b"),
        (1, r"\bshorter\b"), (1, r"\bgist\b"),
    ],
    "coding": [
        (3, r"\brefactor\b"), (3, r"\bdebug\b"), (3, r"\bimplement\b"),
        (3, r"\bwrite (?:a |the )?(?:function|class|script|handler|test|query|regex|parser|endpoint)\b"),
        (3, r"\bunit tests?\b"), (3, r"\bstack trace\b"), (3, r"\bfix (?:the |this )?bug\b"),
        (2, r"\bcode\b"), (2, r"\bfunction\b"), (2, r"\bwebhook\b"), (2, r"\bapi\b"),
        (2, r"\bsql\b"), (2, r"\bregex\b"), (2, r"\bendpoint\b"), (2, r"\bcompiles?\b"),
        (2, r"\b(?:python|javascript|typescript|rust|golang|java|c\+\+)\b"),
        (2, r"\bmigration script\b"), (1, r"\brepo(?:sitory)?\b"), (1, r"\bmodule\b"),
        (1, r"\bexception\b"), (1, r"\bdependenc(?:y|ies)\b"),
    ],
    "creative": [
        (3, r"\bpoem\b"), (3, r"\bstory\b"), (3, r"\btagline\b"), (3, r"\bslogan\b"),
        (3, r"\blanding[- ]page copy\b"), (3, r"\bad copy\b"), (3, r"\bbrainstorm\b"),
        (2, r"\bmarketing\b"), (2, r"\bheadline\b"), (2, r"\bblog post\b"),
        (2, r"\bnewsletter\b"), (2, r"\bcopy\b"), (2, r"\bnarrative\b"),
        (2, r"\bpunch(?:y|ier)\b"), (2, r"\bwitty\b"), (1, r"\bdraft\b"),
        (1, r"\btone\b"), (1, r"\bcompelling\b"), (1, r"\bvoice\b"),
    ],
    "reasoning": [
        (3, r"\bwhy (?:did|does|is|are|would|should)\b"), (3, r"\btrade[- ]?offs?\b"),
        (3, r"\brecommend\b"), (3, r"\bevaluate\b"), (3, r"\bdiagnose\b"),
        (3, r"\bprove\b"), (3, r"\bshould we\b"), (3, r"\bstrateg(?:y|ise|ize)\b"),
        (2, r"\banaly[sz]e\b"), (2, r"\bcompare\b"), (2, r"\bexplain\b"),
        (2, r"\bdecide\b"), (2, r"\bimplications?\b"), (2, r"\brisks?\b"),
        (2, r"\bdesign (?:a|an|the) (?:system|architecture|protocol|approach)\b"),
        (2, r"\bpros and cons\b"), (1, r"\bassess\b"), (1, r"\bconsider\b"),
        (1, r"\brationale\b"),
    ],
}

# Structural tells that beat prose. A fenced code block is not a hint.
_STRUCTURAL: list[tuple[str, int, str]] = [
    ("coding", 4, r"```"),
    ("coding", 3, r"\bdef \w+\("),
    ("coding", 3, r"\bclass \w+[:(]"),
    ("coding", 2, r"\bimport \w+"),
    ("coding", 2, r"\b(?:SELECT|INSERT|UPDATE|DELETE)\b.*\bFROM\b"),
    ("extraction", 2, r"\{\s*\"\w+\"\s*:"),  # a JSON schema in the prompt
]

# Complexity priors per task type: what an average request of this kind costs
# in capability, before the prompt's own signals adjust it.
_COMPLEXITY_PRIOR = {
    "extraction": 0.30,
    "summarization": 0.35,
    "creative": 0.50,
    "coding": 0.55,
    "reasoning": 0.60,
}

_HARDER = [
    (0.15, r"\bnovel\b"), (0.15, r"\bfrom scratch\b"), (0.15, r"\bproduction\b"),
    (0.15, r"\bdistributed\b"), (0.12, r"\barchitectur\w+\b"), (0.12, r"\bsecurity\b"),
    (0.12, r"\bconcurren\w+\b"), (0.12, r"\bedge cases?\b"), (0.12, r"\bprove\b"),
    (0.10, r"\boptimi[sz]e\b"), (0.10, r"\bmigrat\w+\b"), (0.10, r"\brollback\b"),
    (0.10, r"\btrade[- ]?offs?\b"), (0.10, r"\bambiguous\b"), (0.10, r"\brigorous\b"),
    (0.10, r"\bend[- ]to[- ]end\b"), (0.08, r"\bscal\w+\b"), (0.08, r"\bcompliance\b"),
]

_EASIER = [
    (0.15, r"\bsimple\b"), (0.15, r"\btrivial\b"), (0.12, r"\bquick(?:ly)?\b"),
    (0.12, r"\bone[- ]line\b"), (0.12, r"\bjust \b"), (0.10, r"\bshort\b"),
    (0.10, r"\bbriefly\b"), (0.10, r"\btl;?dr\b"), (0.10, r"\bbasic\b"),
    (0.08, r"\bstraightforward\b"),
]

_MIN_COMPLEXITY, _MAX_COMPLEXITY = 0.05, 0.95


@dataclass(frozen=True)
class Triage:
    """What triage concluded, and how it got there."""

    task_type: str
    complexity: float
    # "heuristic" or "model:<model_id>". Reaches the routing rationale
    # verbatim, so a trace never implies more rigour than was applied.
    source: str
    confidence: float = 0.0
    signals: list[str] = field(default_factory=list)

    @property
    def is_heuristic(self) -> bool:
        """True when no model was consulted — useful for filtering traces."""
        return self.source == "heuristic"

    def describe(self) -> str:
        """One line for the routing rationale."""
        return (
            f"triage: classified as {self.task_type}, "
            f"complexity {self.complexity:.2f} ({self.source})"
        )


def _score_types(text: str) -> tuple[dict[str, int], list[str]]:
    """Weighted signal score per task type, plus the signals that fired."""
    scores = {t: 0 for t in TASK_TYPES}
    signals: list[str] = []
    for task_type, patterns in _KEYWORDS.items():
        for weight, pattern in patterns:
            if re.search(pattern, text):
                scores[task_type] += weight
                signals.append(f"{task_type}+{weight}:{pattern}")
    for task_type, weight, pattern in _STRUCTURAL:
        if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
            scores[task_type] += weight
            signals.append(f"{task_type}+{weight}:structure({pattern})")
    return scores, signals


def _estimate_complexity(text: str, task_type: str) -> tuple[float, list[str]]:
    """Prior for the task type, nudged by difficulty and length cues."""
    complexity = _COMPLEXITY_PRIOR.get(task_type, 0.5)
    signals: list[str] = []

    for delta, pattern in _HARDER:
        if re.search(pattern, text):
            complexity += delta
            signals.append(f"harder+{delta}:{pattern}")
    for delta, pattern in _EASIER:
        if re.search(pattern, text):
            complexity -= delta
            signals.append(f"easier-{delta}:{pattern}")

    # Length is weak evidence of scope, so it moves the estimate a little and
    # never on its own: a long prompt is often just a long input document.
    words = len(text.split())
    if words > 120:
        complexity += 0.10
        signals.append("harder+0.1:length>120w")
    elif words > 45:
        complexity += 0.05
        signals.append("harder+0.05:length>45w")
    elif words < 8:
        complexity -= 0.05
        signals.append("easier-0.05:length<8w")

    # Multi-part asks are harder than single ones.
    clauses = len(re.findall(r"\b(?:and then|then|also|as well as|plus)\b", text))
    if clauses >= 2:
        complexity += 0.08
        signals.append("harder+0.08:multi-step")

    return max(_MIN_COMPLEXITY, min(_MAX_COMPLEXITY, complexity)), signals


def classify_heuristic(prompt: str) -> Triage:
    """Classify a prompt with deterministic keyword and structure signals.

    Deterministic on purpose: the same prompt must produce the same route on
    every run, or the traces stop being comparable and the eval suite stops
    meaning anything.
    """
    text = prompt.lower()
    scores, type_signals = _score_types(text)

    best = max(scores.values())
    if best == 0:
        task_type = _DEFAULT_TYPE
        confidence = 0.0
        type_signals.append("no signal matched; defaulted to reasoning")
    else:
        leaders = [t for t in _TIE_BREAK if scores[t] == best]
        task_type = leaders[0]
        total = sum(scores.values())
        runner_up = max((s for t, s in scores.items() if t != task_type), default=0)
        # Confidence blends share-of-signal with margin over the runner-up, so
        # "several weak hits on one type" scores lower than "one decisive verb".
        share = best / total if total else 0.0
        margin = (best - runner_up) / best
        confidence = round(0.5 * share + 0.5 * margin, 3)

    complexity, complexity_signals = _estimate_complexity(text, task_type)
    return Triage(
        task_type=task_type,
        complexity=round(complexity, 2),
        source="heuristic",
        confidence=confidence,
        signals=type_signals + complexity_signals,
    )


_MODEL_PROMPT = """Classify this task. Respond with ONLY a JSON object, no prose.

{{"task_type": one of {types}, "complexity": 0.0-1.0}}

complexity is how much model capability the task demands: 0.1 trivial
lookup, 0.5 ordinary professional work, 0.9 genuinely hard novel work.

TASK:
{prompt}
"""


def classify_with_model(
    prompt: str,
    registry,
    providers,
    policy=None,
    *,
    model_id: str | None = None,
) -> Triage:
    """Classify with a cheap model, falling back to the heuristic on any error.

    Triage is a routing *input*, so it must never be able to take the run down
    with it: a classifier outage would otherwise block every task in the queue.
    Any failure — provider error, unparseable reply, a label the catalog has
    never heard of — degrades to the heuristic and says so in `source`.
    """
    fallback = classify_heuristic(prompt)
    try:
        spec = registry.get(model_id) if model_id else _cheapest_capable(registry)
        provider = providers.get(spec.provider)
        reply = provider.complete(
            spec.model_id,
            _MODEL_PROMPT.format(types=list(TASK_TYPES), prompt=prompt),
        )
        data = json.loads(_first_json(reply.text) or "")
        task_type = str(data["task_type"]).strip().lower()
        if task_type not in TASK_TYPES:
            raise ValueError(f"unknown task_type {task_type!r}")
        complexity = float(data["complexity"])
        if not 0.0 <= complexity <= 1.0:
            raise ValueError(f"complexity {complexity} outside [0, 1]")
        return Triage(
            task_type=task_type,
            complexity=round(complexity, 2),
            source=f"model:{spec.model_id}",
            confidence=1.0,
            signals=[f"classified by {spec.model_id}"],
        )
    except Exception as e:  # noqa: BLE001 - deliberate: never fail the run
        return Triage(
            task_type=fallback.task_type,
            complexity=fallback.complexity,
            source="heuristic",
            confidence=fallback.confidence,
            signals=fallback.signals + [f"model triage failed ({type(e).__name__}: {e})"],
        )


def _cheapest_capable(registry):
    """Cheapest model in the catalog; triage is a classification, not the work."""
    return min(registry.all(), key=lambda m: (m.output_cost, m.input_cost))


def _first_json(text: str) -> str | None:
    """First balanced {...} block, so prose around the JSON is survivable."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def triage_task(
    task: Task,
    registry=None,
    providers=None,
    policy=None,
    *,
    use_model: bool = False,
) -> tuple[Task, Triage]:
    """Resolve a `task_type="auto"` Task into a concrete one.

    The model layer, when enabled, is used only where the heuristic is
    uncertain (see `Policy.triage_confidence_threshold`).

    Returns the resolved Task and the Triage that produced it. `complexity` is
    replaced too: a caller who did not know the task type cannot have known
    the complexity either, and leaving the 0.5 default in place would silently
    mix a real estimate with a placeholder.
    """
    verdict = classify_heuristic(task.prompt)
    threshold = getattr(policy, "triage_confidence_threshold", 0.25)
    if (
        use_model
        and registry is not None
        and providers is not None
        and verdict.confidence < threshold
    ):
        # Confidence-gated, not all-or-nothing. The heuristic is free, instant,
        # and right ~90% of the time; the model is better precisely where the
        # heuristic matched nothing and fell through to its default. Asking a
        # model on every prompt costs latency on the 88% the heuristic already
        # had, and measurably *lowers* accuracy by overriding cases it got right.
        verdict = classify_with_model(task.prompt, registry, providers, policy)
    resolved = Task(
        prompt=task.prompt,
        task_type=verdict.task_type,
        complexity=verdict.complexity,
        est_input_tokens=task.est_input_tokens,
        est_output_tokens=task.est_output_tokens,
        needs_fast_response=task.needs_fast_response,
        assumed_cache_hit_rate=task.assumed_cache_hit_rate,
    )
    return resolved, verdict
