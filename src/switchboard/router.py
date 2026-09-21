"""The router: pick the best model for a task under a policy.

Design principles:
1. Every decision is explainable. The router returns a rationale and the full
   ranked list, never just a model id. If you can't explain a routing choice
   to a stakeholder, you can't debug it either.
2. Hard gates before soft scores. Complexity above the policy's frontier gate
   filters to frontier models *before* any scoring happens — a cheap model
   should never win a task it isn't qualified for just because it's cheap.
3. Gates degrade upward, never open. If a gate would leave zero candidates,
   the fallback is the most capable tier available plus a loud warning — not
   a silent return to the full catalog. Opening the field back up means cost
   decides, which is exactly what the gate existed to prevent.
4. Costs are normalized on a log scale, because model prices span orders of
   magnitude and a linear scale would make cost dominate every decision.
5. Quality is normalized against a fixed floor (UNKNOWN_CAPABILITY_PRIOR),
   never against the observed candidates. A per-decision candidate range
   makes the same model score differently depending who else is competing —
   the same artifact already fixed for cost, reintroduced on another axis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .policies import NoQualifiedModelError, Policy, Task
from .registry import TIER_RANK, UNKNOWN_CAPABILITY_PRIOR, ModelSpec, Registry

_LATENCY_SCORE = {"fast": 1.0, "medium": 0.6, "slow": 0.2}


@dataclass(frozen=True)
class ScoredModel:
    """One candidate's score with its components kept separate.

    The breakdown is retained rather than collapsed to a total so a decision
    can be argued with: "it lost on latency, not quality" is debuggable.

    ``*_component`` fields are always the normalized [0, 1] terms actually
    multiplied by the policy's weights — ``quality_component`` included.
    Raw capability is still one call away via ``spec.capability_for(task_type)``,
    the same way ``est_cost_usd`` sits next to the normalized ``cost_component``.
    """

    spec: ModelSpec
    score: float
    quality_component: float
    cost_component: float
    latency_component: float
    est_cost_usd: float


@dataclass(frozen=True)
class RoutingDecision:
    """Where the task is going, and the full case for why.

    Carries the entire ranked list, not just the winner, because the runner-up
    and its margin are what tell you whether a weight change would flip it.
    """

    chosen: ModelSpec
    ranked: list[ScoredModel]
    rationale: str
    policy_name: str
    # Gates that actually fired, in order. Useful for grouping traces.
    gates: list[str] = field(default_factory=list)
    # Conditions the caller should know about but that did not stop routing:
    # unknown task type, empty tier, nothing qualified. Also folded into the
    # rationale so a single printed line stays sufficient.
    warnings: list[str] = field(default_factory=list)
    # True when no candidate cleared the qualification gate. The task ran on
    # the best available model, but the catalog claims nothing is rated for it.
    underqualified: bool = False


def estimate_cost(task: Task, spec: ModelSpec) -> float:
    """Estimated USD cost of running this task on this model."""
    return (
        task.est_input_tokens * spec.input_cost
        + task.est_output_tokens * spec.output_cost
    ) / 1_000_000


def actual_cost(spec: ModelSpec, input_tokens: int, output_tokens: int) -> float:
    """USD cost of a completed call, from observed token counts."""
    return (input_tokens * spec.input_cost + output_tokens * spec.output_cost) / 1_000_000


def _cost_score(cost: float, min_cost: float, max_cost: float) -> float:
    """Map cost to [0, 1] on a log scale; cheapest -> 1.0, priciest -> 0.0."""
    if max_cost <= min_cost:
        return 1.0
    lo, hi = math.log10(max(min_cost, 1e-9)), math.log10(max(max_cost, 1e-9))
    val = math.log10(max(cost, 1e-9))
    return 1.0 - (val - lo) / (hi - lo)


def _quality_score(capability: float) -> float:
    """Stretch raw capability onto the range it actually varies over.

    Capability is declared 0-1, but no honest catalog scores a real model
    below UNKNOWN_CAPABILITY_PRIOR — that number *is* "we have no idea,"
    so anything a catalog maintainer actually rates sits at or above it.
    The working range is therefore [UNKNOWN_CAPABILITY_PRIOR, 1.0], not
    [0, 1], while cost (log-scaled to the full catalog) and latency (three
    discrete steps) both already use their full [0, 1] range. Used raw,
    capability under-uses its axis: a weight of 0.85 on an axis that only
    ever moves through ~0.3 of its nominal span buys a fraction of the
    influence the number implies (ROADMAP item 1b).

    ROADMAP item 1b also records why the obvious fix — min-max normalizing
    against the *candidates actually being compared* — was tried and
    rejected: on a narrow catalog it turns a real 0.10 spread into a full
    0-to-1 swing, and a task's candidate set is exactly the thing that
    changes from one routing call to the next, so the same models could
    score differently against each other depending who else showed up.
    Anchoring on UNKNOWN_CAPABILITY_PRIOR instead of the observed
    candidate range fixes that: the floor is a fixed catalog-wide constant,
    not a function of who is competing this time, so it does not
    reintroduce the candidate-count artifact already fixed for cost.
    """
    floor = UNKNOWN_CAPABILITY_PRIOR
    return max(0.0, (capability - floor) / (1.0 - floor))


def _highest_tier(models: list[ModelSpec]) -> list[ModelSpec]:
    """The subset sitting in the most capable tier present."""
    best = max(TIER_RANK.get(m.tier, 0) for m in models)
    return [m for m in models if TIER_RANK.get(m.tier, 0) == best]


def score_models(
    task: Task, candidates: list[ModelSpec], registry: Registry, policy: Policy
) -> list[ScoredModel]:
    """Score candidates under a policy, best first.

    Costs are normalized over the FULL catalog, not just the candidates:
    candidate-only min-max made scores extreme (with 2 candidates one is always
    0.0), letting a normalization artifact flip decisions.

    Public because escalation needs the identical calculation. Picking the
    escalation target by raw capability instead meant routing honoured the
    policy and escalation ignored it — so a cost-first run could fail an audit
    and jump straight to the priciest model in the catalog.
    """
    all_costs = [estimate_cost(task, m) for m in registry.all()]
    min_cost, max_cost = min(all_costs), max(all_costs)

    scored: list[ScoredModel] = []
    for spec in candidates:
        cost = estimate_cost(task, spec)
        quality_s = _quality_score(spec.capability_for(task.task_type))
        cost_s = _cost_score(cost, min_cost, max_cost)
        latency_s = _LATENCY_SCORE[spec.latency]
        if task.needs_fast_response and spec.latency == "slow":
            latency_s = 0.0
        total = (
            policy.quality_weight * quality_s
            + policy.cost_weight * cost_s
            + policy.latency_weight * latency_s
        )
        scored.append(ScoredModel(spec, total, quality_s, cost_s, latency_s, cost))

    scored.sort(key=lambda s: s.score, reverse=True)
    return scored


@dataclass(frozen=True)
class ComponentMargin:
    """One axis of a counterfactual: where the winner and a rival stood on a
    single scoring component, and what the gap was worth once the policy's
    weight was applied.

    ``weighted_margin`` is ``weight * (chosen_component - other_component)``:
    positive means the winner led on this axis and the rival lost ground here;
    negative means the rival was actually the *better* option on this axis and
    its overall deficit came from elsewhere. Keeping the sign is the point —
    "you lost on cost but were ahead on quality" is the honest answer, and
    collapsing it to an absolute gap would hide half of it.
    """

    component: str  # "quality" | "cost" | "latency"
    weight: float
    chosen_component: float
    other_component: float
    weighted_margin: float


@dataclass(frozen=True)
class Counterfactual:
    """Why a specific model did not win a routing decision.

    Reconstructed entirely from the decision's existing ranked list — no
    re-routing — so it costs nothing and cannot silently disagree with the
    decision it explains. The per-axis ``weighted_margin`` values sum to
    ``total_margin`` by construction, because the score they came from is
    exactly their weighted sum.
    """

    model_id: str
    chosen_id: str
    # False when the model is absent from the ranked list: a gate filtered it
    # before scoring, or it is not in this catalog. A gated model has no
    # component breakdown, so answering "it scored low" would be a lie.
    scored: bool
    is_winner: bool
    total_margin: float  # chosen.score - other.score; 0.0 if this model won
    components: list[ComponentMargin]
    rationale: str


def explain_not_chosen(
    decision: RoutingDecision, model_id: str, policy: Policy
) -> Counterfactual:
    """Answer "why not this model?" from a decision already made.

    The rationale on a ``RoutingDecision`` argues why the winner won. The
    question people actually ask is why *their* preferred model lost, and the
    ranked list already holds every number needed to answer it: each
    ``ScoredModel`` keeps its normalized components, so the losing margin can
    be attributed axis by axis without re-running the router (ROADMAP item 13).

    ``policy`` supplies the weights the components were combined under. It must
    be the policy that produced ``decision`` — a different one would attribute
    the gap using weights that never applied, so a name mismatch is refused
    rather than silently producing a plausible-looking wrong answer.

    A model that a gate removed before scoring is reported with
    ``scored=False`` and no component breakdown: it never earned a score to
    lose by, and pretending otherwise would invent a comparison that did not
    happen.
    """
    if policy.name != decision.policy_name:
        raise ValueError(
            f"policy {policy.name!r} did not produce this decision "
            f"(policy_name={decision.policy_name!r}); the weighted margins would "
            f"be computed under weights that never applied to this ranking"
        )

    chosen = decision.ranked[0]
    chosen_id = chosen.spec.model_id
    target = next(
        (s for s in decision.ranked if s.spec.model_id == model_id), None
    )

    if target is None:
        if decision.gates:
            note = (
                f"{model_id} is not among the {len(decision.ranked)} scored "
                f"candidate(s): a gate removed it before scoring, or it is not in "
                f"this catalog (gates fired: {', '.join(decision.gates)})"
            )
        else:
            note = (
                f"{model_id} is not in this catalog: no gate fired, so every "
                f"catalog model was scored and would appear in the ranked list"
            )
        return Counterfactual(
            model_id=model_id,
            chosen_id=chosen_id,
            scored=False,
            is_winner=False,
            total_margin=0.0,
            components=[],
            rationale=note,
        )

    if model_id == chosen_id:
        return Counterfactual(
            model_id=model_id,
            chosen_id=chosen_id,
            scored=True,
            is_winner=True,
            total_margin=0.0,
            components=[],
            rationale=f"{model_id} was the chosen model under {policy.name}",
        )

    axes = (
        ("quality", policy.quality_weight, chosen.quality_component, target.quality_component),
        ("cost", policy.cost_weight, chosen.cost_component, target.cost_component),
        ("latency", policy.latency_weight, chosen.latency_component, target.latency_component),
    )
    components = [
        ComponentMargin(name, weight, chosen_c, other_c, weight * (chosen_c - other_c))
        for name, weight, chosen_c, other_c in axes
    ]
    # Biggest source of the deficit first, so the rationale leads with the axis
    # that actually cost this model the decision.
    components.sort(key=lambda c: c.weighted_margin, reverse=True)
    total_margin = chosen.score - target.score

    phrases = []
    for c in components:
        pair = f"{c.other_component:.2f} vs {c.chosen_component:.2f}"
        if c.weighted_margin > 1e-9:
            phrases.append(f"lost on {c.component}: {pair}, worth {c.weighted_margin:.2f}")
        elif c.weighted_margin < -1e-9:
            phrases.append(f"led on {c.component}: {pair}, worth {-c.weighted_margin:.2f} its way")
        else:
            phrases.append(f"tied on {c.component}: {pair}")
    rationale = (
        f"{model_id} lost to {chosen_id} under {policy.name} by "
        f"{total_margin:.3f}: " + "; ".join(phrases)
    )

    return Counterfactual(
        model_id=model_id,
        chosen_id=chosen_id,
        scored=True,
        is_winner=False,
        total_margin=total_margin,
        components=components,
        rationale=rationale,
    )


def route(task: Task, registry: Registry, policy: Policy) -> RoutingDecision:
    """Rank all eligible models and return an explained decision."""
    candidates = registry.all()
    if not candidates:
        raise ValueError("registry is empty")
    if task.task_type == "auto":
        # Routing an unresolved task would score every model on the flat prior
        # for a task type called "auto" and quietly hand the decision to cost.
        # Triage is the Broker's job; say so rather than producing a confident
        # meaningless ranking.
        raise ValueError(
            "task_type='auto' must be resolved before routing — use Broker.run(), "
            "which runs triage first, or call switchboard.triage_task(task) yourself"
        )

    gates: list[str] = []
    warnings: list[str] = []

    # A task type nobody has scored means every model returns the same prior,
    # so the quality term is dead weight and cost silently decides. That used
    # to happen invisibly; now it is stated.
    if not registry.has_capability_data(task.task_type):
        warnings.append(
            f"no capability data for task_type={task.task_type!r} anywhere in the "
            f"catalog — every model scored on the default prior, so the quality "
            f"term cannot discriminate; add scores for this task type"
        )

    # Gate 1: frontier. Complexity above the policy threshold is frontier-only,
    # regardless of cost pressure.
    if task.complexity >= policy.frontier_gate:
        frontier = [m for m in candidates if m.tier == "frontier"]
        if frontier:
            candidates = frontier
            gates.append("frontier gate applied")
        else:
            candidates = _highest_tier(candidates)
            gates.append("frontier gate applied (no frontier tier; kept highest tier available)")
            warnings.append(
                f"complexity {task.complexity:.2f} triggered the frontier gate but the "
                f"catalog has no frontier models; fell back to tier "
                f"{candidates[0].tier!r}"
            )

    # Gate 2: qualification. Capability must clear complexity by a margin.
    # Found by the eval suite: without this, cost pressure routed
    # mid-complexity work to underqualified cheap models.
    # The margin is a safety buffer above the raw requirement. Near the top of
    # the scale there is no headroom for one — complexity 0.9 plus a 0.1 margin
    # demands a capability of 1.0, which no honest catalog claims — so the
    # buffer is dropped rather than making the gate unsatisfiable by
    # construction and flagging every hard task as underqualified. The raw
    # requirement still applies: a complexity-1.0 task really does have nothing
    # rated for it, and that is worth saying.
    threshold = task.complexity + policy.capability_margin
    if threshold >= 1.0:
        threshold = task.complexity
    qualified = [m for m in candidates if m.capability_for(task.task_type) >= threshold]
    underqualified = False

    if qualified:
        if len(qualified) < len(candidates):
            gates.append("qualification filter applied")
        candidates = qualified
    else:
        # Nothing clears the bar. Fail upward, not open: previously this branch
        # fell through with the full candidate list, so cost weight picked the
        # cheapest model for work nothing was rated to handle.
        underqualified = True
        best_capability = max(m.capability_for(task.task_type) for m in candidates)
        warnings.append(
            f"no model clears capability {threshold:.2f} for task_type="
            f"{task.task_type!r} (best available: {best_capability:.2f}); "
            f"routing to the strongest option and flagging the result"
        )
        if policy.on_no_qualified_model == "raise":
            raise NoQualifiedModelError("; ".join(warnings))
        if policy.on_no_qualified_model == "best_capability":
            candidates = [
                m for m in candidates if m.capability_for(task.task_type) >= best_capability
            ]
            gates.append("qualification filter applied (none qualified; kept most capable)")
        else:  # "escalate_tier"
            candidates = _highest_tier(candidates)
            gates.append("qualification filter applied (none qualified; degraded upward)")

    scored = score_models(task, candidates, registry, policy)
    top = scored[0]

    parts = [
        f"policy={policy.name}",
        f"task_type={task.task_type}",
        f"complexity={task.complexity:.2f}" + "".join(f" ({g})" for g in gates),
        f"chose {top.spec.model_id}: quality={top.spec.capability_for(task.task_type):.2f} "
        f"(score {top.quality_component:.2f}), "
        f"cost=${top.est_cost_usd:.4f} (score {top.cost_component:.2f}), "
        f"latency={top.spec.latency}",
    ]
    if len(scored) > 1:
        runner = scored[1]
        parts.append(
            f"runner-up {runner.spec.model_id} scored "
            f"{runner.score:.3f} vs {top.score:.3f}"
        )
    parts.extend(f"WARNING: {w}" for w in warnings)

    return RoutingDecision(
        chosen=top.spec,
        ranked=scored,
        rationale="; ".join(parts),
        policy_name=policy.name,
        gates=gates,
        warnings=warnings,
        underqualified=underqualified,
    )
