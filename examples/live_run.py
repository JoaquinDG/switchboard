"""Run real tasks against real APIs. This one spends money.

    # 1. free: check the catalog's model ids actually exist
    PYTHONPATH=src python3 examples/live_check.py --catalog examples/starter_catalog.json

    # 2. free: see the plan and the worst-case bill, without calling anyone
    PYTHONPATH=src python3 examples/live_run.py --catalog examples/starter_catalog.json

    # 3. spends: real calls, capped at 50 cents
    PYTHONPATH=src python3 examples/live_run.py --catalog examples/starter_catalog.json \\
        --live --budget-usd 0.50

Why this exists: everything else in the repo is honest about being mocked, and
that honesty has a cost — `verified: True` from `MockProvider` means a canned
string was graded by a canned grader. It demonstrates the machinery, not the
idea. This script is where the claims get tested: a real model produces real
output, a *different real model from a different lab* grades it, escalation
fires on genuine disagreement, and the cost accounting reports an actual bill.

The traces it writes are the input to trace-driven capability scoring
(ROADMAP item 1) — the path from estimated scores to measured ones.

Safety:
* Dry-run is the default. `--live` is required to make any billable call.
* `--budget-usd` is a hard cap, checked before each task against that task's
  worst case, not its expected case.
* `--max-tokens` bounds output per call; that is the term that dominates cost.
* Keys come from the environment only. Never printed, never traced.
"""

from __future__ import annotations

import argparse
import sys

from switchboard import (
    PRESETS,
    Broker,
    Completion,
    ProviderError,
    Registry,
    Task,
    actual_cost,
    demo_registry,
    estimate_cost,
)
from switchboard.providers.live import KNOWN_PROVIDERS, live_pool, usable_registry

# Short prompts with checkable answers: an audit can only be meaningful if the
# task has a right answer the grader can actually assess.
SUITE = [
    ("extraction", Task(
        prompt="Extract every date from this text as ISO-8601, one per line, nothing else: "
               "'The kickoff is 5 January 2026, the review follows on 12 February 2026, "
               "and we ship 3 March 2026.'",
        task_type="extraction", complexity=0.2,
        est_input_tokens=80, est_output_tokens=40)),
    ("summarization", Task(
        prompt="Summarise in exactly one sentence: 'Switchboard routes each task to the "
               "cheapest model qualified for it, then has a different model from a "
               "different provider grade the output. Failed audits escalate one tier up, "
               "carrying the auditor's findings into the retry.'",
        task_type="summarization", complexity=0.3,
        est_input_tokens=120, est_output_tokens=60)),
    ("coding", Task(
        prompt="Write a Python function `median(xs: list[float]) -> float` that returns the "
               "median. Handle the even-length case correctly and raise ValueError on an "
               "empty list. Code only, no explanation.",
        task_type="coding", complexity=0.4,
        est_input_tokens=90, est_output_tokens=140)),
    ("reasoning", Task(
        prompt="A team sends every request to a frontier model at $5/$25 per million tokens. "
               "Their median request is 2000 input and 500 output tokens, and 70% of requests "
               "are simple extraction. Should they route? Give the reasoning and one number "
               "that justifies it. Be concise.",
        task_type="reasoning", complexity=0.6,
        est_input_tokens=140, est_output_tokens=200)),
    ("auto (triage decides)", Task(
        prompt="Refactor this to remove the duplicated retry logic, keeping behaviour "
               "identical. Code only.",
        task_type="auto", complexity=0.5,
        est_input_tokens=80, est_output_tokens=160)),
]


class _CappedProvider:
    """Forces a max_tokens ceiling on every call.

    The Broker does not take a token budget — it is a routing layer, not a
    spend controller. For a live exercise the ceiling matters more than
    anything else, since output tokens dominate the bill, so it is imposed
    here at the edge rather than by widening the library's API for a demo.
    """

    def __init__(self, inner, max_tokens: int) -> None:
        self.inner = inner
        self.max_tokens = max_tokens

    @property
    def name(self) -> str:
        return self.inner.name

    def complete(self, model_id: str, prompt: str, max_tokens: int = 1024) -> Completion:
        return self.inner.complete(model_id, prompt, min(max_tokens, self.max_tokens))


def worst_case_cost(task: Task, registry: Registry, policy, max_tokens: int) -> float:
    """Ceiling for one task: priciest model, every escalation, plus audits.

    Deliberately pessimistic. A budget guard that assumes the expected case
    is not a guard.
    """
    priciest = max(registry.all(), key=lambda m: m.output_cost)
    one_call = actual_cost(priciest, task.est_input_tokens, max_tokens)
    attempts = 1 + policy.max_escalations
    audits = attempts if policy.audit_enabled else 0
    # An audit prompt carries the task prompt plus the output plus the rubric.
    audit_call = actual_cost(priciest, task.est_input_tokens + max_tokens + 200, 120)
    return one_call * attempts + audit_call * audits


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run real tasks through Switchboard against live vendor APIs.",
        epilog="Dry-run by default. --live is required to spend anything.",
    )
    parser.add_argument("--catalog", help="catalog JSON (default: built-in demo catalog)")
    parser.add_argument("--policy", default="balanced", choices=sorted(PRESETS))
    parser.add_argument("--live", action="store_true",
                        help="actually call the APIs. Without this, nothing is billed.")
    parser.add_argument("--budget-usd", type=float, default=0.50,
                        help="hard spend cap for the whole run (default: 0.50). The guard\n     is priced at the worst case; real spend is typically 10-30x lower because\n     routing sends most of the suite to cheap models.")
    parser.add_argument("--max-tokens", type=int, default=400,
                        help="output ceiling per call (default: 400)")
    parser.add_argument("--trace", default="traces/live.jsonl")
    args = parser.parse_args()

    registry = Registry.from_json(args.catalog) if args.catalog else demo_registry()
    policy = PRESETS[args.policy]

    pool, skipped = live_pool(registry)
    if skipped:
        print("Providers skipped for want of a key:")
        for name in skipped:
            spec = KNOWN_PROVIDERS.get(name)
            env = spec.env_var if spec else "unknown"
            print(f"  {name:<12} set {env}" + (f"  -> {spec.signup_url}" if spec else ""))
        print()

    # A dry run is still worth doing with no keys at all — seeing the plan and
    # the ceiling is how you decide whether to go and get keys. Only a real
    # run needs one.
    if args.live and not pool.names():
        print("No provider keys are set, so there is nothing to run.")
        print("Export at least one of: " +
              ", ".join(s.env_var for s in KNOWN_PROVIDERS.values()))
        return 1

    live_registry = usable_registry(registry, pool)
    if not live_registry.all():
        if args.live:
            print("Keys are set, but no catalog model belongs to a keyed provider.")
            return 1
        # Nothing keyed: cost the plan against the full catalog so the ceiling
        # is still meaningful, and say that is what happened.
        print("(no keys set — costing the plan against the full catalog)")
        live_registry = registry

    usable_note = (
        f"{len(live_registry)}/{len(registry)} models usable with the keys present"
        if pool.names() else f"{len(registry)} models, none currently keyed"
    )
    print(f"catalog:   {args.catalog or 'built-in demo'} ({usable_note})")
    print(f"providers: {', '.join(pool.names()) or '(none keyed)'}")
    print(f"policy:    {policy.name}")
    print(f"budget:    ${args.budget_usd:.2f} hard cap, {args.max_tokens} output tokens per call")

    projected = sum(worst_case_cost(t, live_registry, policy, args.max_tokens) for _, t in SUITE)
    print(f"worst case for {len(SUITE)} tasks: ${projected:.4f} "
          f"(priciest model, all escalations, all audits)")

    if not args.live:
        print("\n--- DRY RUN: no API calls made, nothing billed ---")
        for label, task in SUITE:
            ceiling = worst_case_cost(task, live_registry, policy, args.max_tokens)
            print(f"  {label:<24} worst case ${ceiling:.4f}   {task.prompt[:52]}...")
        print("\nAdd --live to run it for real.")
        return 0

    if projected > args.budget_usd:
        print(f"\nSTOP: worst case ${projected:.4f} exceeds the ${args.budget_usd:.2f} budget.")
        print("Raise --budget-usd, lower --max-tokens, or trim SUITE.")
        return 1

    broker = Broker(live_registry, pool, policy, trace_path=args.trace)
    spent = 0.0
    rows = []

    print(f"\n--- LIVE: calling real APIs, capped at ${args.budget_usd:.2f} ---\n")
    for label, task in SUITE:
        ceiling = worst_case_cost(task, live_registry, policy, args.max_tokens)
        if spent + ceiling > args.budget_usd:
            print(f"{label}: SKIPPED — would risk exceeding budget "
                  f"(${spent:.4f} spent, ${ceiling:.4f} worst case)")
            continue

        capped = type(pool)([_CappedProvider(pool.get(n), args.max_tokens) for n in pool.names()])
        try:
            result = Broker(live_registry, capped, policy, trace_path=args.trace).run(task)
        except ProviderError as e:
            print(f"{label}: FAILED — every routing option exhausted: {e}\n")
            continue

        spent += result.total_cost_usd
        rows.append((label, result))
        estimated = estimate_cost(task, live_registry.get(result.final_model))

        print(f"{label}")
        if result.triage:
            print(f"   triage:   {result.triage.task_type} "
                  f"@ {result.triage.complexity} ({result.triage.source})")
        print(f"   routed:   {result.final_model}")
        for i, a in enumerate(result.attempts, 1):
            note = f"error={a.error}" if a.error else f"audit={a.audit_passed} ({a.audit_score})"
            lab = ""
            if a.auditor_model:
                lab = f" by {a.auditor_model} ({'cross-lab' if a.cross_lab_audit else 'same-lab'})"
            print(f"   attempt {i}: {a.model_id} [{a.role}] {note}{lab} "
                  f"{a.input_tokens}->{a.output_tokens} tok  ${a.total_cost_usd:.5f}")
        if result.attempts and result.attempts[-1].audit_issues:
            for issue in result.attempts[-1].audit_issues[:3]:
                print(f"     issue: {issue}")
        print(f"   verified: {result.verified}")
        print(f"   cost:     ${result.total_cost_usd:.5f} actual "
              f"vs ${estimated:.5f} estimated from declared tokens")
        print(f"   output:   {result.final_text.strip()[:90].replace(chr(10), ' ')}...")
        print()

    print("=" * 78)
    print(f"tasks run:        {len(rows)}/{len(SUITE)}")
    print(f"verified:         {sum(1 for _, r in rows if r.verified)}/{len(rows)}")
    print(f"escalated:        {sum(1 for _, r in rows if r.escalated)}")
    print(f"cross-lab audits: {sum(1 for _, r in rows if r.attempts[-1].cross_lab_audit)}")

    # Decomposed, because a single "saving" number here is misleading: the
    # baseline is generation-only, so comparing it against a total that
    # includes audits charges routing for verification the baseline never
    # paid for. Routing and verification are separate economic decisions and
    # the report has to keep them separate.
    gen = sum(r.generation_cost_usd for _, r in rows)
    aud = sum(r.audit_cost_usd for _, r in rows)
    baseline = sum(r.baseline_cost_usd for _, r in rows)
    share = aud / (gen + aud) * 100 if (gen + aud) else 0.0
    print(f"generation, routed:               ${gen:.5f}")
    print(f"generation, strongest model:      ${baseline:.5f}")
    if baseline:
        print(f"  -> routing saved on generation: ${baseline - gen:+.5f} "
              f"({(baseline - gen) / baseline * 100:+.0f}%)")
    print(f"audit overhead:                   ${aud:.5f}  ({share:.0f}% of total spend)")
    print(f"total spent:                      ${spent:.5f} of ${args.budget_usd:.2f} budget")
    print()
    print("Audits are a policy choice, not a routing cost. On small tasks the audit")
    print("prompt (task + output + rubric, on a frontier grader) dwarfs the work it")
    print("grades — try --policy cost_first or Policy(auditor_selection=")
    print("'cheapest_qualified') and compare.")
    print(f"\nTraces: {args.trace}")
    print("These are real audit outcomes — the input for measured capability scores.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
