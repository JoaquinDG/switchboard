"""Planner evals: does decomposition help, and does it ever hurt?

False-split rate is reported first and gates the run, because the two failure
modes are not symmetric. Failing to split a compound request costs you a
saving you could have had. Splitting a simple one costs you money you did not
have to spend — extra calls, extra audits, extra escalation surface — and it
does so quietly, on work that was already fine.

So the target for false splits is zero, and a regression fails CI.

Deterministic and offline: the heuristic planner makes no model calls, so this
runs free and identically every time.

Usage:
    PYTHONPATH=src python3 evals/planner_eval.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from planner_cases import (  # noqa: E402
    ALL_CASES,
    COMPOUND,
    COMPOUND_UNMARKED,
    NOT_COMPOUND,
)

from switchboard import (  # noqa: E402
    BALANCED,
    TIER_RANK,
    Registry,
    Task,
    demo_registry,
    estimate_cost,
    route,
)
from switchboard.planner import plan_heuristic  # noqa: E402

# A single false split is a regression, not a rounding error.
MAX_FALSE_SPLIT_RATE = 0.0
MIN_SPLIT_COVERAGE = 0.80


def _plan_cost(plan, registry: Registry) -> float:
    """What this plan's steps would cost at their routed models."""
    from switchboard import route

    total = 0.0
    for step in plan.steps:
        task = Task(
            prompt=step.prompt, task_type=step.task_type, complexity=step.complexity,
            est_input_tokens=step.est_input_tokens, est_output_tokens=step.est_output_tokens,
        )
        total += estimate_cost(task, route(task, registry, BALANCED).chosen)
    return total


def run() -> int:
    registry = demo_registry()

    # ---- 1. False splits. Reported first because it is the expensive error.
    print("=" * 100)
    print("FALSE SPLITS — simple requests the planner chopped up anyway")
    print("=" * 100)
    false_splits = []
    for case in NOT_COMPOUND:
        plan = plan_heuristic(case.request)
        if plan.is_split:
            false_splits.append((case, plan))
            print(f"  [FAIL] split into {len(plan.steps)}: {case.request[:66]}...")
            print(f"         {plan.rationale}")
            if case.note:
                print(f"         label note: {case.note}")
    rate = len(false_splits) / len(NOT_COMPOUND)
    if not false_splits:
        print(f"  none. {len(NOT_COMPOUND)}/{len(NOT_COMPOUND)} simple requests left alone.")
    print(f"\n  false-split rate: {rate:.0%} (target {MAX_FALSE_SPLIT_RATE:.0%})")

    # ---- 2. Split coverage on the genuinely compound set.
    print("\n" + "=" * 100)
    print("SPLIT COVERAGE — compound requests the planner decomposed")
    print("=" * 100)
    print(f"{'request':<64} {'want':>5} {'got':>4} {'conf':>5}  result")
    print("-" * 100)
    covered = 0
    misses = []
    for case in COMPOUND:
        plan = plan_heuristic(case.request)
        want = len(case.expected_types or ())
        ok = plan.is_split
        covered += ok
        if not ok:
            misses.append((case, plan))
        shown = case.request.replace("\n", " ⏎ ")
        shown = shown if len(shown) <= 62 else shown[:59] + "..."
        print(f"{shown:<64} {want:>5} {len(plan.steps):>4} {plan.confidence:>5.2f}  "
              f"{'PASS' if ok else 'MISS'}")
    coverage = covered / len(COMPOUND)
    print("-" * 100)
    print(f"  split coverage: {covered}/{len(COMPOUND)} = {coverage:.0%} "
          f"(target {MIN_SPLIT_COVERAGE:.0%})")
    for case, plan in misses:
        print(f"\n  MISS: {case.request[:70]}")
        print(f"        {plan.rationale}")
        if case.note:
            print(f"        label note: {case.note}")

    # ---- 2b. The set the heuristic does not claim to handle.
    print("\n" + "=" * 100)
    print("UNMARKED COMPOUNDS — genuinely compound, no structure to cut on")
    print("=" * 100)
    print("  The heuristic splits on explicit structure by design, so declining")
    print("  these is correct behaviour, not a miss. What matters is whether it")
    print("  KNOWS it cannot judge — low confidence opens the model gate; high")
    print("  confidence means the model is never asked.")
    print()
    gate_would_open = 0
    for case in COMPOUND_UNMARKED:
        plan = plan_heuristic(case.request)
        opens = plan.confidence < 0.25
        gate_would_open += opens
        print(f"  conf={plan.confidence:.2f}  gate {'OPENS ' if opens else 'STAYS SHUT'}  "
              f"{case.request[:52]}")
        print(f"            {case.note}")
    print(f"\n  model gate would fire on {gate_would_open}/{len(COMPOUND_UNMARKED)}")
    print("  A blind spot is worse than a known unknown: the gate cannot rescue")
    print("  a case the heuristic is confident about.")

    # ---- 3. Step-type agreement with the hand labels.
    print("\n" + "=" * 100)
    print("STEP CLASSIFICATION — planner types vs hand labels")
    print("=" * 100)
    matched = compared = 0
    count_mismatch = 0
    confusion: Counter[tuple[str, str]] = Counter()
    for case in COMPOUND:
        plan = plan_heuristic(case.request)
        expected = case.expected_types or ()
        got = tuple(s.task_type for s in plan.steps)
        if len(got) != len(expected):
            count_mismatch += 1
            continue
        for want, have in zip(expected, got):
            compared += 1
            matched += want == have
            if want != have:
                confusion[(want, have)] += 1
    agreement = matched / compared if compared else 0.0
    print(f"  step-count agreement: {len(COMPOUND) - count_mismatch}/{len(COMPOUND)} cases")
    print(f"  type agreement on those: {matched}/{compared} = {agreement:.0%}")
    if confusion:
        print("  disagreements (label -> planner):")
        for (want, have), n in confusion.most_common():
            print(f"    {want} -> {have}  x{n}")

    # ---- 4. What the split is worth, and what the planner costs to get it.
    print("\n" + "=" * 100)
    print("VALUE AND OVERHEAD")
    print("=" * 100)
    # Three columns, because "is decomposition cheaper" turns out to be the
    # wrong question. Routing a compound request whole is often CHEAPER — but
    # only because the classifier labels it by its dominant verb, routes it to
    # a small model, and never sends the hard sub-task anywhere qualified.
    # The qualification gate cannot help: it sees an easy extraction task.
    #
    # So the honest comparison is against what a correct single call would
    # need, and the metric that matters is how often routing whole silently
    # under-routes.
    IN, OUT = 12_000, 4_000
    print(f"{'case':<40} {'whole':>9} {'whole@hard':>11} {'planned':>9}  under-routed?")
    print("-" * 84)
    under = 0
    tot_whole = tot_correct = tot_split = 0.0
    for case in COMPOUND:
        plan = plan_heuristic(case.request, est_input_tokens=IN, est_output_tokens=OUT)
        whole = _whole_plan(case.request, IN, OUT)
        hardest = max(plan.steps, key=lambda st: st.complexity)

        c_whole = _plan_cost(whole, registry)
        c_split = _plan_cost(plan, registry)
        # The same single call, but labelled at the hardest sub-task's
        # requirement — what one call would have to be to answer correctly.
        correct = _plan_cost(
            _relabel(whole, hardest.task_type, hardest.complexity), registry
        )
        tot_whole += c_whole
        tot_correct += correct
        tot_split += c_split

        whole_tier = _tier_of(whole.steps[0], registry)
        need_tier = _tier_of(hardest, registry)
        short = TIER_RANK[whole_tier] < TIER_RANK[need_tier]
        under += short
        print(f"{case.request[:39]:<40} ${c_whole:>8.4f} ${correct:>10.4f} ${c_split:>8.4f}  "
              f"{'YES ' + whole_tier + '<' + need_tier if short else 'no'}")
    print("-" * 84)
    print(f"{'TOTAL':<40} ${tot_whole:>8.4f} ${tot_correct:>10.4f} ${tot_split:>8.4f}")
    print()
    print(f"  under-routing rate: {under}/{len(COMPOUND)} = {under / len(COMPOUND):.0%}")
    print("    routing whole sent the request to a tier below what its hardest")
    print("    sub-task needs. That is a correctness failure the qualification")
    print("    gate cannot catch, because the whole request looks easy.")
    print()
    if tot_split:
        print(f"  vs a CORRECT single call, decomposition is {tot_correct / tot_split:.1f}x cheaper")
        print(f"  vs the cheap MISLABELLED single call, it is {tot_whole / tot_split:.1f}x "
              f"({'more expensive' if tot_whole < tot_split else 'cheaper'})")
    print()
    print("  Read that as: decomposition is a CORRECTNESS mechanism first and a")
    print("  cost mechanism second. It surfaces the hard sub-task so the router")
    print("  can qualify it. The saving follows only when the whole request was")
    print("  going to be labelled hard anyway.")
    print()
    print("  planner overhead: $0.0000 — the heuristic makes no model calls.")
    print("  (the model layer's overhead is measured in examples/planner_ab.py)")

    # ---- 5. Determinism. Same request, same plan, or the traces stop comparing.
    stable = all(
        len({tuple(s.task_type for s in plan_heuristic(c.request).steps) for _ in range(5)}) == 1
        for c in ALL_CASES
    )
    print(f"\n  deterministic across repeated calls: {stable}")

    failed = rate > MAX_FALSE_SPLIT_RATE or coverage < MIN_SPLIT_COVERAGE or not stable
    print("\n" + "=" * 100)
    print(f"{'FAIL' if failed else 'PASS'}: planner evals "
          f"({len(ALL_CASES)} cases: {len(COMPOUND)} compound, {len(NOT_COMPOUND)} not)")
    return 1 if failed else 0


def _tier_of(step, registry: Registry) -> str:
    """Which tier this step actually routes to."""
    task = Task(prompt=step.prompt, task_type=step.task_type, complexity=step.complexity,
                est_input_tokens=step.est_input_tokens, est_output_tokens=step.est_output_tokens)
    return route(task, registry, BALANCED).chosen.tier


def _relabel(plan, task_type: str, complexity: float):
    """The same single step, labelled at a different requirement."""
    from dataclasses import replace as _replace

    step = _replace(plan.steps[0], task_type=task_type, complexity=complexity)
    return _replace(plan, steps=(step,))


def _whole_plan(request: str, est_in: int | None = None, est_out: int | None = None):
    """The same request as a single step — the no-decomposition baseline."""
    from switchboard.planner import no_split_plan

    return no_split_plan(
        request, "baseline: routed whole",
        est_input_tokens=est_in, est_output_tokens=est_out,
    )


if __name__ == "__main__":
    sys.exit(run())
