"""Measure the planner layers against each other on real APIs.

    PYTHONPATH=src python3 examples/planner_ab.py --catalog examples/starter_catalog.json

Costs a fraction of a cent. Planning runs on the cheapest model in the catalog,
because deciding how to split a job is a classification, not the job.

The same shape as `triage_ab.py`, and asking the same question: the heuristic
is free and offline, the model layer costs money and latency, and the useful
answer is not "which is better" but "where is each one right, and can the
heuristic tell you which case you are in".

For the planner the stakes are asymmetric in a way triage's are not. A wrong
task type routes one call to a slightly wrong model. A wrong *split* multiplies
calls, audits and latency — or silently drops half the request. So false splits
are reported first here too.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))

from planner_cases import ALL_CASES, COMPOUND, NOT_COMPOUND  # noqa: E402

from switchboard import BALANCED, Registry, actual_cost, demo_registry  # noqa: E402
from switchboard.planner import plan_heuristic, plan_with_model  # noqa: E402
from switchboard.providers.live import live_pool, usable_registry  # noqa: E402

THRESHOLDS = (0.0, 0.25, 0.5, 1.01)


def main() -> int:
    parser = argparse.ArgumentParser(description="A/B the planner layers on live APIs.")
    parser.add_argument("--catalog", help="catalog JSON (default: built-in demo)")
    args = parser.parse_args()

    registry = Registry.from_json(args.catalog) if args.catalog else demo_registry()
    pool, _ = live_pool(registry)
    if not pool.names():
        print("No provider keys set — this experiment needs at least one.")
        return 1
    usable = usable_registry(registry, pool)
    cheapest = min(usable.all(), key=lambda m: (m.output_cost, m.input_cost))
    print(f"planner model: {cheapest.model_id} "
          f"(${cheapest.input_cost}/${cheapest.output_cost} per 1M)\n")

    rows = []
    started = time.time()
    for case in ALL_CASES:
        heuristic = plan_heuristic(case.request)
        model, discarded = plan_with_model(
            case.request, usable, pool, BALANCED, fallback=heuristic
        )
        cost = sum(d.get("cost_usd", 0.0) for d in discarded)
        if model.planned_by.startswith("model"):
            cost += actual_cost(cheapest, len(case.request) // 3 + 260, 220)
        rows.append((case, heuristic, model, cost, discarded))
    elapsed = time.time() - started

    print("where the two layers disagree on WHETHER to split:")
    for case, heuristic, model, _, _ in rows:
        if heuristic.is_split == model.is_split:
            continue
        truth = "compound" if case.should_split else "NOT compound"
        winner = ("model" if model.is_split == case.should_split else "heuristic")
        print(f"  [{winner:<9} right] {truth:<12} heuristic={'split' if heuristic.is_split else 'keep':<5} "
              f"model={'split' if model.is_split else 'keep':<5} {case.request[:40]}")

    rejected = [(c, d) for c, _, _, _, d in rows if d]
    if rejected:
        print(f"\nmodel plans rejected by validation: {len(rejected)}")
        for case, discarded in rejected[:5]:
            for attempt in discarded:
                tag = "repair" if attempt.get("repair") else "first"
                print(f"  [{tag}] {attempt['reason'][:82]}")

    print(f"\n{'threshold':>10} {'false splits':>13} {'coverage':>10} {'model calls':>12} {'cost':>10}")
    for threshold in THRESHOLDS:
        false_splits = covered = calls = 0
        cost = 0.0
        for case, heuristic, model, call_cost, _ in rows:
            use_model = heuristic.confidence < threshold
            chosen = model if use_model else heuristic
            if use_model:
                calls += 1
                cost += call_cost
            if case.should_split:
                covered += chosen.is_split
            else:
                false_splits += chosen.is_split
        note = ("  heuristic only" if threshold == 0.0
                else "  model always" if threshold > 1.0 else "")
        print(f"{threshold:>10.2f} {false_splits:>10}/{len(NOT_COMPOUND)} "
              f"{covered:>7}/{len(COMPOUND)} {calls:>12} ${cost:>9.5f}{note}")

    print(f"\n{len(rows)} requests, {elapsed:.0f}s for the model pass "
          f"({elapsed / len(rows):.1f}s per plan).")
    print("As with triage, latency is the real cost of the model layer, not money.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
