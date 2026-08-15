"""Measure the triage layers against each other on real APIs.

    PYTHONPATH=src python3 examples/triage_ab.py --catalog examples/starter_catalog.json

Costs roughly $0.001. Every classification runs on the cheapest model in the
catalog, because triage is a routing hint, not the work.

The question this answers: the offline heuristic scores 100% on the set it was
built against and 60% on a held-out set. Is a model worth calling instead? The
answer turned out to be "sometimes, and you can tell exactly when" — the
heuristic's own confidence separates its errors perfectly, so gating on it
beats either layer used alone.

Re-run this after any change to the keyword table or the classifier prompt.
The numbers in the README came from here.
"""

from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from triage_eval import HELD_OUT, LABELED  # noqa: E402

from switchboard import (  # noqa: E402
    Registry,
    actual_cost,
    classify_heuristic,
    classify_with_model,
    demo_registry,
)
from switchboard.providers.live import live_pool, usable_registry  # noqa: E402

THRESHOLDS = (0.0, 0.25, 0.4, 0.5, 0.7, 1.01)


def main() -> int:
    parser = argparse.ArgumentParser(description="A/B the triage layers on live APIs.")
    parser.add_argument("--catalog", help="catalog JSON (default: built-in demo)")
    args = parser.parse_args()

    registry = Registry.from_json(args.catalog) if args.catalog else demo_registry()
    pool, skipped = live_pool(registry)
    if not pool.names():
        print("No provider keys set — this experiment needs at least one.")
        return 1
    usable = usable_registry(registry, pool)
    cheapest = min(usable.all(), key=lambda m: (m.output_cost, m.input_cost))
    print(f"classifier model: {cheapest.model_id} "
          f"(${cheapest.input_cost}/${cheapest.output_cost} per 1M)\n")

    cases = [(p, e, "tuned") for p, e in LABELED] + [(p, e, "held-out") for p, e in HELD_OUT]
    rows = []
    started = time.time()
    for prompt, expected, setname in cases:
        heuristic = classify_heuristic(prompt)
        model = classify_with_model(prompt, usable, pool)
        rows.append((setname, expected, heuristic, model, prompt))
    elapsed = time.time() - started

    def call_cost(prompt: str) -> float:
        # Classifier prompt is the task plus a short rubric; reply is tiny.
        return actual_cost(cheapest, len(prompt) // 3 + 90, 25)

    print("where the two layers disagree:")
    for setname, expected, h, m, prompt in rows:
        if h.task_type == m.task_type:
            continue
        winner = ("model" if m.task_type == expected
                  else "heuristic" if h.task_type == expected else "neither")
        print(f"  [{winner:<9}] expected={expected:<14} heuristic={h.task_type:<14} "
              f"model={m.task_type:<14} {prompt[:44]}")

    print("\ncan the heuristic tell when it is wrong?")
    for label, want in (("correct", True), ("wrong", False)):
        confs = [h.confidence for _, e, h, _, _ in rows if (h.task_type == e) is want]
        if confs:
            print(f"  heuristic {label:<8} n={len(confs):<3} "
                  f"mean confidence {sum(confs)/len(confs):.2f}, max {max(confs):.2f}")

    print(f"\n{'threshold':>10} {'accuracy':>9} {'held-out':>9} {'model calls':>12} {'cost':>10}")
    for threshold in THRESHOLDS:
        correct = calls = held_ok = held_n = 0
        cost = 0.0
        for setname, expected, h, m, prompt in rows:
            use_model = h.confidence < threshold
            chosen = m.task_type if use_model else h.task_type
            if use_model:
                calls += 1
                cost += call_cost(prompt)
            correct += chosen == expected
            if setname == "held-out":
                held_n += 1
                held_ok += chosen == expected
        note = ("  heuristic only" if threshold == 0.0
                else "  model always" if threshold > 1.0 else "")
        print(f"{threshold:>10.2f} {correct/len(rows):>8.0%} {held_ok/held_n:>8.0%} "
              f"{calls:>12} ${cost:>9.5f}{note}")

    print(f"\n{len(rows)} prompts, {elapsed:.0f}s wall clock for the model pass "
          f"({elapsed/len(rows):.1f}s per classification).")
    print("Latency, not money, is what the model layer actually costs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
