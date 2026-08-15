"""Dispatch a multi-step agentic workflow through Switchboard.

    PYTHONPATH=src python3 examples/agentic_workflow.py
    PYTHONPATH=src python3 examples/agentic_workflow.py --catalog examples/starter_catalog.json
    PYTHONPATH=src python3 examples/agentic_workflow.py --catalog examples/starter_catalog.json --policy cost_first

What is real here and what is not, so the numbers are read correctly:

* The **routing** is real. Real catalog, real vendor prices, the real router.
  Which model each step lands on, and why, is exactly what production would do.
* The **execution** is mocked. `MockProvider` returns a canned string without
  calling anyone, so nothing here reflects real output quality, real latency,
  or real token counts.
* The **costs** are therefore *estimates at the token volumes you declared* on
  each Task (`est_input_tokens` / `est_output_tokens`) priced at real rates —
  a dispatch plan's budget, not a bill. Wire in a real provider and
  `BrokerResult.total_cost_usd` reports observed tokens instead, audits
  included.
"""

from __future__ import annotations

import argparse

from switchboard import (
    PRESETS,
    Broker,
    Registry,
    Task,
    demo_registry,
    estimate_cost,
    mock_pool,
)

# ---- EDIT ME: your workflow, one Task per step ------------------------------
# task_type: reasoning / coding / summarization / extraction / creative
# complexity: 0.0 (trivial) to 1.0 (very hard)
WORKFLOW = [
    ("1. Research: pull competitor pricing from 5 saved pages",
     Task(prompt="Extract plan names, prices and limits from these 5 competitor pages: ...",
          task_type="extraction", complexity=0.3, est_input_tokens=8000, est_output_tokens=1200)),
    ("2. Synthesize: summarize findings into a comparison brief",
     Task(prompt="Summarize the extracted pricing data into a one-page competitive brief.",
          task_type="summarization", complexity=0.5, est_input_tokens=3000, est_output_tokens=800)),
    ("3. Strategize: recommend our pricing response",
     Task(prompt="Given the brief, recommend a pricing response with tradeoffs and risks.",
          task_type="reasoning", complexity=0.85, est_input_tokens=2000, est_output_tokens=1500)),
    ("4. Write: draft the new pricing page copy",
     Task(prompt="Write landing-page copy for the new pricing, friendly but direct.",
          task_type="creative", complexity=0.6, est_input_tokens=1500, est_output_tokens=1000)),
    ("5. Integrate: generate the webhook code to update the website",
     Task(prompt="Write a webhook handler that updates the pricing table in the CMS via its API.",
          task_type="coding", complexity=0.65, est_input_tokens=1200, est_output_tokens=1500)),
]
# ------------------------------------------------------------------------------


def load_registry(path: str | None) -> tuple[Registry, str]:
    """Load a catalog from disk, or fall back to the synthetic demo one."""
    if path is None:
        return demo_registry(), "built-in demo catalog (synthetic prices)"
    registry = Registry.from_json(path)
    verified = registry.last_verified
    age = registry.age_in_days()
    provenance = f"{path} ({len(registry)} models"
    if verified is not None:
        provenance += f", prices verified {verified}, {age} days ago"
    provenance += ")"
    return registry, provenance


def best_model_for(registry: Registry, task: Task):
    """The model a team would reach for if it never thought about cost.

    This is the baseline the routing saving is measured against — "always use
    the strongest thing available", which is one of the two bad equilibria the
    README describes.
    """
    return max(registry.all(), key=lambda m: m.capability_for(task.task_type))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dispatch an agentic workflow through Switchboard.",
        epilog="Execution is mocked; routing and prices are real. See the module docstring.",
    )
    parser.add_argument(
        "--catalog",
        help="path to a catalog JSON (default: the built-in synthetic demo catalog)",
    )
    parser.add_argument(
        "--policy", default="balanced", choices=sorted(PRESETS),
        help="routing policy preset (default: balanced)",
    )
    parser.add_argument(
        "--trace", default="traces/workflow.jsonl", help="where to append JSONL decision traces"
    )
    args = parser.parse_args()

    registry, provenance = load_registry(args.catalog)
    policy = PRESETS[args.policy]
    # One offline mock per provider the catalog names, so a real catalog
    # runs end to end — cross-lab audits included — without any API keys.
    broker = Broker(registry, mock_pool(registry), policy, trace_path=args.trace)

    print("=== Agentic workflow dispatch ===")
    print(f"catalog: {provenance}")
    print(f"policy:  {policy.name}")
    print("costs:   estimated at each step's declared token volumes, at real catalog prices")
    print("         (execution is mocked — routing is real, output quality is not)\n")

    rows = []
    routed_cost = baseline_cost = 0.0
    for label, task in WORKFLOW:
        result = broker.run(task)
        chosen = registry.get(result.final_model)
        step_cost = estimate_cost(task, chosen)
        step_baseline = estimate_cost(task, best_model_for(registry, task))
        routed_cost += step_cost
        baseline_cost += step_baseline
        rows.append((label, result, chosen, step_cost))

        print(label)
        print(f"   -> dispatched to {result.final_model} ({chosen.tier}, {chosen.provider}), "
              f"est. ${step_cost:.4f}")
        if result.failed_over:
            print("   -> provider outage, rerouted")
        if result.escalated:
            print(f"   -> audit failed, escalated: "
                  f"{' -> '.join(a.model_id for a in result.attempts)}")
        if result.underqualified:
            print("   -> WARNING: no model in this catalog is rated for this task")
        audit = result.attempts[-1]
        if audit.auditor_model:
            print(f"   -> audited by {audit.auditor_model} "
                  f"({'cross-lab' if audit.cross_lab_audit else 'SAME-LAB'})")
        print(f"   -> verified: {result.verified}")
        print(f"   -> why: {result.routing_rationale}\n")

    print("=" * 78)
    print("DISPATCH PLAN SUMMARY")
    for label, result, chosen, cost in rows:
        flag = "OK " if result.verified else "FLAG"
        print(f"  [{flag}] {label:<56} {chosen.model_id:<26} ${cost:.4f}")
    print("-" * 78)
    saved = baseline_cost - routed_cost
    pct = 100 * saved / baseline_cost if baseline_cost else 0.0
    print(f"  Routed pipeline, estimated:       ${routed_cost:.4f}")
    print(f"  Best-model-every-step, estimated: ${baseline_cost:.4f}")
    print(f"  Estimated saving from routing:    ${saved:.4f}  ({pct:.0f}%)")
    print()
    print("  Estimates exclude audit cost, which roughly doubles the audited path;")
    print("  BrokerResult.total_cost_usd accounts for it on real runs.")
    print("  Any step marked FLAG needs human review before shipping.")
    print(f"  Full decision log: {args.trace}")


if __name__ == "__main__":
    main()
