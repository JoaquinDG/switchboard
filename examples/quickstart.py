"""Switchboard quickstart — runs fully offline, no API keys needed.

    PYTHONPATH=src python3 examples/quickstart.py

Swap MockProvider for AnthropicProvider / OpenAICompatibleProvider and load
your own catalog (see catalog.example.json) to run against real APIs.
"""

from switchboard import (
    BALANCED,
    COST_FIRST,
    QUALITY_FIRST,
    Broker,
    ModelSpec,
    MockProvider,
    ProviderPool,
    ProviderUnavailable,
    Registry,
    ScriptedProvider,
    Task,
    demo_registry,
)


class OtherLabProvider(MockProvider):
    """A second vendor, so the cross-lab auditor has somewhere to cross to."""

    name = "otherlab"


def show(title: str, result) -> None:
    print(f"\n=== {title} ===")
    print(f"routing:   {result.routing_rationale}")
    print(f"model:     {result.final_model}")
    print(
        f"verified:  {result.verified}   escalated: {result.escalated}   "
        f"failed over: {result.failed_over}   underqualified: {result.underqualified}"
    )
    for i, a in enumerate(result.attempts, 1):
        detail = f"error={a.error}" if a.error else f"audit_passed={a.audit_passed}"
        if a.auditor_model:
            lab = "cross-lab" if a.cross_lab_audit else "same-lab"
            detail += f" by {a.auditor_model} ({lab})"
        if a.had_audit_feedback:
            detail += " [given prior audit findings]"
        print(
            f"  attempt {i}: {a.model_id} ({a.tier}, {a.role}) {detail} "
            f"${a.total_cost_usd:.6f}"
        )
    print(
        f"cost:      ${result.total_cost_usd:.6f} "
        f"(generation ${result.generation_cost_usd:.6f} + audit ${result.audit_cost_usd:.6f})"
    )
    print(
        f"baseline:  ${result.baseline_cost_usd:.6f} on {result.baseline_model} "
        f"-> saved ${result.savings_vs_baseline_usd:+.6f}"
    )
    for w in result.warnings:
        print(f"  ! {w}")


def main() -> None:
    broker = Broker(
        registry=demo_registry(),
        providers=ProviderPool([MockProvider()]),
        policy=BALANCED,
        trace_path="traces/quickstart.jsonl",
    )

    # 1. An easy task routes cheap and passes audit.
    show(
        "Easy extraction",
        broker.run(Task(prompt="Extract all dates from: meeting on Jan 5, follow-up Feb 12.",
                        task_type="extraction", complexity=0.2)),
    )

    # 2. A hard task hits the frontier gate.
    show(
        "Hard reasoning",
        broker.run(Task(prompt="Design a migration plan with rollback for a live billing system.",
                        task_type="reasoning", complexity=0.9)),
    )

    # 3. A failed audit triggers escalation (the mock provider is told to fail).
    cost_broker = Broker(demo_registry(), ProviderPool([MockProvider()]), COST_FIRST,
                         trace_path="traces/quickstart.jsonl")
    show(
        "Failed audit -> escalation",
        cost_broker.run(Task(prompt="FORCE_AUDIT_FAIL summarize this contract",
                             task_type="summarization", complexity=0.3)),
    )

    # 4. A task type the catalog has never been scored on. Nothing qualifies,
    #    so routing degrades *upward* and flags the result rather than letting
    #    cost weight quietly hand unrated work to the cheapest model.
    quality_broker = Broker(demo_registry(), ProviderPool([MockProvider()]), QUALITY_FIRST,
                            trace_path="traces/quickstart.jsonl")
    show(
        "Unscored task type -> conservative route + warning",
        quality_broker.run(Task(prompt="Review this lease for unusual clauses.",
                                task_type="legal_analysis", complexity=0.75)),
    )

    # 5. A provider outage is an availability problem, not a quality one: the
    #    task reroutes to the next-ranked model without spending escalation
    #    budget. ScriptedProvider injects the outage offline.
    flaky = ScriptedProvider(
        {
            "atlas-small": [ProviderUnavailable("atlas-small: 503 from upstream")],
            "atlas-mid": ["[atlas-mid] completed after failover"],
            "atlas-frontier": ['{"pass": true, "score": 0.92, "issues": []}'],
        },
        name="mock",
    )
    show(
        "Provider outage -> failover",
        Broker(demo_registry(), ProviderPool([flaky]), BALANCED,
               trace_path="traces/quickstart.jsonl").run(
            Task(prompt="Extract the invoice totals.", task_type="extraction", complexity=0.2)
        ),
    )

    # 6. Audit independence has two degrees. Every audit above used a
    #    different *model* — but the demo catalog is one synthetic vendor, so
    #    they were all same-lab, and the attempts say so. Add a second lab and
    #    the auditor crosses to it, because models from one lab share training
    #    data and alignment: their blind spots correlate.
    two_lab = Registry([
        ModelSpec("lab-a-frontier", "mock", "frontier", 3.0, 15.0,
                  capabilities={"summarization": 0.9, "audit": 0.9}),
        ModelSpec("lab-b-frontier", "otherlab", "frontier", 3.0, 15.0,
                  capabilities={"summarization": 0.9, "audit": 0.9}),
    ])
    show(
        "Two labs in the catalog -> cross-lab audit",
        Broker(two_lab, ProviderPool([MockProvider(), OtherLabProvider()]), BALANCED,
               trace_path="traces/quickstart.jsonl").run(
            Task(prompt="Summarize the Q3 board memo.", task_type="summarization",
                 complexity=0.5)
        ),
    )

    print("\nDecision traces written to traces/quickstart.jsonl")


if __name__ == "__main__":
    main()
