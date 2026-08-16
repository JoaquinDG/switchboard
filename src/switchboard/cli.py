"""A one-line way to see Switchboard route, run, and price a single prompt.

    switchboard "extract every date from this email"

This is the 30-second first experience the README promises: no catalog file,
no provider keys, no script to write. It runs the exact same pipeline as
`Broker.run` — triage, routing, auditing, escalation — through the built-in
demo catalog and `MockProvider`, which costs nothing and touches no network.

Dry-run by default, the same rule `examples/live_run.py` uses: nothing is
billed to a real vendor unless `--live` is passed, and `--live` still refuses
to run without both a provider key and an explicit `--budget-usd` cap.
"""

from __future__ import annotations

import argparse
import sys

from .broker import Broker, BrokerResult
from .policies import PRESETS, Task
from .providers.base import Completion, ProviderError, mock_pool
from .providers.live import KNOWN_PROVIDERS, live_pool, usable_registry
from .registry import Registry, demo_registry
from .router import actual_cost


class _CappedProvider:
    """Forces a max_tokens ceiling on every call to a real vendor.

    `Broker._complete` does not pass `max_tokens`, so a live call would
    otherwise run at the provider's own default. Output tokens dominate the
    bill, so the ceiling is imposed here at the edge rather than by widening
    the library's API for one CLI flag — the same approach live_run.py uses.
    """

    def __init__(self, inner, max_tokens: int) -> None:
        self.inner = inner
        self.max_tokens = max_tokens

    @property
    def name(self) -> str:
        return self.inner.name

    def complete(self, model_id: str, prompt: str, max_tokens: int = 1024) -> Completion:
        return self.inner.complete(model_id, prompt, min(max_tokens, self.max_tokens))


def _worst_case_cost(task: Task, registry: Registry, policy, max_tokens: int) -> float:
    """Ceiling for one task: priciest model, every escalation, plus audits.

    Deliberately pessimistic, mirroring `examples/live_run.py`'s guard — a
    budget check that assumes the expected case is not a guard.
    """
    priciest = max(registry.all(), key=lambda m: m.output_cost)
    one_call = actual_cost(priciest, task.est_input_tokens, max_tokens)
    attempts = 1 + policy.max_escalations
    audits = attempts if policy.audit_enabled else 0
    audit_call = actual_cost(priciest, task.est_input_tokens + max_tokens + 200, 120)
    return one_call * attempts + audit_call * audits


def _print_result(result: BrokerResult, *, mocked: bool) -> None:
    if result.triage:
        print(f"triage:   {result.triage.task_type} "
              f"@ {result.triage.complexity} ({result.triage.source})")
    print(f"routing:  {result.routing_rationale}")
    print(f"model:    {result.final_model}")
    print(f"verified: {result.verified}   escalated: {result.escalated}   "
          f"failed over: {result.failed_over}")
    for w in result.warnings:
        print(f"  ! {w}")
    print(f"cost:     ${result.total_cost_usd:.6f} "
          f"(generation ${result.generation_cost_usd:.6f} + "
          f"audit ${result.audit_cost_usd:.6f})")
    print(f"baseline: ${result.baseline_cost_usd:.6f} on {result.baseline_model} "
          f"-> saved ${result.savings_vs_baseline_usd:+.6f}")
    label = "MOCKED output (no real model was called)" if mocked else "output"
    print(f"\n{label}:")
    print(result.final_text.strip())
    if mocked:
        print("\n(offline demo: the text above is a canned MockProvider stand-in — "
              "routing, auditing and cost above are the real logic. Pass --live to "
              "spend real money on a real vendor.)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="switchboard",
        description="Route one prompt through Switchboard and print what it decided.",
        epilog="Dry-run by default (MockProvider: zero cost, zero network). "
               "Pass --live to spend real money on a real vendor.",
    )
    parser.add_argument("prompt", nargs="+", help="the task to route")
    parser.add_argument("--policy", default="balanced", choices=sorted(PRESETS),
                        help="routing policy preset (default: balanced)")
    parser.add_argument("--task-type", default="auto",
                        help="skip triage and force a task type "
                             "(default: auto, triage decides)")
    parser.add_argument("--catalog", help="catalog JSON (default: built-in demo catalog)")
    parser.add_argument("--live", action="store_true",
                        help="call a real vendor API instead of the offline mock; "
                             "requires a provider key and --budget-usd")
    parser.add_argument("--budget-usd", type=float, default=0.0,
                        help="hard spend cap for --live (must be > 0 to run live)")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="output ceiling per call in --live mode (default: 1024)")
    parser.add_argument("--trace", help="append a decision trace to this JSONL path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    prompt = " ".join(args.prompt)
    policy = PRESETS[args.policy]
    registry = Registry.from_json(args.catalog) if args.catalog else demo_registry()
    task = Task(prompt=prompt, task_type=args.task_type)

    if args.live:
        if args.budget_usd <= 0:
            print("--live requires --budget-usd > 0 (a hard spend cap).")
            return 1
        pool, skipped = live_pool(registry)
        if skipped:
            print("Providers skipped for want of a key:")
            for name in skipped:
                spec = KNOWN_PROVIDERS.get(name)
                env = spec.env_var if spec else "unknown"
                print(f"  {name:<12} set {env}")
        if not pool.names():
            print("No provider keys are set. Export at least one of: " +
                  ", ".join(s.env_var for s in KNOWN_PROVIDERS.values()))
            return 1
        registry = usable_registry(registry, pool)
        if not registry.all():
            print("Keys are set, but no catalog model belongs to a keyed provider.")
            return 1
        ceiling = _worst_case_cost(task, registry, policy, args.max_tokens)
        if ceiling > args.budget_usd:
            print(f"STOP: worst case ${ceiling:.4f} exceeds the "
                  f"${args.budget_usd:.2f} budget. Raise --budget-usd or lower --max-tokens.")
            return 1
        pool = type(pool)([_CappedProvider(pool.get(n), args.max_tokens) for n in pool.names()])
        mocked = False
    else:
        pool = mock_pool(registry)
        mocked = True

    broker = Broker(registry, pool, policy, trace_path=args.trace)
    try:
        result = broker.run(task)
    except ProviderError as e:
        print(f"routing failed: {e}")
        return 1

    _print_result(result, mocked=mocked)
    return 0


if __name__ == "__main__":
    sys.exit(main())
