"""Shadow-routing report: what would a second policy have chosen?

ROADMAP item 8. `Broker(..., shadow_policy=<Policy>)` scores every task under
a SECOND policy using the exact same routing math (`router.route`) the real
decision uses — but the shadow decision is never executed, never audited,
and never allowed to influence the real choice. It is pure arithmetic over
the registry already loaded in memory, so it costs nothing extra to compute.
`Broker._trace` records both sides of every task: the real decision's own
estimated cost (`chosen_est_cost_usd`) next to what the shadow policy would
have chosen and its estimated cost (`shadow_chosen_model` /
`shadow_est_cost_usd`), plus whether the two agree (`shadow_agrees`).

This script reads `traces/*.jsonl`, keeps the records where a shadow policy
actually ran, and reports how often the two policies agree and how their
ESTIMATED costs compare. It stops there on purpose. The shadow decision is
never executed, so it has no audit verdict, no escalation, no observed
token count on its side — only the real, executed decision has any of that.
Reporting a "shadow would have saved you $X" as anything other than an
estimate-vs-estimate comparison would be exactly the honesty trap this
mechanism exists to avoid: agreeing on a MODEL is not the same claim as
agreeing on an OUTCOME.

Usage:
    PYTHONPATH=src python3 evals/shadow_routing_report.py
    PYTHONPATH=src python3 evals/shadow_routing_report.py --traces traces

Traces are gitignored, so a fresh checkout has none. Generate some by
constructing a Broker with `shadow_policy` set and running tasks through it
— `examples/quickstart.py` and the live examples both work, once the
`shadow_policy` argument is threaded through by whoever calls them; this
script does not do that wiring itself, since the point of the feature is
that a caller opts a specific Policy comparison in.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from switchboard.replay import task_records


@dataclass
class PairStats:
    real_policy: str
    shadow_policy: str
    n: int = 0
    agree: int = 0
    # shadow's estimated cost minus the real decision's own estimated cost,
    # summed. Positive means the shadow policy would have spent more.
    cost_delta_sum: float = 0.0
    cost_delta_samples: int = 0

    @property
    def agree_rate(self) -> float:
        return self.agree / self.n if self.n else 0.0

    @property
    def mean_cost_delta(self) -> float | None:
        if not self.cost_delta_samples:
            return None
        return self.cost_delta_sum / self.cost_delta_samples


def load_records(traces_dir: Path) -> list[dict]:
    records = []
    files = [traces_dir] if traces_dir.is_file() else sorted(traces_dir.glob("*.jsonl"))
    for path in files:
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"warning: {path}:{lineno}: skipping malformed line ({e})", file=sys.stderr)
    return records


def aggregate(shadow_records: list[dict]) -> dict[tuple[str, str], PairStats]:
    pairs: dict[tuple[str, str], PairStats] = {}
    for record in shadow_records:
        real_policy = record.get("policy") or "?"
        shadow_policy = record.get("shadow_policy") or "?"
        stats = pairs.setdefault(
            (real_policy, shadow_policy), PairStats(real_policy, shadow_policy)
        )
        stats.n += 1
        if record.get("shadow_agrees"):
            stats.agree += 1
        chosen_cost = record.get("chosen_est_cost_usd")
        shadow_cost = record.get("shadow_est_cost_usd")
        if chosen_cost is not None and shadow_cost is not None:
            stats.cost_delta_sum += shadow_cost - chosen_cost
            stats.cost_delta_samples += 1
    return pairs


def print_report(pairs: dict[tuple[str, str], PairStats]) -> None:
    print(f"{'real policy':<16} {'shadow policy':<16} {'n':>5} {'agree':>8} "
          f"{'mean est. cost delta':>22}")
    print("-" * 72)
    for key in sorted(pairs):
        s = pairs[key]
        delta = s.mean_cost_delta
        delta_s = "n/a" if delta is None else f"{delta:+.6f}"
        print(f"{s.real_policy:<16} {s.shadow_policy:<16} {s.n:>5} "
              f"{s.agree_rate:>8.0%} {delta_s:>22}")
    print("-" * 72)


def run() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--traces", default="traces", help="directory of *.jsonl traces, or a single file")
    args = parser.parse_args()

    traces_dir = Path(args.traces)
    if not traces_dir.exists():
        print(f"no traces found at {traces_dir} — nothing to report.")
        print("construct Broker(..., shadow_policy=<Policy>) to start collecting shadow data.")
        return 0

    records = load_records(traces_dir)
    if not records:
        print(f"{traces_dir} exists but contains no trace records — nothing to report.")
        return 0

    tasks = task_records(records)
    shadow_tasks = [r for r in tasks if r.get("shadow_chosen_model") is not None]

    print(f"read {len(tasks)} task record(s) from {traces_dir}; "
          f"{len(shadow_tasks)} ran with a shadow policy")
    print()

    if not shadow_tasks:
        print("none of these runs configured a shadow policy — nothing to compare.")
        print("this is expected unless a caller passed shadow_policy=... to Broker().")
        return 0

    pairs = aggregate(shadow_tasks)
    print_report(pairs)

    print(
        "\nmean est. cost delta = mean(shadow's estimated cost − the real decision's\n"
        "own estimated cost), both priced at routing time before anything ran.\n"
        "Positive means the shadow policy would have spent more; negative, less.\n"
        "This is NOT a verified saving: the shadow decision never ran, so neither\n"
        "figure reflects what actually happened after routing — audits, escalation,\n"
        "or provider failover. Agreement is about which MODEL a policy would pick,\n"
        "not about which one would have produced the better OUTCOME."
    )
    return 0


if __name__ == "__main__":
    sys.exit(run())
