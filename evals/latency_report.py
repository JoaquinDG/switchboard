"""Measured latency classes: what does observed wall-clock time say?

`ModelSpec.latency` is `fast` / `medium` / `slow`, assigned by tier when the
catalog was written and never checked against anything real — yet latency is
a full third of the `balanced` policy's weighting (ROADMAP item 11). Every
`Broker.run()` now times each generation attempt and writes it to the trace
as `latency_ms` (see `Attempt.latency_ms` in `broker.py`). This script reads
`traces/*.jsonl`, computes observed p50/p95 wall-clock latency per model, and
reports it next to the catalog's assigned class — it never writes anything
back.

Usage:
    PYTHONPATH=src python3 evals/latency_report.py
    PYTHONPATH=src python3 evals/latency_report.py --traces traces --catalog examples/starter_catalog.json
    PYTHONPATH=src python3 evals/latency_report.py --min-sample 10

Traces are gitignored, so a fresh checkout has none — this prints "no traces
found" rather than fabricating a report. Generate real ones with:

    PYTHONPATH=src python3 examples/live_run.py --catalog examples/starter_catalog.json --live

Running the offline examples (quickstart.py, agentic_workflow.py) also
appends to traces/*.jsonl, but MockProvider returns in microseconds — that is
a measurement of a dict lookup, not of a vendor. Every attempt in a trace
carries a `synthetic` flag for exactly this reason (see `Provider.synthetic`
in providers/base.py); this script drops any attempt where that flag is true
or missing before it touches the aggregates below, and it refuses to print
p50/p95 for a model with zero real samples rather than silently reporting
mock timings as if they meant something.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from switchboard import Registry

# Below this many real samples, a percentile is noise dressed as a number.
# Latency is noisier call-to-call than audit pass/fail, so this asks for
# fewer samples than catalog_feedback.py's threshold, not more.
DEFAULT_MIN_SAMPLE = 10


def _percentile(sorted_samples: list[float], p: float) -> float:
    """Linear-interpolation percentile. `sorted_samples` must be non-empty and sorted."""
    if len(sorted_samples) == 1:
        return sorted_samples[0]
    rank = p * (len(sorted_samples) - 1)
    lo, hi = int(rank), min(int(rank) + 1, len(sorted_samples) - 1)
    frac = rank - lo
    return sorted_samples[lo] + (sorted_samples[hi] - sorted_samples[lo]) * frac


@dataclass
class LatencyObservation:
    model_id: str
    samples: list[float] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.samples)

    @property
    def p50_ms(self) -> float:
        return _percentile(sorted(self.samples), 0.50)

    @property
    def p95_ms(self) -> float:
        return _percentile(sorted(self.samples), 0.95)


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


def aggregate(records: list[dict]) -> tuple[dict[str, LatencyObservation], dict[str, int]]:
    """Fold trace records into per-model latency observations.

    Returns the observations plus a count of how many attempts were seen and
    why each excluded one was dropped, so the report can say what it left out
    instead of quietly looking more complete than it is.
    """
    observations: dict[str, LatencyObservation] = {}
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        for attempt in record.get("attempts", []):
            counts["attempts_seen"] += 1
            # Fail closed: no flag at all is not evidence the call was real.
            if attempt.get("synthetic", True):
                counts["synthetic"] += 1
                continue
            if attempt.get("error"):
                # An errored call measures time-to-failure (a timeout, a
                # rejected request), not the response latency the routing
                # weight is trying to model. Counted, not folded in.
                counts["errored"] += 1
                continue
            latency_ms = attempt.get("latency_ms")
            if latency_ms is None:
                counts["missing_latency"] += 1  # trace predates this field
                continue
            model_id = attempt.get("model_id")
            if not model_id:
                counts["missing_model_id"] += 1
                continue
            obs = observations.setdefault(model_id, LatencyObservation(model_id))
            obs.samples.append(latency_ms)
            counts["measured"] += 1
    return observations, counts


def load_catalog_latency_class(path: str | None) -> dict[str, str]:
    """model_id -> the catalog's assigned latency class, if a catalog was given."""
    if not path:
        return {}
    registry = Registry.from_json(path)
    return {m.model_id: m.latency for m in registry.all()}


def print_report(
    observations: dict[str, LatencyObservation],
    catalog_class: dict[str, str],
    min_sample: int,
) -> None:
    print(f"{'model':<28} {'n':>5} {'p50_ms':>10} {'p95_ms':>10} {'catalog class':>14}  note")
    print("-" * 90)
    for model_id in sorted(observations):
        obs = observations[model_id]
        cls = catalog_class.get(model_id, "n/a")
        note = "" if obs.n >= min_sample else f"n<{min_sample}: too small to act on"
        print(
            f"{obs.model_id:<28} {obs.n:>5} {obs.p50_ms:>10.0f} {obs.p95_ms:>10.0f} "
            f"{cls:>14}  {note}"
        )
    print("-" * 90)


def run() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--traces", default="traces", help="directory of *.jsonl traces, or a single file")
    parser.add_argument("--catalog", default=None, help="catalog JSON to show the assigned class alongside")
    parser.add_argument("--min-sample", type=int, default=DEFAULT_MIN_SAMPLE)
    args = parser.parse_args()

    traces_dir = Path(args.traces)
    if not traces_dir.exists():
        print(f"no traces found at {traces_dir} — nothing to report.")
        print("generate some with: PYTHONPATH=src python3 examples/live_run.py --live")
        return 0

    records = load_records(traces_dir)
    if not records:
        print(f"{traces_dir} exists but contains no trace records — nothing to report.")
        return 0

    observations, counts = aggregate(records)
    catalog_class = load_catalog_latency_class(args.catalog)

    print(f"read {len(records)} trace record(s) from {traces_dir}")
    print(
        f"attempts seen: {counts['attempts_seen']}  "
        f"measured: {counts['measured']}  "
        f"excluded — synthetic: {counts['synthetic']}, "
        f"errored: {counts['errored']}, "
        f"missing latency_ms: {counts['missing_latency']}"
    )
    print()

    if not observations:
        print("every attempt was synthetic, errored, or predates latency_ms — no measured "
              "data to report.")
        print("this is expected for traces from quickstart.py / agentic_workflow.py "
              "(MockProvider returns instantly, which is not a latency measurement);")
        print("it is not expected for traces from `live_run.py --live`.")
        return 0

    print_report(observations, catalog_class, args.min_sample)

    confident = [o for o in observations.values() if o.n >= args.min_sample]
    print()
    print(f"{len(confident)}/{len(observations)} model(s) clear n>={args.min_sample}; "
          f"the rest are directional at best.")
    print(
        "\nThis table is p50/p95 of REAL provider wall-clock time only — synthetic "
        "(MockProvider/ScriptedProvider) attempts are excluded before a single number is "
        "computed, per the trap in ROADMAP.md item 11. It does not update the catalog; "
        "reclassifying a model's `latency` field (fast/medium/slow in registry.py) from "
        "these numbers is a manual, by-hand judgement call once a model clears "
        "--min-sample, the same way item 1's catalog_feedback.py leaves capability scores "
        "to a human."
    )
    return 0


if __name__ == "__main__":
    sys.exit(run())
