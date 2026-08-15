"""Trace-driven catalog feedback: what did audits actually say about each model?

The starter catalog's capability scores are estimates — the file says so in
block capitals. Every real run through the Broker writes a JSONL trace that
records, per attempt, which model produced the output and whether a *different*
model's audit passed it. That is the raw material for a measured number, and
it has been sitting unused in `traces/`.

This script reads `traces/*.jsonl`, aggregates observed audit pass-rate and
mean audit score per (model, task_type), and prints it next to the catalog's
current estimate when one is available. It never writes anything back — see
"closing the loop" below for why, and what to do by hand instead.

Usage:
    PYTHONPATH=src python3 evals/catalog_feedback.py
    PYTHONPATH=src python3 evals/catalog_feedback.py --traces traces --catalog examples/starter_catalog.json
    PYTHONPATH=src python3 evals/catalog_feedback.py --min-sample 15

Traces are gitignored, so a fresh checkout has none — this prints "no traces
found" rather than fabricating a report. Generate real ones first:

    PYTHONPATH=src python3 examples/live_check.py --catalog examples/starter_catalog.json
    PYTHONPATH=src python3 examples/live_run.py --catalog examples/starter_catalog.json --live

Running the offline examples (quickstart.py, agentic_workflow.py) also
appends to traces/*.jsonl, but their audit verdicts come from MockProvider's
canned grader — always a flat 0.90 pass or 0.35 fail, never a judgement about
the text it was handed. Every attempt in a trace carries a `synthetic` flag
for exactly this reason (see `Provider.synthetic` in providers/base.py); this
script drops any attempt where that flag is true, or missing, before it
touches the aggregates below. A trace with no synthetic flag at all predates
that field and is treated the same way: not proof of anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from switchboard import Registry

# Below this many audited samples, a pass-rate is noise dressed as a number.
# Chosen the same way as the rest of the repo's thresholds: a round number
# that is honest about being a guess, not a statistically derived one.
DEFAULT_MIN_SAMPLE = 15


@dataclass
class Observation:
    model_id: str
    task_type: str
    n: int = 0
    passes: int = 0
    score_sum: float = 0.0

    @property
    def pass_rate(self) -> float:
        return self.passes / self.n if self.n else 0.0

    @property
    def mean_score(self) -> float:
        return self.score_sum / self.n if self.n else 0.0


def load_records(traces_dir: Path) -> list[dict]:
    records = []
    if traces_dir.is_file():
        files = [traces_dir]
    else:
        files = sorted(traces_dir.glob("*.jsonl"))
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


def aggregate(records: list[dict]) -> tuple[dict[tuple[str, str], Observation], dict[str, int]]:
    """Fold trace records into per-(model, task_type) observations.

    Returns the observations plus a count of how many attempts were seen and
    why each excluded one was dropped, so the report can say what it left out
    instead of quietly looking more complete than it is.
    """
    observations: dict[tuple[str, str], Observation] = {}
    counts = defaultdict(int)
    for record in records:
        task_type = record.get("task_type")
        for attempt in record.get("attempts", []):
            counts["attempts_seen"] += 1
            audit_passed = attempt.get("audit_passed")
            if audit_passed is None:
                counts["unaudited"] += 1
                continue
            # Fail closed: no flag at all is not evidence the run was real.
            if attempt.get("synthetic", True):
                counts["synthetic"] += 1
                continue
            if not task_type:
                counts["missing_task_type"] += 1
                continue
            model_id = attempt.get("model_id")
            if not model_id:
                counts["missing_model_id"] += 1
                continue
            key = (model_id, task_type)
            obs = observations.setdefault(key, Observation(model_id, task_type))
            obs.n += 1
            obs.passes += 1 if audit_passed else 0
            audit_score = attempt.get("audit_score")
            if audit_score is not None:
                obs.score_sum += audit_score
            counts["scored"] += 1
    return observations, counts


def load_catalog_capability(path: str | None) -> dict[tuple[str, str], float]:
    """model_id, task_type -> current catalog capability, if a catalog was given."""
    if not path:
        return {}
    registry = Registry.from_json(path)
    return {
        (m.model_id, task_type): score
        for m in registry.all()
        for task_type, score in m.capabilities.items()
    }


def print_report(
    observations: dict[tuple[str, str], Observation],
    catalog_capability: dict[tuple[str, str], float],
    min_sample: int,
) -> None:
    print(f"{'model':<28} {'task_type':<16} {'n':>5} {'pass_rate':>10} {'mean_score':>11} "
          f"{'catalog':>8}  note")
    print("-" * 100)
    for key in sorted(observations):
        obs = observations[key]
        current = catalog_capability.get(key)
        current_s = f"{current:.2f}" if current is not None else "n/a"
        note = "" if obs.n >= min_sample else f"n<{min_sample}: too small to act on"
        print(
            f"{obs.model_id:<28} {obs.task_type:<16} {obs.n:>5} "
            f"{obs.pass_rate:>10.0%} {obs.mean_score:>11.2f} {current_s:>8}  {note}"
        )
    print("-" * 100)


def run() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--traces", default="traces", help="directory of *.jsonl traces, or a single file")
    parser.add_argument("--catalog", default=None, help="catalog JSON to show alongside observed numbers")
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
    catalog_capability = load_catalog_capability(args.catalog)

    print(f"read {len(records)} trace record(s) from {traces_dir}")
    print(
        f"attempts seen: {counts['attempts_seen']}  "
        f"scored: {counts['scored']}  "
        f"excluded — synthetic: {counts['synthetic']}, "
        f"unaudited: {counts['unaudited']}, "
        f"no task_type: {counts['missing_task_type']}"
    )
    print()

    if not observations:
        print("every attempt was synthetic or unaudited — no measured data to report.")
        print("this is expected for traces from quickstart.py / agentic_workflow.py;")
        print("it is not expected for traces from `live_run.py --live`.")
        return 0

    print_report(observations, catalog_capability, args.min_sample)

    confident = [o for o in observations.values() if o.n >= args.min_sample]
    print()
    print(f"{len(confident)}/{len(observations)} (model, task_type) pairs clear "
          f"n>={args.min_sample}; the rest are directional at best.")

    print(
        "\nClosing the loop (by hand, on purpose — see the trap in ROADMAP.md item 1):\n"
        "  1. Only touch rows above with n >= --min-sample. A small sample can look\n"
        "     confident and still be wrong; report it, do not act on it.\n"
        "  2. For those rows, treat mean_score as the candidate new capability score —\n"
        "     it is graded on the same audit rubric the current estimate is trying to\n"
        "     predict, and it is more informative than the binary pass_rate alone.\n"
        "  3. Edit examples/starter_catalog.json (or your own catalog) by hand: replace\n"
        "     the estimated capabilities[task_type] value for that model_id, and update\n"
        "     the _disclaimer / _last_verified notes so the file still says honestly\n"
        "     which numbers are measured and which are still priors.\n"
        "  4. Re-run this script periodically as traces/ grows; a score measured on 15\n"
        "     samples is worth revisiting once you have 150."
    )
    return 0


if __name__ == "__main__":
    sys.exit(run())
