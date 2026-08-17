"""Cross-check a catalog against OpenRouter's public model index. Free.

    PYTHONPATH=src python3 examples/catalog_crosscheck.py --catalog examples/starter_catalog.json
    PYTHONPATH=src python3 examples/catalog_crosscheck.py --catalog examples/starter_catalog.json --suggest

`live_check.py` asks the vendors "does this model id exist and answer". This
asks an independent index four different questions the vendors' own pages do
not make easy:

1. **Has a price moved?** Reported as a *disagreement*, never as a correction.
2. **What is the real context window?** Our catalog admits its values are
   conservative placeholders; the index has the actual numbers.
3. **Which models spend thinking tokens by default?** This is the property
   behind the truncation failures found repeatedly in this repo — a reasoning
   model burns its whole `max_tokens` budget thinking and returns nothing
   visible. Knowing it in advance turns that from a thing we detect after
   paying into a thing we prevent.
4. **What can each model actually do?** Structured outputs, tool calls —
   binary requirements a capability *score* cannot express.

## What this deliberately does NOT do

**It does not import benchmark scores as capability scores.** OpenRouter
publishes GPQA Diamond and τ²-Bench results, which measure graduate-science
multiple choice and agentic airline support. Switchboard routes on extraction,
summarization, coding, creative and reasoning. Four of those five have no
benchmark at all, and the fifth has a narrow proxy that is not the same thing.
Copying those numbers in would put "estimated presented as measured" straight
back into the claim this repo is most careful about — and `evals/catalog_
feedback.py` already measures capability from *your own* traffic, which is
better evidence than anyone's public leaderboard.

**It does not rewrite prices.** A third-party index may add margin, quote a
different tier, or lag. When it disagrees with the catalog, the correct action
is to re-read the vendor's own pricing page — which is what `_source` on every
catalog entry is for. Disagreement is a prompt to go and look, not a number to
copy.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

from switchboard import Registry, demo_registry

INDEX_URL = "https://openrouter.ai/api/v1/models"

# Their namespace does not always match the `provider` field a catalog uses.
_PROVIDER_ALIASES = {"moonshot": "moonshotai"}

# Vendors sometimes publish a dated id while the index carries the canonical
# one. Listed explicitly rather than fuzzy-matched: guessing which model a
# name refers to is exactly the kind of silent wrongness this repo avoids.
_MODEL_ALIASES = {
    "claude-haiku-4-5-20251001": "anthropic/claude-haiku-4.5",
}


def fetch_index() -> dict[str, dict]:
    """The public model index. No key required."""
    req = urllib.request.Request(INDEX_URL, headers={"user-agent": "switchboard-crosscheck"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return {m["id"]: m for m in json.loads(resp.read())["data"]}


def index_id(model_id: str, provider: str) -> str:
    return _MODEL_ALIASES.get(model_id) or f"{_PROVIDER_ALIASES.get(provider, provider)}/{model_id}"


def per_million(pricing: dict, key: str) -> float | None:
    raw = pricing.get(key)
    try:
        return float(raw) * 1_000_000
    except (TypeError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cross-check a catalog against OpenRouter's public index. Free, no key.",
    )
    parser.add_argument("--catalog", help="catalog JSON (default: built-in demo)")
    parser.add_argument(
        "--suggest", action="store_true",
        help="print a JSON patch for context_window only. Prices are never "
             "suggested — go and re-read the vendor page instead.",
    )
    args = parser.parse_args()

    registry = Registry.from_json(args.catalog) if args.catalog else demo_registry()
    try:
        index = fetch_index()
    except (urllib.error.URLError, OSError) as e:
        print(f"could not reach the index ({type(e).__name__}: {e}). Nothing checked.")
        return 1
    print(f"index: {len(index)} models\ncatalog: {len(registry)} models\n")

    matched, unmatched = [], []
    for spec in registry.all():
        entry = index.get(index_id(spec.model_id, spec.provider))
        (matched if entry else unmatched).append((spec, entry))

    # ---- 1. Prices. Disagreement is a signal, not a correction.
    print("=" * 92)
    print("PRICE DISAGREEMENTS — go re-read the vendor page; do not copy these numbers")
    print("=" * 92)
    drifted = 0
    for spec, entry in matched:
        their_in = per_million(entry.get("pricing") or {}, "prompt")
        their_out = per_million(entry.get("pricing") or {}, "completion")
        if their_in is None or their_out is None:
            continue
        if abs(their_in - spec.input_cost) < 0.005 and abs(their_out - spec.output_cost) < 0.005:
            continue
        drifted += 1
        print(f"  {spec.model_id:<28} catalog ${spec.input_cost:g}/${spec.output_cost:g}"
              f"   index ${their_in:g}/${their_out:g}")
    if not drifted:
        print("  none — every matched model agrees with the catalog.")
    else:
        print(f"\n  {drifted} disagreement(s). Causes differ and matter: an index may add")
        print("  margin, quote a different tier, or lag a change. Only the vendor's own")
        print("  page settles it, which is what each entry's _source field is for.")

    # ---- 2. Context windows. Ours are admitted placeholders.
    print("\n" + "=" * 92)
    print("CONTEXT WINDOWS — the catalog calls its own values conservative placeholders")
    print("=" * 92)
    patch = {}
    for spec, entry in matched:
        theirs = entry.get("context_length")
        max_out = (entry.get("top_provider") or {}).get("max_completion_tokens")
        if not theirs or theirs == spec.context_window:
            continue
        patch[spec.model_id] = theirs
        print(f"  {spec.model_id:<28} catalog {spec.context_window:>9,}   "
              f"index {theirs:>9,}   max_output {max_out or '?':>8}")
    if not patch:
        print("  none — every matched model already carries the indexed value.")

    # ---- 3. The property behind this repo's most repeated bug.
    print("\n" + "=" * 92)
    print("THINKING TOKENS — models that spend the output budget before saying anything")
    print("=" * 92)
    print("  Found three times in this repo: a reasoning model burns its whole")
    print("  max_tokens allowance thinking and returns zero visible characters,")
    print("  which reads downstream as an empty or malformed answer. These models")
    print("  need headroom, not a bigger tier.\n")
    thinkers = []
    for spec, entry in matched:
        reasoning = entry.get("reasoning") or {}
        if reasoning.get("default_enabled"):
            thinkers.append(spec.model_id)
            print(f"  {spec.model_id:<28} reasoning ON by default   "
                  f"efforts={','.join(reasoning.get('supported_efforts') or []) or 'n/a'}")
    if not thinkers:
        print("  none of the matched models default to reasoning.")
    else:
        print(f"\n  {len(thinkers)}/{len(matched)} matched models. Give these a generous")
        print("  max_tokens, and treat a truncated reply as mechanical, not poor quality.")

    # ---- 4. Binary requirements a score cannot express.
    print("\n" + "=" * 92)
    print("HARD CAPABILITIES — binary, so they belong in a gate rather than a score")
    print("=" * 92)
    print(f"  {'model':<28} {'structured_outputs':>19} {'tools':>7} {'reasoning':>10}")
    for spec, entry in matched:
        params = set(entry.get("supported_parameters") or [])
        print(f"  {spec.model_id:<28} {'yes' if 'structured_outputs' in params else 'NO':>19} "
              f"{'yes' if 'tools' in params else 'NO':>7} "
              f"{'yes' if 'reasoning' in params else 'NO':>10}")

    if unmatched:
        print("\n" + "=" * 92)
        print("NOT INDEXED — unknown, which is not the same as wrong")
        print("=" * 92)
        for spec, _ in unmatched:
            print(f"  {spec.model_id:<28} ({spec.provider}) — no entry under "
                  f"{index_id(spec.model_id, spec.provider)!r}")
        print("\n  A model absent from a third-party index may simply not be carried")
        print("  there. Verify these against the vendor with live_check.py --probe.")

    if args.suggest and patch:
        print("\n" + "=" * 92)
        print("SUGGESTED PATCH — context_window only, and still worth checking")
        print("=" * 92)
        print(json.dumps(patch, indent=2))
        print("\n  Apply by hand. Prices are deliberately absent from this patch.")

    print(f"\n{len(matched)}/{len(registry)} matched, {drifted} price disagreement(s), "
          f"{len(patch)} context window(s) differing, {len(thinkers)} default-reasoning model(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
