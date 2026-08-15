"""Verify a catalog against the live APIs. Costs nothing.

    PYTHONPATH=src python3 examples/live_check.py --catalog examples/starter_catalog.json

Run this before `live_run.py`, and any time you edit a catalog.

Every model id in a catalog is a claim that a string will be accepted by a
vendor. Nothing in the offline test suite can check that claim — `MockProvider`
answers to any id you give it — so a typo, a renamed model, or an id copied
from a pricing page that uses display names rather than API names survives all
186 tests and then fails as a hard 404 the first time it routes real traffic.

This script lists the models each key can actually reach and diffs them against
the catalog. It only ever issues GETs to the models endpoints, so it generates
no tokens and costs nothing.

Keys are read from the environment and never printed. The only thing reported
about a key is whether it is set.
"""

from __future__ import annotations

import argparse
import difflib
import sys

from switchboard import (
    ProviderConfigError,
    ProviderError,
    ProviderRateLimited,
    Registry,
    demo_registry,
)
from switchboard.providers.live import KNOWN_PROVIDERS, build_provider, key_status


# Reasoning models spend thinking tokens from the same budget, so a 1-token
# probe cannot produce output. OpenAI rejects that with a 400 outright while
# Anthropic returns a truncated 200 — either way it says nothing about whether
# the model exists. Small, but not that small.
PROBE_MAX_TOKENS = 16


def probe(provider, model_id: str) -> tuple[bool, str]:
    """Actually call a model once, minimally. Returns (usable, explanation).

    Listing is free but not sufficient: a vendor can keep a retired model in
    `/models` and still refuse it at the chat endpoint (observed with
    gemini-2.5-flash-lite, listed and 404 "no longer available"). Only a real
    call distinguishes "exists" from "works".

    A 429 is deliberately NOT treated as broken — that is your account quota,
    not a bad catalog entry.
    """
    provider.max_retries = 0  # a probe should report, not sit through backoff
    try:
        provider.complete(model_id, "Hi", max_tokens=PROBE_MAX_TOKENS)
        return True, "usable"
    except ProviderRateLimited:
        return True, "rate-limited (quota, not a catalog problem)"
    except ProviderConfigError as e:
        return False, f"auth/config: {e}"
    except ProviderError as e:
        message = str(e)
        if "max_tokens" in message or "output limit" in message:
            # The ceiling was too low for this model to say anything. That is a
            # fact about the probe, not about the catalog.
            return True, "usable (needs more than a token budget to respond)"
        return False, message[:120]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check a catalog's model ids against the live vendor APIs. Free.",
    )
    parser.add_argument("--catalog", help="catalog JSON (default: built-in demo catalog)")
    parser.add_argument(
        "--show-available", action="store_true",
        help="also print every model id each key can reach",
    )
    parser.add_argument(
        "--probe", action="store_true",
        help="make one 1-token call per model to prove it actually works. "
             "Listing alone cannot catch a retired model that is still listed. "
             "COSTS A FRACTION OF A CENT.",
    )
    args = parser.parse_args()

    registry = Registry.from_json(args.catalog) if args.catalog else demo_registry()
    catalog_providers = sorted({m.provider for m in registry.all()})

    print("=== Key status (presence only — values are never read or printed) ===")
    status = key_status(catalog_providers)
    for name in catalog_providers:
        spec = KNOWN_PROVIDERS.get(name)
        if spec is None:
            print(f"  {name:<12} UNKNOWN PROVIDER — not in KNOWN_PROVIDERS, cannot verify")
            continue
        if status.get(name):
            print(f"  {name:<12} {spec.env_var} is set")
        else:
            print(f"  {name:<12} {spec.env_var} NOT set   -> {spec.signup_url}")

    print("\n=== Catalog vs live APIs ===")
    problems = 0
    unverified = 0

    for name in catalog_providers:
        catalog_ids = sorted(m.model_id for m in registry.all() if m.provider == name)
        spec = KNOWN_PROVIDERS.get(name)

        if spec is None or not status.get(name):
            reason = "unknown provider" if spec is None else f"{spec.env_var} not set"
            print(f"\n{name}: SKIPPED ({reason}) — {len(catalog_ids)} model(s) unverified")
            unverified += len(catalog_ids)
            continue

        try:
            raw = build_provider(name).list_models()
            # Google returns resource names ("models/gemini-3.7-flash") while
            # catalogs and the chat endpoint use the bare id. Comparing the two
            # forms literally marked every Google model unreachable — a false
            # positive that would have sent someone editing a correct catalog.
            available = {m for m in raw} | {m.split("/", 1)[-1] for m in raw}
        except Exception as e:  # noqa: BLE001 - report and continue to the next vendor
            print(f"\n{name}: could not list models ({type(e).__name__}: {e})")
            unverified += len(catalog_ids)
            continue

        print(f"\n{name}: {len(raw)} model(s) reachable")
        if args.show_available:
            for model_id in sorted(available):
                print(f"    - {model_id}")

        prober = build_provider(name) if args.probe else None
        for model_id in catalog_ids:
            if model_id in available:
                if prober is None:
                    print(f"  [ OK ] {model_id}  (listed; not probed)")
                else:
                    ok, why = probe(prober, model_id)
                    if ok:
                        print(f"  [ OK ] {model_id}  ({why})")
                    else:
                        problems += 1
                        print(f"  [FAIL] {model_id}  listed but NOT usable -- {why}")
            else:
                problems += 1
                close = difflib.get_close_matches(model_id, sorted(available), n=3, cutoff=0.5)
                hint = f"  did you mean: {', '.join(close)}" if close else ""
                print(f"  [FAIL] {model_id}  -- not reachable with this key{hint}")

    print("\n" + "=" * 70)
    if problems:
        print(f"{problems} catalog model id(s) do not exist on the vendor's API.")
        print("Fix them before running anything live — each one is a guaranteed 404.")
    elif args.probe:
        print("Every catalog model id resolves AND answered a real call.")
    else:
        print("Every verifiable catalog model id is listed by its vendor.")
        print("Listing is not proof it works — re-run with --probe to actually call each one.")
    if unverified:
        print(f"{unverified} model(s) could not be checked (missing key or unknown provider).")
        print("Unchecked is not the same as correct.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
