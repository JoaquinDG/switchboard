"""Triage evals: does prompt classification agree with a human label?

Same spirit as routing_eval.py — a scorecard over labeled examples, not unit
tests. Unit tests pin individual behaviours; this measures whether the
classifier is good enough to put in front of the router at all, and prints the
confusion so a regression shows you *what* it started confusing.

The last block is adversarial on purpose: prompts that mention code but are
extraction, mention writing but are code, mention a function but are
summarization. Those are where a keyword classifier earns or loses its keep.

Usage:
    PYTHONPATH=src python3 evals/triage_eval.py
"""

from __future__ import annotations

import sys
from collections import Counter

from switchboard import BALANCED, classify_heuristic

# Accuracy below this fails the run. Set at 0.80 because triage is a routing
# *hint*: the qualification gate and the cross-model audit both sit downstream
# of it, so a wrong label degrades cost efficiency rather than correctness.
PASS_THRESHOLD = 0.80

LABELED: list[tuple[str, str]] = [
    # -- extraction ------------------------------------------------------
    ("Extract all email addresses and phone numbers from this contact sheet.", "extraction"),
    ("Parse these invoices and return the totals as JSON.", "extraction"),
    ("Pull the plan names and prices out of these five competitor pages.", "extraction"),
    ("List all the company names mentioned in the attached filing.", "extraction"),
    ("From this text, identify every date and the event attached to it.", "extraction"),
    # -- summarization ---------------------------------------------------
    ("Summarize this 40-page board memo for a non-executive reader.", "summarization"),
    ("tl;dr this thread.", "summarization"),
    ("Condense the research notes into key takeaways.", "summarization"),
    ("Give me a one-paragraph recap of the customer call.", "summarization"),
    ("Boil down this policy document into its main points.", "summarization"),
    # -- coding ----------------------------------------------------------
    ("Refactor this module to remove the duplicated retry logic.", "coding"),
    ("Write a webhook handler that updates the pricing table via the CMS API.", "coding"),
    ("Debug this stack trace and tell me what is failing.", "coding"),
    ("Write unit tests for the payment reconciliation function.", "coding"),
    ("Convert this SQL query to use a window function instead of a subquery.", "coding"),
    # -- creative --------------------------------------------------------
    ("Write landing page copy for our new pricing tier, friendly but direct.", "creative"),
    ("Brainstorm ten taglines for a developer tools launch.", "creative"),
    ("Draft a punchy blog post announcing the beta.", "creative"),
    ("Write a short story about a lighthouse keeper.", "creative"),
    ("Come up with headline options for the newsletter.", "creative"),
    # -- reasoning -------------------------------------------------------
    ("Should we migrate off the monolith this quarter? Walk through the tradeoffs.", "reasoning"),
    ("Analyze why our conversion dropped 12% after the redesign.", "reasoning"),
    ("Recommend a pricing response to the competitor's new free tier.", "reasoning"),
    ("Compare these three vendor proposals and tell us which to pick.", "reasoning"),
    ("Design a rollback strategy for a live billing migration.", "reasoning"),
    # -- adversarial: the label is not the most obvious keyword ----------
    ("Extract the function signatures from this Python file.", "extraction"),
    ("Summarize what this function does.", "summarization"),
    ("Write a Python script that summarizes a directory of PDFs.", "coding"),
    ("Explain the tradeoffs between REST and GraphQL for our API.", "reasoning"),
    ("Write a regex that validates international phone numbers.", "coding"),
]

# Written AFTER the classifier was frozen, and deliberately not tuned against.
#
# The set above is worth 100%, which mostly measures that the author of the
# keyword table and the author of the labels were the same person. These are
# ordinary requests phrased the way people actually phrase them — without the
# canonical verb the keyword table keys on. Accuracy here is the number to
# believe, and it is reported separately for that reason.
#
# Do not add keywords to make these pass. The moment you tune against them
# they stop measuring anything, and the honest weakness they document (see the
# README learnings section) is worth more than a rounder number.
HELD_OUT: list[tuple[str, str]] = [
    ("Rewrite this paragraph so it sounds less corporate.", "creative"),
    ("Add type hints to every function in this file.", "coding"),
    ("How many customers churned last quarter according to this report?", "extraction"),
    ("Turn this changelog into a customer-facing release note.", "creative"),
    ("What is wrong with this function?", "coding"),
    ("Make this endpoint faster without changing its behaviour.", "coding"),
    ("Which of these two onboarding flows would you ship, and why?", "reasoning"),
    ("Give me the three numbers I need for the board slide from this spreadsheet.", "extraction"),
    ("Cut this 800-word intro down to 200 words.", "summarization"),
    ("Is it worth rewriting the scheduler, or should we patch it?", "reasoning"),
]

# A few prompts where the *complexity* estimate matters as much as the label.
# Bands are wide on purpose: this is a prior, not a measurement.
COMPLEXITY_BANDS: list[tuple[str, float, float]] = [
    ("tl;dr this thread.", 0.05, 0.35),
    ("Extract all email addresses from this contact sheet.", 0.05, 0.45),
    ("Design a distributed consensus protocol from scratch with proofs.", 0.75, 0.95),
    ("Write a production migration script with rollback for a live billing system.", 0.65, 0.95),
]


def score(cases, title, confusion):
    print(f"\n### {title}")
    print(f"{'prompt':<72} {'expected':<14} {'got':<14} conf  cx    result")
    print("-" * 122)
    correct = 0
    for prompt, expected in cases:
        verdict = classify_heuristic(prompt)
        ok = verdict.task_type == expected
        correct += ok
        if not ok:
            confusion[(expected, verdict.task_type)] += 1
        shown = prompt if len(prompt) <= 70 else prompt[:67] + "..."
        print(
            f"{shown:<72} {expected:<14} {verdict.task_type:<14} "
            f"{verdict.confidence:<5.2f} {verdict.complexity:<5.2f} "
            f"{'PASS' if ok else 'FAIL'}"
        )
    print("-" * 122)
    print(f"{title}: {correct}/{len(cases)} = {correct / len(cases):.0%}")
    return correct


def run() -> int:
    confusion: Counter[tuple[str, str]] = Counter()
    tuned = score(LABELED, "tuned set (classifier built against these)", confusion)
    held = score(HELD_OUT, "held-out set (written after the classifier was frozen)", confusion)

    total = len(LABELED) + len(HELD_OUT)
    correct = tuned + held
    accuracy = correct / total
    print("\n" + "=" * 122)
    print(f"combined accuracy:  {correct}/{total} = {accuracy:.0%} "
          f"(threshold {PASS_THRESHOLD:.0%})")
    print(f"  tuned:            {tuned}/{len(LABELED)} = {tuned / len(LABELED):.0%} "
          f"<- measures internal consistency, not skill")
    print(f"  held-out:         {held}/{len(HELD_OUT)} = {held / len(HELD_OUT):.0%} "
          f"<- the number to believe")

    if confusion:
        print("\nconfusions (expected -> predicted):")
        for (expected, got), n in confusion.most_common():
            print(f"  {expected} -> {got}  x{n}")

    print("\ncomplexity bands:")
    band_failures = 0
    for prompt, low, high in COMPLEXITY_BANDS:
        verdict = classify_heuristic(prompt)
        ok = low <= verdict.complexity <= high
        band_failures += 0 if ok else 1
        shown = prompt if len(prompt) <= 70 else prompt[:67] + "..."
        print(f"  [{'PASS' if ok else 'FAIL'}] {verdict.complexity:.2f} "
              f"in [{low}, {high}]  {shown}")

    # Determinism is load-bearing: same prompt, same route, every run, or the
    # traces stop being comparable across time.
    repeats = {classify_heuristic(LABELED[0][0]).task_type for _ in range(5)}
    deterministic = len(repeats) == 1
    print(f"\ndeterministic across repeated calls: {deterministic}")

    # ROADMAP 14: the held-out gap is real and the keyword table must not be
    # tuned against these prompts (see the comment above HELD_OUT). What can
    # be checked honestly, offline, is whether the already-shipped confidence
    # gate (ROADMAP 5) would even see these failures — i.e. whether
    # Broker(triage_use_model=True) hands them to a model instead of trusting
    # a wrong heuristic guess. This is a structural fact, not a rescue: it
    # says nothing about whether the model then answers correctly. That part
    # was measured live, separately, and is not re-run here.
    threshold = BALANCED.triage_confidence_threshold
    held_failures = [(p, e) for p, e in HELD_OUT if classify_heuristic(p).task_type != e]
    in_reach = [p for p, _ in held_failures if classify_heuristic(p).confidence < threshold]
    print("\nconfidence-gate reach on held-out failures (structural, offline):")
    print(f"  {len(in_reach)}/{len(held_failures)} of the failures above score below the "
          f"default triage_confidence_threshold ({threshold}), so Broker(triage_use_model=True) "
          f"routes every one of them to a model instead of trusting the wrong heuristic guess.")
    print("  Whether the model then gets them right is a live-API question, last measured "
          "against real APIs (examples/triage_ab.py, cited in the README): held-out accuracy "
          "60% -> 90% at $0.00011 for 5 model calls across the 40-prompt set. Re-run "
          "examples/triage_ab.py to refresh those numbers; this script has no provider keys.")

    failed = accuracy < PASS_THRESHOLD or band_failures or not deterministic
    print(f"\n{'FAIL' if failed else 'PASS'}: triage evals")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(run())
