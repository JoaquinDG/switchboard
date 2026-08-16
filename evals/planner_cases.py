"""Labeled requests for the planner eval.

Three sets: compound requests with explicit structure, compound requests
without it, and requests that only look compound. The negatives matter
more than the positives: a planner that splits something simple buys extra
audits, extra escalations and extra latency for work that needed one call, and
it does so silently. Under-splitting costs you a saving; over-splitting costs
you money you did not have to spend.

Sources: the workflow in `examples/agentic_workflow.py`, the request shapes in
the README, and ordinary phrasings people actually type.

`expected_types` is a hand label, not an oracle. Where the planner disagrees
with it, read both before assuming the planner is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    request: str
    should_split: bool
    # Hand-labeled decomposition. None for negatives.
    expected_types: tuple[str, ...] | None = None
    note: str = ""


# The hardest negative in the set: long, multi-clause, and still one job.
# Length is not structure, and a planner that treats it as such will shred
# every long document request in production.
_LONG_SINGLE = (
    "Summarize the attached annual report for a non-executive reader. It runs "
    "to about ninety pages and covers revenue by segment, headcount changes, "
    "churn and retention, pipeline coverage, regional performance, the "
    "competitive landscape, capital expenditure, the outlook for the coming "
    "year, and a lengthy appendix of accounting notes that you can safely skim "
    "but should not ignore entirely if something material appears in it."
)


COMPOUND: tuple[Case, ...] = (
    Case(
        "1. Extract the payment terms from the contract.\n"
        "2. Summarize the obligations.\n"
        "3. Recommend whether we should sign.",
        True, ("extraction", "summarization", "reasoning"),
        "enumeration: the writer already decomposed it",
    ),
    Case(
        "Pull the pricing out of these five competitor pages, then summarise it "
        "into a brief, then recommend how we should respond, and finally write "
        "the new landing page copy.",
        True, ("extraction", "summarization", "reasoning", "creative"),
        "the agentic_workflow pipeline as one sentence",
    ),
    Case(
        "Extract the dates from the contract, then draft a reminder email.",
        True, ("extraction", "creative"),
    ),
    Case(
        "First, summarize the research notes. Second, recommend next steps.",
        True, ("summarization", "reasoning"), "ordinals",
    ),
    Case(
        "Parse the invoices and then write a summary email to finance.",
        True, ("extraction", "creative"),
    ),
    Case(
        "Read the support tickets, then group them by theme, and finally draft "
        "a reply template for each theme.",
        True, ("extraction", "reasoning", "creative"),
    ),
    Case(
        "Summarize the incident report, then recommend preventative measures.",
        True, ("summarization", "reasoning"),
    ),
    Case(
        "Extract the metrics from the dashboard export. Then write a short "
        "update for the board.",
        True, ("extraction", "creative"), "sentence-initial connective",
    ),
    Case(
        "- Extract the competitor prices\n"
        "- Summarize where we are exposed\n"
        "- Recommend our response",
        True, ("extraction", "summarization", "reasoning"), "bullets",
    ),
    Case(
        "Analyze why churn rose last quarter, then draft an email to the team "
        "explaining what we found.",
        True, ("reasoning", "creative"),
    ),
    Case(
        "Summarize this contract. Afterwards, tell me whether we should sign it.",
        True, ("summarization", "reasoning"),
    ),
    Case(
        "Refactor the retry logic to remove the duplication, then write unit "
        "tests covering the backoff path.",
        True, ("coding", "coding"),
        "two coding jobs: same KIND of work, different pieces of work",
    ),
)


# Genuinely compound, but with NO enumeration, ordinal or sequence connective
# to cut on. The heuristic's stated contract is that it splits on explicit
# structure, so it is right to decline these — measuring it against them and
# calling that a failure would misrepresent what it claims to do.
#
# They are scored separately for that reason, and reported rather than hidden:
# this is the model layer's territory, and the only set on which an A/B between
# the layers can measure anything at all.
COMPOUND_UNMARKED: tuple[Case, ...] = (
    Case(
        "Read this contract, extract the payment terms, and tell me if we "
        "should sign it.",
        True, ("extraction", "reasoning"),
        "heuristic KNOWS it cannot judge: confidence 0.20, which opens the gate",
    ),
    Case(
        "Take these support tickets, work out which themes are growing, write "
        "me something I can send to the team.",
        True, ("reasoning", "creative"),
        "was a blind spot (scored 0.95, confidently wrong) until the work-verb "
        "vocabulary learned 'work out' and 'write me/us'; now 0.20, gate opens",
    ),
)


NOT_COMPOUND: tuple[Case, ...] = (
    Case("Summarize this board memo for a non-executive reader.", False),
    Case(
        "Extract all the names, email addresses and phone numbers from this "
        "contact sheet.",
        False, note="comma chain inside ONE extraction",
    ),
    Case("Refactor this module to remove the duplicated retry logic.", False),
    Case(
        "Write landing page copy for our new pricing tier, friendly but direct.",
        False,
    ),
    Case(_LONG_SINGLE, False, note="LONG but single-task — the hardest negative"),
    Case(
        "Pull the plan names, prices and limits out of these five competitor pages.",
        False, note="comma chain inside ONE extraction",
    ),
    Case(
        "Debug this stack trace and tell me what is failing.",
        False, note="'and' joining one diagnostic job",
    ),
    Case(
        "Write a Python function that parses ISO dates, handles the empty case, "
        "and raises on bad input.",
        False, note="three clauses, one function",
    ),
    Case(
        "Compare these three vendor proposals and tell us which one to pick.",
        False, note="'and' joining one comparison",
    ),
    Case("Explain the tradeoffs between REST and GraphQL for our API.", False),
)


ALL_CASES: tuple[Case, ...] = COMPOUND + COMPOUND_UNMARKED + NOT_COMPOUND
