# Roadmap

The working backlog, in priority order. **This file is the single source of truth** — the scheduled maintenance agent reads it and takes the first unchecked item, so editing this file is how you steer what gets built next.

Reorder freely. Add items freely. Tick a box when the work lands on `main`.

**Claimed items.** An item marked `[~]` and **CLAIMED** is being built right now
in an interactive session. The scheduled agent must skip it and take the next
unclaimed one, or two people build the same thing twice.

## How to read an item

Each entry states **why it matters** (the gap it closes), **done looks like** (acceptance criteria), and where relevant a **trap** — a way of satisfying the letter of the item while making the repo worse. The traps are the important part. Most of these items have an easy wrong version.

## Ground rules for every item

These are not negotiable and apply to all work below:

- Zero runtime dependencies; the package stays stdlib-only and runs fully offline via `MockProvider`.
- `unittest discover`, `routing_eval.py`, and `triage_eval.py` all pass before anything opens for review.
- Honest claims only. Estimated is never presented as measured; mocked is never presented as real.
- New routing logic extends the rationale string and the ranked list. It never bypasses them.
- One logical change per PR.

---

## Tier 0 — make it usable

Added after a non-technical reader asked three questions the repo could not
answer: *how do I use it, does it run in the background, and what if my task
has several tasks inside it?* All three were fair. The first two are packaging
problems; the third is a real hole in the thesis.

### U1. Say who it is for — **DONE**
- [x] README now opens with "Is this for me?", which states plainly that
  Switchboard sits between *your code* and the vendor APIs, that a flat
  subscription has no routing decision in it, and that a reader using
  ChatGPT or Claude through their apps is not the user.
- **Why it mattered:** the honest answer to "how do I use it" is sometimes
  "you don't". A repo that trades on honesty should say so before someone
  spends an afternoon finding out.

### U2. OpenAI-compatible local proxy — `[~]` **CLAIMED**
- [~] **Why it matters:** the integration today is "write Python and construct
  a Registry". A proxy exposing `/v1/chat/completions` on localhost lets any
  existing tool — an SDK, an editor, a framework — get routing, cross-lab
  auditing and cost accounting by changing one base URL and no code. This is
  the single largest adoption unlock available, and it costs no dependencies:
  `http.server` is in the stdlib.
- **Done looks like:** `switchboard serve`; a request to `/v1/chat/completions`
  routes through triage + router + broker and returns an OpenAI-shaped
  response; routing rationale and cost returned in headers or an extension
  field; `/v1/models` lists the catalog. Offline test coverage with no sockets
  to real vendors.
- **Trap — auditing and streaming are incompatible.** You cannot audit a
  response you have not finished receiving. Either refuse `stream: true`, or
  serve it with audits disabled and say so in the response. Silently dropping
  verification while still reporting success would be the worst option, and it
  is the easy one.
- **Second trap:** auditing doubles latency, which is fine in a batch job and
  unusable in an editor. Default audits **off** in the proxy and make them
  opt-in per request; the library default stays on.

### U3. Composite tasks — decomposition — **DONE (v1)**
- [x] Shipped: `planner.py` (Plan schema, strict validation, anti-split heuristic), `Broker.run_plan` (dependency-ordered dispatch through the unchanged broker path, context threading with named truncation, three labelled cost figures), `replay.py` (full reconstruction from trace alone), `evals/planner_eval.py` (22 labeled cases, false-split rate 0%, in CI).
- **What it changed about the thesis:** the eval showed decomposition is a *correctness* mechanism first. Routing a compound request whole under-routed it 42% of the time — cheaper, but below the tier its hardest sub-task needs, and invisible to the qualification gate. See the README findings.
- **Also shipped:** gated model planner with repair-once and honest fallback (`plan_with_model`), `examples/planner_ab.py`, `--plan` on the agentic example, and the optional plan-level final audit behind `Policy.plan_final_audit`.
- **Measured (examples/planner_ab.py, live):** the model layer never over-splits (0 false splits at every threshold) and rescues unmarked compounds the heuristic declines. Gating at 0.25 catches 1 of 2 for one model call; asking always catches 2 of 2 for 24 calls and 312s.
- **Known limit:** confidence gating is bounded by the heuristic's self-knowledge. One unmarked compound scores 0.20 (gate opens, model rescues it); the other scores 0.95 while being wrong, so the gate never opens. Widening the work-verb vocabulary would help; a learned confidence would help more. Tracked as its own item below.
- **Deferred to v2:** parallel execution of independent steps (blocked on async), step-output summarisation for long chains, and recursive planning.
- **Original rationale:** measured, not hypothesised. Switchboard routes one
  task to one model, so a composite request routes at its *hardest* sub-task
  and everything runs at frontier prices: the four-part example in the README
  costs **$0.1600 as one task vs $0.0545 as four — 2.9x**. Triage does not
  catch it either. "Read this contract, extract the payment terms, and tell me
  if we should sign it" classifies as `extraction, complexity 0.30` with
  **confidence 1.00**, and the confidence gate cannot help because it guards
  against *no signal*, not *several signals*. The headline savings therefore
  depend on a decomposition the library does not perform — `agentic_workflow.py`
  hand-writes its five steps.
- **Done looks like:** a `Plan` layer that splits a request into steps, routes
  each one, threads outputs between them, and reports the plan alongside the
  result. Measured against routing the same request whole.
- **Trap:** keep it out of the router. The router routes; planning is a
  separate concern with a different failure mode, and folding them together
  makes both untestable.
- **Second trap:** an LLM planner is unreliable in exactly the way triage is,
  so it gets the same treatment — show the plan, name the layer that produced
  it, and let the caller approve or override before anything is spent.
- **Third:** a step-wise plan also makes audits cheaper to repair. One verdict
  on a four-part answer fails the whole thing for one bad part and re-runs
  everything; per-step audits repair only the step that failed.

### U4. A one-line CLI — **DONE**
- [x] Shipped as `switchboard.cli:main`, wired up via `[project.scripts]` in
  `pyproject.toml`. `switchboard "prompt"` triages (auto by default,
  `--task-type` to skip it), routes, runs, and prints the rationale, verdict,
  and full cost breakdown against the built-in demo catalog — no keys, no
  script, no `PYTHONPATH` juggling once installed.
- **How "dry-run by default" was read:** the default run executes the full
  `Broker.run` pipeline (triage, routing, audit, escalation) against
  `MockProvider`, so routing, auditing and cost accounting are the real
  logic — only the generated text is canned, and the output says so.
  Nothing is billed or sent over the network unless `--live` is passed, and
  `--live` refuses to run without both a keyed provider and an explicit
  `--budget-usd > 0`, mirroring `examples/live_run.py`'s worst-case cost
  guard so a single mistyped flag can't produce a surprise bill.
- **Tests:** `tests/test_cli.py` — default mocked run, forced `--task-type`
  skips triage, custom catalog loading, opt-in tracing, and three
  live-mode safety tests (no budget, no keys, budget of zero) that assert no
  provider is ever touched, run with provider keys scrubbed from the
  environment so the test can't pass by accident.

---

## Tier 1 — close known gaps in what already exists

These are the highest-value items because each one fixes something the repo currently gets *wrong or unmeasured*, not something it merely lacks.

### 0. Real-API exercise harness — **DONE**
- [x] `examples/live_check.py` verifies every catalog `model_id` against the vendor's models endpoint for free, and `examples/live_run.py` runs a real task suite under a hard budget cap. `providers/live.py` maps catalog provider names to real adapters and per-vendor env vars. Live integration tests in `tests/test_live.py` are opt-in via `SWITCHBOARD_LIVE_TESTS=1`.
- **Why it mattered:** nothing offline could check that a `model_id` is a string a vendor will accept, and every `verified: True` in the repo came from a canned grader. This is also what makes item 1 possible — real audit outcomes are the raw material for measured capability scores.

### 1. Trace-driven catalog feedback
- [x] **Why it matters:** The starter catalog's capability scores are estimates, and the file says so in block capitals. That is the weakest claim in the repo and the first thing a reviewer will poke at. Traces already record audit outcomes per model per task type — the measurement is sitting there unused.
- **Done looks like:** a script in `evals/` that reads `traces/*.jsonl` and reports observed audit pass-rate and mean audit score, per model, per task type, with sample counts. Plus a documented path from that report to updated catalog numbers.
- **Note:** generate real traces first with `examples/live_run.py --live`; `traces/` is gitignored, so a fresh checkout has none. Running the offline examples also produces traces, but their audit outcomes are canned and must not be scored.
- **Trap:** do not auto-write scores back into the catalog from a handful of samples. Report the numbers *and the sample size*, and say plainly when `n` is too small to act on. A confidently wrong measured score is worse than an honestly labelled estimate.

### 1b. Scoring terms are not on comparable scales — **DONE**
- [x] **Why it matters:** cost and latency are normalized to span [0, 1]; capability is used raw and clusters tightly. Measured on the starter catalog for `extraction`: capability ranges 0.78-0.90 (spread 0.12, so an 0.85 weight buys 0.102 of influence) while latency spans 0.20-1.00 (spread 0.80, so an 0.10 weight buys 0.080). The weights do not mean what they say. Concretely, `quality_first` on `reasoning@0.6` picks `gemini-3.7-flash` (capability 0.84, fast) over `claude-opus-5` (0.95, slow): a 0.093 quality advantage loses to an 0.080 latency advantage. A policy named quality-first should not do that.
- **Done looks like:** the weights behave as advertised, demonstrated on both the 3-model demo catalog and the 16-model starter catalog, with the routing evals still passing.
- **Trap — this one has already been walked into.** The obvious fix is to min-max normalize capability the way cost is normalized. It works on a wide catalog and *breaks on a narrow one*: with three models spanning 0.78-0.88, normalizing turns a 0.10 capability difference into a full 0.0-to-1.0 swing, and `balanced` starts sending easy extraction to the frontier model. That is the same artifact already fixed for cost ("with two candidates one is always 0.0"), reintroduced on another axis. Capability is a genuinely absolute 0-1 quantity; its observed range in a given catalog is not a meaningful denominator.
- **Other directions worth trying:** narrow the latency score spread (currently fast=1.0 / medium=0.6 / slow=0.2); normalize against the full 0-1 capability scale rather than the observed range; or leave scoring alone and recalibrate the preset weights, accepting that weights are catalog-dependent and saying so.
- **Shipped as:** `router._quality_score` in `router.py`, anchoring capability's normalization on `UNKNOWN_CAPABILITY_PRIOR` (0.5) instead of the observed candidate range. This is *not* the rejected fix: the floor is a fixed catalog-wide constant — no honest catalog rates a real model below "no idea" — not a function of who else is competing for this task, so it doesn't reintroduce the candidate-count artifact. Verified: `quality_first` on `reasoning@0.6` now picks `claude-opus-5` over `gemini-3.7-flash` on the 16-model starter catalog (`tests/test_router.py::QualityScoreScaleTests`, `evals/routing_eval.py`), and the narrow 3-model demo catalog still keeps easy extraction cheap under `cost_first` — the exact case the rejected fix broke.

### 1c. The planner's confidence has blind spots
- [ ] **Why it matters:** the model gate can only fire where the heuristic reports low confidence, and measured on unmarked compound requests it is right about its own uncertainty half the time. "Read this contract, extract the payment terms, and tell me if we should sign it" scores 0.20 and the gate rescues it. "Take these support tickets, work out which themes are growing, write me something I can send" scores **0.95 while being wrong**: no connective, and the phrasings match none of the work-verb patterns, so it sees one job and is sure of it. A blind spot is worse than a known unknown.
- **Done looks like:** the unmarked-compound gate-fire rate in `planner_eval.py` improves without the false-split rate moving off 0.
- **Trap:** do not fix this by lowering the confidence the heuristic reports across the board. That opens the gate everywhere, which is "model always" with extra steps — measured at 24 calls and 312s to gain one case.

### 1d. A verified step can be worthless to everything downstream
- [ ] **Why it matters:** measured live. Asked for data absent from the source, every model correctly refused to fabricate and the auditor correctly passed each refusal — then the plan spent **$0.02531 producing three variations of "that data is not here"**. `plan_halt_dependents_on_failure` cannot help: nothing failed. The auditor named it, scoring the second step 0.72 with "adds no value over step s1 — essentially a verbatim restatement, so the step is redundant", but `verified` is a boolean and 0.72 clears the 0.70 threshold.
- **Done looks like:** a plan stops paying for steps that can only restate their input, without ever suppressing a step that had something to add.
- **Trap:** the obvious fix — raise `audit_pass_threshold` so 0.72 fails — makes every borderline-but-useful answer fail too. The signal is not "low score", it is "this output adds nothing over its input", and those are different measurements.
- **Second trap:** do not detect it with string similarity between a step's output and its injected context. A correct summarisation legitimately restates its input; the difference is whether it *adds* anything, which similarity cannot see.
- **Worth trying:** the auditor already produces the judgement in prose. Asking it for a structured `adds_value` field alongside `pass` is one call's worth of change and needs no new machinery.

### 2. Context-window awareness in routing
- [ ] **Why it matters:** `ModelSpec.context_window` is stored, validated, and **never read by the router** (verified: no reference outside `registry.py`). A task with 400k estimated input tokens can be routed today to a 200k model, which fails at the provider with a hard error rather than being caught as a routing constraint.
- **Done looks like:** a hard gate — models whose context window cannot hold `est_input_tokens` plus `est_output_tokens` are ineligible, with the exclusion named in the rationale. Gate ordering and the degrade-upward rule must be preserved. Eval scenario covering it.
- **Trap:** this is a *gate*, not a score. Do not add "context headroom" as a weighted term; a model that cannot fit the input is not a worse choice, it is not a choice.
- **Second trap, now avoided:** the catalog's context windows were conservative placeholders understating six models by 5x. Building this gate against them would have excluded models that can do the work — a correctness bug caused entirely by trusting a field nothing had ever read. Corrected on 2026-08-16 from a public index (see `examples/catalog_crosscheck.py`); re-run it before relying on the values.
- **Worth using:** the same index reports `max_completion_tokens` per model, which is the ceiling this gate has to respect on the output side, and `reasoning.default_enabled` — 10 of 15 catalog models spend thinking tokens from that same budget.

### 3. Tokenizer-aware cost comparison
- [ ] **Why it matters:** `examples/starter_catalog.json` already documents that Claude 4.7 and later use a tokenizer producing roughly 30% more tokens for the same text. Per-token price therefore understates their real cost, and the router compares vendors on price alone. Cross-vendor cost comparison is currently biased.
- **Done looks like:** an optional per-model `token_multiplier` (default 1.0) applied in `estimate_cost`, populated for the models the catalog flags, documented as an estimate. Actual billing still uses observed tokens and must not be double-counted.
- **Trap:** `actual_cost` operates on token counts the provider already reported. Applying the multiplier there would inflate real bills. It belongs in the *estimate* path only.

### 4. Batch-API pricing — **DONE**
- [x] **Why it matters:** Vendors offer ~50% off for asynchronous batch processing — the single largest discount available, and the router cannot see it. A brokerage that ignores the biggest cost lever in the market is incomplete.
- **Done looks like:** a `Task` flag for latency-insensitive work, a catalog field for the batch discount, and routing that prices eligible tasks at the batch rate. `needs_fast_response` and batch eligibility are mutually exclusive; enforce that.
- **Trap:** do not claim batch *execution* support. This is pricing-model work. Actually submitting to batch endpoints is a separate, larger item — say so rather than implying it works.
- **Shipped as:** `Task.batch_eligible` (`policies.py`, raises in `__post_init__` if set alongside `needs_fast_response`), `ModelSpec.batch_discount` (`registry.py`, validated to `[0, 1)`, default `0.0` meaning "not modeled" rather than "no discount"), and `router.estimate_cost` scaling both token estimates by `1 - batch_discount` when `task.batch_eligible`. `router.actual_cost` is untouched — it prices tokens a provider actually billed for a call that actually happened at the standard synchronous rate, and Switchboard never submits to a real batch endpoint, so applying the discount there would claim a saving nobody received. The chosen model's rationale line names the batch rate when one applies, and warns when a batch-eligible task lands on a model with no `batch_discount` in the catalog.
- **Populated in `examples/starter_catalog.json`:** only the four Anthropic models, at `0.5`, sourced from `platform.claude.com/docs/en/about-claude/pricing#batch-processing` and fetched live during this session (2026-08-18) — the page states batch input/output are exactly half of standard for every model listed. OpenAI's, Google's, DeepSeek's, and Moonshot's pricing pages were unreachable from this environment (network egress blocked to those hosts), so those providers stay at the field's `0.0` default rather than guessing parity with Anthropic's rate. Noted in `_pricing_caveats`.

---

## Tier 2 — extend the core mechanisms

### 5. Confidence-gated triage — **DONE**
- [x] Shipped as `Policy.triage_confidence_threshold` (default 0.25), measured by `examples/triage_ab.py`. Across 40 labeled prompts the heuristic's confidence separated its errors perfectly (every wrong answer scored 0.00, correct ones averaged 0.93), so gating reaches 98% accuracy on 5 model calls instead of 40 — better than either layer alone, since asking a model on every prompt overrides cases the heuristic got right.
- **Original rationale:** Triage is currently all-heuristic or all-model per Broker. But the heuristic already reports its own confidence, and the held-out eval shows its failures cluster exactly where confidence is `0.00` — prompts that matched no keyword and fell through to the `reasoning` default. Spending a cheap model call *only* on the uncertain cases targets the real weakness at a fraction of the cost.
- **Done looks like:** a policy threshold below which triage escalates from heuristic to model; a measurement of resulting accuracy and cost per classification; the rationale still naming which layer decided each time.
- **Trap:** the honesty rule holds. A prompt classified by the heuristic must still say `heuristic`, even when the policy *would have* escalated but the model call failed.

### 6. Budget-aware policy
- [ ] **Why it matters:** Policies are static, but budgets are not. A team three weeks into a monthly cap wants different weights from one on day one, and today that means editing code mid-month.
- **Done looks like:** a policy that reads cumulative spend from traces and shifts cost weight as the cap approaches, with the current budget position visible in the rationale.
- **Trap:** it must degrade gracefully to a normal policy when no trace history exists. A cold start must not behave like an exhausted budget.

### 7. Multi-auditor consensus for high-stakes tasks
- [ ] **Why it matters:** One auditor is one opinion. For high-complexity or explicitly flagged work, N independent auditors and a majority verdict is meaningfully stronger evidence — and the cross-lab machinery to pick genuinely independent auditors already exists.
- **Done looks like:** a policy knob for auditor count, verdicts aggregated by majority with disagreement recorded rather than averaged away, and the extra cost reported in the existing cost accounting.
- **Trap:** disagreement is signal. Do not collapse a 2–1 split into a clean pass; the split is exactly what a reviewer needs to see.

### 8. Shadow routing / policy A-B
- [ ] **Why it matters:** The README argues these traces are the training data for a future learned router. Today they only record the road taken. Logging what a *second* policy would have chosen — without executing it — produces comparison data at zero extra inference cost.
- **Done looks like:** an optional shadow policy on the Broker; both decisions in the trace; a small report comparing chosen vs shadow cost and outcomes.
- **Trap:** the shadow must never execute, never audit, and never affect the real decision. Zero added spend is the entire point.

---

## Tier 3 — infrastructure and fidelity

### 9. Prompt-cache-aware costing
- [ ] **Why it matters:** Cache reads cost ~10% of standard input on major vendors, and the starter catalog already notes it is quoting cache-miss rates. Agentic workloads re-send large stable prefixes constantly, so the modelled cost of exactly the workload this library targets is systematically too high.
- **Done looks like:** optional cache-hit-rate assumptions per task, applied in estimation, clearly labelled as assumptions rather than observations.
- **Trap:** an assumed cache hit rate is an input, not a measurement. Never let it appear in a "measured cost" field.

### 10. Async provider layer
- [ ] **Why it matters:** Every call is blocking. A five-step pipeline with independent steps runs strictly serially, and audits serialise behind generation.
- **Done looks like:** `asyncio` variants of Provider and Broker, the existing sync API unchanged and still tested, no new dependencies.
- **Trap:** do not fork the codebase into two half-maintained paths. Share the routing, auditing, and accounting logic.

### 11. Measured latency classes
- [ ] **Why it matters:** `fast` / `medium` / `slow` are assigned by tier and never measured, yet latency is a full third of the `balanced` policy weighting. One of three routing inputs is currently a guess.
- **Done looks like:** wall-clock timing recorded per attempt in traces, and a report of observed p50/p95 per model to inform the catalog's latency classes.
- **Trap:** mock timings are meaningless. The report must state whether the traces came from real providers, and refuse to draw conclusions from mock runs.

### 12. Hard capability flags
- [ ] **Why it matters:** Capabilities are all soft 0–1 scores, but some requirements are binary: JSON mode, tool use, vision, a minimum context. A model either supports structured output or it does not, and no amount of cheapness compensates.
- **Done looks like:** optional boolean feature flags on `ModelSpec`, requestable per Task, enforced as a gate with the exclusion explained in the rationale.
- **Trap:** flags are gates. Resist folding them into the weighted score.

---

## Tier 4 — explainability polish

### 13. Counterfactual "why not X?" explainer
- [ ] **Why it matters:** The rationale explains why the winner won. The question people actually ask is why their preferred model lost, and the ranked list holds every number needed to answer it.
- **Done looks like:** a function taking a decision and a model id, returning the specific losing margin by component — "lost on cost: 0.31 vs 0.88, worth 0.17 under this policy".
- **Trap:** compute it from the existing ranked list. Do not re-run routing.

### 14. Close the triage held-out gap
- [ ] **Why it matters:** Triage scores 100% on its tuned set and 60% held out. The gap is honestly reported, but 60% is still the real number.
- **Done looks like:** a genuine generalisation improvement, *or* a measured recommendation to default to the model-based layer, backed by accuracy and cost numbers.
- **Trap:** **do not add keywords that make the held-out prompts pass.** There is a comment in `evals/triage_eval.py` saying exactly this. Tuning against the held-out set destroys the only honest measurement in the repository and converts a real result into a decorative one. If the set ever becomes tuned, it must be relabelled as tuned.

---

## When this list is empty

Do not invent work. No refactors for their own sake, no speculative abstractions, no README inflation. Run the maintenance pass instead:

- Re-check every price in `examples/starter_catalog.json` against the `_source` URL on that entry; update values and `_last_verified` if they moved, and call out each change.
- Verify README claims still match reality: test counts, eval percentages, captured example output.
- If everything is accurate and nothing above is unchecked, open an issue describing the repo's state and what you would suggest next, and stop.

An empty run is a good outcome. Padding the log is not.
