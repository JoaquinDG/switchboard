"""Plan schema, validation, and the anti-split bias.

The planner's most important property is not that it splits well — it is that
it *refuses* to split when it is not sure. A planner that shreds a simple
request into six billable steps with six audits destroys the economics the
router exists to protect, and it does so silently. So the negative cases here
outnumber the positive ones on purpose.
"""

import unittest

from switchboard.planner import (
    MAX_STEPS,
    Plan,
    PlanStep,
    PlanValidationError,
    no_split_plan,
    parse_plan,
    plan_heuristic,
    topological_order,
    validate_plan,
)


def step(sid, prompt="do a thing", task_type="reasoning", complexity=0.5,
         est_in=100, est_out=200, depends=()):
    return PlanStep(sid, prompt, task_type, complexity, est_in, est_out, tuple(depends))


def plan(*steps, request="a request", planned_by="heuristic", confidence=0.9):
    return Plan(request, tuple(steps), planned_by, confidence, "because")


class SchemaValidationTests(unittest.TestCase):
    """Strict about meaning. A plan that half-parses is worse than none: it
    spends money on a shape nobody checked."""

    def test_a_well_formed_plan_validates(self):
        validate_plan(plan(step("s1"), step("s2", depends=["s1"])))

    def test_empty_plan_rejected(self):
        with self.assertRaises(PlanValidationError):
            validate_plan(plan(request="x"))

    def test_empty_request_rejected(self):
        with self.assertRaises(PlanValidationError):
            validate_plan(plan(step("s1"), request="   "))

    def test_gapped_step_ids_rejected(self):
        with self.assertRaises(PlanValidationError) as ctx:
            validate_plan(plan(step("s1"), step("s3")))
        self.assertIn("contiguous", str(ctx.exception))

    def test_misordered_step_ids_rejected(self):
        with self.assertRaises(PlanValidationError):
            validate_plan(plan(step("s2"), step("s1")))

    def test_empty_prompt_rejected(self):
        # The spec's example: a six-step plan where step four is empty.
        steps = [step(f"s{i}") for i in range(1, 7)]
        steps[3] = step("s4", prompt="   ")
        with self.assertRaises(PlanValidationError) as ctx:
            validate_plan(plan(*steps))
        self.assertIn("s4", str(ctx.exception))

    def test_unknown_task_type_rejected(self):
        # A label the catalog has no scores for makes the router fall back to
        # a flat prior for every model — worse than a wrong known label.
        with self.assertRaises(PlanValidationError) as ctx:
            validate_plan(plan(step("s1", task_type="legal_analysis")))
        self.assertIn("task_type", str(ctx.exception))

    def test_complexity_out_of_range_rejected(self):
        for bad in (-0.1, 1.5):
            with self.subTest(complexity=bad):
                with self.assertRaises(PlanValidationError):
                    validate_plan(plan(step("s1", complexity=bad)))

    def test_negative_token_estimate_rejected(self):
        with self.assertRaises(PlanValidationError):
            validate_plan(plan(step("s1", est_in=-1)))

    def test_dangling_dependency_rejected(self):
        with self.assertRaises(PlanValidationError) as ctx:
            validate_plan(plan(step("s1"), step("s2", depends=["s9"])))
        self.assertIn("s9", str(ctx.exception))

    def test_self_dependency_rejected(self):
        with self.assertRaises(PlanValidationError) as ctx:
            validate_plan(plan(step("s1", depends=["s1"])))
        self.assertIn("itself", str(ctx.exception))

    def test_duplicate_dependency_rejected(self):
        with self.assertRaises(PlanValidationError):
            validate_plan(plan(step("s1"), step("s2", depends=["s1", "s1"])))

    def test_cycle_rejected(self):
        with self.assertRaises(PlanValidationError) as ctx:
            validate_plan(plan(step("s1", depends=["s2"]), step("s2", depends=["s1"])))
        self.assertIn("cycle", str(ctx.exception))

    def test_too_many_steps_rejected(self):
        steps = [step(f"s{i}") for i in range(1, MAX_STEPS + 2)]
        with self.assertRaises(PlanValidationError) as ctx:
            validate_plan(plan(*steps))
        self.assertIn("malfunction", str(ctx.exception))


class TopologicalOrderTests(unittest.TestCase):
    def test_linear_chain_keeps_its_order(self):
        p = plan(step("s1"), step("s2", depends=["s1"]), step("s3", depends=["s2"]))
        self.assertEqual([s.step_id for s in topological_order(p)], ["s1", "s2", "s3"])

    def test_independent_steps_are_ordered_deterministically(self):
        # Two runs of the same plan must dispatch in the same sequence or the
        # traces stop being comparable.
        p = plan(step("s1"), step("s2"), step("s3", depends=["s1", "s2"]))
        runs = {tuple(s.step_id for s in topological_order(p)) for _ in range(5)}
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs.pop(), ("s1", "s2", "s3"))

    def test_cycle_raises(self):
        with self.assertRaises(PlanValidationError):
            topological_order(plan(step("s1", depends=["s2"]), step("s2", depends=["s1"])))


class AntiSplitBiasTests(unittest.TestCase):
    """The expensive failure is splitting something that did not need it."""

    NOT_COMPOUND = [
        "Summarize this board memo for a non-executive reader.",
        "Extract all the names, email addresses and phone numbers from this sheet.",
        "Refactor this module to remove the duplicated retry logic.",
        "Write landing page copy for our new pricing tier, friendly but direct.",
        "Pull the plan names, prices and limits out of these five competitor pages.",
    ]

    def test_simple_requests_are_not_split(self):
        for request in self.NOT_COMPOUND:
            with self.subTest(request=request[:40]):
                self.assertFalse(plan_heuristic(request).is_split)

    def test_comma_chains_are_not_structure(self):
        # "extract names, emails and phones" is one extraction task. Treating
        # commas as seams is the classic over-split.
        p = plan_heuristic("Extract the names, the emails, the phone numbers and the roles.")
        self.assertFalse(p.is_split)

    def test_length_alone_never_splits(self):
        # The hardest negative: long but single-task. Length is not structure.
        long_single = (
            "Summarize the attached quarterly report. " + "It covers revenue, "
            "headcount, churn, pipeline, regional performance, and the outlook "
            "for the coming period across every business unit. " * 12
        )
        p = plan_heuristic(long_single)
        self.assertFalse(p.is_split, p.rationale)

    def test_a_not_split_plan_still_has_one_step(self):
        # Uniform execution: run_plan always walks a list.
        p = plan_heuristic("Summarize this memo.")
        self.assertEqual(len(p.steps), 1)
        self.assertEqual(p.steps[0].prompt, "Summarize this memo.")

    def test_not_splitting_is_explained(self):
        p = plan_heuristic("Summarize this memo.")
        self.assertTrue(p.rationale)
        self.assertIn("single task", p.rationale)


class SplittingTests(unittest.TestCase):
    def test_sequence_connective_splits(self):
        p = plan_heuristic(
            "Extract the pricing from these pages, then summarize the findings, "
            "and finally draft the new landing page copy."
        )
        self.assertTrue(p.is_split)
        self.assertGreaterEqual(len(p.steps), 2)
        self.assertEqual(p.planned_by, "heuristic")

    def test_numbered_list_splits(self):
        p = plan_heuristic(
            "1. Extract the payment terms from the contract.\n"
            "2. Summarize the obligations.\n"
            "3. Recommend whether we should sign."
        )
        self.assertTrue(p.is_split)
        self.assertEqual(len(p.steps), 3)

    def test_enumeration_is_the_most_confident_signal(self):
        # The writer already did the decomposition; we are only reading it.
        numbered = plan_heuristic("1. Extract the dates.\n2. Recommend a deadline.")
        prose = plan_heuristic("Extract the dates, then recommend a deadline.")
        self.assertGreater(numbered.confidence, prose.confidence)

    def test_split_steps_form_a_linear_chain(self):
        p = plan_heuristic(
            "1. Extract the pricing.\n2. Summarize it.\n3. Recommend a response."
        )
        self.assertEqual(p.steps[0].depends_on, ())
        self.assertEqual(p.steps[1].depends_on, ("s1",))
        self.assertEqual(p.steps[2].depends_on, ("s2",))

    def test_adjacent_same_kind_fragments_are_merged(self):
        # Two sentences describing one job should not buy two audits.
        p = plan_heuristic(
            "1. Extract the names from the sheet.\n"
            "2. Extract the phone numbers from the sheet."
        )
        self.assertFalse(p.is_split, p.rationale)

    def test_every_split_plan_validates(self):
        for request in (
            "Extract the data, then summarize it, then recommend an action.",
            "1. Parse the invoices.\n2. Draft a summary email.",
            "First, extract the totals. Second, tell me whether to approve it.",
        ):
            with self.subTest(request=request[:40]):
                validate_plan(plan_heuristic(request))


class FragmentCoverageTests(unittest.TestCase):
    """A plan that silently omits part of the request is worse than no plan."""

    def test_short_fragments_are_absorbed_not_dropped(self):
        # Regression, found in phase-2 smoke testing: splitting on `then` gave
        # a two-word middle fragment ("summarize it") which a minimum-length
        # filter discarded, producing a plan that never summarised anything.
        request = ("Extract the pricing from these pages, then summarize it, "
                   "then recommend a response.")
        p = plan_heuristic(request)
        joined = " ".join(step.prompt.lower() for step in p.steps)
        for word in ("extract", "summarize", "recommend"):
            self.assertIn(word, joined, f"{word!r} vanished from the plan")

    def test_absorption_is_reported_in_the_signals(self):
        p = plan_heuristic(
            "Extract the pricing, then summarize it, then recommend a response."
        )
        self.assertTrue(any("absorbed" in sig for sig in p.signals), p.signals)

    def test_no_step_is_ever_empty(self):
        for request in (
            "Extract the data, then summarize, then draft the copy, then finally send it.",
            "1. Extract.\n2. Summarize.\n3. Recommend.",
            "First, extract. Second, decide.",
        ):
            with self.subTest(request=request[:40]):
                for step in plan_heuristic(request).steps:
                    self.assertTrue(step.prompt.strip())


class ConfidenceGateTests(unittest.TestCase):
    """Low confidence is the signal that opens the model gate. It has to fire
    on the case the heuristic genuinely cannot judge."""

    def test_unmarked_compound_declines_to_split_with_low_confidence(self):
        p = plan_heuristic(
            "Read this contract, extract the payment terms, and tell me if we should sign it."
        )
        self.assertFalse(p.is_split)
        self.assertLess(p.confidence, 0.25)
        self.assertIn("declining to guess", p.rationale)

    def test_a_clearly_single_task_is_high_confidence(self):
        p = plan_heuristic("Summarize this memo in one paragraph.")
        self.assertGreater(p.confidence, 0.9)

    def test_low_confidence_names_the_work_kinds_it_saw(self):
        p = plan_heuristic(
            "Read this contract, extract the payment terms, and tell me if we should sign it."
        )
        self.assertTrue(any("work-kinds" in s for s in p.signals))


class ParsePlanTests(unittest.TestCase):
    """Tolerant about transport, strict about meaning."""

    GOOD = """{"confidence": 0.8, "rationale": "two steps", "steps": [
        {"step_id": "s1", "prompt": "extract", "task_type": "extraction",
         "complexity": 0.3, "est_input_tokens": 100, "est_output_tokens": 200},
        {"step_id": "s2", "prompt": "summarise", "task_type": "summarization",
         "complexity": 0.4, "est_input_tokens": 100, "est_output_tokens": 200,
         "depends_on": ["s1"]}]}"""

    def test_plain_json_parses(self):
        p = parse_plan(self.GOOD, "the request", "model:m1")
        self.assertEqual(len(p.steps), 2)
        self.assertEqual(p.planned_by, "model:m1")
        self.assertEqual(p.steps[1].depends_on, ("s1",))

    def test_fenced_json_parses(self):
        p = parse_plan(f"```json\n{self.GOOD}\n```", "r", "model:m1")
        self.assertEqual(len(p.steps), 2)

    def test_prose_wrapped_json_parses(self):
        p = parse_plan(f"Sure, here you go:\n{self.GOOD}\nHope that helps.", "r", "model:m1")
        self.assertEqual(len(p.steps), 2)

    def test_depends_on_as_a_bare_string_is_tolerated(self):
        text = self.GOOD.replace('"depends_on": ["s1"]', '"depends_on": "s1"')
        self.assertEqual(parse_plan(text, "r", "model:m1").steps[1].depends_on, ("s1",))

    def test_not_json_rejected(self):
        with self.assertRaises(PlanValidationError):
            parse_plan("I think you should do three things", "r", "model:m1")

    def test_missing_steps_key_rejected(self):
        with self.assertRaises(PlanValidationError):
            parse_plan('{"confidence": 0.9}', "r", "model:m1")

    def test_invalid_plan_content_is_rejected_after_parsing(self):
        # Parses as JSON, fails on meaning — a cycle.
        text = """{"steps": [
            {"step_id": "s1", "prompt": "a", "task_type": "reasoning", "complexity": 0.5,
             "est_input_tokens": 1, "est_output_tokens": 1, "depends_on": ["s2"]},
            {"step_id": "s2", "prompt": "b", "task_type": "reasoning", "complexity": 0.5,
             "est_input_tokens": 1, "est_output_tokens": 1, "depends_on": ["s1"]}]}"""
        with self.assertRaises(PlanValidationError) as ctx:
            parse_plan(text, "r", "model:m1")
        self.assertIn("cycle", str(ctx.exception))

    def test_unknown_task_type_from_a_model_is_rejected(self):
        text = self.GOOD.replace('"extraction"', '"legal_analysis"')
        with self.assertRaises(PlanValidationError):
            parse_plan(text, "r", "model:m1")


class HonestyTests(unittest.TestCase):
    def test_heuristic_plans_are_credited_to_the_heuristic(self):
        self.assertEqual(plan_heuristic("Summarize this.").planned_by, "heuristic")

    def test_no_split_plan_defaults_to_none(self):
        p = no_split_plan("do a thing", "because")
        self.assertEqual(p.planned_by, "none")

    def test_describe_names_the_layer(self):
        self.assertIn("heuristic", plan_heuristic("Summarize this.").describe())

    def test_empty_request_cannot_be_planned(self):
        with self.assertRaises(PlanValidationError):
            plan_heuristic("   ")


if __name__ == "__main__":
    unittest.main()
