import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from switchboard import cli


def run_cli(argv):
    """Run cli.main and return (exit_code, stdout)."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = cli.main(argv)
    return code, out.getvalue()


class OneLinerDefaultsTests(unittest.TestCase):
    def test_default_run_is_mocked_and_prints_routing_and_cost(self):
        code, out = run_cli(["Summarize this contract in one sentence."])
        self.assertEqual(code, 0)
        self.assertIn("routing:", out)
        self.assertIn("model:", out)
        self.assertIn("cost:", out)
        self.assertIn("MOCKED output", out)
        # Auto task_type means triage ran and is shown in the output.
        self.assertIn("triage:", out)

    def test_prompt_words_are_joined_without_quoting(self):
        code, out = run_cli(["extract", "the", "dates", "from", "this", "email"])
        self.assertEqual(code, 0)
        self.assertIn("routing:", out)

    def test_no_network_and_no_key_needed_by_default(self):
        # Scrub every known provider key from the environment so a default
        # run cannot accidentally succeed by touching a real vendor.
        env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
        with mock.patch.dict(os.environ, env, clear=True):
            code, out = run_cli(["write a short poem about routing"])
        self.assertEqual(code, 0)
        self.assertIn("MOCKED output", out)

    def test_forced_task_type_skips_triage(self):
        code, out = run_cli(["--task-type", "coding", "some ambiguous prompt"])
        self.assertEqual(code, 0)
        self.assertNotIn("triage:", out)

    def test_policy_choice_is_validated_by_argparse(self):
        with self.assertRaises(SystemExit):
            run_cli(["--policy", "not-a-real-policy", "hello"])

    def test_custom_catalog_is_loaded(self):
        catalog = {
            "models": [
                {
                    "model_id": "solo-model",
                    "provider": "onlylab",
                    "tier": "frontier",
                    "input_cost": 1.0,
                    "output_cost": 2.0,
                    "capabilities": {"reasoning": 0.9},
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "catalog.json"
            path.write_text(json.dumps(catalog))
            code, out = run_cli(["--catalog", str(path), "--task-type", "reasoning",
                                  "explain the tradeoff"])
        self.assertEqual(code, 0)
        self.assertIn("solo-model", out)

    def test_trace_is_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "cli.jsonl"
            code, _ = run_cli(["--trace", str(trace_path), "extract the totals"])
            self.assertEqual(code, 0)
            self.assertTrue(trace_path.exists())
            lines = trace_path.read_text().strip().splitlines()
            self.assertEqual(len(lines), 1)


class LiveModeSafetyTests(unittest.TestCase):
    """--live is the one path that can spend money, so it gets its own guardrails."""

    def test_live_without_budget_refuses_before_touching_providers(self):
        code, out = run_cli(["--live", "hello"])
        self.assertEqual(code, 1)
        self.assertIn("--budget-usd", out)

    def test_live_with_no_keys_set_refuses(self):
        env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
        with mock.patch.dict(os.environ, env, clear=True):
            code, out = run_cli(["--live", "--budget-usd", "0.5", "hello"])
        self.assertEqual(code, 1)
        self.assertIn("No provider keys are set", out)

    def test_live_never_calls_a_provider_when_budget_is_zero_or_missing(self):
        # Even with a key present, --budget-usd defaults to 0 and must refuse
        # before any provider is even constructed.
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-fake-not-real"}):
            code, out = run_cli(["--live", "hello"])
        self.assertEqual(code, 1)
        self.assertIn("--budget-usd", out)


if __name__ == "__main__":
    unittest.main()
