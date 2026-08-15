"""Provider failure handling: retries in the adapter, rerouting in the broker.

A routing layer that cannot route around a provider outage is not doing the
one job that distinguishes it from a hardcoded model id. None of these tests
touch the network — urlopen is patched and sleeps are injected.
"""

import email.message
import io
import json
import unittest
import urllib.error
from unittest import mock

from switchboard import (
    BALANCED,
    AnthropicProvider,
    Broker,
    OpenAICompatibleProvider,
    Policy,
    ProviderConfigError,
    ProviderError,
    ProviderPool,
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
    ScriptedProvider,
    Task,
    demo_registry,
)

PASS_VERDICT = '{"pass": true, "score": 0.9, "issues": []}'


class FakeResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def http_error(code, *, retry_after=None, body="upstream detail"):
    headers = email.message.Message()
    if retry_after is not None:
        headers["retry-after"] = str(retry_after)
    return urllib.error.HTTPError(
        "https://example.test/v1/messages", code, "err", headers, io.BytesIO(body.encode())
    )


ANTHROPIC_OK = {
    "content": [{"type": "text", "text": "hello"}],
    "usage": {"input_tokens": 11, "output_tokens": 7},
}


class RecordingSleep:
    def __init__(self):
        self.delays = []

    def __call__(self, seconds):
        self.delays.append(seconds)


class HTTPRetryTests(unittest.TestCase):
    def make(self, **kw):
        self.sleep = RecordingSleep()
        return AnthropicProvider(api_key="k", sleep=self.sleep, **kw)

    def test_retries_a_429_then_succeeds(self):
        provider = self.make(max_retries=2)
        with mock.patch(
            "switchboard.providers.http.urllib.request.urlopen",
            side_effect=[http_error(429), FakeResponse(ANTHROPIC_OK)],
        ) as urlopen:
            result = provider.complete("some-model", "hi")
        self.assertEqual(result.text, "hello")
        self.assertEqual(result.input_tokens, 11)
        self.assertEqual(urlopen.call_count, 2)
        self.assertEqual(len(self.sleep.delays), 1)

    def test_retries_a_529_overload(self):
        provider = self.make(max_retries=1)
        with mock.patch(
            "switchboard.providers.http.urllib.request.urlopen",
            side_effect=[http_error(529), FakeResponse(ANTHROPIC_OK)],
        ):
            self.assertEqual(provider.complete("m", "hi").text, "hello")

    def test_exhausted_retries_raise_typed_error(self):
        provider = self.make(max_retries=2)
        with mock.patch(
            "switchboard.providers.http.urllib.request.urlopen",
            side_effect=[http_error(429)] * 3,
        ) as urlopen:
            with self.assertRaises(ProviderRateLimited):
                provider.complete("m", "hi")
        self.assertEqual(urlopen.call_count, 3)  # initial + 2 retries

    def test_server_error_surfaces_as_unavailable(self):
        provider = self.make(max_retries=0)
        with mock.patch(
            "switchboard.providers.http.urllib.request.urlopen",
            side_effect=http_error(503),
        ):
            with self.assertRaises(ProviderUnavailable):
                provider.complete("m", "hi")

    def test_retry_after_header_is_honored(self):
        provider = self.make(max_retries=1, backoff_cap=30.0)
        with mock.patch(
            "switchboard.providers.http.urllib.request.urlopen",
            side_effect=[http_error(429, retry_after=4), FakeResponse(ANTHROPIC_OK)],
        ):
            provider.complete("m", "hi")
        self.assertEqual(self.sleep.delays, [4.0])

    def test_retry_after_is_capped(self):
        provider = self.make(max_retries=1, backoff_cap=5.0)
        with mock.patch(
            "switchboard.providers.http.urllib.request.urlopen",
            side_effect=[http_error(429, retry_after=600), FakeResponse(ANTHROPIC_OK)],
        ):
            provider.complete("m", "hi")
        self.assertEqual(self.sleep.delays, [5.0])

    def test_backoff_grows_and_stays_under_the_cap(self):
        provider = self.make(max_retries=3, backoff_base=1.0, backoff_cap=4.0)
        with mock.patch(
            "switchboard.providers.http.urllib.request.urlopen",
            side_effect=[http_error(503)] * 4,
        ):
            with self.assertRaises(ProviderUnavailable):
                provider.complete("m", "hi")
        self.assertEqual(len(self.sleep.delays), 3)
        self.assertTrue(all(d <= 4.0 for d in self.sleep.delays), self.sleep.delays)

    def test_auth_failure_is_a_config_error_and_is_not_retried(self):
        # Retrying a bad key just burns latency; rerouting hides the bug.
        provider = self.make(max_retries=3)
        with mock.patch(
            "switchboard.providers.http.urllib.request.urlopen",
            side_effect=http_error(401),
        ) as urlopen:
            with self.assertRaises(ProviderConfigError):
                provider.complete("m", "hi")
        self.assertEqual(urlopen.call_count, 1)
        self.assertEqual(self.sleep.delays, [])

    def test_bad_request_is_not_retried(self):
        provider = self.make(max_retries=3)
        with mock.patch(
            "switchboard.providers.http.urllib.request.urlopen",
            side_effect=http_error(400),
        ) as urlopen:
            with self.assertRaises(ProviderError):
                provider.complete("m", "hi")
        self.assertEqual(urlopen.call_count, 1)

    def test_connection_failure_is_unavailable(self):
        provider = self.make(max_retries=0)
        with mock.patch(
            "switchboard.providers.http.urllib.request.urlopen",
            side_effect=urllib.error.URLError("no route to host"),
        ):
            with self.assertRaises(ProviderUnavailable):
                provider.complete("m", "hi")

    def test_timeout_is_typed_as_timeout(self):
        provider = self.make(max_retries=0)
        with mock.patch(
            "switchboard.providers.http.urllib.request.urlopen",
            side_effect=TimeoutError("timed out"),
        ):
            with self.assertRaises(ProviderTimeout):
                provider.complete("m", "hi")

    def test_missing_key_is_a_config_error(self):
        with self.assertRaises(ProviderConfigError):
            AnthropicProvider(api_key="").complete("m", "hi")

    def test_garbage_response_body_is_a_provider_error(self):
        provider = self.make(max_retries=0)
        bad = mock.MagicMock()
        bad.__enter__ = lambda s: s
        bad.__exit__ = lambda s, *a: False
        bad.read = lambda: b"<html>gateway</html>"
        with mock.patch("switchboard.providers.http.urllib.request.urlopen", return_value=bad):
            with self.assertRaises(ProviderError):
                provider.complete("m", "hi")


class OpenAIAdapterTests(unittest.TestCase):
    OK = {
        "choices": [{"message": {"content": "hi there"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3},
    }

    def sent_body(self, provider):
        with mock.patch(
            "switchboard.providers.http.urllib.request.urlopen",
            return_value=FakeResponse(self.OK),
        ) as urlopen:
            provider.complete("gpt-x", "hi")
        return json.loads(urlopen.call_args[0][0].data)

    def test_openai_endpoint_uses_max_completion_tokens(self):
        # OpenAI's newer models reject max_tokens outright.
        body = self.sent_body(OpenAICompatibleProvider(api_key="k"))
        self.assertIn("max_completion_tokens", body)
        self.assertNotIn("max_tokens", body)

    def test_compatible_vendors_keep_max_tokens(self):
        body = self.sent_body(
            OpenAICompatibleProvider(api_key="k", base_url="https://api.groq.com/openai")
        )
        self.assertIn("max_tokens", body)
        self.assertNotIn("max_completion_tokens", body)

    def test_parameter_name_is_overridable(self):
        body = self.sent_body(
            OpenAICompatibleProvider(api_key="k", max_tokens_param="max_tokens")
        )
        self.assertIn("max_tokens", body)

    def test_response_is_parsed(self):
        with mock.patch(
            "switchboard.providers.http.urllib.request.urlopen",
            return_value=FakeResponse(self.OK),
        ):
            result = OpenAICompatibleProvider(api_key="k").complete("gpt-x", "hi")
        self.assertEqual(result.text, "hi there")
        self.assertEqual(result.output_tokens, 3)


class BrokerFailoverTests(unittest.TestCase):
    def broker(self, script, policy=BALANCED):
        return Broker(demo_registry(), ProviderPool([ScriptedProvider(script, name="mock")]), policy)

    def easy_task(self):
        return Task(prompt="pull the dates", task_type="extraction", complexity=0.2)

    def test_outage_reroutes_to_the_next_ranked_model(self):
        result = self.broker(
            {
                "atlas-small": [ProviderUnavailable("small is down")],
                "atlas-mid": ["mid output"],
                "atlas-frontier": [PASS_VERDICT],
            }
        ).run(self.easy_task())
        self.assertEqual(result.final_model, "atlas-mid")
        self.assertTrue(result.failed_over)
        self.assertTrue(result.verified)
        self.assertEqual([a.role for a in result.attempts], ["initial", "failover"])
        self.assertIn("small is down", result.attempts[0].error)

    def test_failover_does_not_consume_escalation_budget(self):
        # Availability and quality are different budgets; conflating them
        # means one outage silently disables escalation for that task.
        result = self.broker(
            {
                "atlas-small": [ProviderUnavailable("down")],
                "atlas-mid": ["mid output"],
                "atlas-frontier": [PASS_VERDICT],
            }
        ).run(self.easy_task())
        self.assertFalse(result.escalated)

    def test_failover_budget_is_respected(self):
        policy = Policy("nofail", 0.5, 0.3, 0.2, max_provider_failovers=0)
        with self.assertRaises(ProviderError):
            self.broker(
                {
                    "atlas-small": [ProviderUnavailable("down")],
                    "atlas-mid": ["mid output"],
                    "atlas-frontier": [PASS_VERDICT],
                },
                policy,
            ).run(self.easy_task())

    def test_total_outage_raises_after_exhausting_options(self):
        with self.assertRaises(ProviderError) as ctx:
            self.broker(
                {
                    "atlas-small": [ProviderUnavailable("down")],
                    "atlas-mid": [ProviderUnavailable("down")],
                    "atlas-frontier": [ProviderUnavailable("down")],
                }
            ).run(self.easy_task())
        self.assertIn("exhausted", str(ctx.exception))

    def test_config_error_aborts_instead_of_rerouting(self):
        # A missing key should not quietly move traffic to a pricier provider.
        with self.assertRaises(ProviderError):
            self.broker(
                {
                    "atlas-small": [ProviderConfigError("ANTHROPIC_API_KEY is not set")],
                    "atlas-mid": ["mid output"],
                    "atlas-frontier": [PASS_VERDICT],
                }
            ).run(self.easy_task())

    def test_auditor_outage_fails_closed_rather_than_passing(self):
        # Verification must not evaporate exactly when the platform is sick.
        result = self.broker(
            {
                "atlas-small": ["small output"],
                "atlas-mid": ["mid output"],
                "atlas-frontier": [ProviderUnavailable("auditor down")],
            }
        ).run(self.easy_task())
        self.assertFalse(result.verified)
        self.assertTrue(
            any("auditor unavailable" in i for a in result.attempts for i in a.audit_issues)
        )


if __name__ == "__main__":
    unittest.main()
