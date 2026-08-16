"""Real API adapters (Anthropic + OpenAI-compatible), stdlib-only.

These are deliberately thin: authentication, request shape, response parsing,
and the retry behaviour every network client needs. Model ids, costs, and
capability scores all live in *your* registry catalog, not here — vendors
change models and prices faster than code should.

Failures are translated into the typed errors in `base.py`. That translation
is the point: a 429 or a 529 is an availability problem the broker can route
around, while a missing API key is a configuration bug that should stop the
run. Collapsing both into a raw urllib exception makes a routing layer that
cannot route around the one thing it exists to route around.

Both adapters read API keys from environment variables and are only exercised
when you wire them into a ProviderPool; the test suite never touches the
network.
"""

from __future__ import annotations

import json
import os
import random
import socket
import time
import urllib.error
import urllib.request

from .base import (
    Completion,
    ProviderConfigError,
    ProviderError,
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
)

# Status codes worth trying again: transient by definition.
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


class _HTTPProviderBase:
    """Shared transport: retries, backoff, and error translation."""

    name = "http"
    # A real vendor call, not a canned reply. See Provider.synthetic in base.py.
    synthetic = False

    def __init__(
        self,
        *,
        timeout: float = 120.0,
        max_retries: int = 2,
        backoff_base: float = 0.5,
        backoff_cap: float = 8.0,
        sleep=time.sleep,
    ) -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self._sleep = sleep  # injectable so tests do not actually wait

    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        """Exponential backoff with jitter, unless the server named a delay."""
        if retry_after is not None:
            return min(retry_after, self.backoff_cap)
        delay = min(self.backoff_base * (2**attempt), self.backoff_cap)
        return delay * (0.5 + random.random() / 2)  # full-ish jitter

    @staticmethod
    def _retry_after(headers) -> float | None:
        raw = headers.get("retry-after") if headers else None
        if not raw:
            return None
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            return None  # HTTP-date form; fall back to computed backoff

    def _request(
        self,
        url: str,
        body: bytes | None,
        headers: dict[str, str],
        model_id: str,
        method: str = "POST",
    ) -> dict:
        """Send JSON with retries; raise a typed ProviderError on failure.

        GET is supported so catalog verification (listing a vendor's models)
        reuses the same auth, retry, and error-translation path as inference.
        """
        last: ProviderError | None = None
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(url, data=body, headers=headers, method=method)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = e.read().decode("utf-8", "replace")[:500]
                except Exception:  # noqa: BLE001 - body is best-effort context
                    pass
                finally:
                    e.close()  # release the connection before retrying
                message = f"{self.name}: HTTP {e.code} for {model_id}: {detail or e.reason}"
                if e.code in (401, 403):
                    raise ProviderConfigError(
                        message, provider=self.name, model_id=model_id
                    ) from None
                if e.code == 429:
                    last = ProviderRateLimited(message, provider=self.name, model_id=model_id)
                elif e.code in _RETRYABLE_STATUS:
                    last = ProviderUnavailable(message, provider=self.name, model_id=model_id)
                else:
                    # 400, 404, 422: our request is wrong. Retrying re-sends
                    # the same bad request; rerouting hides a real bug.
                    raise ProviderError(
                        message, provider=self.name, model_id=model_id
                    ) from None
                retry_after = self._retry_after(e.headers)
            except socket.timeout:
                last = ProviderTimeout(
                    f"{self.name}: timed out after {self.timeout}s for {model_id}",
                    provider=self.name,
                    model_id=model_id,
                )
                retry_after = None
            except urllib.error.URLError as e:
                reason = getattr(e, "reason", e)
                if isinstance(reason, socket.timeout):
                    last = ProviderTimeout(
                        f"{self.name}: timed out after {self.timeout}s for {model_id}",
                        provider=self.name,
                        model_id=model_id,
                    )
                else:
                    last = ProviderUnavailable(
                        f"{self.name}: connection failed for {model_id}: {reason}",
                        provider=self.name,
                        model_id=model_id,
                    )
                retry_after = None
            except json.JSONDecodeError as e:
                raise ProviderError(
                    f"{self.name}: response was not valid JSON for {model_id}: {e}",
                    provider=self.name,
                    model_id=model_id,
                ) from None
            except OSError as e:
                # ConnectionResetError, BrokenPipeError and friends. These are
                # OSErrors but NOT URLErrors: urllib wraps failures during
                # connection *setup*, while a reset mid-response surfaces raw.
                # Found live — a reset while reading an audit response escaped
                # untyped, bypassing both this retry loop and the broker's
                # failover, and took a whole plan run down with it. A dropped
                # connection is the most ordinary transient failure there is;
                # it belongs in the same bucket as a 503.
                last = ProviderUnavailable(
                    f"{self.name}: connection dropped for {model_id}: "
                    f"{type(e).__name__}: {e}",
                    provider=self.name,
                    model_id=model_id,
                )
                retry_after = None

            if attempt < self.max_retries:
                self._sleep(self._backoff(attempt, retry_after))

        assert last is not None  # only reachable after a retryable failure
        raise last


class AnthropicProvider(_HTTPProviderBase):
    """Adapter for the Anthropic Messages API.

    Reads ANTHROPIC_API_KEY from the environment. See
    https://docs.claude.com/en/api/overview for current API details.
    """

    name = "anthropic"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.anthropic.com",
        *,
        version: str = "2023-06-01",
        **transport,
    ) -> None:
        super().__init__(**transport)
        # `is not None`, not `or`: an explicitly passed empty string means
        # "no key", and must NOT fall through to the ambient environment.
        # With `or`, a caller passing a config value that failed to load would
        # silently bill whatever account happens to be exported in the shell.
        self.api_key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.version = version

    def _headers(self) -> dict[str, str]:
        return {
            "content-type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": self.version,
        }

    def list_models(self) -> list[str]:
        """Model ids this key can actually reach, newest first.

        Free to call — no tokens are generated. This is what makes a catalog
        checkable against reality: a `model_id` that looks plausible but does
        not exist fails as a hard 404 at inference time, and a routing layer
        should find that out before it routes production traffic there.
        """
        if not self.api_key:
            raise ProviderConfigError(
                "ANTHROPIC_API_KEY is not set", provider=self.name
            )
        data = self._request(
            f"{self.base_url}/v1/models?limit=1000", None, self._headers(), "", method="GET"
        )
        return [m["id"] for m in data.get("data", []) if "id" in m]

    def complete(self, model_id: str, prompt: str, max_tokens: int = 1024) -> Completion:
        if not self.api_key:
            raise ProviderConfigError(
                "ANTHROPIC_API_KEY is not set", provider=self.name, model_id=model_id
            )
        body = json.dumps(
            {
                "model": model_id,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode()
        data = self._request(
            f"{self.base_url}/v1/messages", body, self._headers(), model_id
        )
        text = "".join(
            block.get("text", "")
            for block in data.get("content", [])
            if block.get("type") == "text"
        )
        usage = data.get("usage", {})
        return Completion(
            text=text,
            model_id=model_id,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            stop_reason=data.get("stop_reason", "") or "",
        )


class OpenAICompatibleProvider(_HTTPProviderBase):
    """Adapter for OpenAI-compatible chat-completions endpoints.

    Works with any vendor exposing the /v1/chat/completions shape. Reads
    OPENAI_API_KEY by default; pass api_key/base_url for other vendors.

    OpenAI's own newer models reject `max_tokens` and require
    `max_completion_tokens`, while most compatible vendors only accept
    `max_tokens`. The parameter name is therefore chosen from the base URL and
    overridable — guessing wrong is a hard 400, not a degraded response.
    """

    name = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com",
        name: str | None = None,
        *,
        max_tokens_param: str | None = None,
        env_var: str = "OPENAI_API_KEY",
        **transport,
    ) -> None:
        super().__init__(**transport)
        # env_var lets one adapter serve every OpenAI-shaped vendor: DeepSeek
        # reads DEEPSEEK_API_KEY, Google reads GEMINI_API_KEY, and no vendor's
        # key is ever silently used against another vendor's endpoint.
        self.env_var = env_var
        # See AnthropicProvider: an explicit "" means no key, never "fall back
        # to the environment". Silently substituting an ambient credential for
        # one a caller deliberately passed is a billing bug, not a convenience.
        self.api_key = api_key if api_key is not None else os.environ.get(env_var, "")
        self.base_url = base_url.rstrip("/")
        if name:
            self.name = name
        self.max_tokens_param = max_tokens_param or (
            "max_completion_tokens" if "api.openai.com" in self.base_url else "max_tokens"
        )

    def _headers(self) -> dict[str, str]:
        return {
            "content-type": "application/json",
            "authorization": f"Bearer {self.api_key}",
        }

    def list_models(self) -> list[str]:
        """Model ids this key can reach. Free — no tokens generated."""
        if not self.api_key:
            raise ProviderConfigError(
                f"{self.env_var} is not set for provider {self.name}", provider=self.name
            )
        data = self._request(
            f"{self.base_url}/v1/models", None, self._headers(), "", method="GET"
        )
        return [m["id"] for m in data.get("data", []) if "id" in m]

    def complete(self, model_id: str, prompt: str, max_tokens: int = 1024) -> Completion:
        if not self.api_key:
            raise ProviderConfigError(
                f"{self.env_var} is not set for provider {self.name}",
                provider=self.name,
                model_id=model_id,
            )
        body = json.dumps(
            {
                "model": model_id,
                self.max_tokens_param: max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode()
        data = self._request(
            f"{self.base_url}/v1/chat/completions", body, self._headers(), model_id
        )
        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content", "") or ""
        usage = data.get("usage", {})
        return Completion(
            text=text,
            model_id=model_id,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            stop_reason=choice.get("finish_reason", "") or "",
        )
