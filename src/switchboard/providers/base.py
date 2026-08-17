"""Provider abstraction.

A Provider is anything that can complete a prompt on a given model. Keeping
this surface tiny (one method) means adding a new vendor is a ~30 line
adapter, and the whole system runs offline against MockProvider — which is
also how the test suite and evals stay free and deterministic.

Providers signal availability failures by raising ProviderError. That is the
contract the broker relies on to reroute: a vendor outage should move the task
to the next-ranked model, not surface as an unhandled urllib exception.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..prompts import AUDIT_PROMPT_HEADER


class ProviderError(RuntimeError):
    """A provider call failed. The broker may reroute to another model."""

    def __init__(self, message: str, *, provider: str = "", model_id: str = "") -> None:
        super().__init__(message)
        self.provider = provider
        self.model_id = model_id


class ProviderTimeout(ProviderError):
    """The provider did not respond in time."""


class ProviderRateLimited(ProviderError):
    """The provider rejected the call for rate/quota reasons (429)."""


class ProviderUnavailable(ProviderError):
    """The provider is down or overloaded (5xx, connection failure)."""


class ProviderConfigError(ProviderError):
    """Missing key or bad configuration. Not retryable, not reroutable."""


@dataclass(frozen=True)
class Completion:
    """A provider's reply plus the token counts the bill is computed from."""

    text: str
    model_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    # Why generation stopped, verbatim from the vendor ("end_turn", "stop",
    # "max_tokens", "length"). Discarding this made truncation indistinguishable
    # from a complete answer: a reasoning model that spends its whole budget
    # thinking returns little or no visible text, the auditor fails it as poor
    # quality, and the broker pays for an escalation that will truncate again.
    stop_reason: str = ""

    @property
    def truncated(self) -> bool:
        """True when the vendor cut generation off at the token ceiling.

        A truncated output is a mechanical failure, not a quality one. Raising
        max_tokens fixes it; a bigger model does not.
        """
        return self.stop_reason in ("max_tokens", "length")


class Provider(Protocol):
    """Anything that can complete a prompt on a named model.

    One method on purpose: a new vendor should be a ~30 line adapter, and a
    wide interface would push vendor differences into the broker.

    An implementation may also declare a class or instance attribute
    ``synthetic: bool = True`` to mark itself as a canned stand-in rather than
    a real vendor call (see ``MockProvider`` / ``ScriptedProvider`` below).
    It is read with ``getattr(provider, "synthetic", False)`` rather than
    required here, so existing third-party adapters do not have to declare it
    to keep satisfying this protocol. The broker copies it onto every
    ``Attempt`` and audit verdict it produces, so a trace can tell a measured
    outcome from a demo one without guessing from model ids or scores.
    """

    name: str

    def complete(self, model_id: str, prompt: str, max_tokens: int = 1024) -> Completion:
        """Complete `prompt`, raising ProviderError if the call cannot be made.

        Raising the typed error rather than returning a sentinel is what lets
        the broker tell "reroute around this" from "stop, you have a bug".
        """
        ...


class ProviderPool:
    """Maps provider names (as used in the registry) to Provider instances."""

    def __init__(self, providers: list[Provider]) -> None:
        self._providers = {p.name: p for p in providers}

    def get(self, name: str) -> Provider:
        """Resolve a provider name from the catalog to a live adapter."""
        try:
            return self._providers[name]
        except KeyError:
            raise KeyError(
                f"no provider registered for {name!r}; "
                f"available: {sorted(self._providers)}"
            ) from None

    def has(self, name: str) -> bool:
        """Whether a provider is wired up, without raising if it is not."""
        return name in self._providers

    def names(self) -> list[str]:
        """Registered provider names, sorted — handy in error messages."""
        return sorted(self._providers)


class MockProvider:
    """Deterministic offline provider for tests, evals, and demos.

    Behavior hooks (all keyed off the prompt text, so tests can steer it):
    - audit prompts (recognised by AUDIT_PROMPT_HEADER, a real line of the
      real audit prompt) return a failing verdict when the audited text
      contains 'FORCE_AUDIT_FAIL', a passing verdict with adds_value=false
      when it contains 'FORCE_NO_ADDED_VALUE', otherwise a passing verdict
      with adds_value=true.
    - everything else echoes a canned completion tagged with the model id.
    """

    name = "mock"
    # Every verdict this provider hands back is a fixed 0.9/0.35, not a real
    # model's judgement. Downstream trace consumers (evals/catalog_feedback.py)
    # rely on this to keep canned outcomes out of measured statistics.
    synthetic = True

    AUDIT_SENTINEL = AUDIT_PROMPT_HEADER

    def complete(self, model_id: str, prompt: str, max_tokens: int = 1024) -> Completion:
        if self.AUDIT_SENTINEL in prompt:
            if "FORCE_AUDIT_FAIL" in prompt:
                text = (
                    '{"pass": false, "score": 0.35, '
                    '"issues": ["forced failure for testing"], "adds_value": null}'
                )
            elif "FORCE_NO_ADDED_VALUE" in prompt:
                text = (
                    '{"pass": true, "score": 0.9, '
                    '"issues": ["adds no value over its input"], "adds_value": false}'
                )
            else:
                text = '{"pass": true, "score": 0.9, "issues": [], "adds_value": true}'
        else:
            text = f"[{model_id}] completed: {prompt[:80]}"
        return Completion(
            text=text,
            model_id=model_id,
            input_tokens=max(1, len(prompt) // 4),
            output_tokens=max(1, len(text) // 4),
        )


def mock_pool(providers: list[str] | object) -> ProviderPool:
    """A ProviderPool of offline mocks, one per provider name.

    A real catalog names real vendors, so a `ProviderPool([MockProvider()])`
    fails on the first model that is not `mock`. This builds a stand-in for
    every provider a catalog mentions, which is what lets the whole system be
    exercised against a real catalog — including cross-lab auditing and
    provider failover — without a single API key.

    Accepts a list of provider names or anything with an `all()` returning
    ModelSpecs (i.e. a Registry).
    """
    if hasattr(providers, "all"):
        names = sorted({m.provider for m in providers.all()})  # type: ignore[union-attr]
    else:
        names = sorted(set(providers))  # type: ignore[arg-type]
    pool = []
    for name in names:
        provider = MockProvider()
        provider.name = name  # shadow the class attribute per instance
        pool.append(provider)
    return ProviderPool(pool)


class ScriptedProvider:
    """Offline provider that replays a queued script per model.

    MockProvider only knows how to pass or fail wholesale, which means tests
    written against it verify plumbing rather than judgement. ScriptedProvider
    lets a test say exactly what each model returns on each call — including
    malformed audit verdicts and injected outages — so audit parsing,
    escalation, and failover can be exercised for real.

        provider = ScriptedProvider({
            "atlas-small":    ["draft answer"],
            "atlas-mid":      ['{"pass": true, "score": 0.9, "issues": []}'],
        })

    Queue entries are either a string (returned as completion text) or an
    exception instance (raised). The last entry repeats once exhausted, so a
    short script does not have to anticipate retry counts.
    """

    def __init__(
        self,
        script: dict[str, list[str | Exception]] | None = None,
        name: str = "mock",
        default: str | Exception | None = None,
    ) -> None:
        self.name = name
        self.synthetic = True
        self._script: dict[str, list[str | Exception]] = {
            k: list(v) for k, v in (script or {}).items()
        }
        self._default = default
        self.calls: list[tuple[str, str]] = []  # (model_id, prompt), for assertions

    def complete(self, model_id: str, prompt: str, max_tokens: int = 1024) -> Completion:
        self.calls.append((model_id, prompt))
        queue = self._script.get(model_id)
        if queue:
            item = queue.pop(0) if len(queue) > 1 else queue[0]
        elif self._default is not None:
            item = self._default
        else:
            raise ProviderError(
                f"ScriptedProvider has no script for {model_id!r}; "
                f"scripted models: {sorted(self._script)}",
                provider=self.name,
                model_id=model_id,
            )
        if isinstance(item, Exception):
            raise item
        return Completion(
            text=item,
            model_id=model_id,
            input_tokens=max(1, len(prompt) // 4),
            output_tokens=max(1, len(item) // 4),
        )


@dataclass
class FlakyProvider:
    """Wraps a provider and fails its first `fail_times` calls.

    Used to test that retry and failover paths do what they claim without
    reaching the network.
    """

    inner: Provider
    fail_times: int = 1
    error: Exception = field(default_factory=lambda: ProviderUnavailable("injected outage"))
    calls: int = 0

    @property
    def name(self) -> str:  # type: ignore[override]
        return self.inner.name

    @property
    def synthetic(self) -> bool:
        # Wraps whatever it is given; a flaky wrapper around a real adapter is
        # still a real adapter once it stops injecting failures.
        return getattr(self.inner, "synthetic", False)

    def complete(self, model_id: str, prompt: str, max_tokens: int = 1024) -> Completion:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.error
        return self.inner.complete(model_id, prompt, max_tokens)
