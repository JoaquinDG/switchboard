"""Wiring a real catalog to real vendors.

`mock_pool(registry)` gives every provider in a catalog an offline stand-in.
This is its counterpart: the same catalog, pointed at the actual APIs.

Keys come from environment variables and nowhere else. They are never read
from a file, never written to one, never logged, and never included in a
trace. The only thing this module reports about a key is whether it is set.

Every vendor here speaks either the Anthropic Messages shape or the OpenAI
chat-completions shape, so two adapters cover four providers. The mapping
below is the only place that knows which is which.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .base import ProviderPool
from .http import AnthropicProvider, OpenAICompatibleProvider


@dataclass(frozen=True)
class ProviderSpec:
    """How to build a live adapter for one provider name in a catalog."""

    env_var: str
    base_url: str | None = None
    # OpenAI's own newer models reject `max_tokens`; most compatible vendors
    # only accept it. None means "let the adapter decide from the base URL".
    max_tokens_param: str | None = None
    signup_url: str = ""


# provider name (as it appears in a catalog's `provider` field) -> how to build it
KNOWN_PROVIDERS: dict[str, ProviderSpec] = {
    "anthropic": ProviderSpec(
        env_var="ANTHROPIC_API_KEY",
        signup_url="https://console.anthropic.com/settings/keys",
    ),
    "openai": ProviderSpec(
        env_var="OPENAI_API_KEY",
        signup_url="https://platform.openai.com/api-keys",
    ),
    "deepseek": ProviderSpec(
        env_var="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
        max_tokens_param="max_tokens",
        signup_url="https://platform.deepseek.com/api_keys",
    ),
    "google": ProviderSpec(
        env_var="GEMINI_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        max_tokens_param="max_tokens",
        signup_url="https://aistudio.google.com/apikey",
    ),
    "moonshot": ProviderSpec(
        # Kimi. The vendor rebranded from moonshot.ai to kimi.ai, but the API
        # host and the env var everyone's shell already exports both still say
        # moonshot, so that is what the catalog keys on.
        env_var="MOONSHOT_API_KEY",
        base_url="https://api.moonshot.ai",
        max_tokens_param="max_tokens",
        signup_url="https://platform.kimi.ai/console/api-keys",
    ),
}


def key_status(providers: list[str] | None = None) -> dict[str, bool]:
    """Which providers have a key set. Reports presence only, never values."""
    names = providers if providers is not None else sorted(KNOWN_PROVIDERS)
    status = {}
    for name in names:
        spec = KNOWN_PROVIDERS.get(name)
        status[name] = bool(spec and os.environ.get(spec.env_var))
    return status


def build_provider(name: str):
    """Build one live adapter by catalog provider name."""
    spec = KNOWN_PROVIDERS.get(name)
    if spec is None:
        raise KeyError(
            f"unknown provider {name!r}; known: {sorted(KNOWN_PROVIDERS)}. "
            f"Add it to KNOWN_PROVIDERS, or build the adapter yourself and "
            f"pass it to ProviderPool directly."
        )
    if name == "anthropic":
        return AnthropicProvider()
    return OpenAICompatibleProvider(
        base_url=spec.base_url or "https://api.openai.com",
        name=name,
        env_var=spec.env_var,
        max_tokens_param=spec.max_tokens_param,
    )


def live_pool(registry, *, skip_missing_keys: bool = True) -> tuple[ProviderPool, list[str]]:
    """Real adapters for every provider a catalog names.

    Returns the pool and the list of provider names skipped for want of a key.

    Skipping rather than failing is deliberate: a catalog spanning four
    vendors is still useful with two keys, and routing around the absent ones
    is exactly the behaviour the broker already has for an outage. The caller
    gets the skip list so it can say so out loud instead of quietly routing a
    narrower catalog than the user believes they configured.
    """
    names = sorted({m.provider for m in registry.all()})
    built, skipped = [], []
    for name in names:
        spec = KNOWN_PROVIDERS.get(name)
        if spec is None:
            if not skip_missing_keys:
                raise KeyError(f"unknown provider {name!r} in catalog")
            skipped.append(name)
            continue
        if not os.environ.get(spec.env_var):
            if not skip_missing_keys:
                raise KeyError(f"{spec.env_var} is not set (needed for provider {name!r})")
            skipped.append(name)
            continue
        built.append(build_provider(name))
    return ProviderPool(built), skipped


def usable_registry(registry, pool: ProviderPool):
    """The subset of a catalog whose providers are actually wired up.

    Without this, routing can pick a model whose provider has no key and only
    discover it at call time. The broker would treat that as an outage and
    fail over, which works but muddies a live exercise: you want to measure
    routing, not key coverage.
    """
    from ..registry import Registry

    return Registry(
        [m for m in registry.all() if pool.has(m.provider)],
        metadata=dict(registry.metadata),
    )
