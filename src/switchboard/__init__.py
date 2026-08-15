"""Switchboard: an LLM model brokerage.

Route every task to the most efficient model under an explicit policy,
verify outputs with cross-model audits, escalate when quality fails, reroute
when a provider falls over, and account for every dollar it took.
"""

from .auditor import AUDIT_PROMPT_TEMPLATE, AuditVerdict, audit, pick_auditor
from .broker import Attempt, Broker, BrokerResult
from .policies import (
    BALANCED,
    COST_FIRST,
    PRESETS,
    QUALITY_FIRST,
    NoQualifiedModelError,
    Policy,
    Task,
)
from .prompts import AUDIT_PROMPT_HEADER, ESCALATION_RETRY_TEMPLATE, build_retry_prompt
from .providers.base import (
    Completion,
    FlakyProvider,
    MockProvider,
    Provider,
    ProviderConfigError,
    ProviderError,
    ProviderPool,
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
    ScriptedProvider,
    mock_pool,
)
from .providers.http import AnthropicProvider, OpenAICompatibleProvider
from .providers.live import build_provider, key_status, live_pool, usable_registry
from .registry import (
    CATALOG_STALE_AFTER_DAYS,
    LATENCY_CLASSES,
    TIER_RANK,
    TIERS,
    UNKNOWN_CAPABILITY_PRIOR,
    CatalogStaleWarning,
    ModelSpec,
    Registry,
    demo_registry,
)
from .triage import (
    TASK_TYPES,
    Triage,
    classify_heuristic,
    classify_with_model,
    triage_task,
)
from .router import (
    RoutingDecision,
    ScoredModel,
    actual_cost,
    estimate_cost,
    route,
    score_models,
)

__version__ = "0.3.0"

__all__ = [
    "AUDIT_PROMPT_HEADER",
    "AUDIT_PROMPT_TEMPLATE",
    "AnthropicProvider",
    "Attempt",
    "AuditVerdict",
    "BALANCED",
    "Broker",
    "BrokerResult",
    "CATALOG_STALE_AFTER_DAYS",
    "COST_FIRST",
    "CatalogStaleWarning",
    "Completion",
    "ESCALATION_RETRY_TEMPLATE",
    "FlakyProvider",
    "ModelSpec",
    "MockProvider",
    "NoQualifiedModelError",
    "OpenAICompatibleProvider",
    "PRESETS",
    "Policy",
    "Provider",
    "ProviderConfigError",
    "ProviderError",
    "ProviderPool",
    "ProviderRateLimited",
    "ProviderTimeout",
    "ProviderUnavailable",
    "QUALITY_FIRST",
    "Registry",
    "RoutingDecision",
    "ScoredModel",
    "LATENCY_CLASSES",
    "ScriptedProvider",
    "TASK_TYPES",
    "TIERS",
    "TIER_RANK",
    "Task",
    "Triage",
    "UNKNOWN_CAPABILITY_PRIOR",
    "actual_cost",
    "audit",
    "build_provider",
    "build_retry_prompt",
    "classify_heuristic",
    "classify_with_model",
    "demo_registry",
    "estimate_cost",
    "key_status",
    "live_pool",
    "mock_pool",
    "pick_auditor",
    "route",
    "score_models",
    "triage_task",
    "usable_registry",
]
