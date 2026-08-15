"""Model registry: the catalog Switchboard routes against.

Every model is described by cost, latency class, capability scores, and tier.
Capability scores are *your* judgments (or eval results) on a 0-1 scale per
task type. The registry deliberately makes these explicit and editable:
routing quality is only as good as the catalog, and the catalog is a living
document you should update as models and prices change.

The catalog is also the only place that knows which task types you have
capability data for. The router asks (`has_capability_data`) so that an
unknown or misspelled task type surfaces as a warning instead of silently
collapsing every model onto the same prior.
"""

from __future__ import annotations

import datetime
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

TIERS = ("frontier", "mid", "small")

# A catalog older than this is warned about on load. Vendor prices move, models
# are retired, and a routing decision made on last quarter's price list is a
# confident wrong answer — the failure mode this whole project exists to avoid.
CATALOG_STALE_AFTER_DAYS = 60


class CatalogStaleWarning(UserWarning):
    """The catalog's `_last_verified` date is old enough to distrust."""
LATENCY_CLASSES = ("fast", "medium", "slow")

# Ordering used for escalation and for "degrade upward" fallbacks. Single
# source of truth: the broker and router both read it rather than each
# keeping their own notion of which tier outranks which.
TIER_RANK = {"small": 0, "mid": 1, "frontier": 2}

# Capability assumed for a task type the catalog has no score for. Deliberately
# mid-scale: high enough not to disqualify everything, low enough that the
# router's qualification gate notices.
UNKNOWN_CAPABILITY_PRIOR = 0.5


@dataclass(frozen=True)
class ModelSpec:
    """A single routable model."""

    model_id: str
    provider: str
    tier: str  # one of TIERS
    input_cost: float  # USD per 1M input tokens
    output_cost: float  # USD per 1M output tokens
    latency: str = "medium"  # one of LATENCY_CLASSES
    context_window: int = 200_000
    # 0-1 capability score per task type, e.g. {"reasoning": 0.9, "extraction": 0.7}
    capabilities: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must be a non-empty string")
        if not self.provider:
            raise ValueError(f"{self.model_id}: provider must be a non-empty string")
        if self.tier not in TIERS:
            raise ValueError(f"tier must be one of {TIERS}, got {self.tier!r}")
        if self.latency not in LATENCY_CLASSES:
            raise ValueError(
                f"latency must be one of {LATENCY_CLASSES}, got {self.latency!r}"
            )
        for label, value in (("input_cost", self.input_cost), ("output_cost", self.output_cost)):
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(
                    f"{self.model_id}: {label} must be a non-negative number, got {value!r}"
                )
        if not isinstance(self.context_window, int) or self.context_window <= 0:
            raise ValueError(
                f"{self.model_id}: context_window must be a positive int, "
                f"got {self.context_window!r}"
            )
        # Capability scores are the input the router trusts most; garbage here
        # produces confidently wrong routing, so validate at construction.
        for task_type, score in self.capabilities.items():
            if not isinstance(task_type, str) or not task_type:
                raise ValueError(
                    f"{self.model_id}: capability keys must be non-empty strings, "
                    f"got {task_type!r}"
                )
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError(
                    f"{self.model_id}: capability {task_type!r} must be a number, "
                    f"got {score!r}"
                )
            if not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"{self.model_id}: capability {task_type!r} must be in [0, 1], "
                    f"got {score}"
                )

    def capability_for(self, task_type: str) -> float:
        """Score for a task type; unknown task types fall back to a prior."""
        return self.capabilities.get(task_type, UNKNOWN_CAPABILITY_PRIOR)


class Registry:
    """Holds the model catalog and answers lookup queries."""

    def __init__(
        self,
        models: list[ModelSpec] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self._models: dict[str, ModelSpec] = {}
        # Top-level underscore keys from a loaded catalog (_last_verified,
        # _sources, _disclaimer). Provenance travels with the data so a
        # reviewer can ask where a number came from without leaving the repo.
        self.metadata: dict[str, object] = dict(metadata or {})
        for m in models or []:
            self.add(m)

    @property
    def last_verified(self) -> datetime.date | None:
        """Date the catalog's prices were last checked, if it says."""
        raw = self.metadata.get("_last_verified")
        if not isinstance(raw, str):
            return None
        try:
            return datetime.date.fromisoformat(raw)
        except ValueError:
            return None

    def age_in_days(self, today: datetime.date | None = None) -> int | None:
        """Days since the catalog was last verified, or None if undated."""
        verified = self.last_verified
        if verified is None:
            return None
        return ((today or datetime.date.today()) - verified).days

    def add(self, spec: ModelSpec) -> None:
        """Register a model. Duplicate ids are rejected rather than merged."""
        if spec.model_id in self._models:
            raise ValueError(f"duplicate model_id: {spec.model_id}")
        self._models[spec.model_id] = spec

    def get(self, model_id: str) -> ModelSpec:
        """Look up one model, raising with the offending id if absent."""
        try:
            return self._models[model_id]
        except KeyError:
            raise KeyError(f"unknown model_id: {model_id}") from None

    def all(self) -> list[ModelSpec]:
        """Every model, as a fresh list the caller may sort or filter."""
        return list(self._models.values())

    def by_tier(self, tier: str) -> list[ModelSpec]:
        """Models in one tier — the rungs escalation climbs."""
        return [m for m in self._models.values() if m.tier == tier]

    def known_task_types(self) -> set[str]:
        """Every task type any model in the catalog has a score for."""
        types: set[str] = set()
        for m in self._models.values():
            types.update(m.capabilities)
        return types

    def has_capability_data(self, task_type: str) -> bool:
        """True if at least one model has a real score for this task type.

        When this is False every model scores the same prior, so the quality
        term carries no signal and cost alone decides. The router warns rather
        than pretending the decision was informed.
        """
        return any(task_type in m.capabilities for m in self._models.values())

    def __len__(self) -> int:
        return len(self._models)

    @classmethod
    def from_json(
        cls, path: str | Path, *, today: datetime.date | None = None
    ) -> "Registry":
        """Load a catalog from JSON (see examples/catalog.example.json).

        Errors name the offending entry — a catalog is hand-maintained, and a
        stack trace pointing at ``ModelSpec(**entry)`` tells you nothing about
        which of forty models is malformed.

        Keys beginning with ``_`` are treated as metadata and ignored by the
        loader, at both the top level and inside a model entry, so provenance
        (``_source``, ``_note``) can live beside the number it justifies.

        If the catalog carries ``_last_verified`` and that date is more than
        ``CATALOG_STALE_AFTER_DAYS`` old, a ``CatalogStaleWarning`` is raised.
        ``today`` is injectable so the check can be tested without waiting.
        """
        p = Path(path)
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(f"{p}: not valid JSON ({e})") from None
        if not isinstance(data, dict) or "models" not in data:
            raise ValueError(f"{p}: expected a JSON object with a 'models' key")
        entries = data["models"]
        if not isinstance(entries, list):
            raise ValueError(f"{p}: 'models' must be a list, got {type(entries).__name__}")

        known_fields = set(ModelSpec.__dataclass_fields__)
        models: list[ModelSpec] = []
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError(f"{p}: models[{i}] must be an object, got {entry!r}")
            label = entry.get("model_id", f"models[{i}]")
            fields = {k: v for k, v in entry.items() if not k.startswith("_")}
            unknown = set(fields) - known_fields
            if unknown:
                raise ValueError(
                    f"{p}: {label} has unknown field(s) {sorted(unknown)}; "
                    f"valid fields are {sorted(known_fields)} "
                    f"(prefix a key with '_' to keep it as a note)"
                )
            missing = {"model_id", "provider", "tier", "input_cost", "output_cost"} - set(fields)
            if missing:
                raise ValueError(f"{p}: {label} is missing required field(s) {sorted(missing)}")
            try:
                models.append(ModelSpec(**fields))
            except (ValueError, TypeError) as e:
                raise ValueError(f"{p}: {label}: {e}") from None
        if not models:
            raise ValueError(f"{p}: catalog contains no models")

        registry = cls(models, metadata={k: v for k, v in data.items() if k.startswith("_")})
        age = registry.age_in_days(today)
        if age is not None and age > CATALOG_STALE_AFTER_DAYS:
            warnings.warn(
                f"{p}: prices last verified {registry.last_verified} ({age} days ago). "
                f"Vendor pricing changes and models get retired — re-check the "
                f"catalog against current pricing pages before trusting its routing.",
                CatalogStaleWarning,
                stacklevel=2,
            )
        return registry


def demo_registry() -> Registry:
    """A synthetic catalog used by tests, evals, and the quickstart.

    Costs and scores are illustrative, not claims about real models.
    Swap in your own catalog (examples/catalog.example.json) for real use.
    """
    return Registry(
        [
            ModelSpec(
                model_id="atlas-frontier",
                provider="mock",
                tier="frontier",
                input_cost=3.00,
                output_cost=15.00,
                latency="slow",
                capabilities={
                    "reasoning": 0.95,
                    "coding": 0.93,
                    "creative": 0.90,
                    "summarization": 0.90,
                    "extraction": 0.88,
                    "audit": 0.95,
                },
            ),
            ModelSpec(
                model_id="atlas-mid",
                provider="mock",
                tier="mid",
                input_cost=0.80,
                output_cost=4.00,
                latency="medium",
                capabilities={
                    "reasoning": 0.80,
                    "coding": 0.82,
                    "creative": 0.78,
                    "summarization": 0.85,
                    "extraction": 0.84,
                    "audit": 0.80,
                },
            ),
            ModelSpec(
                model_id="atlas-small",
                provider="mock",
                tier="small",
                input_cost=0.10,
                output_cost=0.50,
                latency="fast",
                capabilities={
                    "reasoning": 0.55,
                    "coding": 0.60,
                    "creative": 0.55,
                    "summarization": 0.75,
                    "extraction": 0.78,
                    "audit": 0.50,
                },
            ),
        ]
    )
