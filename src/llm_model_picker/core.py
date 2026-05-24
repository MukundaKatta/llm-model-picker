"""Select the best LLM model from a catalog given cost and capability constraints.

Build a :class:`ModelCatalog` (or use the built-in one), set your
:class:`PickConstraints`, and call :meth:`~ModelCatalog.pick` to get
a ranked list of matching :class:`ModelSpec` objects.

Example::

    from llm_model_picker import ModelCatalog, PickConstraints

    catalog = ModelCatalog.default()
    constraints = PickConstraints(
        min_context_length=100_000,
        requires_vision=True,
        prefer_cheapest=True,
    )
    results = catalog.pick(constraints)
    if results:
        print(results[0].best.model_id)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelSpec:
    """Specification for a single LLM model.

    Costs are USD per 1 million tokens.

    Attributes:
        model_id:              Canonical model identifier
                               (e.g. ``"claude-sonnet-4-5"``).
        provider:              Provider name (e.g. ``"anthropic"``).
        input_cost_per_1m:     Input token cost in USD per 1M tokens.
        output_cost_per_1m:    Output token cost in USD per 1M tokens.
        context_length:        Maximum context window in tokens.
        supports_vision:       Whether the model accepts image inputs.
        supports_tools:        Whether the model supports tool/function calling.
        supports_json_mode:    Whether the model supports structured JSON output.
        supports_streaming:    Whether the model supports streaming responses.
        notes:                 Optional free-text notes.
    """

    model_id: str
    provider: str
    input_cost_per_1m: float
    output_cost_per_1m: float
    context_length: int
    supports_vision: bool = False
    supports_tools: bool = True
    supports_json_mode: bool = True
    supports_streaming: bool = True
    notes: str = ""

    @property
    def combined_cost_per_1m(self) -> float:
        """Simple sum of input + output costs (rough budget proxy)."""
        return self.input_cost_per_1m + self.output_cost_per_1m

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "model_id": self.model_id,
            "provider": self.provider,
            "input_cost_per_1m": self.input_cost_per_1m,
            "output_cost_per_1m": self.output_cost_per_1m,
            "context_length": self.context_length,
            "supports_vision": self.supports_vision,
            "supports_tools": self.supports_tools,
            "supports_json_mode": self.supports_json_mode,
            "supports_streaming": self.supports_streaming,
            "combined_cost_per_1m": self.combined_cost_per_1m,
        }

    def __repr__(self) -> str:
        return (
            f"ModelSpec(model_id={self.model_id!r},"
            f" provider={self.provider!r},"
            f" combined_cost={self.combined_cost_per_1m:.2f},"
            f" context={self.context_length})"
        )


@dataclass
class PickConstraints:
    """Constraints for selecting a model.

    All fields are optional — omit to skip that filter.

    Attributes:
        max_input_cost_per_1m:   Reject models more expensive than this for input.
        max_output_cost_per_1m:  Reject models more expensive than this for output.
        min_context_length:      Reject models with a shorter context window.
        requires_vision:         Only accept models that support image input.
        requires_tools:          Only accept models that support tool calls.
        requires_json_mode:      Only accept models that support JSON mode.
        requires_streaming:      Only accept models that support streaming.
        providers:               If non-empty, only accept these providers.
        prefer_cheapest:         Sort results by combined cost (cheapest first).
        prefer_largest_context:  Sort results by context length (largest first).
    """

    max_input_cost_per_1m: float | None = None
    max_output_cost_per_1m: float | None = None
    min_context_length: int | None = None
    requires_vision: bool = False
    requires_tools: bool = False
    requires_json_mode: bool = False
    requires_streaming: bool = False
    providers: list[str] = field(default_factory=list)
    prefer_cheapest: bool = True
    prefer_largest_context: bool = False


@dataclass
class PickResult:
    """A single result from :meth:`ModelCatalog.pick`.

    Attributes:
        best:  The selected :class:`ModelSpec`.
        rank:  1-based rank among all matching models.
        score: Internal sort key (lower is better when prefer_cheapest).
    """

    best: ModelSpec
    rank: int
    score: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "rank": self.rank,
            "score": self.score,
            "model": self.best.to_dict(),
        }

    def __repr__(self) -> str:
        return f"PickResult(rank={self.rank}, model_id={self.best.model_id!r})"


class ModelCatalog:
    """A registry of :class:`ModelSpec` objects.

    Args:
        models: Initial list of models.

    Raises:
        ValueError: If two models share the same ``model_id``.
    """

    def __init__(self, models: list[ModelSpec] | None = None) -> None:
        self._models: dict[str, ModelSpec] = {}
        for m in models or []:
            self.add(m)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, model: ModelSpec) -> None:
        """Register *model* in the catalog.

        Raises:
            ValueError: If ``model.model_id`` is already registered.
        """
        if model.model_id in self._models:
            raise ValueError(f"Model {model.model_id!r} is already in the catalog")
        self._models[model.model_id] = model

    def remove(self, model_id: str) -> None:
        """Remove a model by id.

        Raises:
            KeyError: If ``model_id`` is not in the catalog.
        """
        if model_id not in self._models:
            raise KeyError(f"Model {model_id!r} not found")
        del self._models[model_id]

    def get(self, model_id: str) -> ModelSpec | None:
        """Return the :class:`ModelSpec` for *model_id*, or ``None``."""
        return self._models.get(model_id)

    @property
    def all_models(self) -> list[ModelSpec]:
        """All registered models in insertion order."""
        return list(self._models.values())

    def __len__(self) -> int:
        return len(self._models)

    # ------------------------------------------------------------------
    # Picking
    # ------------------------------------------------------------------

    def pick(self, constraints: PickConstraints) -> list[PickResult]:
        """Return models that satisfy *constraints*, ranked by preference.

        Args:
            constraints: Filtering and sorting parameters.

        Returns:
            Ranked list of :class:`PickResult`.  Empty if nothing matches.
        """
        candidates = list(self._models.values())

        # --- hard filters ---
        if constraints.max_input_cost_per_1m is not None:
            candidates = [
                m
                for m in candidates
                if m.input_cost_per_1m <= constraints.max_input_cost_per_1m
            ]
        if constraints.max_output_cost_per_1m is not None:
            candidates = [
                m
                for m in candidates
                if m.output_cost_per_1m <= constraints.max_output_cost_per_1m
            ]
        if constraints.min_context_length is not None:
            candidates = [
                m
                for m in candidates
                if m.context_length >= constraints.min_context_length
            ]
        if constraints.requires_vision:
            candidates = [m for m in candidates if m.supports_vision]
        if constraints.requires_tools:
            candidates = [m for m in candidates if m.supports_tools]
        if constraints.requires_json_mode:
            candidates = [m for m in candidates if m.supports_json_mode]
        if constraints.requires_streaming:
            candidates = [m for m in candidates if m.supports_streaming]
        if constraints.providers:
            allowed = {p.lower() for p in constraints.providers}
            candidates = [m for m in candidates if m.provider.lower() in allowed]

        # --- sorting ---
        if constraints.prefer_largest_context:
            candidates.sort(key=lambda m: -m.context_length)
        elif constraints.prefer_cheapest:
            candidates.sort(key=lambda m: m.combined_cost_per_1m)

        return [
            PickResult(best=m, rank=i + 1, score=_score(m, constraints))
            for i, m in enumerate(candidates)
        ]

    def __repr__(self) -> str:
        return f"ModelCatalog(models={len(self._models)})"

    # ------------------------------------------------------------------
    # Built-in catalog
    # ------------------------------------------------------------------

    @classmethod
    def default(cls) -> ModelCatalog:
        """Return a catalog pre-populated with well-known models.

        Prices are approximate as of mid-2025; always verify with the
        provider before using them for billing decisions.
        """
        models = [
            # Anthropic
            ModelSpec(
                "claude-haiku-4-5",
                "anthropic",
                input_cost_per_1m=0.80,
                output_cost_per_1m=4.00,
                context_length=200_000,
                supports_vision=True,
                supports_tools=True,
                supports_json_mode=True,
            ),
            ModelSpec(
                "claude-sonnet-4-5",
                "anthropic",
                input_cost_per_1m=3.00,
                output_cost_per_1m=15.00,
                context_length=200_000,
                supports_vision=True,
                supports_tools=True,
                supports_json_mode=True,
            ),
            ModelSpec(
                "claude-opus-4-5",
                "anthropic",
                input_cost_per_1m=15.00,
                output_cost_per_1m=75.00,
                context_length=200_000,
                supports_vision=True,
                supports_tools=True,
                supports_json_mode=True,
            ),
            # OpenAI
            ModelSpec(
                "gpt-4o-mini",
                "openai",
                input_cost_per_1m=0.15,
                output_cost_per_1m=0.60,
                context_length=128_000,
                supports_vision=True,
                supports_tools=True,
                supports_json_mode=True,
            ),
            ModelSpec(
                "gpt-4o",
                "openai",
                input_cost_per_1m=5.00,
                output_cost_per_1m=15.00,
                context_length=128_000,
                supports_vision=True,
                supports_tools=True,
                supports_json_mode=True,
            ),
            ModelSpec(
                "gpt-5",
                "openai",
                input_cost_per_1m=2.00,
                output_cost_per_1m=8.00,
                context_length=128_000,
                supports_vision=True,
                supports_tools=True,
                supports_json_mode=True,
            ),
            # Google
            ModelSpec(
                "gemini-2.5-flash",
                "google",
                input_cost_per_1m=0.15,
                output_cost_per_1m=0.60,
                context_length=1_048_576,
                supports_vision=True,
                supports_tools=True,
                supports_json_mode=True,
            ),
            ModelSpec(
                "gemini-2.5-pro",
                "google",
                input_cost_per_1m=1.25,
                output_cost_per_1m=10.00,
                context_length=1_048_576,
                supports_vision=True,
                supports_tools=True,
                supports_json_mode=True,
            ),
            # Groq (fast inference)
            ModelSpec(
                "llama-3.3-70b-versatile",
                "groq",
                input_cost_per_1m=0.59,
                output_cost_per_1m=0.79,
                context_length=128_000,
                supports_vision=False,
                supports_tools=True,
                supports_json_mode=True,
            ),
        ]
        return cls(models)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _score(model: ModelSpec, constraints: PickConstraints) -> float:
    """Compute a numeric sort score for a model (lower = better)."""
    if constraints.prefer_largest_context:
        return -float(model.context_length)
    # default: cheapest first
    return model.combined_cost_per_1m
