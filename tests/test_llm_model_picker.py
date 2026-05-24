"""Tests for llm-model-picker."""

from __future__ import annotations

import pytest

from llm_model_picker import ModelCatalog, ModelSpec, PickConstraints, PickResult


def _cheap_model(**kw: object) -> ModelSpec:
    """Helper: build a cheap model spec."""
    defaults = dict(
        model_id="cheap-model",
        provider="test",
        input_cost_per_1m=0.5,
        output_cost_per_1m=1.0,
        context_length=8_000,
    )
    defaults.update(kw)  # type: ignore[arg-type]
    return ModelSpec(**defaults)  # type: ignore[arg-type]


def _expensive_model(**kw: object) -> ModelSpec:
    """Helper: build an expensive model spec."""
    defaults = dict(
        model_id="expensive-model",
        provider="test",
        input_cost_per_1m=15.0,
        output_cost_per_1m=75.0,
        context_length=200_000,
    )
    defaults.update(kw)  # type: ignore[arg-type]
    return ModelSpec(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ModelSpec
# ---------------------------------------------------------------------------


def test_model_spec_combined_cost():
    m = ModelSpec(
        "test",
        "provider",
        input_cost_per_1m=3.0,
        output_cost_per_1m=15.0,
        context_length=100_000,
    )
    assert m.combined_cost_per_1m == 18.0


def test_model_spec_defaults():
    m = ModelSpec("m", "p", 1.0, 2.0, 4096)
    assert m.supports_tools is True
    assert m.supports_json_mode is True
    assert m.supports_streaming is True
    assert m.supports_vision is False
    assert m.notes == ""


def test_model_spec_to_dict():
    m = ModelSpec("m", "p", 1.0, 2.0, 4096)
    d = m.to_dict()
    assert d["model_id"] == "m"
    assert d["combined_cost_per_1m"] == 3.0


def test_model_spec_repr():
    m = ModelSpec("mymodel", "openai", 5.0, 15.0, 128_000)
    r = repr(m)
    assert "mymodel" in r
    assert "openai" in r


# ---------------------------------------------------------------------------
# PickConstraints
# ---------------------------------------------------------------------------


def test_pick_constraints_defaults():
    c = PickConstraints()
    assert c.prefer_cheapest is True
    assert c.prefer_largest_context is False
    assert c.requires_vision is False
    assert c.providers == []


# ---------------------------------------------------------------------------
# PickResult
# ---------------------------------------------------------------------------


def test_pick_result_to_dict():
    m = _cheap_model()
    pr = PickResult(best=m, rank=1, score=1.5)
    d = pr.to_dict()
    assert d["rank"] == 1
    assert "model" in d


def test_pick_result_repr():
    m = _cheap_model()
    pr = PickResult(best=m, rank=1, score=1.5)
    assert "1" in repr(pr)


# ---------------------------------------------------------------------------
# ModelCatalog — construction
# ---------------------------------------------------------------------------


def test_empty_catalog():
    c = ModelCatalog()
    assert len(c) == 0
    assert c.all_models == []


def test_catalog_with_models():
    c = ModelCatalog([_cheap_model()])
    assert len(c) == 1


def test_duplicate_model_raises():
    with pytest.raises(ValueError, match="already in the catalog"):
        ModelCatalog([_cheap_model(), _cheap_model()])


def test_catalog_repr():
    c = ModelCatalog()
    assert "ModelCatalog" in repr(c)


# ---------------------------------------------------------------------------
# ModelCatalog — add / remove / get
# ---------------------------------------------------------------------------


def test_add_model():
    c = ModelCatalog()
    m = _cheap_model()
    c.add(m)
    assert len(c) == 1


def test_add_duplicate_raises():
    c = ModelCatalog()
    c.add(_cheap_model())
    with pytest.raises(ValueError):
        c.add(_cheap_model())


def test_remove_model():
    c = ModelCatalog([_cheap_model()])
    c.remove("cheap-model")
    assert len(c) == 0


def test_remove_missing_raises():
    c = ModelCatalog()
    with pytest.raises(KeyError):
        c.remove("nonexistent")


def test_get_existing():
    m = _cheap_model()
    c = ModelCatalog([m])
    got = c.get("cheap-model")
    assert got is m


def test_get_missing_returns_none():
    c = ModelCatalog()
    assert c.get("x") is None


def test_all_models_preserves_order():
    m1 = _cheap_model(model_id="a")
    m2 = _cheap_model(model_id="b")
    c = ModelCatalog([m1, m2])
    ids = [m.model_id for m in c.all_models]
    assert ids == ["a", "b"]


# ---------------------------------------------------------------------------
# ModelCatalog.pick — no constraints
# ---------------------------------------------------------------------------


def test_pick_no_constraints_returns_all():
    c = ModelCatalog([_cheap_model(), _expensive_model()])
    results = c.pick(PickConstraints())
    assert len(results) == 2


def test_pick_empty_catalog_returns_empty():
    c = ModelCatalog()
    results = c.pick(PickConstraints())
    assert results == []


def test_pick_rank_starts_at_1():
    c = ModelCatalog([_cheap_model()])
    results = c.pick(PickConstraints())
    assert results[0].rank == 1


# ---------------------------------------------------------------------------
# ModelCatalog.pick — cost filters
# ---------------------------------------------------------------------------


def test_pick_max_input_cost():
    cheap = _cheap_model()
    expensive = _expensive_model()
    c = ModelCatalog([cheap, expensive])
    results = c.pick(PickConstraints(max_input_cost_per_1m=1.0))
    assert len(results) == 1
    assert results[0].best.model_id == "cheap-model"


def test_pick_max_output_cost():
    cheap = _cheap_model()
    expensive = _expensive_model()
    c = ModelCatalog([cheap, expensive])
    results = c.pick(PickConstraints(max_output_cost_per_1m=5.0))
    assert len(results) == 1
    assert results[0].best.model_id == "cheap-model"


def test_pick_both_cost_limits():
    m1 = _cheap_model(model_id="m1", input_cost_per_1m=0.5, output_cost_per_1m=1.0)
    m2 = _cheap_model(model_id="m2", input_cost_per_1m=1.5, output_cost_per_1m=1.0)
    m3 = _cheap_model(model_id="m3", input_cost_per_1m=0.5, output_cost_per_1m=5.0)
    c = ModelCatalog([m1, m2, m3])
    results = c.pick(
        PickConstraints(max_input_cost_per_1m=1.0, max_output_cost_per_1m=2.0)
    )
    assert len(results) == 1
    assert results[0].best.model_id == "m1"


# ---------------------------------------------------------------------------
# ModelCatalog.pick — context length
# ---------------------------------------------------------------------------


def test_pick_min_context_length():
    small = _cheap_model(model_id="small", context_length=4_000)
    large = _cheap_model(model_id="large", context_length=200_000)
    c = ModelCatalog([small, large])
    results = c.pick(PickConstraints(min_context_length=100_000))
    assert len(results) == 1
    assert results[0].best.model_id == "large"


# ---------------------------------------------------------------------------
# ModelCatalog.pick — capability flags
# ---------------------------------------------------------------------------


def test_pick_requires_vision():
    no_vis = _cheap_model(model_id="no-vis", supports_vision=False)
    has_vis = _cheap_model(model_id="has-vis", supports_vision=True)
    c = ModelCatalog([no_vis, has_vis])
    results = c.pick(PickConstraints(requires_vision=True))
    assert len(results) == 1
    assert results[0].best.model_id == "has-vis"


def test_pick_requires_tools():
    no_tools = _cheap_model(model_id="no-tools", supports_tools=False)
    has_tools = _cheap_model(model_id="has-tools", supports_tools=True)
    c = ModelCatalog([no_tools, has_tools])
    results = c.pick(PickConstraints(requires_tools=True))
    assert len(results) == 1
    assert results[0].best.model_id == "has-tools"


def test_pick_requires_json_mode():
    no_json = _cheap_model(model_id="no-json", supports_json_mode=False)
    has_json = _cheap_model(model_id="has-json", supports_json_mode=True)
    c = ModelCatalog([no_json, has_json])
    results = c.pick(PickConstraints(requires_json_mode=True))
    assert len(results) == 1
    assert results[0].best.model_id == "has-json"


def test_pick_requires_streaming():
    no_stream = _cheap_model(model_id="no-stream", supports_streaming=False)
    has_stream = _cheap_model(model_id="has-stream", supports_streaming=True)
    c = ModelCatalog([no_stream, has_stream])
    results = c.pick(PickConstraints(requires_streaming=True))
    assert len(results) == 1
    assert results[0].best.model_id == "has-stream"


# ---------------------------------------------------------------------------
# ModelCatalog.pick — providers filter
# ---------------------------------------------------------------------------


def test_pick_by_provider():
    ant = _cheap_model(model_id="ant", provider="anthropic")
    oai = _cheap_model(model_id="oai", provider="openai")
    c = ModelCatalog([ant, oai])
    results = c.pick(PickConstraints(providers=["anthropic"]))
    assert len(results) == 1
    assert results[0].best.provider == "anthropic"


def test_pick_provider_case_insensitive():
    ant = _cheap_model(model_id="ant", provider="Anthropic")
    c = ModelCatalog([ant])
    results = c.pick(PickConstraints(providers=["ANTHROPIC"]))
    assert len(results) == 1


def test_pick_multiple_providers():
    ant = _cheap_model(model_id="ant", provider="anthropic")
    oai = _cheap_model(model_id="oai", provider="openai")
    goo = _cheap_model(model_id="goo", provider="google")
    c = ModelCatalog([ant, oai, goo])
    results = c.pick(PickConstraints(providers=["anthropic", "openai"]))
    assert len(results) == 2


# ---------------------------------------------------------------------------
# ModelCatalog.pick — sorting
# ---------------------------------------------------------------------------


def test_pick_prefer_cheapest_default():
    cheap = _cheap_model(
        model_id="cheap", input_cost_per_1m=0.5, output_cost_per_1m=0.5
    )
    pricey = _cheap_model(
        model_id="pricey", input_cost_per_1m=5.0, output_cost_per_1m=10.0
    )
    c = ModelCatalog([pricey, cheap])  # add pricey first
    results = c.pick(PickConstraints(prefer_cheapest=True))
    assert results[0].best.model_id == "cheap"
    assert results[1].best.model_id == "pricey"


def test_pick_prefer_largest_context():
    small = _cheap_model(model_id="small", context_length=4_000)
    large = _cheap_model(model_id="large", context_length=200_000)
    c = ModelCatalog([small, large])
    results = c.pick(
        PickConstraints(prefer_largest_context=True, prefer_cheapest=False)
    )
    assert results[0].best.model_id == "large"


# ---------------------------------------------------------------------------
# ModelCatalog.default()
# ---------------------------------------------------------------------------


def test_default_catalog_has_models():
    c = ModelCatalog.default()
    assert len(c) >= 5


def test_default_catalog_has_anthropic():
    c = ModelCatalog.default()
    providers = {m.provider for m in c.all_models}
    assert "anthropic" in providers


def test_default_catalog_has_openai():
    c = ModelCatalog.default()
    providers = {m.provider for m in c.all_models}
    assert "openai" in providers


def test_default_catalog_pick_cheap_vision():
    c = ModelCatalog.default()
    results = c.pick(PickConstraints(requires_vision=True, prefer_cheapest=True))
    assert len(results) > 0
    assert results[0].best.supports_vision is True
    # first result should be cheaper than last
    if len(results) > 1:
        assert (
            results[0].best.combined_cost_per_1m
            <= results[-1].best.combined_cost_per_1m
        )


def test_default_catalog_large_context():
    c = ModelCatalog.default()
    results = c.pick(PickConstraints(min_context_length=500_000))
    # only gemini models should qualify
    assert len(results) > 0
    for r in results:
        assert r.best.context_length >= 500_000
