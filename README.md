# llm-model-picker

Select the best LLM model from a catalog given cost and capability constraints. Zero dependencies.

## Install

```bash
pip install llm-model-picker
```

## Quick start

```python
from llm_model_picker import ModelCatalog, PickConstraints

catalog = ModelCatalog.default()   # pre-populated with Anthropic/OpenAI/Google/Groq models

constraints = PickConstraints(
    min_context_length=100_000,
    requires_vision=True,
    prefer_cheapest=True,
)

results = catalog.pick(constraints)
for r in results:
    m = r.best
    print(f"{r.rank}. {m.model_id} ({m.provider}) "
          f"${m.combined_cost_per_1m:.2f}/1M ctx={m.context_length:,}")
```

## API

### `ModelCatalog`

| Method | Description |
|---|---|
| `ModelCatalog.default()` | Built-in catalog with ~9 well-known models |
| `ModelCatalog(models)` | Custom catalog from a list of `ModelSpec` |
| `add(model)` | Register a model |
| `remove(model_id)` | Unregister a model |
| `get(model_id)` | Look up a model by id |
| `pick(constraints)` | Return ranked `PickResult` list |

### `PickConstraints`

| Field | Type | Description |
|---|---|---|
| `max_input_cost_per_1m` | `float \| None` | Max USD per 1M input tokens |
| `max_output_cost_per_1m` | `float \| None` | Max USD per 1M output tokens |
| `min_context_length` | `int \| None` | Minimum context window |
| `requires_vision` | `bool` | Must support image input |
| `requires_tools` | `bool` | Must support tool calls |
| `requires_json_mode` | `bool` | Must support JSON mode |
| `requires_streaming` | `bool` | Must support streaming |
| `providers` | `list[str]` | Filter to these providers |
| `prefer_cheapest` | `bool` | Sort by combined cost (default `True`) |
| `prefer_largest_context` | `bool` | Sort by context window |

### `ModelSpec`

`model_id`, `provider`, `input_cost_per_1m`, `output_cost_per_1m`, `context_length`, `supports_vision`, `supports_tools`, `supports_json_mode`, `supports_streaming`, `combined_cost_per_1m`.

## License

MIT
