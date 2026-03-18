# ragnarok Optional Dependency Stacks

ragnarok has four optional extras defined in `pyproject.toml`. The default install flow uses `cloud`.

## Extras

### `cloud` (default for dev setup)

API-based LLM backends — lightweight, no GPU needed.

```bash
uv sync --extra cloud
pip install -e ".[cloud]"
```

| Package | Purpose |
|---------|---------|
| `cohere` | Cohere API backend |
| `openai` | OpenAI/Azure API backend |
| `tiktoken` | Token counting for OpenAI models |

### `local`

Local model inference — requires GPU and significant disk space.

```bash
uv sync --extra local
pip install -e ".[local]"
```

| Package | Purpose |
|---------|---------|
| `fschat` | FastChat model serving |
| `spacy` | NLP pipeline (pinned 3.7.2) |
| `stanza` | Stanford NLP toolkit |
| `torch`, `torchaudio`, `torchvision` | PyTorch stack |
| `transformers` | HuggingFace model loading |
| `vllm` | Fast local LLM inference |

### `api`

Web API and UI serving.

```bash
uv sync --extra api
pip install -e ".[api]"
```

| Package | Purpose |
|---------|---------|
| `flask` | REST API server |
| `gradio` | Interactive web UI |
| `pandas` | Data handling for API responses |

### `pyserini`

Pyserini integration for retrieval in evaluation workflows.

```bash
uv sync --extra pyserini
pip install -e ".[pyserini]"
```

| Package | Purpose |
|---------|---------|
| `pyserini` | Lucene-based retrieval (requires Java 21) |

### `all`

Everything — union of all extras above.

```bash
uv sync --extra all
pip install -e ".[all]"
```

## Dev Dependencies (dependency-group)

| Package | Purpose |
|---------|---------|
| `pre-commit` | Git hook management |
| `pytest` | Test runner |
| `shtab` | Shell tab-completion generation |

## Combining Extras

Multiple extras can be combined:

```bash
uv sync --group dev --extra cloud --extra api
pip install -e ".[cloud,api]"
```
