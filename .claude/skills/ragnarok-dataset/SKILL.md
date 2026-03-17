---
name: ragnarok-dataset
description: Use when working with ragnarok dataset-backed generation — retrieval configuration, pyserini dependency setup, topk multi-stage pipelines, dataset naming conventions, and retrieval method selection. Use when setting up or debugging dataset-backed workflows.
---

# Ragnarok Dataset Workflow

Detailed reference for ragnarok's dataset-backed generation mode, which combines retrieval and answer generation in a single pipeline.

## When to Use

- Setting up a new dataset-backed generation run
- Debugging retrieval issues (empty results, wrong topk, missing index)
- Choosing retrieval methods and topk values
- Understanding pyserini dependency requirements

## Quick Start

```bash
# 1. Install pyserini extra
uv sync --extra pyserini

# 2. Check readiness
ragnarok doctor --output json

# 3. Run dataset-backed generation
ragnarok generate --dataset rag24.raggy-dev \
  --retrieval-method bm25,rank_zephyr_rho --topk 100,20 \
  --model gpt-4o --prompt-mode ragnarok_v4 --output-file answers.jsonl
```

## Reference Files

- `references/datasets.md` — Known dataset names, retrieval methods, and topk conventions

## Pipeline Architecture

```
Dataset topics
    │
    ▼
[Stage 1: Sparse retrieval]  ← --retrieval-method bm25 --topk 100
    │ top 100 candidates
    ▼
[Stage 2: Neural reranking]  ← --retrieval-method rank_zephyr_rho --topk 20
    │ top 20 candidates
    ▼
[LLM generation]             ← --model gpt-4o --prompt-mode ragnarok_v4
    │
    ▼
Cited answers (JSONL)
```

## Key Configuration

### Multi-stage topk

`--topk 100,20` with `--retrieval-method bm25,rank_zephyr_rho`:
- Stage 1 (bm25): retrieve top 100
- Stage 2 (rank_zephyr_rho): rerank to top 20
- LLM receives the final 20 candidates

### Single-stage

`--topk 100` with `--retrieval-method bm25`:
- BM25 retrieves top 100
- All 100 fed to LLM (may hit context limits on smaller models)

## Gotchas

- **pyserini required**: `uv sync --extra pyserini` or `pip install ragnarok[pyserini]`. Without it, `--dataset` fails with an import error.
- **Java 21 required**: pyserini depends on Lucene via JNI. Install OpenJDK 21.
- **No async**: `--execution-mode async` is not supported with `--dataset`. Use file-backed mode for async.
- **topk alignment**: Number of values in `--topk` must match number of values in `--retrieval-method`.
- **Index download**: First run for a dataset may download large indexes. Check disk space.
- **Output capture**: When `--output-file` is omitted, output goes to stdout — may be lost if mixed with progress bars. Always use `--output-file` for batch runs.
- **Config file**: Defaults can be set in `.ragnarok.toml` or `~/.config/ragnarok/config.toml`. CLI flags override.
