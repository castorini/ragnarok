# Ragnarok Dataset-Backed Generation

## Overview

Dataset-backed generation combines retrieval and answer generation in a single command. Instead of providing a pre-built JSONL of query+candidates, you specify a dataset name and retrieval method — ragnarok handles the full pipeline.

## Basic Usage

```bash
ragnarok generate --dataset rag24.raggy-dev \
  --retrieval-method bm25,rank_zephyr_rho --topk 100,20 \
  --model gpt-4o --prompt-mode ragnarok_v4 --output-file answers.jsonl
```

## How It Works

1. Load topics from the named dataset
2. For each topic, run the retrieval pipeline (multi-stage if multiple methods specified)
3. Feed retrieved candidates to the LLM for answer generation
4. Write cited answers to output

## Key Flags

| Flag | Example | Description |
|------|---------|-------------|
| `--dataset` | `rag24.raggy-dev` | Named dataset with prebuilt index |
| `--retrieval-method` | `bm25,rank_zephyr_rho` | Comma-separated retrieval stages |
| `--topk` | `100,20` | Comma-separated depths per stage |

## Retrieval Methods

| Method | Type | Description |
|--------|------|-------------|
| `bm25` | Sparse | BM25 first-stage retrieval (pyserini) |
| `rank_zephyr` | Neural reranker | Zephyr-based reranking |
| `rank_zephyr_rho` | Neural reranker | Zephyr-Rho variant |
| `rank_vicuna` | Neural reranker | Vicuna-based reranking |
| `gpt-4o` | LLM reranker | GPT-4o listwise reranking |
| `gpt-4` | LLM reranker | GPT-4 listwise reranking |
| `gpt-3.5-turbo` | LLM reranker | GPT-3.5 listwise reranking |

## Multi-Stage Pipeline

`--retrieval-method bm25,rank_zephyr_rho --topk 100,20` means:

1. BM25 retrieves top 100 candidates
2. rank_zephyr_rho reranks those 100 down to top 20
3. Top 20 candidates are fed to the LLM for answer generation

## Pyserini Dependency

Dataset-backed mode requires pyserini:

```bash
uv sync --extra pyserini
# or
pip install ragnarok[pyserini]
```

Check readiness with:

```bash
ragnarok doctor --output json
```

## Limitations

- `--execution-mode async` is **not supported** with `--dataset`
- Dataset names must match prebuilt index configurations
- Requires pyserini and its Java dependencies (JDK 21)
