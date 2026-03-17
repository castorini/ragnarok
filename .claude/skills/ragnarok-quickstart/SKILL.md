---
name: ragnarok-quickstart
description: Use when working with ragnarok CLI commands (generate, validate, convert), prompt modes (chatqa, ragnarok_v4, etc.), dataset-backed or file-backed generation, TREC RAG output formats, or introspection (doctor, describe, schema). Covers all entry points, flags, and generation workflows.
---

# Ragnarok Quickstart

Reference for the `ragnarok` CLI — the tool for RAG answer generation with citation support, TREC evaluation, and multi-stage retrieval pipelines.

## CLI Entry Point

```bash
ragnarok <command> [options]
```

## Primary Commands

| Command | Purpose |
|---------|---------|
| `generate` | Generate cited answers from query + candidate passages |
| `validate` | Validate request payloads or TREC RAG output files |
| `convert` | Convert output to TREC submission formats |

## Introspection Commands

| Command | Purpose |
|---------|---------|
| `doctor` | Check Python version, API keys, backend readiness |
| `describe <cmd>` | Machine-readable command contract |
| `schema <name>` | Print JSON Schema for inputs/outputs |
| `prompt list\|show\|render` | Inspect and render prompt templates |
| `view <path>` | Inspect existing artifact files |

## Two Generation Modes

### File-backed (batch JSONL)
```bash
ragnarok generate --input-file requests.jsonl --output-file answers.jsonl \
  --model gpt-4o --prompt-mode ragnarok_v4
```

### Dataset-backed (retrieval + generation)
```bash
ragnarok generate --dataset rag24.raggy-dev \
  --retrieval-method bm25,rank_zephyr_rho --topk 100,20 \
  --model gpt-4o --prompt-mode ragnarok_v4 --output-file answers.jsonl
```

## Reference Files

Read these on demand for details:

- `references/cli-examples.md` — Common invocations for each command
- `references/input-output-examples.md` — JSONL format examples
- `references/prompt-modes.md` — Prompt mode descriptions and selection guide
- `references/dataset-workflow.md` — Dataset-backed generation details

## Key Concepts

- **Prompt modes**: Template families controlling answer format and citation style (e.g., `ragnarok_v4`, `chatqa`)
- **Cited answers**: Each sentence has a `citations` array referencing candidate indices
- **topk**: Comma-separated pipeline depths (e.g., `100,20` = retrieve 100, rerank to 20)
- **Backends**: OpenAI/compatible (default), Cohere (`command-r*` models), open-source (`llama`, `mistral`, `qwen`)

## Gotchas

- `--model` and `--prompt-mode` are both **required** for `generate`.
- `--dataset` mode does not support `--execution-mode async` yet.
- `--topk` is comma-separated: first value is retrieval depth, last value is generation depth. For file-backed mode, only the last value matters.
- Model name detection is automatic: `command-r*` → Cohere backend, `llama`/`mistral`/`qwen` → open-source backend, everything else → OpenAI-compatible.
- `--retrieval-method` is comma-separated and must align with `--topk` for multi-stage pipelines.
- TREC RAG 2025 validation (`validate rag25-output`) has `--fix-length`, `--fix-citations`, and `--apply-fixes` flags for repairing submissions.
- Config file (`.ragnarok.toml` or `~/.config/ragnarok/config.toml`) can set defaults; CLI flags override.
