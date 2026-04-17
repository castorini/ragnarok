# Repo Project Instructions

## Scope
- This repository is a Python package for retrieval-augmented generation (RAG) pipelines and TREC RAG workflows.
- Keep changes focused on `src/ragnarok/**`, `test/**`, and docs that match the feature/fix.

## Language and Runtime
- Primary language: Python.
- Packaging uses `pyproject.toml` + `setuptools` (`src` layout).
- Required Python version is `>=3.11` (`pyproject.toml`).
- README examples assume a Python 3.11 `uv` environment, with conda kept as an optional path.

## Repository Structure
- `src/ragnarok/`: core package code.
- `examples/`: runnable user-facing demos. `rag_demo.py` is the default async inline-hit smoke test and `sync_rag_demo.py` is the synchronous variant.
- `src/ragnarok/scripts/`: runnable scripts (entrypoints for RAG workflows and format/check utilities).
- `src/ragnarok/api/`: API and Gradio/web-server-related code.
- `src/ragnarok/generate/`: model integrations (OpenAI/Cohere/open-source/vLLM paths).
- `src/ragnarok/retrieve_and_rerank/`: retrieval and reranking pipeline components.
- `test/`: `unittest`-style test suites.
- `docs/`: track-specific run/eval docs (`rag24.md`, `rag25.md`, `elo.md`).

## Install and Setup
- Canonical contributor setup uses `uv`:
  - `uv python install 3.11`
  - `uv venv --python 3.11`
  - `source .venv/bin/activate`
  - `uv sync --group dev`
- `uv run ...` is the no-activation fallback for docs and CI.
- Install optional runtime stacks only when needed:
  - `uv sync --extra cloud` for OpenAI/Cohere paths
  - `uv sync --extra local` for open-weight local model paths
  - `uv sync --extra api` for Flask/Gradio and related UI helpers
  - `uv sync --extra pyserini` for `pyserini`-backed retrieval helpers
- Keep the optional conda path in the README for contributors who need it.
- Keep dependency declarations in sync:
  - Base runtime deps: `[project.dependencies]` in `pyproject.toml`
  - Optional runtime stacks: `[project.optional-dependencies]` in `pyproject.toml`
  - Dev deps: `[dependency-groups].dev` in `pyproject.toml`

## Formatting and Linting
- Pre-commit is the canonical formatter/lint entrypoint.
- Install hooks once per clone: `pre-commit install`.
- Run all checks before PR: `uv run pre-commit run --all-files`.
- `pytest` is the canonical test runner, even though much of the existing suite
  still uses `unittest.TestCase` style.
- Configured tools in `.pre-commit-config.yaml`:
  - `uv lock --check`
  - `ruff-check --fix`
  - `ruff-format`
  - `mypy` (repo-local config in `pyproject.toml`, checking `src` and `test`)

## Testing
- Test tiers:
  - `core`: `uv run pytest -q -m core test`
  - `integration`: `uv run pytest -q -m integration test`
  - `live`: opt-in smoke tests such as `RAGNAROK_LIVE_OPENAI_SMOKE=1 uv run pytest -q -m live test`
- Quick example validation:
  - `uv run python examples/rag_demo.py --help`
  - `uv run python examples/sync_rag_demo.py --help`
- Prefer adding/maintaining tests in `test/**` that remain compatible with
  `pytest`; continuing to use `unittest.TestCase` in existing modules is fine.
- Keep `core` and `integration` coverage offline and deterministic; provider-backed checks belong in the `live` tier.
- Use the shared pytest markers `core`, `integration`, and `live` at the module level so CI and local commands stay consistent with the other Castorini Python repos.
- Some tests depend on optional/external packages or data paths; ensure required deps are installed in the active environment before claiming failures/regressions.

## Running Main Pipelines
- Main script: `src/ragnarok/scripts/run_ragnarok.py`.
- Lightweight smoke-test examples: `examples/rag_demo.py` for the async-first path and `examples/sync_rag_demo.py` for synchronous compatibility.
- Typical invocation is in README/docs; key args include:
  - `--model`
  - `--dataset`
  - `--retrieval_method` (comma-separated enum values)
  - `--topk` (comma-separated ints, aligned with retrieval stages)
  - `--prompt_mode`
- Keep `topk` length and retrieval-method stages aligned for multi-stage retrieval+rereank flows.

## Output and Evaluation Conventions
- Track docs define expected output formats and workflow:
  - Retrieval output (TREC run files)
  - JSONL reranker requests
  - Augmented generation outputs with references/citations
- Validation utilities:
  - `src/ragnarok/scripts/check_trec_rag24_gen.py`
  - `src/ragnarok/scripts/validate_trec_rag25_gen.py`
  - `src/ragnarok/scripts/convert_to_trec25_format.py`
- Preserve expected field names and citation/reference constraints when changing generation output code.

## External Integrations and Secrets
- Integrates with OpenAI, Azure OpenAI, Cohere, and open-source model runtimes (vLLM/FastChat style paths).
- Never hardcode API keys or tokens.
- Use environment variables for provider credentials/settings (README and script help text reference required Azure/OpenAI vars).

## Contribution Workflow
- Follow `CONTRIBUTING.md` and `PULL_REQUEST_TEMPLATE.md`.
- For code changes:
  - Update docs when behavior/CLI/output changes.
  - Add or update tests for non-trivial behavior changes.
  - Add or update `docs/release-notes/` entries for user-visible changes.
  - Run formatting/lint + test commands locally before PR.
- Keep PRs scoped and explicit about:
  - Behavior change
  - Any output format impact
  - Any dependency impact

## Practical Guardrails for Contributors
- Do not silently change output schemas used by TREC submission/eval scripts.
- Avoid introducing new heavyweight dependencies unless necessary and justified.
- Preserve CLI backward compatibility for existing script flags when possible.
- If CLI flags/defaults, output schemas, or validator behavior change, document the migration path in the release note.
- When adjusting retrieval/rerank/generation flow, verify downstream scripts still consume outputs correctly.
