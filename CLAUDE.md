# Repo Project Instructions

## Scope
- This repository is a Python package for retrieval-augmented generation (RAG) pipelines and TREC RAG workflows.
- Keep changes focused on `src/ragnarok/**`, `test/**`, and docs that match the feature/fix.

## Language and Runtime
- Primary language: Python.
- Packaging uses `pyproject.toml` + `setuptools` (`src` layout).
- Required Python version is `>=3.11` (`pyproject.toml`).
- README examples assume a Python 3.11 conda environment.

## Repository Structure
- `src/ragnarok/`: core package code.
- `src/ragnarok/scripts/`: runnable scripts (entrypoints for RAG workflows and format/check utilities).
- `src/ragnarok/api/`: API and Gradio/web-server-related code.
- `src/ragnarok/generate/`: model integrations (OpenAI/Cohere/open-source/vLLM paths).
- `src/ragnarok/retrieve_and_rerank/`: retrieval and reranking pipeline components.
- `test/`: `unittest`-style test suites.
- `docs/`: track-specific run/eval docs (`rag24.md`, `rag25.md`, `elo.md`).

## Install and Setup
- Source install pattern:
  - `pip install -r requirements.txt`
  - `pip install -e .`
- If using GPU models, follow README’s PyTorch install guidance first.
- Keep dependency declarations in sync:
  - Runtime deps: `requirements.txt` (loaded dynamically by `pyproject.toml`).

## Formatting and Linting
- Pre-commit is the canonical formatter/lint entrypoint.
- Install hooks once per clone: `pre-commit install`.
- Run all checks before PR: `pre-commit run --all-files`.
- Configured tools in `.pre-commit-config.yaml`:
  - `black` (Python 3.11)
  - `isort --profile=black` (Python 3.11)
  - `flake8` (currently configured to ignore `E501` and select `F401`).

## Testing
- Default test command (also reflected in `pr-format.yml`):
  - `python -m unittest discover -s test`
- Prefer adding/maintaining `unittest` tests in `test/**` to match current style.
- Some tests depend on optional/external packages or data paths; ensure required deps are installed in the active environment before claiming failures/regressions.

## Running Main Pipelines
- Main script: `src/ragnarok/scripts/run_ragnarok.py`.
- Typical invocation is in README/docs; key args include:
  - `--model_path`
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
  - Run formatting/lint + test commands locally before PR.
- Keep PRs scoped and explicit about:
  - Behavior change
  - Any output format impact
  - Any dependency impact

## Practical Guardrails for Contributors
- Do not silently change output schemas used by TREC submission/eval scripts.
- Avoid introducing new heavyweight dependencies unless necessary and justified.
- Preserve CLI backward compatibility for existing script flags when possible.
- When adjusting retrieval/rerank/generation flow, verify downstream scripts still consume outputs correctly.
