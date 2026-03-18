---
name: ragnarok-install
description: Set up a ragnarok development environment — checks Python 3.11+, installs via uv or pip with cloud extras, and verifies with doctor. Use when someone is onboarding, setting up a fresh clone, or troubleshooting their environment.
---

# ragnarok Install

Development environment setup for [ragnarok](https://github.com/castorini/ragnarok) — RAG answer generation and TREC evaluation.

## Prerequisites

- Python 3.11+
- Git (SSH access to `github.com:castorini`)

## Verify Runtime

```bash
python3 --version   # must be 3.11+
command -v uv       # if present, use uv path; otherwise recommend uv
```

If `uv` is on PATH, use it silently. If not, ask the user once: install uv or proceed with pip.

## Clone (if needed)

If no `pyproject.toml` in cwd:

```bash
git clone git@github.com:castorini/ragnarok.git && cd ragnarok
```

## Install (source — preferred)

### uv path

```bash
uv venv --python 3.11
source .venv/bin/activate
uv sync --group dev --extra cloud
```

### pip path

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[cloud]"
pip install pre-commit pytest
```

### PyPI alternative (mention but don't default to)

```bash
pip install pyragnarok
```

Note: the PyPI package name is `pyragnarok`, not `ragnarok`.

## Smoke Test

```bash
ragnarok doctor --output json
ragnarok --help
```

## Pre-commit (source installs)

```bash
pre-commit install
```

## Reference Files

- `references/extras.md` — Optional dependency stacks (cloud, local, api, pyserini)

## Gotchas

- **PyPI name mismatch**: the package is published as `pyragnarok` on PyPI, but the CLI binary and import name are both `ragnarok`.
- ragnarok uses an async-first design — many internal APIs are coroutines.
- Dev dependencies use PEP 735 `[dependency-groups]` — only `uv sync --group dev` resolves them natively. With pip, install each package manually.
- The `local` extra pulls in PyTorch, vLLM, and transformers — large downloads, only needed for local model inference.
- Test directory is `test/` (no trailing s).
