# Contributing to Ragnarök

Thank you for contributing to Ragnarök. This repository packages retrieval-augmented generation workflows, TREC Retrieval-Augmented Generation (RAG) utilities, and output validators, so changes should preserve both developer ergonomics and artifact compatibility.

## Before You Start

- Open an issue or reference an existing one for bug fixes, features, and larger refactors whenever possible.
- Keep pull requests scoped to one behavioral change or one tightly related documentation or maintenance update.
- If a change affects command-line behavior, generated output schemas, or evaluation utilities, update the relevant documentation in `README.md`, `docs/`, and inline help text.

## Development Setup

Ragnarök uses `uv` and the `dev` dependency group defined in `pyproject.toml`.

```bash
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
uv sync --group dev
pre-commit install
```

`.python-version` pins this repository to Python 3.12 for `uv`-aware tooling.
If you prefer not to activate the virtual environment, run commands through `uv run`.
`uv.lock` resolves the base dependencies, the default `dev` dependency group,
and the declared extras, but extras remain opt-in during sync and are only
installed when you pass `--extra ...`.
If you use GPU-backed models locally, install the PyTorch build appropriate for your environment before installing the optional runtime stack you need, as documented in the README.

## Local Quality Gate

Run these commands before opening a pull request:

```bash
uv lock --check
uv run pre-commit run --all-files
uv run ragnarok-quality-gate
```

`pre-commit` is the canonical formatter and lint entrypoint for this repository. The hook stack now uses Ruff for autofix plus the repo-local `ragnarok-quality-gate` entrypoint, which runs Ruff check, Ruff format check, core tests, integration tests, and MyPy in that order inside the `uv` environment.

## Testing Expectations

- Add or update tests for non-trivial behavior changes.
- `pytest` is the canonical runner; `unittest.TestCase` remains acceptable when extending existing modules incrementally.
- Keep tests in one of these layers:
  - `core`: fast deterministic unit and CLI coverage that always runs in PR CI
  - `integration`: deterministic offline regressions that exercise end-to-end CLI flows with frozen fixtures
  - `live`: provider-backed smoke tests gated behind explicit environment variables
- Apply the shared pytest markers (`core`, `integration`, `live`) at the module level when adding or moving tests so CI and local commands stay aligned across Castorini Python repos.
- Keep default automated coverage offline and deterministic whenever possible.
- If a change touches retrieval, reranking, generation post-processing, or TREC validation logic, include regression coverage or explain clearly in the pull request why coverage was not practical.

## Documentation Expectations

- Update `README.md` when install steps, public entrypoints, or common workflows change.
- Update files in `docs/` when track-specific execution or validation behavior changes.
- Call out output-format changes explicitly. Downstream scripts depend on stable field names and submission formats.
- Add or update a file in `docs/release-notes/` for user-visible changes.
- If a pull request changes CLI flags/defaults, output schemas, or validator behavior, document the migration path in the release note as well as in the pull request description.

## Artifact and API Safety

- Do not silently change JSON, JSONL, or TREC run-file schemas consumed by downstream tooling.
- Preserve existing command-line flags when possible. If you must change a flag or default, document the migration path in the pull request and release notes.
- Never hardcode provider credentials. Use environment variables for OpenAI, Azure OpenAI, Cohere, or other external services.
- Avoid introducing heavyweight dependencies unless they are necessary and justified in the pull request description.

## Pull Request Checklist

Before submitting:

1. Rebase or merge the latest `main` branch state into your working branch as needed.
2. Run the local quality gate commands listed above.
3. Summarize the behavioral change, documentation impact, and any dependency impact in the pull request body.
4. Include benchmarks when a change may affect retrieval quality, generation quality, latency, or cost.

## Reporting Issues

GitHub issues are the public tracker for bugs and feature requests. Good reports include:

- the exact command or code path used
- input files or a minimal reproduction
- expected behavior
- observed behavior
- relevant logs, tracebacks, or sample output artifacts

## License

By contributing to Ragnarök, you agree that your contributions will be licensed under the `LICENSE` file in the root of this repository.
