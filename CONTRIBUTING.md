# Contributing to Ragnarök

Thank you for contributing to Ragnarök. This repository packages retrieval-augmented generation workflows, TREC Retrieval-Augmented Generation (RAG) utilities, and output validators, so changes should preserve both developer ergonomics and artifact compatibility.

## Before You Start

- Open an issue or reference an existing one for bug fixes, features, and larger refactors whenever possible.
- Keep pull requests scoped to one behavioral change or one tightly related documentation or maintenance update.
- If a change affects command-line behavior, generated output schemas, or evaluation utilities, update the relevant documentation in `README.md`, `docs/`, and inline help text.

## Development Setup

Ragnarök currently uses `requirements.txt` plus editable installs.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pre-commit install
```

If you use GPU-backed models locally, install the PyTorch build appropriate for your environment before `pip install -r requirements.txt`, as documented in the README.

## Local Quality Gate

Run these commands before opening a pull request:

```bash
pre-commit run --all-files
python -m unittest discover -s test
```

`pre-commit` is the canonical formatter and lint entrypoint for this repository. The current hooks run `black`, `isort`, and `flake8`.

## Testing Expectations

- Add or update tests for non-trivial behavior changes.
- Prefer `unittest` tests under `test/` to match the existing suite.
- Keep tests offline and deterministic whenever possible.
- If a change touches retrieval, reranking, generation post-processing, or TREC validation logic, include regression coverage or explain clearly in the pull request why coverage was not practical.

## Documentation Expectations

- Update `README.md` when install steps, public entrypoints, or common workflows change.
- Update files in `docs/` when track-specific execution or validation behavior changes.
- Call out output-format changes explicitly. Downstream scripts depend on stable field names and submission formats.

## Artifact and API Safety

- Do not silently change JSON, JSONL, or TREC run-file schemas consumed by downstream tooling.
- Preserve existing command-line flags when possible. If you must change a flag or default, document the migration path in the pull request.
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
