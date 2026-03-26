# Ragnarök v0.0.1

Initial release-note scaffold for the packaged CLI era.

## Included In This Baseline

- Packaged `ragnarok` CLI with generation, validation, conversion, prompt inspection, doctor, and view commands.
- FastAPI `ragnarok serve` command exposing `GET /healthz` and `POST /v1/generate` on port `8084` by default.
- Direct `generate` input now also accepts Anserini REST candidates where `candidates[].doc` is a plain string, so Anserini search results can be piped directly into `POST /v1/generate` without a `jq` reshape step.
- Direct `generate` input now also accepts single-record `castorini.cli.v1` envelopes from upstream tools such as `rank_llm`, so `search | rerank | generate` can be piped through `POST /v1/generate` without unwrapping `.artifacts[0].value[0]` first.
- Offline-first contributor workflow built around `uv`.
- Track validators and format-conversion utilities for TREC RAG workflows.

## Migration Notes

This baseline establishes the release-note policy for future changes.

Document a migration note in this directory whenever a change affects:

- CLI flags or default behavior
- output JSON, JSONL, or TREC artifact formats
- validator behavior or auto-fix behavior
- required dependencies or environment variables
