# Ragnarok CLI For Agents

`ragnarok ...` is the canonical offline command-line interface for this
repository. Prefer it over `src/ragnarok/scripts/*.py`. In an activated
environment, run `ragnarok ...` directly. If the virtual environment is not
activated, use `uv run ragnarok ...`.

## Command Overview

- `ragnarok generate`: Run dataset-backed retrieval plus generation, batch
  request-file generation, or a single direct JSON request.
- `ragnarok validate`: Validate generate payloads or TREC output artifacts.
- `ragnarok convert trec25-format`: Convert older Ragnarok JSONL into the newer
  TREC 2025 format.
- `ragnarok describe`: Inspect command metadata and examples.
- `ragnarok schema`: Print JSON schemas for inputs, outputs, and the shared CLI
  envelope.
- `ragnarok doctor`: Report Python, environment-variable, optional dependency,
  and Java or `pyserini` readiness.

## Command Mapping

Old:
```bash
python src/ragnarok/scripts/run_ragnarok.py --model_path=gpt-4o --dataset=rag24.raggy-dev --retrieval_method=bm25 --topk=20 --prompt_mode=chatqa
```

New:
```bash
ragnarok generate --model-path gpt-4o --dataset rag24.raggy-dev --retrieval-method bm25 --topk 20 --prompt-mode chatqa
```

Old:
```bash
python src/ragnarok/scripts/check_trec_rag24_gen.py --topicfile topics.tsv run.jsonl
```

New:
```bash
ragnarok validate rag24-output --topicfile topics.tsv --runfile run.jsonl
```

Old:
```bash
python src/ragnarok/scripts/validate_trec_rag25_gen.py --input run.jsonl --topics topics.jsonl
```

New:
```bash
ragnarok validate rag25-output --input run.jsonl --topics topics.jsonl
```

Old:
```bash
python src/ragnarok/scripts/convert_to_trec25_format.py --input_file old.jsonl --output_file new.jsonl --prompt_file prompts.jsonl
```

New:
```bash
ragnarok convert trec25-format --input-file old.jsonl --output-file new.jsonl --prompt-file prompts.jsonl
```

## Direct And Batch Examples

Direct single request:

```bash
ragnarok generate \
  --model-path gpt-4o \
  --input-json '{"query":"how long is life cycle of flea","candidates":["The life cycle of a flea can last anywhere from 20 days to an entire year."]}' \
  --prompt-mode chatqa \
  --output json
```

Batch request file:

```bash
ragnarok generate \
  --model-path gpt-4o \
  --input-file requests.jsonl \
  --output-file results.jsonl \
  --prompt-mode chatqa
```

Async request-file generation:

```bash
ragnarok generate \
  --model-path gpt-4o \
  --input-file requests.jsonl \
  --output-file results.jsonl \
  --prompt-mode chatqa \
  --execution-mode async \
  --max-concurrency 8
```

Dataset-backed retrieval and generation:

```bash
ragnarok generate \
  --model-path gpt-4o \
  --dataset rag24.raggy-dev \
  --retrieval-method bm25,rank_zephyr_rho \
  --topk 100,5 \
  --prompt-mode chatqa
```

## Introspection And Validation

```bash
ragnarok describe generate --output json
ragnarok schema generate-direct-input --output json
ragnarok validate generate --input-json '{"query":"q","candidates":["p"]}' --output json
ragnarok doctor --output json
```

## Notes

- JSON output uses the shared Castorini CLI envelope shape.
- `generate --execution-mode async` is currently supported for direct JSON and
  request-file generation. Dataset-backed retrieval plus generation remains
  synchronous for now and returns a validation error if async mode is requested.
- `generate --dry-run` and `generate --validate-only` resolve inputs without
  running a model.
- Write policies mirror the other packaged CLIs: default fail if the output file
  already exists, opt in to `--overwrite` or `--resume` when needed.
- Existing scripts remain available as compatibility entrypoints, but new docs
  and automation should target `ragnarok ...`.
