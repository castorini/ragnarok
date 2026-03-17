# Ragnarok CLI Examples

## generate

```bash
# File-backed generation
ragnarok generate --input-file requests.jsonl --output-file answers.jsonl \
  --model gpt-4o --prompt-mode ragnarok_v4

# Dataset-backed (retrieve + generate)
ragnarok generate --dataset rag24.raggy-dev \
  --retrieval-method bm25,rank_zephyr_rho --topk 100,20 \
  --model gpt-4o --prompt-mode ragnarok_v4 --output-file answers.jsonl

# Direct input (single query, stdout)
ragnarok generate --input-json '{"query":"What is IR?","candidates":["Information retrieval is..."]}' \
  --model gpt-4o --prompt-mode ragnarok_v4 --output json

# With resume
ragnarok generate --input-file requests.jsonl --output-file answers.jsonl \
  --model gpt-4o --prompt-mode ragnarok_v4 --resume

# Async execution
ragnarok generate --input-file requests.jsonl --output-file answers.jsonl \
  --model gpt-4o --prompt-mode ragnarok_v4 --execution-mode async --max-concurrency 8

# OpenRouter backend
ragnarok generate --input-file requests.jsonl --output-file answers.jsonl \
  --model openai/gpt-4o --prompt-mode ragnarok_v4 --use-openrouter

# With trace and reasoning
ragnarok generate --input-file requests.jsonl --output-file answers.jsonl \
  --model gpt-4o --prompt-mode ragnarok_v4 --include-trace --include-reasoning

# Dry run
ragnarok generate --input-file requests.jsonl --output-file answers.jsonl \
  --model gpt-4o --prompt-mode ragnarok_v4 --dry-run

# No-citation mode
ragnarok generate --input-file requests.jsonl --output-file answers.jsonl \
  --model gpt-4o --prompt-mode ragnarok_v4_no_cite

# Biogen mode (biological generation)
ragnarok generate --input-file requests.jsonl --output-file answers.jsonl \
  --model gpt-4o --prompt-mode ragnarok_v4_biogen
```

## validate

```bash
# Validate generate input
ragnarok validate generate --input-file requests.jsonl

# Validate TREC RAG 2024 output
ragnarok validate rag24-output --topicfile topics.tsv --runfile run.jsonl

# Validate TREC RAG 2025 output
ragnarok validate rag25-output --input answers.jsonl --topics topics.jsonl

# Validate and fix TREC 2025 output
ragnarok validate rag25-output --input answers.jsonl --topics topics.jsonl \
  --fix-length --fix-citations --apply-fixes
```

## convert

```bash
# Convert to TREC 2025 submission format
ragnarok convert trec25-format --input-file answers.jsonl --output-file submission.jsonl

# With prompt sidecar
ragnarok convert trec25-format --input-file answers.jsonl --output-file submission.jsonl \
  --prompt-file prompts.jsonl
```

## Introspection

```bash
# Environment check
ragnarok doctor
ragnarok doctor --output json

# Command contract
ragnarok describe generate --output json

# JSON Schemas
ragnarok schema generate-direct-input
ragnarok schema generate-batch-input-record
ragnarok schema generate-output-record

# Prompt inspection
ragnarok prompt list
ragnarok prompt show --prompt-mode ragnarok_v4
ragnarok prompt render --prompt-mode ragnarok_v4 --model gpt-4o \
  --input-json '{"query":"test","candidates":["passage"]}' --part user

# View artifacts
ragnarok view answers.jsonl --records 5
```
