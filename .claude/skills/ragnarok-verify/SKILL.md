---
name: ragnarok-verify
description: Use when validating ragnarok generate outputs — checks TREC format compliance, citation integrity (valid indices, non-empty references), response length bounds, and JSONL structure. Wraps `ragnarok validate` plus custom assertions. Use after running generate or convert to verify output correctness.
---

# Ragnarok Verify

Validates ragnarok generation outputs for TREC compliance, citation integrity, and structural correctness.

## When to Use

- After `ragnarok generate` — verify answer output integrity
- After `ragnarok convert trec25-format` — verify TREC submission format
- Before submitting to TREC RAG shared tasks
- When comparing outputs across models or prompt modes

## What It Checks

### JSONL Integrity
- Every line is valid JSON
- No trailing commas, no truncated records
- Consistent field presence across records

### Generate Output
- Every record has `topic_id`, `topic`, `references`, `response_length`, and `answer` array
- `answer` is an array of objects with `text` and `citations`
- Citation indices are valid (within `references` array bounds)
- No empty `answer` arrays
- No duplicate `topic_id` values
- `response_length` matches actual word count (within tolerance)

### Citation Integrity
- All citation indices reference valid positions in `references`
- No orphaned references (referenced by no citation)
- No out-of-range citation indices

### TREC RAG 2025 Format
- Response ≤ 400 words
- References ≤ 100
- Document IDs match MS MARCO v2.1 format
- Required metadata fields present (`team_id`, `run_id`, `narrative_id`, `type`)

## Usage

Run the verification script:

```bash
bash .claude/skills/ragnarok-verify/scripts/verify.sh <artifact-path> [--trec25]
```

Or use the built-in validators:

```bash
ragnarok validate generate --input-file requests.jsonl
ragnarok validate rag25-output --input answers.jsonl --topics topics.jsonl
```

## Verification Script

See `scripts/verify.sh` for the runnable verification wrapper.

## Gotchas

- `ragnarok validate generate` checks *input* requests. The verify script checks *output* answers.
- `ragnarok validate rag25-output` is the built-in TREC 2025 output validator with `--fix-length`, `--fix-citations`, `--apply-fixes` for automatic repair.
- Citation indices are 0-based in the output, referencing positions in the `references` array.
- `response_length` is word count, not character count.
- `--prompt-mode ragnarok_v4_no_cite` and `chatqa` produce answers without citations — citation checks should be skipped for these modes.
