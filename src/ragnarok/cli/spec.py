from __future__ import annotations

EXIT_CODES = {
    "invalid_arguments": 2,
    "missing_resource": 4,
    "validation_error": 5,
    "runtime_error": 6,
}

KNOWN_COMMANDS = (
    "generate",
    "validate",
    "convert",
    "view",
    "prompt",
    "describe",
    "schema",
    "doctor",
)

TOP_LEVEL_EXAMPLES = (
    (
        "ragnarok generate --model gpt-4o --dataset rag24.raggy-dev "
        "--retrieval-method bm25 --topk 20 --prompt-mode chatqa"
    ),
    (
        "ragnarok generate --model gpt-4o --input-json "
        '\'{"query":"q","candidates":["p"]}\' --output json'
    ),
    "ragnarok prompt show --prompt-mode chatqa",
    "ragnarok doctor --output json",
)
