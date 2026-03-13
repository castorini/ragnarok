from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

from .io import read_json, read_jsonl
from .normalize import normalize_direct_generate_input


COMMAND_DESCRIPTIONS: dict[str, dict[str, Any]] = {
    "generate": {
        "summary": (
            "Run Ragnarok generation from dataset-backed retrieval, direct JSON "
            "input, or batch request files."
        ),
        "execution_mode_default": "sync",
        "examples": [
            (
                "ragnarok generate --model-path gpt-4o --dataset rag24.raggy-dev "
                "--retrieval-method bm25 --topk 20 --prompt-mode chatqa"
            ),
            (
                "ragnarok generate --model-path gpt-4o --input-json "
                '\'{"query":"q","candidates":["p"]}\' --output json'
            ),
        ],
        "input_modes": ["dataset", "input-file", "stdin", "input-json"],
    },
    "validate": {
        "summary": "Validate generate requests or TREC output artifacts without running models.",
        "targets": ["generate", "rag24-output", "rag25-output"],
    },
    "convert": {
        "summary": "Convert older Ragnarok artifacts into the newer TREC 2025 format.",
        "targets": ["trec25-format"],
    },
    "view": {
        "summary": "Inspect Ragnarok artifact files with a human-readable preview.",
        "examples": [
            "ragnarok view results.jsonl",
            "ragnarok view results.jsonl --records 1",
        ],
        "supported_types": ["generate-output-record"],
    },
}


SCHEMAS: dict[str, dict[str, Any]] = {
    "generate-direct-input": {
        "type": "object",
        "required": ["query", "candidates"],
        "properties": {
            "query": {
                "oneOf": [
                    {"type": "string"},
                    {
                        "type": "object",
                        "required": ["text"],
                        "properties": {
                            "qid": {"type": ["string", "integer"]},
                            "text": {"type": "string"},
                        },
                    },
                ]
            },
            "candidates": {
                "type": "array",
                "items": {
                    "oneOf": [
                        {"type": "string"},
                        {
                            "type": "object",
                            "required": ["text"],
                            "properties": {
                                "text": {"type": "string"},
                                "docid": {"type": ["string", "integer"]},
                                "score": {"type": "number"},
                            },
                        },
                        {
                            "type": "object",
                            "required": ["doc"],
                            "properties": {
                                "docid": {"type": ["string", "integer"]},
                                "score": {"type": "number"},
                                "doc": {
                                    "type": "object",
                                    "required": ["segment"],
                                    "properties": {"segment": {"type": "string"}},
                                },
                            },
                        },
                    ]
                },
            },
        },
    },
    "generate-batch-input-record": {
        "type": "object",
        "required": ["query", "candidates"],
        "properties": {
            "query": {
                "type": "object",
                "required": ["qid", "text"],
                "properties": {
                    "qid": {"type": ["string", "integer"]},
                    "text": {"type": "string"},
                },
            },
            "candidates": {"type": "array"},
        },
    },
    "generate-output-record": {
        "type": "object",
        "required": ["run_id", "topic_id", "topic", "references", "response_length", "answer"],
    },
    "trec24-output-record": {
        "type": "object",
        "required": ["run_id", "topic_id", "topic", "references", "response_length", "answer"],
    },
    "trec25-converted-output-record": {
        "type": "object",
        "required": ["metadata", "references", "answer"],
    },
    "view-summary": {
        "type": "object",
        "required": ["path", "artifact_type", "summary", "sampled_records"],
    },
    "cli-envelope": {
        "type": "object",
        "required": [
            "schema_version",
            "repo",
            "command",
            "mode",
            "status",
            "exit_code",
            "inputs",
            "resolved",
            "artifacts",
            "validation",
            "metrics",
            "warnings",
            "errors",
        ],
    },
}


def doctor_report() -> dict[str, Any]:
    env_path = Path(".env")
    return {
        "python_version": sys.version.split()[0],
        "python_ok": sys.version_info >= (3, 11),
        "env_file_present": env_path.exists(),
        "provider_keys": {
            "openai": bool(os.getenv("OPENAI_API_KEY")),
            "azure_openai": bool(
                os.getenv("AZURE_OPENAI_API_BASE")
                and os.getenv("AZURE_OPENAI_API_VERSION")
                and os.getenv("AZURE_OPENAI_API_KEY")
            ),
            "cohere": bool(os.getenv("CO_API_KEY")),
        },
        "optional_dependencies": {
            "torch": importlib.util.find_spec("torch") is not None,
            "cohere": importlib.util.find_spec("cohere") is not None,
            "openai": importlib.util.find_spec("openai") is not None,
            "pyserini": importlib.util.find_spec("pyserini") is not None,
        },
        "java_home_present": bool(os.getenv("JAVA_HOME")),
    }


def validate_generate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalize_direct_generate_input(payload)
    return {"valid": True, "record_count": 1}


def validate_generate_batch_file(path: str) -> dict[str, Any]:
    reader = read_jsonl if path.endswith(".jsonl") else read_json
    payload = reader(path)
    records = payload if isinstance(payload, list) else [payload]
    valid = True
    for record in records:
        query = record.get("query")
        candidates = record.get("candidates")
        if not (
            isinstance(query, dict)
            and "text" in query
            and "qid" in query
            and isinstance(candidates, list)
        ):
            valid = False
            break
    return {"valid": valid, "record_count": len(records)}
