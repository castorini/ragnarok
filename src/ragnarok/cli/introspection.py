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
        "inspection_safe": False,
        "examples": [
            (
                "ragnarok generate --model gpt-4o --dataset rag24.raggy-dev "
                "--retrieval-method bm25 --topk 20 --prompt-mode chatqa"
            ),
            (
                "ragnarok generate --model gpt-4o --input-json "
                '\'{"query":"q","candidates":["p"]}\' --output json'
            ),
        ],
        "input_modes": ["dataset", "input-file", "stdin", "input-json"],
    },
    "validate": {
        "summary": "Validate generate requests or TREC output artifacts without running models.",
        "targets": ["generate", "rag24-output", "rag25-output"],
        "inspection_safe": True,
    },
    "convert": {
        "summary": "Convert older Ragnarok artifacts into the newer TREC 2025 format.",
        "targets": ["trec25-format"],
        "inspection_safe": True,
    },
    "view": {
        "summary": "Inspect Ragnarok artifact files with a human-readable preview.",
        "examples": [
            "ragnarok view results.jsonl",
            "ragnarok view results.jsonl --records 1",
        ],
        "supported_types": ["generate-output-record"],
        "inspection_safe": True,
    },
    "prompt": {
        "summary": "Inspect Ragnarok prompt-mode definitions.",
        "examples": [
            "ragnarok prompt list",
            "ragnarok prompt show --prompt-mode chatqa",
            'ragnarok prompt render --prompt-mode chatqa --model gpt-4o --input-json \'{"query":"q","candidates":["p"]}\'',
            "ragnarok prompt show --prompt-mode ragnarok_v4 --output json",
        ],
        "inspection_safe": True,
        "subcommands": ["list", "show", "render"],
    },
    "describe": {
        "summary": "Inspect structured metadata for a public Ragnarok command.",
        "inspection_safe": True,
    },
    "schema": {
        "summary": "Print JSON schemas for supported inputs, outputs, and envelopes.",
        "inspection_safe": True,
    },
    "doctor": {
        "summary": "Report environment, dependency, and backend readiness for the packaged CLI.",
        "inspection_safe": True,
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
        "required": [
            "run_id",
            "topic_id",
            "topic",
            "references",
            "response_length",
            "answer",
        ],
    },
    "trec24-output-record": {
        "type": "object",
        "required": [
            "run_id",
            "topic_id",
            "topic",
            "references",
            "response_length",
            "answer",
        ],
    },
    "trec25-converted-output-record": {
        "type": "object",
        "required": ["metadata", "references", "answer"],
    },
    "view-summary": {
        "type": "object",
        "required": ["path", "artifact_type", "summary", "sampled_records"],
    },
    "prompt-catalog": {
        "type": "array",
        "items": {"type": "object", "required": ["prompt_mode", "instruction"]},
    },
    "prompt-mode": {
        "type": "object",
        "required": [
            "prompt_mode",
            "instruction",
            "chat_system_message",
            "chat_system_message_no_cite",
            "chatqa_system_message",
            "input_context_template",
        ],
    },
    "rendered-prompt": {
        "type": "object",
        "required": ["prompt", "inputs"],
    },
    "doctor-output": {
        "type": "object",
        "required": [
            "python_version",
            "python_ok",
            "env_file_present",
            "backend_readiness",
            "command_readiness",
            "overall_status",
        ],
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
    python_ok = sys.version_info >= (3, 11)
    openai_env_ready = bool(os.getenv("OPENAI_API_KEY"))
    azure_env_ready = bool(
        os.getenv("AZURE_OPENAI_API_BASE")
        and os.getenv("AZURE_OPENAI_API_VERSION")
        and os.getenv("AZURE_OPENAI_API_KEY")
    )
    cohere_env_ready = bool(os.getenv("CO_API_KEY"))
    torch_ready = importlib.util.find_spec("torch") is not None
    openai_dep_ready = importlib.util.find_spec("openai") is not None
    cohere_dep_ready = importlib.util.find_spec("cohere") is not None
    pyserini_dep_ready = importlib.util.find_spec("pyserini") is not None

    def status(
        *,
        ready: bool,
        missing_env: list[str] | None = None,
        missing_deps: list[str] | None = None,
    ) -> dict[str, Any]:
        missing_env = missing_env or []
        missing_deps = missing_deps or []
        if ready:
            state = "ready"
        elif missing_env:
            state = "missing_env"
        elif missing_deps:
            state = "missing_dependency"
        else:
            state = "blocked"
        return {
            "status": state,
            "missing_env": missing_env,
            "missing_dependencies": missing_deps,
        }

    backend_readiness = {
        "openai": status(
            ready=python_ok and openai_env_ready and openai_dep_ready,
            missing_env=[] if openai_env_ready else ["OPENAI_API_KEY"],
            missing_deps=[] if openai_dep_ready else ["openai"],
        ),
        "azure_openai": status(
            ready=python_ok and azure_env_ready and openai_dep_ready,
            missing_env=[]
            if azure_env_ready
            else [
                "AZURE_OPENAI_API_BASE",
                "AZURE_OPENAI_API_VERSION",
                "AZURE_OPENAI_API_KEY",
            ],
            missing_deps=[] if openai_dep_ready else ["openai"],
        ),
        "cohere": status(
            ready=python_ok and cohere_env_ready and cohere_dep_ready,
            missing_env=[] if cohere_env_ready else ["CO_API_KEY"],
            missing_deps=[] if cohere_dep_ready else ["cohere"],
        ),
        "local_open_weights": status(
            ready=python_ok and torch_ready,
            missing_deps=[] if torch_ready else ["torch"],
        ),
    }
    command_readiness = {
        "generate": status(ready=python_ok),
        "validate": status(ready=python_ok),
        "convert": status(ready=python_ok),
        "view": status(ready=python_ok),
        "describe": status(ready=python_ok),
        "schema": status(ready=python_ok),
        "doctor": status(ready=python_ok),
    }
    overall_status = "ready" if python_ok else "blocked"
    return {
        "python_version": sys.version.split()[0],
        "python_ok": python_ok,
        "env_file_present": env_path.exists(),
        "provider_keys": {
            "openai": openai_env_ready,
            "azure_openai": azure_env_ready,
            "cohere": cohere_env_ready,
        },
        "optional_dependencies": {
            "torch": torch_ready,
            "cohere": cohere_dep_ready,
            "openai": openai_dep_ready,
            "pyserini": pyserini_dep_ready,
        },
        "java_home_present": bool(os.getenv("JAVA_HOME")),
        "backend_readiness": backend_readiness,
        "command_readiness": command_readiness,
        "overall_status": overall_status,
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
