from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ANSI_CODES = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
}

GENERATE_REQUIRED_KEYS = {
    "run_id",
    "topic_id",
    "topic",
    "references",
    "response_length",
    "answer",
}


class ViewError(ValueError):
    """Raised when a file cannot be rendered as a supported Ragnarok artifact."""


def _color_enabled(color: str) -> bool:
    if color == "always":
        return True
    if color == "never":
        return False
    return sys.stdout.isatty()


def _style(text: str, color: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"{ANSI_CODES[color]}{text}{ANSI_CODES['reset']}"


def _truncate(text: str, limit: int = 140) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1] + "\u2026"


def load_records(path: str) -> list[dict[str, Any]]:
    file_path = Path(path)
    try:
        raw_text = file_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ViewError(f"path does not exist: {path}") from exc

    if not raw_text.strip():
        raise ViewError(f"file is empty: {path}")

    try:
        if file_path.suffix == ".json":
            payload = json.loads(raw_text)
            if isinstance(payload, dict):
                return [payload]
            if isinstance(payload, list):
                return payload
            raise ViewError(f"unsupported JSON payload in {path}")
        records = [json.loads(line) for line in raw_text.splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        raise ViewError(f"file is not valid JSON/JSONL: {path}") from exc

    if not records:
        raise ViewError(f"file is empty: {path}")
    return records


def detect_artifact_type(
    records: list[dict[str, Any]], requested_type: str | None
) -> str:
    if requested_type is not None:
        if requested_type != "generate-output-record":
            raise ViewError(
                "unsupported --type for ragnarok view; expected generate-output-record"
            )
        return requested_type

    if GENERATE_REQUIRED_KEYS.issubset(records[0].keys()):
        return "generate-output-record"
    raise ViewError(
        "could not detect Ragnarok artifact type; use --type generate-output-record"
    )


def build_view_summary(
    path: str, records: list[dict[str, Any]], artifact_type: str, *, record_limit: int
) -> dict[str, Any]:
    limit = max(record_limit, 0)
    run_ids = sorted(
        {str(record.get("run_id", "")) for record in records if record.get("run_id")}
    )
    sampled_records: list[dict[str, Any]] = []
    for record in records[:limit]:
        sampled_records.append(
            {
                "run_id": record["run_id"],
                "topic_id": record["topic_id"],
                "topic": _truncate(str(record["topic"]), 150),
                "response_length": record["response_length"],
                "references": list(record["references"]),
                "answer": [
                    {
                        "text": _truncate(str(sentence.get("text", "")), 180),
                        "citations": sentence.get("citations", []),
                    }
                    for sentence in record.get("answer", [])
                ],
            }
        )
    return {
        "path": str(Path(path)),
        "artifact_type": artifact_type,
        "summary": {
            "record_count": len(records),
            "run_ids": run_ids,
        },
        "sampled_records": sampled_records,
        "requested_records": limit,
    }


def render_view_summary(view: dict[str, Any], *, color: str) -> str:
    enabled = _color_enabled(color)
    run_ids = view["summary"]["run_ids"]
    lines = [
        _style("Ragnarok View", "bold", enabled),
        f"path: {view['path']}",
        f"type: {view['artifact_type']}",
        f"records: {view['summary']['record_count']}",
    ]
    if len(run_ids) == 1:
        lines.append(f"run_id: {run_ids[0]}")
    elif run_ids:
        lines.append(f"run_ids: {', '.join(run_ids)}")

    for index, record in enumerate(view["sampled_records"], start=1):
        lines.append("")
        lines.append(
            f"[{index}] topic_id={_style(str(record['topic_id']), 'cyan', enabled)} "
            f"response_length={_style(str(record['response_length']), 'green', enabled)}"
        )
        lines.append(f"topic: {record['topic']}")
        lines.append(f"references: {', '.join(map(str, record['references']))}")
        for sentence_index, sentence in enumerate(record["answer"], start=1):
            citations = ",".join(str(item) for item in sentence["citations"])
            lines.append(
                f"{sentence_index}. {sentence['text']} "
                f"[citations: {citations or 'none'}]"
            )
    return "\n".join(lines)
