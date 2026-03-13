from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast


def read_json(path: str) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(Path(path).read_text(encoding="utf-8")))


def read_jsonl(path: str) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: str, records: list[dict[str, Any]]) -> None:
    Path(path).write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def append_jsonl(path: str, records: list[dict[str, Any]]) -> None:
    with Path(path).open("a", encoding="utf-8") as file_obj:
        for record in records:
            file_obj.write(json.dumps(record) + "\n")
