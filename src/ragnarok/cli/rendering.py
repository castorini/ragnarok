from __future__ import annotations

import json
from typing import Any, cast

from .prompt_view import (
    render_prompt_catalog_text,
    render_prompt_mode_text,
    render_rendered_prompt_text,
)
from .responses import CommandResponse
from .view import render_view_summary


def format_text_response(response: CommandResponse) -> str:
    if response.command == "generate":
        if not response.artifacts:
            return ""
        data = response.artifacts[0].get("data", [])
        if not data:
            return ""
        rendered_records: list[str] = []
        for record in data:
            answer_lines: list[str] = []
            for answer in record.get("answer", []):
                text = answer.get("text", "")
                if not text:
                    continue
                citations = answer.get("citations", [])
                citation_suffix = ""
                if citations:
                    human_citations = ",".join(
                        str(int(citation) + 1) for citation in citations
                    )
                    citation_suffix = f" [{human_citations}]"
                answer_lines.append(f"{text}{citation_suffix}")
            lines = [
                f"query: {record.get('topic', '')}",
                f"answer: {' '.join(answer_lines).strip()}",
                f"references: [{', '.join(str(reference) for reference in record.get('references', []))}]",
            ]
            reasoning_traces = [
                trace.strip()
                for trace in record.get("reasoning_traces", [])
                if isinstance(trace, str) and trace.strip()
            ]
            if reasoning_traces:
                lines.append(f"reasoning: {reasoning_traces[0]}")
            rendered_records.append("\n".join(lines).rstrip())
        return "\n\n".join(rendered_records).rstrip()
    if response.command in ("describe", "schema"):
        return json.dumps(response.artifacts[0]["data"], indent=2)
    if response.command == "doctor":
        return json.dumps(response.metrics, indent=2)
    if response.command == "validate":
        return json.dumps(response.validation, indent=2)
    if response.command == "view":
        return render_view_summary(
            cast(dict[str, Any], response.artifacts[0]["data"]),
            color=cast(str, response.resolved["color"]),
        )
    if response.command == "prompt":
        if response.artifacts[0]["name"] == "prompt-catalog":
            return render_prompt_catalog_text(
                cast(list[dict[str, Any]], response.artifacts[0]["data"])
            )
        if response.artifacts[0]["name"] == "rendered-prompt":
            return render_rendered_prompt_text(
                cast(dict[str, Any], response.artifacts[0]["data"]),
                part=cast(str, response.resolved["part"]),
            )
        return render_prompt_mode_text(
            cast(dict[str, Any], response.artifacts[0]["data"])
        )
    if response.command == "serve":
        return ""
    return json.dumps(response.to_envelope(), indent=2)
