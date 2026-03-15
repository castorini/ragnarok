from __future__ import annotations

from typing import Any

from ragnarok.generate.llm import PromptMode
from ragnarok.generate.templates.ragnarok_templates import (
    RagnarokTemplates,
    RenderedPrompt,
)


def list_prompt_modes() -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    for prompt_mode in PromptMode:
        if prompt_mode in {PromptMode.UNSPECIFIED, PromptMode.COHERE}:
            continue
        template_obj = RagnarokTemplates(prompt_mode)
        tmpl = template_obj.template
        entry: dict[str, Any] = {
            "prompt_mode": prompt_mode.value,
            "method": tmpl.method if tmpl else prompt_mode.value,
            "source_path": tmpl.source_path if tmpl else None,
            "placeholders": list(tmpl.placeholders) if tmpl else [],
            "template": tmpl.metadata() if tmpl else {},
        }
        catalog.append(entry)
    return catalog


def build_prompt_mode_view(prompt_mode: PromptMode) -> dict[str, Any]:
    template_obj = RagnarokTemplates(prompt_mode)
    tmpl = template_obj.template
    return {
        "prompt_mode": prompt_mode.value,
        "template_name": tmpl.method if tmpl else prompt_mode.value,
        "template": tmpl.metadata() if tmpl else {},
    }


def render_prompt_catalog_text(catalog: list[dict[str, Any]]) -> str:
    lines = ["Ragnarok Prompt Catalog"]
    for entry in catalog:
        lines.append("")
        lines.append(f"- prompt_mode: {entry['prompt_mode']}")
        lines.append(f"  method: {entry['method']}")
        lines.append(f"  source: {entry.get('source_path', '(none)')}")
        lines.append(
            "  placeholders: "
            + (", ".join(entry["placeholders"]) if entry["placeholders"] else "(none)")
        )
    return "\n".join(lines)


def render_prompt_mode_text(view: dict[str, Any]) -> str:
    lines = ["Ragnarok Prompt Template"]
    lines.append(f"prompt_mode: {view['prompt_mode']}")
    lines.append(f"template_name: {view['template_name']}")
    tmpl = view.get("template", {})
    lines.append(f"method: {tmpl.get('method', '')}")
    lines.append(f"source: {tmpl.get('source_path', '(none)')}")
    lines.append(
        "placeholders: "
        + (
            ", ".join(tmpl.get("placeholders", []))
            if tmpl.get("placeholders")
            else "(none)"
        )
    )
    lines.append("")
    lines.append("[system]")
    lines.append(tmpl.get("system_message") or "(empty)")
    lines.append("")
    lines.append("[user]")
    lines.append(tmpl.get("prefix_user", "(empty)"))
    lines.append("")
    lines.append("[instruction]")
    lines.append(tmpl.get("instruction", "(empty)"))
    return "\n".join(lines)


def build_rendered_prompt_view(
    rendered: RenderedPrompt, *, query: str, context_count: int, topk: int
) -> dict[str, Any]:
    messages: list[dict[str, str]] = []
    if rendered.messages is not None:
        messages = rendered.messages
    elif rendered.combined_text is not None:
        messages = [{"role": "user", "content": rendered.combined_text}]
    return {
        "prompt_mode": rendered.prompt_mode,
        "template_name": rendered.prompt_mode,
        "messages": messages,
        "inputs": {
            "query": query,
            "context_count": context_count,
            "topk": topk,
        },
    }


def render_rendered_prompt_text(view: dict[str, Any], *, part: str) -> str:
    inputs = view["inputs"]
    lines = ["Ragnarok Rendered Prompt"]
    lines.append(f"prompt_mode: {view['prompt_mode']}")
    lines.append(f"template_name: {view['template_name']}")
    for name, value in inputs.items():
        lines.append(f"{name}: {value}")
    for message in view["messages"]:
        if part != "all" and message["role"] != part:
            continue
        lines.append("")
        lines.append(f"[{message['role']}]")
        lines.append(message["content"])
    return "\n".join(lines)
