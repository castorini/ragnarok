from __future__ import annotations

from typing import Any

from ragnarok.generate.llm import PromptMode
from ragnarok.generate.templates.ragnarok_templates import RenderedPrompt, RagnarokTemplates


def list_prompt_modes() -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    for prompt_mode in PromptMode:
        if prompt_mode in {PromptMode.UNSPECIFIED, PromptMode.COHERE}:
            continue
        template = RagnarokTemplates(prompt_mode)
        catalog.append(
            {
                "prompt_mode": prompt_mode.value,
                "instruction": template.get_instruction(),
                "chat_system_message": template.system_message_gpt,
                "chat_system_message_no_cite": template.system_message_gpt_no_cite,
                "chatqa_system_message": template.system_message_chatqa,
            }
        )
    return catalog


def build_prompt_mode_view(prompt_mode: PromptMode) -> dict[str, Any]:
    template = RagnarokTemplates(prompt_mode)
    return {
        "prompt_mode": prompt_mode.value,
        "instruction": template.get_instruction(),
        "chat_system_message": template.system_message_gpt,
        "chat_system_message_no_cite": template.system_message_gpt_no_cite,
        "chatqa_system_message": template.system_message_chatqa,
        "input_context_template": template.input_context,
    }


def render_prompt_catalog_text(catalog: list[dict[str, Any]]) -> str:
    lines = ["Ragnarok Prompt Catalog"]
    for entry in catalog:
        lines.append("")
        lines.append(f"- prompt_mode: {entry['prompt_mode']}")
        lines.append(f"  instruction: {entry['instruction']}")
    return "\n".join(lines)


def render_prompt_mode_text(view: dict[str, Any]) -> str:
    lines = ["Ragnarok Prompt Mode"]
    lines.append(f"prompt_mode: {view['prompt_mode']}")
    lines.append("")
    lines.append("[instruction]")
    lines.append(view["instruction"])
    lines.append("")
    lines.append("[chat_system_message]")
    lines.append(view["chat_system_message"])
    lines.append("")
    lines.append("[chat_system_message_no_cite]")
    lines.append(view["chat_system_message_no_cite"])
    lines.append("")
    lines.append("[chatqa_system_message]")
    lines.append(view["chatqa_system_message"])
    lines.append("")
    lines.append("[input_context_template]")
    lines.append(view["input_context_template"])
    return "\n".join(lines)


def build_rendered_prompt_view(
    rendered: RenderedPrompt, *, query: str, context_count: int, topk: int
) -> dict[str, Any]:
    return {
        "prompt": rendered.metadata(),
        "inputs": {
            "query": query,
            "context_count": context_count,
            "topk": topk,
        },
    }


def render_rendered_prompt_text(view: dict[str, Any], *, part: str) -> str:
    prompt = view["prompt"]
    inputs = view["inputs"]
    lines = ["Ragnarok Rendered Prompt"]
    lines.append(f"prompt_mode: {prompt['prompt_mode']}")
    lines.append(f"model: {prompt['model']}")
    lines.append(f"format: {prompt['format']}")
    lines.append(f"query: {inputs['query']}")
    lines.append(f"context_count: {inputs['context_count']}")
    lines.append(f"topk: {inputs['topk']}")
    if prompt["format"] == "chat_messages":
        if part in {"system", "all"}:
            lines.append("")
            lines.append("[system]")
            lines.append(prompt["system_message"] or "(empty)")
        if part in {"user", "all"}:
            lines.append("")
            lines.append("[user]")
            lines.append(prompt["user_message"] or "(empty)")
    else:
        lines.append("")
        lines.append("[prompt]")
        lines.append(prompt["combined_text"] or "")
    return "\n".join(lines)
