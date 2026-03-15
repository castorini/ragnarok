from __future__ import annotations

from typing import Any

from ragnarok.generate.llm import PromptMode
from ragnarok.generate.templates.ragnarok_templates import RagnarokTemplates


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
