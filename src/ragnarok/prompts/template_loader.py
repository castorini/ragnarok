"""
Template loader for YAML-based prompts
"""

from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import Any, Dict

import yaml

# template cache to avoid reloading files
_template_cache: Dict[str, Dict[str, Any]] = {}

# prompt modes that have YAML templates (excludes UNSPECIFIED and COHERE)
_TEMPLATE_MODES = (
    "chatqa",
    "ragnarok_v2",
    "ragnarok_v3",
    "ragnarok_v4",
    "ragnarok_v4_biogen",
    "ragnarok_v4_no_cite",
    "ragnarok_v5_biogen",
    "ragnarok_v5_biogen_no_cite",
)


@dataclass(frozen=True)
class PromptTemplate:
    method: str
    system_message: str
    prefix_user: str
    source_path: str
    # ragnarok-specific extras
    instruction: str
    system_message_no_cite: str
    chatqa_system_message: str

    @property
    def placeholders(self) -> tuple[str, ...]:
        return tuple(
            field_name
            for _, field_name, _, _ in Formatter().parse(self.instruction)
            if field_name is not None
        ) + tuple(
            field_name
            for _, field_name, _, _ in Formatter().parse(self.prefix_user)
            if field_name is not None
        )

    def metadata(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "system_message": self.system_message,
            "prefix_user": self.prefix_user,
            "source_path": self.source_path,
            "placeholders": list(self.placeholders),
            "instruction": self.instruction,
            "system_message_no_cite": self.system_message_no_cite,
            "chatqa_system_message": self.chatqa_system_message,
        }


def _load_raw(template_name: str) -> Dict[str, Any]:
    """
    Load a YAML template from prompt_templates directory (cached)
    """
    if template_name not in _template_cache:
        template_dir = Path(__file__).parent / "prompt_templates"
        template_path = template_dir / f"{template_name}.yaml"

        if not template_path.exists():
            raise FileNotFoundError(
                f"Template {template_name} not found at {template_path}"
            )

        with open(template_path, "r", encoding="utf-8") as f:
            _template_cache[template_name] = yaml.safe_load(f)

    return _template_cache[template_name]


def get_template(template_name: str) -> PromptTemplate:
    raw = _load_raw(template_name)
    template_dir = Path(__file__).parent / "prompt_templates"
    template_path = template_dir / f"{template_name}.yaml"
    return PromptTemplate(
        method=str(raw["method"]),
        system_message=str(raw["system_message"]),
        prefix_user=str(raw["prefix_user"]),
        source_path=str(template_path),
        instruction=str(raw["instruction"]),
        system_message_no_cite=str(raw.get("system_message_no_cite", "")),
        chatqa_system_message=str(raw.get("chatqa_system_message", "")),
    )


def list_templates() -> list[PromptTemplate]:
    return [get_template(name) for name in _TEMPLATE_MODES]
