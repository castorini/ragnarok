from dataclasses import dataclass
from typing import Any

from ftfy import fix_text

from ragnarok.generate.llm import PromptMode
from ragnarok.prompts.template_loader import PromptTemplate, get_template


@dataclass(frozen=True)
class RenderedPrompt:
    format: str
    prompt_mode: str
    model: str
    system_message: str | None
    user_message: str | None
    combined_text: str | None
    messages: list[dict[str, str]] | None
    context_separator: str
    instruction: str

    def runtime_prompt(self) -> Any:
        if self.messages is not None:
            return self.messages
        return self.combined_text

    def metadata(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "prompt_mode": self.prompt_mode,
            "model": self.model,
            "system_message": self.system_message,
            "user_message": self.user_message,
            "combined_text": self.combined_text,
            "messages": self.messages,
            "context_separator": self.context_separator,
            "instruction": self.instruction,
        }


# Maps PromptMode values to YAML template names
_MODE_TO_TEMPLATE: dict[str, str] = {
    "chatqa": "chatqa",
    "ragnarok_v2": "ragnarok_v2",
    "ragnarok_v3": "ragnarok_v3",
    "ragnarok_v4": "ragnarok_v4",
    "ragnarok_v4_biogen": "ragnarok_v4_biogen",
    "ragnarok_v4_no_cite": "ragnarok_v4_no_cite",
    "ragnarok_v5_biogen": "ragnarok_v5_biogen",
    "ragnarok_v5_biogen_no_cite": "ragnarok_v5_biogen_no_cite",
}

_NO_CITE_MODES = {
    PromptMode.RAGNAROK_V4_NO_CITE,
    PromptMode.RAGNAROK_V5_BIOGEN_NO_CITE,
}


class RagnarokTemplates:
    def __init__(self, prompt_mode: PromptMode):
        self.prompt_mode = prompt_mode

        template_name = _MODE_TO_TEMPLATE.get(prompt_mode.value)
        if template_name is not None:
            self._template: PromptTemplate | None = get_template(template_name)
        else:
            self._template = None

        # Load shared system messages from any template (they are identical
        # across all YAML files) or fall back to the chatqa template.
        ref = self._template or get_template("chatqa")
        self.system_message_gpt = ref.system_message
        self.system_message_gpt_no_cite = ref.system_message_no_cite
        self.system_message_chatqa = ref.chatqa_system_message

        self.sep = "\n\n"
        self._instruction_by_mode = {
            PromptMode.CHATQA: get_template("chatqa").instruction,
            PromptMode.RAGNAROK_V2: get_template("ragnarok_v2").instruction,
            PromptMode.RAGNAROK_V3: get_template("ragnarok_v3").instruction,
            PromptMode.RAGNAROK_V4: get_template("ragnarok_v4").instruction,
            PromptMode.RAGNAROK_V4_NO_CITE: get_template(
                "ragnarok_v4_no_cite"
            ).instruction,
            PromptMode.RAGNAROK_V4_BIOGEN: get_template(
                "ragnarok_v4_biogen"
            ).instruction,
            PromptMode.RAGNAROK_V5_BIOGEN: get_template(
                "ragnarok_v5_biogen"
            ).instruction,
            PromptMode.RAGNAROK_V5_BIOGEN_NO_CITE: get_template(
                "ragnarok_v5_biogen_no_cite"
            ).instruction,
        }

    @property
    def template(self) -> PromptTemplate | None:
        return self._template

    @staticmethod
    def _uses_chat_message_format(model: str) -> bool:
        lowered_model = model.lower()
        return not any(
            name in lowered_model
            for name in ("command-r", "chatqa", "llama", "mistral", "qwen")
        )

    def render(self, query: str, context: list[str], model: str) -> RenderedPrompt:
        sep = self.sep
        if not (self._uses_chat_message_format(model) or "chatqa" in model.lower()):
            sep = "\n"
        str_context = sep.join(context)
        instruction = self.get_instruction()

        if self.prompt_mode in _NO_CITE_MODES:
            user_input_context = (
                f"Instruction: {instruction}"
                + sep
                + f"Query: {query}"
                + sep
                + f"Instruction: {instruction}"
            )
        elif self._uses_chat_message_format(model):
            user_input_context = (
                f"Instruction: {instruction}"
                + sep
                + f"Documents: {fix_text(str_context)}"
                + sep
                + f"Query: {query}"
                + sep
                + f"Instruction: {instruction}"
            )
            user_input_context += "\n\nAnswer:"
        else:
            user_input_context = (
                f"Instruction: {instruction}"
                + "\n"
                + f"The following are context references from which you can cite the identifier. References: {fix_text(str_context)}"
                + "\n"
                + f"Query: {query}"
                + "\n"
                + f"Instruction: {instruction}"
            )

        if self._uses_chat_message_format(model):
            messages = []
            system_message = (
                self.system_message_gpt_no_cite
                if self.prompt_mode in _NO_CITE_MODES
                else self.system_message_gpt
            )
            messages.append(
                {
                    "role": "system",
                    "content": system_message,
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": fix_text(user_input_context),
                }
            )
            return RenderedPrompt(
                format="chat_messages",
                prompt_mode=str(self.prompt_mode),
                model=model,
                system_message=system_message,
                user_message=fix_text(user_input_context),
                combined_text=None,
                messages=messages,
                context_separator=sep,
                instruction=instruction,
            )
        elif "chatqa" in model.lower():
            prompt = (
                f"{self.system_message_chatqa}{sep}Context: {str_context}"
                f"{sep}User: {user_input_context}"
            )
        else:
            system_message = (
                self.system_message_gpt_no_cite
                if self.prompt_mode in _NO_CITE_MODES
                else self.system_message_gpt
            )
            messages = []
            messages.append(
                {
                    "role": "system",
                    "content": system_message,
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": fix_text(user_input_context),
                }
            )
            return RenderedPrompt(
                format="chat_messages",
                prompt_mode=str(self.prompt_mode),
                model=model,
                system_message=system_message,
                user_message=fix_text(user_input_context),
                combined_text=None,
                messages=messages,
                context_separator=sep,
                instruction=instruction,
            )
        prompt = fix_text(prompt)
        return RenderedPrompt(
            format="single_string",
            prompt_mode=str(self.prompt_mode),
            model=model,
            system_message=None,
            user_message=None,
            combined_text=prompt,
            messages=None,
            context_separator=sep,
            instruction=instruction,
        )

    def __call__(self, query: str, context: list[str], model: str) -> Any:
        return self.render(query, context, model).runtime_prompt()

    def get_instruction(self) -> str:
        return self._instruction_by_mode.get(
            self.prompt_mode,
            self._instruction_by_mode[PromptMode.CHATQA],
        )
