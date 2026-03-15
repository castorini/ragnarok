from dataclasses import dataclass
from typing import Any, List

from ftfy import fix_text

from ragnarok.generate.llm import PromptMode


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


class RagnarokTemplates:
    def __init__(self, prompt_mode: PromptMode):
        self.prompt_mode = prompt_mode
        self.system_message_gpt = "This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful and detailed answers to the user's question based on the context references. The assistant should also indicate when the answer cannot be found in the context references."
        self.system_message_gpt_no_cite = "This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful and detailed answers to the user's question."
        self.system_message_chatqa = "System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context."
        self.input_context = "Context: {context}"
        self.instruction_official = (
            "Please give a full and complete answer for the question."
        )
        self.instruction_ragnarok = (
            "Please give a full and complete answer for the question. "
            + "Cite each context document inline that supports your answer within brackets [] using the IEEE format. "
            + "Ensure each sentence is properly cited."
        )
        self.instruction_ragnarok_v2 = (
            "Please give a full and complete answer for the question. "
            + "Cite each context document inline that supports your answer within brackets [] using the IEEE format. "
            + "Each sentence should have at most three citations. "
            + "Order the citations in decreasing order of importance. "
            + "Never include or mention anything about references, this is already provided, just answer the question such that each sentence has one or more sentence-level citations and say nothing else."
        )
        self.instruction_ragnarok_v3 = (
            "Provide a concise, information-dense answer to the question. "
            "Your response must not exceed 380 words under any circumstances. "
            "Prioritize the most relevant and impactful information within this strict limit. "
            "Cite supporting context documents inline using IEEE format in square brackets []. "
            "Include 1-3 citations per sentence, ordered by decreasing importance. "
            "Ensure each sentence has at least one citation. "
            "Focus solely on answering the question with properly cited information. "
            "Avoid mentioning references or providing any meta-commentary about the answering process."
        )
        self.instruction_ragnarok_v4 = (
            "Provide a concise, information-dense answer to the question. "
            "Your response must not exceed 380 words under any circumstances. "
            "Prioritize the most relevant and impactful information within this strict limit. "
            "Ensure your answer directly addresses the question and maintains coherence throughout. "
            "Cite supporting context documents inline using IEEE format in square brackets []. "
            "Include 1-3 citations per sentence, ordered by decreasing importance. "
            "Ensure each sentence has at least one citation. "
            "Use multiple sources to provide a well-rounded answer when possible. "
            "If sources contradict each other, acknowledge this and explain the discrepancy. "
            "Express uncertainty when appropriate rather than making unfounded claims. "
            "Prioritize factual accuracy and avoid speculation. "
            "Focus solely on answering the question with properly cited information. "
            "Avoid mentioning references or providing any meta-commentary about the answering process."
        )
        self.instruction_ragnarok_v4_no_cite = (
            "Provide a concise, information-dense answer to the question. "
            "Your response must not exceed 380 words under any circumstances. "
            "Prioritize the most relevant and impactful information within this strict limit. "
            "Ensure your answer directly addresses the question and maintains coherence throughout. "
            "Provide a well-rounded answer when possible. "
            "Express uncertainty when appropriate rather than making unfounded claims. "
            "Prioritize factual accuracy and avoid speculation. "
            "Focus solely on answering the question. "
            "Avoid references or providing any meta-commentary about the answering process."
        )
        self.instruction_ragnarok_v4_biogen = (
            "Provide a concise, information-dense answer to the question in a single cohesive paragraph, avoiding lists and bullet points. "
            "Your response must not exceed 150 words (excluding references) under any circumstances. "
            "Prioritize the most relevant and impactful information within this strict limit. "
            "Ensure your answer directly addresses the question and maintains coherence throughout. "
            "Cite the supporting context PubMed documents inline using IEEE format in square brackets []. "
            "Include 1-3 citations per sentence, ordered by decreasing importance. "
            "Ensure each sentence has at least one citation. "
            "Use multiple sources to provide a well-rounded answer when possible. "
            "If sources contradict each other, acknowledge this and explain the discrepancy. "
            "Express uncertainty when appropriate rather than making unfounded claims. "
            "Prioritize factual accuracy and avoid speculation or potentially harmful advice. "
            "Focus solely on answering the question with properly cited information. "
            "Avoid mentioning references or providing any meta-commentary about the answering process."
        )
        self.instruction_ragnarok_v5_biogen = (
            "Provide a concise, information-dense answer to the biomedical question in a single cohesive paragraph, avoiding lists and bullet points. "
            "Your response must not exceed 150 words (excluding references) under any circumstances. "
            "Prioritize the most relevant and impactful information within this strict limit. "
            "Ensure each sentence is clear, interpretable, and directly relevant to the question. "
            "Cite the supporting context PubMed documents inline using IEEE format in square brackets []. "
            "Include 1-3 citations per sentence, ordered by decreasing importance. "
            "Ensure each sentence has at least one citation. "
            "Use multiple sources to provide a well-rounded answer when possible. "
            "Focus on required and relevant information, avoiding unnecessary or borderline content. "
            "If sources contradict each other, acknowledge this and explain the discrepancy. "
            "Express uncertainty when appropriate rather than making unfounded claims. "
            "Prioritize factual accuracy and avoid speculation or potentially harmful advice. "
            "For patient-oriented questions, provide information suitable for clinician review. "
            "Assume the reader is a healthcare professional, but avoid overly technical jargon. "
            "Do not include trivial statements or generic recommendations to see a health professional, stick to the 150 word limit. "
            "Ensure all information is supported by the provided PubMed abstracts. "
            "Avoid mentioning references or providing any meta-commentary about the answering process."
        )
        self.instruction_ragnarok_v5_biogen_no_cite = (
            "Provide a concise, information-dense answer to the biomedical question in a single cohesive paragraph, avoiding lists and bullet points. "
            "Your response must not exceed 150 words under any circumstances. "
            "Prioritize the most relevant and impactful information within this strict limit. "
            "Ensure each sentence is clear, interpretable, and directly relevant to the question. "
            "Provide a well-rounded answer when possible. "
            "Focus on required and relevant information, avoiding unnecessary or borderline content. "
            "Express uncertainty when appropriate rather than making unfounded claims. "
            "Prioritize factual accuracy and avoid speculation or potentially harmful advice. "
            "For patient-oriented questions, provide information suitable for clinician review. "
            "Assume the reader is a healthcare professional, but avoid overly technical jargon. "
            "Do not include trivial statements or generic recommendations to see a health professional, stick to the 150 word limit. "
            "Avoid providing any meta-commentary about the answering process."
        )
        self.user_input = "{query}"
        self.sep = "\n\n"

    @staticmethod
    def _uses_chat_message_format(model: str) -> bool:
        lowered_model = model.lower()
        return not any(
            name in lowered_model
            for name in ("command-r", "chatqa", "llama", "mistral", "qwen")
        )

    def render(self, query: str, context: List[str], model: str) -> RenderedPrompt:
        sep = self.sep
        if not (self._uses_chat_message_format(model) or "chatqa" in model.lower()):
            sep = "\n"
        str_context = sep.join(context)
        instruction = self.get_instruction()

        if (
            self.prompt_mode == PromptMode.RAGNAROK_V4_NO_CITE
            or self.prompt_mode == PromptMode.RAGNAROK_V5_BIOGEN_NO_CITE
        ):
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
                if self.prompt_mode
                in [
                    PromptMode.RAGNAROK_V4_NO_CITE,
                    PromptMode.RAGNAROK_V5_BIOGEN_NO_CITE,
                ]
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
            prompt = f"{self.system_message_chatqa}{sep}{self.input_context.format(context=str_context)}{sep}User: {user_input_context}"
        else:
            system_message = (
                self.system_message_gpt_no_cite
                if self.prompt_mode
                in [
                    PromptMode.RAGNAROK_V4_NO_CITE,
                    PromptMode.RAGNAROK_V5_BIOGEN_NO_CITE,
                ]
                else self.system_message_gpt
            )
            # conv.set_system_message(system_message)
            # conv.append_message(conv.roles[0], user_input_context)
            # conv.append_message(conv.roles[1], None)
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

    def __call__(self, query: str, context: List[str], model: str) -> Any:
        return self.render(query, context, model).runtime_prompt()

    def get_instruction(self) -> str:
        if self.prompt_mode == PromptMode.RAGNAROK_V2:
            return self.instruction_ragnarok_v2
        elif self.prompt_mode == PromptMode.RAGNAROK_V3:
            return self.instruction_ragnarok_v3
        elif self.prompt_mode == PromptMode.RAGNAROK_V4:
            return self.instruction_ragnarok_v4
        elif self.prompt_mode == PromptMode.RAGNAROK_V4_NO_CITE:
            return self.instruction_ragnarok_v4_no_cite
        elif self.prompt_mode == PromptMode.RAGNAROK_V5_BIOGEN_NO_CITE:
            return self.instruction_ragnarok_v5_biogen_no_cite
        elif self.prompt_mode == PromptMode.RAGNAROK_V4_BIOGEN:
            return self.instruction_ragnarok_v4_biogen
        elif self.prompt_mode == PromptMode.RAGNAROK_V5_BIOGEN:
            return self.instruction_ragnarok_v5_biogen
        else:
            return self.instruction_ragnarok
