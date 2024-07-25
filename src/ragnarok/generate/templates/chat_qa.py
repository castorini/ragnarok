from typing import List

from fastchat.model import get_conversation_template
from ftfy import fix_text


class ChatQATemplate:
    def __init__(self):
        self.system_message_gpt = "This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful and detailed answers to the user's question based on the context references. The assistant should also indicate when the answer cannot be found in the context references."
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
        self.user_input = "{query}"
        self.sep = "\n\n"

    def __call__(self, query: str, context: List[str], model: str) -> List[str]:
        if not ("gpt" in model or "chatqa" in model.lower()):
            self.sep = "\n"
        str_context = self.sep.join(context)
        if "gpt" in model:
            user_input_context = (
                f"Instruction: {self.instruction_ragnarok}"
                + self.sep
                + f"Documents: {fix_text(str_context)}"
                + self.sep
                + f"Query: {query}"
                + self.sep
                + f"Instruction: {self.instruction_ragnarok}"
            )
            user_input_context += "\n\nAnswer:"
        elif "chatqa" in model.lower():
            user_input_context = (
                f"Instruction: {self.instruction_ragnarok}"
                + " "
                + f"Question: {query}{self.sep}Assistant:"
            )
        else:
            user_input_context = (
                f"Instruction: {self.instruction_ragnarok_v2}"
                + "\n"
                + f"The following are context references from which you can cite the identifier. References: {fix_text(str_context)}"
                + "\n"
                + f"Query: {query}"
                + "\n"
                + f"Instruction: {self.instruction_ragnarok_v2}"
            )
        if "gpt" in model:
            messages = []
            messages.append(
                {
                    "role": "system",
                    "content": self.system_message_gpt,
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": fix_text(user_input_context),
                }
            )
            return messages
        elif "chatqa" in model.lower():
            prompt = f"{self.system_message_chatqa}{self.sep}{self.input_context.format(context=str_context)}{self.sep}User: {user_input_context}"
        else:
            conv = get_conversation_template(model)
            conv.set_system_message(self.system_message_gpt)
            conv.append_message(conv.roles[0], user_input_context)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
        prompt = fix_text(prompt)
        return prompt
