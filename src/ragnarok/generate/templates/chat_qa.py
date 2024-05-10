from typing import List
from fastchat.model import get_conversation_template
from ftfy import fix_text

class ChatQATemplate:
    def __init__(self):
        self.system_message = "This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context."
        self.input_context = "{context}"
        self.instruction_official = "Please give a full and complete answer for the question."
        self.instruction_ragnarok = "Please give a full and complete answer for the question. " + \
            "Cite each context document inline that supports your answer within brackets [] using the IEEE format."
        self.user_input = "{query}"
        self.sep = "\n\n"

    def __call__(self, query: str, context: List[str], model: str) -> List[str]:
        str_context = self.sep.join(context)
        user_input_context = f"Instruction: {self.instruction_ragnarok}" + self.sep + f"Context: {self.input_context.format(context=str_context)}" + self.sep + f"Query: {self.query}" \
                + self.sep + f"Instruction: {self.instruction_ragnarok}"
        if model == "gpt":
            messages = []
            messages.append({
                "role": "system",
                "content": self.system_message,
            })
            messages.append({
                "role": "user",
                "content": fix_text(user_input_context),
            })
            return messages
        elif model == "chat-qa":
            prompt = f"{self.system_message}{self.sep}{self.user_input_context.format(context=str_context)}{self.sep}User: {self.user_input.format(query)}Assistant:"
        else:
            conv = get_conversation_template(model)
            conv.set_system_message(self.system_message)
            conv.append_message(conv.roles[0], user_input_context)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
        prompt = fix_text(prompt)
        return prompt
