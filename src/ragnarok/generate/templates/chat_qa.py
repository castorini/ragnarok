from typing import List

from ftfy import fix_text


class ChatQATemplate:
    def __init__(self):
        self.system_message_gpt = "This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful and detailed answers to the user's question based on the context. The assistant should also indicate when the answer cannot be found in the context."
        self.system_message_osllm = "System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context."
        self.input_context = "Context: {context}"
        self.instruction_official = (
            "Please give a full and complete answer for the question."
        )
        self.instruction_ragnarok = (
            "Please give a full and complete answer for the question. "
            + "Cite each context document inline that supports your answer within brackets [] using the IEEE format. "
            + "Ensure each sentence is properly cited."
        )
        self.user_input = "{query}"
        self.sep = "\n\n"

    def __call__(self, query: str, context: List[str], model: str) -> List[str]:
        str_context = self.sep.join(context)
        if "gpt" in model:
            user_input_context = (
                f"Instruction: {self.instruction_ragnarok}"
                + self.sep
                + f"Context: {self.input_context.format(context=str_context)}"
                + self.sep
                + f"Query: {query}"
                + self.sep
                + f"Instruction: {self.instruction_ragnarok}"
            )
            user_input_context += "\n\nAnswer:"
        else:
            user_input_context = (
                f"Instruction: {self.instruction_ragnarok}"
                + " "
                + f"Question: {query}{self.sep}Assistant:"
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
        else:
            prompt = f"{self.system_message_osllm}{self.sep}{self.input_context.format(context=str_context)}{self.sep}User: {user_input_context}"
        # else:
        #     conv = get_conversation_template(model)
        #     conv.set_system_message(self.system_message)
        #     conv.append_message(conv.roles[0], user_input_context)
        #     conv.append_message(conv.roles[1], None)
        #     prompt = conv.get_prompt()
        prompt = fix_text(prompt)
        return prompt
