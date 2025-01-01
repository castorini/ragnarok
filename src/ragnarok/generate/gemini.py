import time
from enum import Enum
from typing import Any, Dict, List, Tuple, Union

import os
import google.generativeai as genai

from ragnarok.data import RAGExecInfo, Request
from ragnarok.generate.llm import LLM, PromptMode
from ragnarok.generate.post_processor import gemini_post_processor
from ragnarok.generate.templates.ragnarok_templates import RagnarokTemplates
from ragnarok.data import CitedSentence

class Gemini(LLM):
    def __init__(
        self,
        model: str,
        context_size: int,
        citation_length: int,
        prompt_mode: PromptMode = PromptMode.GEMINI,
        max_output_tokens: int = 1500,
        num_few_shot_examples: int = 0,
        key: str = None,
    ) -> None:
        """
        Creates instance of the Gemini class, used to make Gemini models perform generation in RAG pipelines. 
        The Gemini 1.5, 1.5-flash, and 2.0-flash-experimental models are the only ones implemented so far.

        Parameters:
        - model (str): The model identifier for the LLM.
        - context_size (int): The maximum number of tokens that the model can handle in a single request.
        - citation_length (int): The number of citations used for generation. 
        - prompt_mode (PromptMode, optional): Specifies the mode of prompt generation, with the default set to GEMINI.
        - max_output_tokens (int, optional): Maximum number of tokens that can be generated in a single response. Defaults to 1500.
        - num_few_shot_examples (int, optional): Number of few-shot learning examples to include in the prompt, allowing for
        the integration of example-based learning to improve model performance. Defaults to 0, indicating no few-shot examples
        by default.
        - key (str, optional): A single Gemini API key.

        Raises:
        - ValueError: If an unsupported prompt mode is provided or if no API key / invalid API key is supplied.
        """

        # Initialize values and check for errors in entered values
        super().__init__(
            model, context_size, prompt_mode, max_output_tokens, num_few_shot_examples
        )
        self.key = str(key)
        self._citation_length = citation_length
        if not (key and isinstance(self.key, str)):
            raise ValueError(f"Gemini api key not provided or in an invalid format. The key provided (if any) is {key}. Assign the appropriate key to the GEMINI_API_KEY env variable.")
        if prompt_mode not in [
            PromptMode.GEMINI,
            PromptMode.CHATQA
        ]:
            raise ValueError(
                f"unsupported prompt mode for Gemini models: {prompt_mode}, expected one of {PromptMode.GEMINI}, {PromptMode.CHATQA}."
            )
        
        # Configure model parameters
        genai.configure(api_key=self.key)
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": max_output_tokens,
            "response_mime_type": "text/plain",
        }
        safety_settings = [
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}
        ]
        system_instruction = "This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context."

        # Initialize model
        self.gen_model = genai.GenerativeModel(
            model_name=model,
            generation_config=generation_config,
            safety_settings=safety_settings,
            system_instruction=system_instruction
        )

    def run_llm(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        logging: bool = False,
    ) -> Tuple[List[CitedSentence], RAGExecInfo]:
        
        chat_session = self.gen_model.start_chat(
            history=[
            ]
        )

        response = chat_session.send_message(prompt).text

        answers, rag_exec_response = gemini_post_processor(response, self._citation_length)

        rag_exec_info = RAGExecInfo(
            prompt=prompt,
            response=rag_exec_response,
            input_token_count=None,
            output_token_count=None,
            candidates=[],
        )

        if logging:
            print(f"Prompt: {prompt}")
            print(f"Response: {response}")
            print(f"Answers: {answers}")
            print(f"RAG Exec Info: {rag_exec_info}")
        
        return answers, rag_exec_info

    def create_prompt(
        self, request: Request, topk: int
    ) -> Tuple[List[Dict[str, str]], int]:
        query = request.query.text
        max_length = (self._context_size - 200) // topk
        rank = 0
        context = []
        for cand in request.candidates[:topk]:
            rank += 1
            content = self.convert_doc_to_prompt_content(cand.doc, max_length)
            context.append(
                f"[{rank}] {self._replace_number(content)}",
            )

        ragnarok_template = RagnarokTemplates(self._prompt_mode)
        messages = ragnarok_template(query, context, "gemini")

        return messages

    def get_num_tokens(self, prompt: Union[str, List[Dict[str, str]]]) -> int:
        """Placeholder function. Returns 1."""
        return 1

    def cost_per_1k_token(self, input_token: bool) -> float:
        """Placeholder function. Returns 1"""
        return 1