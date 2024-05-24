import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import cohere
import os

from ragnarok.generate.llm import PromptMode, LLM
from ragnarok.data import Request
from ragnarok.generate.post_processor import CoherePostProcessor
from ragnarok.generate.api_keys import get_cohere_api_key

class Cohere(LLM):
    def __init__(
        self,
        model: str,
        context_size: int,
        prompt_mode: PromptMode = PromptMode.RAGNAROK,
        max_output_tokens: int = 1500,
        num_few_shot_examples: int = 0,
        key: str = get_cohere_api_key(),
    ) -> None:
        """
        Creates instance of the Cohere class, to deal with Cohere Command R models.

        Parameters:
        - model (str): The model identifier for the LLM (model identifier information can be found via OpenAI's model lists).
        - context_size (int): The maximum number of tokens that the model can handle in a single request.
        - prompt_mode (PromptMode, optional): Specifies the mode of prompt generation, with the default set to RANK_GPT,
         indicating that this class is designed primarily for listwise ranking tasks following the RANK_GPT methodology.
        - max_output_tokens (int, optional): Maximum number of tokens that can be generated in a single response. Defaults to 1500.
        - num_few_shot_examples (int, optional): Number of few-shot learning examples to include in the prompt, allowing for
        the integration of example-based learning to improve model performance. Defaults to 0, indicating no few-shot examples
        by default.
        - key (str, optional): The Cohere API key, defaults to the value of the COHERE_API_KEY environment variable.

        Raises:
        - ValueError: If an unsupported prompt mode is provided or if no Cohere API key / invalid key is supplied.
        """
        if model not in ["command-r-plus", "command-r"]:
            raise ValueError(
                f"Unsupported model: {model}. The only models currently supported are 'command-r' and 'command-r-plus' in Cohere."
            )
        super().__init__(model, context_size, prompt_mode, max_output_tokens, num_few_shot_examples)
        self._client = cohere.Client(key)
        self._post_processor = CoherePostProcessor()

    def run_llm(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        logging: bool = False,
    ) -> Tuple[Any, int]:
        query, top_k_docs = prompt[0]["query"], prompt[0]["context"]
        if logging:
            print(f"Query: {query}")
            print(f"Top K Docs: {top_k_docs}")
        while True:
            try:
                response = self._client.chat(
                    model=self._model,
                    message=query,
                    documents=top_k_docs
                )
                break
            except Exception as e:
                print(str(e))
                time.sleep(60)
        if logging:
            print(f"Response: {response}")
        answers = self._post_processor(response)
        if logging:
            print(f"Answers: {answers}")
        return answers, -1

    def create_prompt(
        self, request: Request, topk: int
    ) -> Tuple[List[Dict[str, Any]], int]:
        query = request.query.text
        max_length = (self._context_size - 200) // topk
        while True:
            rank = 0
            context = []
            for cand in request.candidates[:topk]:
                rank += 1
                content = self.convert_doc_to_prompt_content(cand.doc, max_length)
                context.append(content)
            if self._prompt_mode == PromptMode.COHERE:
                messages = [{"query": query, "context": context}]
            num_tokens = self.get_num_tokens(messages)
            if num_tokens <= self.max_tokens() - self.num_output_tokens():
                break
            else:
                max_length -= max(
                    1,
                    (num_tokens - self.max_tokens() + self.num_output_tokens())
                    // (topk * 4),
                )
        return messages, self.get_num_tokens(messages)

    def get_num_tokens(self, prompt: Union[str, List[Dict[str, str]]]) -> int:
        """Returns the number of tokens used by a list of messages in prompt."""
        # TODO(ronak): Add support
        return -1

    def cost_per_1k_token(self, input_token: bool) -> float:
        # TODO(ronak): Add support
        return -1
