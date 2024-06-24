import re
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from openai import OpenAI, AzureOpenAI
import tiktoken
from ftfy import fix_text

from ragnarok.data import Request

class QueryDecomposeMode(Enum):
    RESEARCHY_QUESTIONS = "researchy_questions"

class SafeOpenaiQueryDecomposer:
    def __init__(
        self,
        model: str,
        context_size: int,
        prompt_mode: QueryDecomposeMode = QueryDecomposeMode.RESEARCHY_QUESTIONS,
        window_size: int = 10,
        keys=None,
        key_start_id=None,
        proxy=None,
        api_type: str = None,
        api_base: str = None,
        api_version: str = None,
    ) -> None:
        """
        Creates instance of the SafeOpenai class, a specialized version of RankLLM designed for safely handling OpenAI API calls with
        support for key cycling, proxy configuration, and Azure AI conditional integration.

        Parameters:
        - model (str): The model identifier for the LLM (model identifier information can be found via OpenAI's model lists).
        - context_size (int): The maximum number of tokens that the model can handle in a single request.
        - prompt_mode (PromptMode, optional): Specifies the mode of prompt generation, with the default set to SUPPORT_GRADE_2,
         indicating that this class is designed primarily for nugget assignment tasks.
        - window_size (int, optional): The window size for handling text inputs. Defaults to 10.
        - keys (Union[List[str], str], optional): A list of OpenAI API keys or a single OpenAI API key.
        - key_start_id (int, optional): The starting index for the OpenAI API key cycle.
        - proxy (str, optional): The proxy configuration for OpenAI API calls.
        - api_type (str, optional): The type of API service, if using Azure AI as the backend.
        - api_base (str, optional): The base URL for the API, applicable when using Azure AI.
        - api_version (str, optional): The API version, necessary for Azure AI integration.

        Raises:
        - ValueError: If an unsupported prompt mode is provided or if no OpenAI API keys / invalid OpenAI API keys are supplied.

        Note:
        - This class supports cycling between multiple OpenAI API keys to distribute quota usage or handle rate limiting.
        - Azure AI integration is depends on the presence of `api_type`, `api_base`, and `api_version`.
        """
        self._model = model
        self._context_size = context_size
        self._prompt_mode = prompt_mode
        if isinstance(keys, str):
            keys = [keys]
        if not keys:
            raise ValueError("Please provide OpenAI Keys.")
        if prompt_mode not in [QueryDecomposeMode.RESEARCHY_QUESTIONS]:
            raise ValueError(
                f"unsupported prompt mode for GPT models: {prompt_mode}, expected {QueryDecomposeMode.RESEARCHY_QUESTIONS} or {QueryDecomposeMode.RESEARCHY_QUESTIONS}."
            )

        self._window_size = window_size
        self._output_token_estimate = None
        self._keys = keys
        self._cur_key_id = key_start_id or 0
        self._cur_key_id = self._cur_key_id % len(self._keys)
        self.openai = OpenAI() if not all([api_type, api_base, api_version]) else AzureOpenAI()
        self.openai.proxy = proxy
        print(self._keys)
        print(proxy)
        print(api_type, api_base, api_version)
        self.openai.api_key = self._keys[self._cur_key_id]
        self.use_azure_ai = True

        if all([api_type, api_base, api_version]):
            self.openai.api_version = api_version
            self.openai.api_type = api_type
            self.openai.api_base = api_base
            print("Azure AI integration enabled.")
            self.use_azure_ai = True

    class CompletionMode(Enum):
        UNSPECIFIED = 0
        CHAT = 1
        TEXT = 2

    def _call_completion(
        self,
        *args,
               completion_mode: CompletionMode,
        return_text=False,
        reduce_length=False,
        **kwargs,
    ) -> Union[str, Dict[str, Any]]:
        while True:
            try:
                if completion_mode == self.CompletionMode.CHAT:
                    completion = self.openai.chat.completions.create(
                        *args, **kwargs, timeout=30
                    )
                elif completion_mode == self.CompletionMode.TEXT:
                    completion = self.openai.Completion.create(*args, **kwargs)
                else:
                    raise ValueError(
                        "Unsupported completion mode: %V" % completion_mode
                    )
                break
            except Exception as e:
                print(str(e))
                if "This model's maximum context length is" in str(e):
                    print("reduce_length")
                    return "ERROR::reduce_length"
                if "The response was filtered" in str(e):
                    print("The response was filtered")
                    return "ERROR::The response was filtered"
                self._cur_key_id = (self._cur_key_id + 1) % len(self._keys)
                self.openai.api_key = self._keys[self._cur_key_id]
                time.sleep(0.1)
        if return_text:
            completion = (
                completion.choices[0].message.content
                if completion_mode == self.CompletionMode.CHAT
                else completion.choices[0].text
            )
        return completion

    def run_llm(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        current_window_size: Optional[int] = None,
    ) -> Tuple[str, int]:
        model_key = "model"
        response = self._call_completion(
            messages=prompt,
            temperature=0,
            completion_mode=self.CompletionMode.CHAT,
            return_text=True,
            max_tokens=512,
            frequency_penalty=0,
            presence_penalty=0,
            **{model_key: self._model},
        )
        try:
            encoding = tiktoken.get_encoding(self._model)
        except:
            encoding = tiktoken.get_encoding("cl100k_base")
        return response, len(encoding.encode(response))

    def _get_prefix_for_researchy_questions_decomposition_prompt(
        self, query: str, num: int
    ) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": "You are QuestionDecomposerLLM, an intelligent assistant that can hierarchically decompose a given question into multiple closed-book sub-questions in Pythonic list format.",
            },
            {
                "role": "user",
                "content": f"Provide a hierarchical decomposition of the given question into multiple closed-book sub-questions which will be helpful to answer the main question. The sub-questions should be in the form of a question and should not be a statement and should not be a yes/no question. The sub-questions should be clear, concise, and should not be redundant. The sub-questions should not contain the main question itself. Return only the final list of {num} sub-questions in a Pythonic list format.\n\nMain Question: {query}\n\nSub-Question List:",
            },
        ]

    def create_prompt(
        self, query: str, num: int=10
    ) -> Tuple[List[Dict[str, str]], int]:
        if self._prompt_mode == QueryDecomposeMode.RESEARCHY_QUESTIONS:
            return self.create_researchy_questions_decomposition_prompt(query)

    def create_researchy_questions_decomposition_prompt(
        self, query: str, num: int=10
    ):
        query = fix_text(query)
        messages = self._get_prefix_for_researchy_questions_decomposition_prompt(query, num)
        return messages
