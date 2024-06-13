import re
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import openai
import tiktoken
from ftfy import fix_text

from ragnarok.data import Request


class NuggetScoreMode(Enum):
    VITAL_OKAY = "vital_okay"


class SafeOpenaiNuggetScorer:
    def __init__(
        self,
        model: str,
        context_size: int,
        prompt_mode: NuggetScoreMode = NuggetScoreMode.VITAL_OKAY,
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
        - prompt_mode (PromptMode, optional): Specifies the mode of prompt generation, with the default set to RANK_GPT,
         indicating that this class is designed primarily for listwise ranking tasks following the RANK_GPT methodology.
        - num_few_shot_examples (int, optional): Number of few-shot learning examples to include in the prompt, allowing for
        the integration of example-based learning to improve model performance. Defaults to 0, indicating no few-shot examples
        by default.
        - window_size (int, optional): The window size for handling text inputs. Defaults to 20.
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
        if prompt_mode not in [NuggetScoreMode.VITAL_OKAY]:
            raise ValueError(
                f"unsupported prompt mode for GPT models: {prompt_mode}, expected {NuggetScoreMode.VITAL_OKAY}."
            )

        self._window_size = window_size
        self._output_token_estimate = None
        self._keys = keys
        self._cur_key_id = key_start_id or 0
        self._cur_key_id = self._cur_key_id % len(self._keys)
        openai.proxy = proxy
        openai.api_key = self._keys[self._cur_key_id]
        self.use_azure_ai = False

        if all([api_type, api_base, api_version]):
            # See https://learn.microsoft.com/en-US/azure/ai-services/openai/reference for list of supported versions
            openai.api_version = api_version
            openai.api_type = api_type
            openai.api_base = api_base
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
                    completion = openai.chat.completions.create(
                        *args, **kwargs, timeout=30
                    )
                elif completion_mode == self.CompletionMode.TEXT:
                    completion = openai.Completion.create(*args, **kwargs)
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
                openai.api_key = self._keys[self._cur_key_id]
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
            completion_mode=SafeOpenaiNuggetScorer.CompletionMode.CHAT,
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

    def _get_prefix_for_vital_okay_prompt(
        self, query: str, num: int
    ) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": "You are NuggetizeScoreLLM, an intelligent assistant that can label a list of atomic nuggets based on their importance for a given search query.",
            },
            {
                "role": "user",
                "content": f"Based on the query, label of the {num} nuggets either a vital or okay based on the following criteria. Vital nuggets represent concepts that must be present in a “good” answer; on the other hand, okay nuggets contribute worthwhile information about the target but are not essential. Return the list of labels in a Pythonic list format (type: List[str]). The list should be in the same order as the input nuggets. Make sure to provide a label for each nugget.\n\n",
            },
        ]

    def create_prompt(
        self, request: Request, rank_start: int, rank_end: int, nuggets: List[str] = []
    ) -> Tuple[List[Dict[str, str]], int]:
        if self._prompt_mode == NuggetScoreMode.VITAL_OKAY:
            return self.create_vital_okay_prompt(request, rank_start, rank_end, nuggets)

    def create_vital_okay_prompt(
        self, query: str, rank_start: int, rank_end: int, nuggets: List[str] = []
    ):
        nuggets = [fix_text(nugget) for nugget in nuggets]
        num = min(rank_end - rank_start, len(nuggets[rank_start:rank_end]))
        nuggets = nuggets[rank_start:rank_end]
        messages = self._get_prefix_for_vital_okay_prompt(query, num)
        message = messages[-1]["content"]
        message += f"Search Query: {query}\nNugget List: {nuggets}\nOnly return the list of labels (List[str]). Do not explain.\nLabels:"
        messages[-1]["content"] = message
        return messages

    def _replace_number(self, s: str) -> str:
        return re.sub(r"\[(\d+)\]", r"(\1)", s)
