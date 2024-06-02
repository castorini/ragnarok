import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import openai
import tiktoken
from ftfy import fix_text
import re
from ragnarok.data import Request

class NuggetMode(Enum):
    ATOMIC = "atomic"
    NOUN_PHRASE = "noun_phrase"
    QUESTION = "question"
    

class SafeOpenaiNuggetizer():
    def __init__(
        self,
        model: str,
        context_size: int,
        prompt_mode: NuggetMode = NuggetMode.ATOMIC,
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
        self._context_size  = context_size
        self._prompt_mode = prompt_mode
        if isinstance(keys, str):
            keys = [keys]
        if not keys:
            raise ValueError("Please provide OpenAI Keys.")
        if prompt_mode not in [NuggetMode.ATOMIC, NuggetMode.QUESTION]:
            raise ValueError(
                f"unsupported prompt mode for GPT models: {prompt_mode}, expected {PromptMode.ATOMIC}."
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
            completion_mode=SafeOpenaiNuggetizer.CompletionMode.CHAT,
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

    def _get_prefix_for_atomic_prompt(
        self, query: str, num: int
    ) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": "You are NuggetizeLLM, an intelligent assistant that can update a list of atomic nuggets to best provide all the information required for the query.",
            },
            {
                "role": "user",
                "content": f"Update the list of atomic nuggets of information (1-12 words), if needed, so they best provide the information required for the query. Leverage only the initial list of nuggets (if exists) and the provided context (this is an iterative process).  Return only the final list of all nuggets in a Pythonic list format (even if no updates). Make sure there is no redundant information. The more nuggets, the merrier. Order them in decreasing order of importance. Prefer nuggets that provide more interesting information.\n\n",
            },
        ] 
    def _get_prefix_for_question_prompt(
        self, query: str, num: int
    ) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": "You are NuggetizeLLM, an intelligent assistant that can update a list of subquery nuggets to best provide all the atomic subqueries that would best lead to information required to answer the main search query.",
            },
            {
                "role": "user",
                "content": f"Update the list of atomic subquery nuggets, if needed, so they best provide the information required for the query. Leverage only the initial list of nuggets (if exists) and the provided context (this is an iterative process to identify such nugget subquestions.  Return only the final list of all subquestion nuggets in a Pythonic list format (even if no updates). Make sure there is no redundant information. Order them in decreasing order of importance. Prefer subquestion nuggets that lead to more interesting information.\n\n",
            },
        ]

    def create_prompt(
        self, request: Request, rank_start: int, rank_end: int,
        nuggets: List[str] = []
    ) -> Tuple[List[Dict[str, str]], int]:
        if self._prompt_mode == NuggetMode.ATOMIC:
            return self.create_atomic_prompt(request, rank_start, rank_end, nuggets)
        elif self._prompt_mode == NuggetMode.QUESTION:
            return self.create_question_prompt(request, rank_start, rank_end, nuggets)

    def create_atomic_prompt(
        self, request: Request, rank_start: int, rank_end: int, nuggets: List[str] = []
    ):
        query = request.query.text
        num = len(request.candidates[rank_start:rank_end])

        max_length = 500 * (self._window_size / (rank_end - rank_start))
        messages = self._get_prefix_for_atomic_prompt(query, num)
        message = messages[-1]["content"]
        rank = 0
        all_context = ""
        for cand in request.candidates[rank_start:rank_end]:
            rank += 1
            content = self.convert_doc_to_prompt_content(cand.doc, max_length)
            content = self._replace_number(content)
            
            all_context += f"[{rank}] {content}\n"
        message += f"Search Query: {request.query.text}\nContext:\n{all_context}"
        message += f"Search Query: {query}\nInitial Nugget List: {nuggets}\nOnly update the list of atomic nuggets (if needed, else return as is). Do not explain. Always answer in short nuggets (not questions). List in the form [\"a\", \"b\", ...] and a and b are strings with no mention of \".\nUpdated Nugget List:"
        messages[-1]["content"] = message
        return messages
    
    def _replace_number(self, s: str) -> str:
        return re.sub(r"\[(\d+)\]", r"(\1)", s)

    def convert_doc_to_prompt_content(self, doc: Dict[str, Any], max_length: int) -> str:
        if "text" in doc:
            content = doc["text"]
        elif "segment" in doc:
            content = doc["segment"]
        elif "contents" in doc:
            content = doc["contents"]
        else:
            content = doc["passage"]
        if "title" in doc and doc["title"]:
            content = "Title: " + doc["title"] + " " + "Content: " + content
        content = content.strip()
        content = fix_text(content)
        content = content.replace("\n", " ")
        content = " ".join(content.split()[: int(max_length)])
        return self._replace_number(content)

    def create_question_prompt(
        self, request: Request, rank_start: int, rank_end: int, nuggets: List[str] = []
    ):
        query = request.query.text
        num = len(request.candidates[rank_start:rank_end])

        max_length = 500 * (self._window_size / (rank_end - rank_start))
        messages = self._get_prefix_for_atomic_prompt(query, num)
        message = messages[-1]["content"]
        rank = 0
        all_context = ""
        for cand in request.candidates[rank_start:rank_end]:
            rank += 1
            content = self.convert_doc_to_prompt_content(cand.doc, max_length)
            content = self._replace_number(content)
            
            all_context += f"[{rank}] {content}\n"
        message += f"Search Query: {request.query.text}\nContext:\n{all_context}"
        message += f"Search Query: {query}\nInitial Nugget List: {nuggets}\nOnly update the list of subquestion nuggets (if needed, else return as is). Do not say any word or explain.\nUpdated Nugget List:"
        messages[-1]["content"] = message
        return messages
    
    def _replace_number(self, s: str) -> str:
        return re.sub(r"\[(\d+)\]", r"(\1)", s)

    def convert_doc_to_prompt_content(self, doc: Dict[str, Any], max_length: int) -> str:
        if "text" in doc:
            content = doc["text"]
        elif "segment" in doc:
            content = doc["segment"]
        elif "contents" in doc:
            content = doc["contents"]
        else:
            content = doc["passage"]
        if "title" in doc and doc["title"]:
            content = "Title: " + doc["title"] + " " + "Content: " + content
        content = content.strip()
        content = fix_text(content)
        content = content.replace("\n", " ")
        content = " ".join(content.split()[: int(max_length)])
        return self._replace_number(content)
