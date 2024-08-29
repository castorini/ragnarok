import random
import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Tuple, Union

from ftfy import fix_text

from ragnarok.data import Request, Result, remove_unused_references


class PromptMode(Enum):
    UNSPECIFIED = "unspecified"
    COHERE = "cohere"
    CHATQA = "chatqa"
    RAGNAROK_V2 = "ragnarok_v2"
    RAGNAROK_V3 = "ragnarok_v3"
    RAGNAROK_V4 = "ragnarok_v4"
    RAGNAROK_V4_NO_CITE = "ragnarok_v4_no_cite"

    def __str__(self):
        return self.value


class LLM(ABC):
    def __init__(
        self,
        model: str,
        context_size: int,
        prompt_mode: PromptMode,
        max_output_tokens: int = 1500,
        num_few_shot_examples: int = 0,
    ) -> None:
        self._model = model
        self._context_size = context_size
        self._prompt_mode = prompt_mode
        self._num_few_shot_examples = num_few_shot_examples
        self._output_token_estimate = max_output_tokens

    def max_tokens(self) -> int:
        """
        Returns the maximum number of tokens for a given model

        Returns:
            int: The maximum token count.
        """
        return self._context_size

    @abstractmethod
    def run_llm(
        self, prompt: Union[str, List[Dict[str, Any]]], logging: bool = False
    ) -> Tuple[Any, int]:
        """
        Abstract method to run the target language model with a passed in prompt.

        Args:
            prompt (Union[str, List[Dict[str, str]]]): The prompt to be processed by the model.
            logging (bool, optional): Flag to enable logging of operations. Defaults to False.

        Returns:
            Tuple[Any, int]: A tuple object containing the answer response and the number of tokens in the response.
        """
        pass

    @abstractmethod
    def create_prompt(
        self, request: Request, topk: int
    ) -> Tuple[Union[str, List[Dict[str, str]]], int]:
        """
        Abstract method to create a prompt based on the request and given topk.

        Args:
            request (Result): The request object containing data for prompt generation.
            topk (int): The topk ranks considered for prompt generation.

        Returns:
            Tuple[Union[str, List[Dict[str, str]]], int]: A tuple object containing the generated prompt and the number of tokens in the generated prompt.
        """
        pass

    @abstractmethod
    def get_num_tokens(self, prompt: Union[str, List[Dict[str, str]]]) -> int:
        """
        Abstract method to calculate the number of tokens contained in the given prompt.

        Args:
            prompt (Union[str, List[Dict[str, str]]]): The prompt for which to compute the token count for.

        Returns:
            int: The number of tokens in the given prompt.
        """
        pass

    @abstractmethod
    def cost_per_1k_token(self, input_token: bool) -> float:
        """
        Abstract method to calculate the cost per 1,000 tokens for the target language model.

        Args:
            input_token (bool): Flag to indicate if the cost is for input tokens or output tokens.

        Returns:
            float: The cost per 1,000 tokens.
        """
        pass

    @abstractmethod
    def num_output_tokens(self) -> int:
        """
        Abstract method to estimate the number of tokens in the model's output, constrained by max tokens for the target language model.

        Returns:
            int: The estimated number of output tokens.
        """
        pass

    def answer(
        self,
        request: Request,
        topk: int,
        shuffle_candidates: bool = False,
        logging: bool = False,
    ) -> Result:
        """
        Answer a given request using the target language model.

        Args:
            request (Request): The request object to process.
            topk (int): The topk ranks to consider for the generation process.
            shuffle_candidates (bool, optional): Flag to shuffle candidates before processing. Defaults to False.
            logging (bool, optional): Flag to enable logging of operations. Defaults to False.

        Returns:
            Result: The result object after answering the request.
        """
        return self.answer_batch(
            [request],
            topk,
            shuffle_candidates=shuffle_candidates,
            logging=logging,
        )[0]

    def answer_batch(
        self,
        requests: List[Request],
        topk: int,
        shuffle_candidates: bool = False,
        logging: bool = False,
        vllm: bool = False,
    ) -> List[Result]:
        """
        Answer a list of requests using the target language model.

        Args:
            requests (List[Request]): The list of requests to process.
            topk (int): The topk ranks to consider for the generation process.
            shuffle_candidates (bool, optional): Flag to shuffle candidates before processing. Defaults to False.
            logging (bool, optional): Flag to enable logging of operations. Defaults to False.
            vllm (bool, optional): Flag to enable VLLM mode. Defaults to False.

        Returns:
            List[Result]: The list of results after answering the requests.
        """
        initial_results = []
        results = []
        for request in requests:
            if shuffle_candidates:
                # First randomly shuffle rerank_result in first topk ranks
                request.candidates[:topk] = random.sample(
                    request.candidates[:topk],
                    len(request.candidates[:topk]),
                )
        if vllm:
            prompt_input_token_count_list = self.create_prompt_batched(requests, topk)
            prompts = [prompt for prompt, _ in prompt_input_token_count_list]
            answer_rag_exec_info_list = self.run_llm_batched(prompts, logging)
            answers = [answer for answer, _ in answer_rag_exec_info_list]
            rag_exec_summary = [
                rag_exec_info for _, rag_exec_info in answer_rag_exec_info_list
            ]
            for request, answer, rag_exec_info in zip(
                requests, answers, rag_exec_summary
            ):
                result = Result(
                    query=request.query,
                    references=[cand.docid for cand in request.candidates[:topk]],
                    answer=answer,
                    rag_exec_summary=rag_exec_info,
                )
                initial_results.append(result)
        else:
            for request in requests:
                prompt, input_token_count = self.create_prompt(request, topk)
                answer, rag_exec_summary = self.run_llm(prompt, logging)
                rag_exec_summary.candidates = [
                    candidate.__dict__ for candidate in request.candidates[:topk]
                ]
                result = Result(
                    query=request.query,
                    references=[cand.docid for cand in request.candidates[:topk]],
                    answer=answer,
                    rag_exec_summary=rag_exec_summary,
                )
                initial_results.append(result)
        for result in initial_results:
            cleaned_result = remove_unused_references(result)
            results.append(cleaned_result)
        return results

    def num_output_tokens(self) -> int:
        return self._output_token_estimate

    def _clean_response(self, response: str) -> str:
        new_response = fix_text(response)
        return new_response

    def _replace_number(self, s: str) -> str:
        return re.sub(r"\[(\d+)\]", r"(\1)", s)

    def convert_doc_to_prompt_content(
        self, doc: Dict[str, Any], max_length: int
    ) -> str:
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
