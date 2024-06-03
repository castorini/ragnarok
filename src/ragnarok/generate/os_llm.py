import os

import json
import random
from typing import Optional, Tuple

import torch
from fastchat.model import get_conversation_template, load_model
try:
    from vllm import LLM as VLLM
    from vllm import SamplingParams
except:
    LLM = None
    SamplingParams = None
from ftfy import fix_text
from transformers.generation import GenerationConfig

from ragnarok.generate.llm import PromptMode, LLM
from ragnarok.data import RAGExecInfo, Request
from ragnarok.generate.post_processor import GPTPostProcessor
from ragnarok.generate.templates.chat_qa import ChatQATemplate

class OSLLM(LLM):
    def __init__(
        self,
        model: str,
        context_size: int = 8192,
        prompt_mode: PromptMode = PromptMode.CHATQA,
        max_output_tokens: int = 1500,
        num_few_shot_examples: int = 0,
        device: str = "cuda",
        num_gpus: int = 1,
        dtype: str = "float16",
    ) -> None:
        """
         Creates instance of the OSLLM class, an extension of LLM designed for performing retrieval-augmented generation using
         a specified open-source language model. Some configurations are supported such as GPU acceleration, and custom system
         messages for generating prompts.

         Parameters:
         - model (str): Identifier for the language model to be used for ranking tasks.
         - context_size (int, optional): Maximum number of tokens that can be handled in a single prompt. Defaults to 4096.
         - prompt_mode (PromptMode, optional): Specifies the mode of prompt generation, with the default set to CHATQA,
         indicating that this class is designed primarily for listwise ranking tasks following the RAGNAROK methodology.
         - max_output_tokens (int, optional): Maximum number of tokens that can be generated in a single response. Defaults to 1500.
         - num_few_shot_examples (int, optional): Number of few-shot learning examples to include in the prompt, allowing for
         the integration of example-based learning to improve model performance. Defaults to 0, indicating no few-shot examples
         by default.
         - device (str, optional): Specifies the device for model computation ('cuda' for GPU or 'cpu'). Defaults to 'cuda'.
         - num_gpus (int, optional): Number of GPUs to use for model loading and inference. Defaults to 1.
         - dtype (str, optional): Specifies the data type for model computation ('float32' or 'float16' or 'bfloat16'). Defaults to 'float32'.
         Raises:
         - AssertionError: If CUDA is specified as the device but is not available on the system.
         - ValueError: If an unsupported prompt mode is provided.

         Note:
         - This class is operates given scenarios where listwise ranking is required, with support for dynamic
         passage handling and customization of prompts through system messages and few-shot examples.
         - GPU acceleration is supported and recommended for faster computations.
        """
        super().__init__(model, context_size, prompt_mode, max_output_tokens, num_few_shot_examples)
        self._device = device
        if self._device == "cuda":
            assert torch.cuda.is_available()
        if prompt_mode not in [PromptMode.CHATQA]:
            raise ValueError(
                f"Unsupported prompt mode: {prompt_mode}. The only prompt mode currently supported is a slight variation of {PromptMode.CHATQA} prompt."
            )
        # ToDo: Make repetition_penalty configurable
        try:
            print(model)
            self._llm = VLLM(model, download_dir=os.getenv("HF_HOME"),
                             enforce_eager=False, tensor_parallel_size=2)
            self._tokenizer = self._llm.get_tokenizer()
        except:
            self._llm, self._tokenizer = load_model(model, device=device, num_gpus=num_gpus)
        self._post_processor = GPTPostProcessor()
        if num_few_shot_examples > 0:
            # TODO(ronak): Add support for few-shot examples
            pass

    def run_llm(
        self, prompt: str, logging: bool = False, vllm: bool = True
    ) -> Tuple[str, int]:
        if logging:
            print(f"Prompt: {prompt}")
        try:
            inputs = self._tokenizer([prompt])
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=self._output_token_estimate,
                min_tokens=100,
            )
            output = self._llm.generate([prompt], sampling_params)
            text = output[0].outputs[0].text
            # Anything after a User: / Context: / References: / Note: remove
            text = text.split("User:")[0].split("Context:")[0].split("References:")[0].split("Note:")[0]
            if logging:
                print(f"Response: {text}")
            answers, rag_exec_response = self._post_processor(text)
            if logging:
                print(f"Answer: {answers}")
            rag_exec_info = RAGExecInfo(prompt=prompt, response=rag_exec_response, input_token_count=self.get_num_tokens(prompt), 
                                    output_token_count=sum([len(ans.text) for ans in answers]), candidates=[]) 
            return answers, rag_exec_info
        except:
            inputs = {k: torch.tensor(v).to(self._device) for k, v in inputs.items()}
            gen_cfg = GenerationConfig.from_model_config(self._llm.config)
            gen_cfg.max_new_tokens = self.num_output_tokens()
            # gen_cfg.temperature = 0
            gen_cfg.do_sample = False
            output_ids = self._llm.generate(**inputs, generation_config=gen_cfg)

            if self._llm.config.is_encoder_decoder:
                output_ids = output_ids[0]
            else:
                output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
            outputs = self._tokenizer.decode(
                output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
            )
            return outputs, output_ids.size(0)

    def create_prompt(
        self, request: Request, topk: int
    ) -> Tuple[str, int]:
        query = request.query.text
        max_length = (self._context_size - 200) // topk
        while True:
            rank = 0
            context = []
            for cand in request.candidates[:topk]:
                rank += 1
                content = self.convert_doc_to_prompt_content(cand.doc, max_length)
                context.append(
                    f"[{rank}] {self._replace_number(content)}", 
                )
            if self._prompt_mode == PromptMode.CHATQA:
                chat_qa_template = ChatQATemplate()
                messages = chat_qa_template(query, context, "chatqa")
            else:
                raise ValueError(
                    f"Unsupported prompt mode: {self._prompt_mode}, expected {PromptMode.CHATQA}."
                )
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

    def get_num_tokens(self, prompt: str) -> int:
        return len(self._tokenizer.encode(prompt))

    def cost_per_1k_token(self, input_token: bool) -> float:
        return 0
