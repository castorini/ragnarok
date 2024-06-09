import json
import random
from typing import Tuple

import torch
from fastchat.model import get_conversation_template, load_model
from ftfy import fix_text
from transformers.generation import GenerationConfig

from ragnarok.data import Request
from ragnarok.generate.llm import LLM, PromptMode


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
        dtype: str = "float32",
    ) -> None:
        """
        Creates instance of the OSLLM class, an extension of LLM designed for performing retrieval-augmented generation using
        a specified open-source language model. Some configurations are supported such as GPU acceleration, and custom system
        messages for generating prompts.

        Parameters:
        - model (str): Identifier for the language model to be used for ranking tasks.
        - context_size (int, optional): Maximum number of tokens that can be handled in a single prompt. Defaults to 4096.
        - prompt_mode (PromptMode, optional): Specifies the mode of prompt generation, with the default set to RAGNAROK,
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
        super().__init__(
            model, context_size, prompt_mode, max_output_tokens, num_few_shot_examples
        )
        self._device = device
        if self._device == "cuda":
            assert torch.cuda.is_available()
        if prompt_mode not in [PromptMode.RAGNAROK]:
            raise ValueError(
                f"Unsupported prompt mode: {prompt_mode}. The only prompt mode currently supported is a slight variation of {PromptMode.RAGNAROK} prompt."
            )
        # ToDo: Make repetition_penalty configurable
        self._llm, self._tokenizer = load_model(model, device=device, num_gpus=num_gpus)
        if num_few_shot_examples > 0:
            # TODO(ronak): Add support for few-shot examples
            pass

    def run_llm(self, prompt: str, logging: bool = False) -> Tuple[str, int]:
        inputs = self._tokenizer([prompt])
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

    def _add_few_shot_examples(self, conv):
        for _ in range(self._num_few_shot_examples):
            ex = random.choice(self._examples)
            obj = json.loads(ex)
            prompt = obj["conversations"][0]["value"]
            response = obj["conversations"][1]["value"]
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], response)
        return conv

    def create_prompt(self, request: Request, topk: int) -> Tuple[str, int]:
        query = request.query.text
        num = len(request.candidates[:topk])
        max_length = (self._context_size - 200 - self.num_output_tokens()) // topk
        while True:
            conv = get_conversation_template(self._model)
            if self._system_message:
                conv.set_system_message(self._system_message)
            conv = self._add_few_shot_examples(conv)
            prefix = self._add_prefix_prompt(query, num)
            rank = 0
            input_context = f"{prefix}\n"
            for cand in request.candidates[:topk]:
                rank += 1
                # For Japanese should cut by character: content = content[:int(max_length)]
                content = self.convert_doc_to_prompt_content(cand.doc, max_length)
                input_context += f"[{rank}] {self._replace_number(content)}\n"

            input_context += self._add_post_prompt(query, num)
            conv.append_message(conv.roles[0], input_context)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            prompt = fix_text(prompt)
            num_tokens = self.get_num_tokens(prompt)
            if num_tokens <= self.max_tokens() - self.num_output_tokens():
                break
            else:
                max_length -= max(
                    1,
                    (num_tokens - self.max_tokens() + self.num_output_tokens())
                    // (topk * 4),
                )
        return prompt, self.get_num_tokens(prompt)

    def get_num_tokens(self, prompt: str) -> int:
        return len(self._tokenizer.encode(prompt))

    def cost_per_1k_token(self, input_token: bool) -> float:
        return 0
