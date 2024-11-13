import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

import torch
from fastchat.model import load_model

try:
    from vllm import LLM as VLLM
    from vllm import SamplingParams
except:
    LLM = None
    SamplingParams = None
from transformers.generation import GenerationConfig

from ragnarok.data import RAGExecInfo, Request
from ragnarok.generate.llm import LLM, PromptMode
from ragnarok.generate.post_processor import GPTPostProcessor
from ragnarok.generate.templates.ragnarok_templates import RagnarokTemplates


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
        dtype: str = "bfloat16",
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
        self._name = model
        if self._device == "cuda":
            assert torch.cuda.is_available()
        if prompt_mode not in [
            PromptMode.CHATQA,
            PromptMode.RAGNAROK_V2,
            PromptMode.RAGNAROK_V3,
            PromptMode.RAGNAROK_V4,
            PromptMode.RAGNAROK_V4_BIOGEN,
            PromptMode.RAGNAROK_V5_BIOGEN,
            PromptMode.RAGNAROK_V4_NO_CITE,
            PromptMode.RAGNAROK_V5_BIOGEN_NO_CITE,
        ]:
            raise ValueError(
                f"Unsupported prompt mode: {prompt_mode}. The only prompt mode currently supported is a slight variation of {PromptMode.CHATQA} prompt."
            )
        # ToDo: Make repetition_penalty configurable
        try:
            print(model)
            ignore_patterns = ["*consolidated*"] if "mistral" in model else []
            self._llm = VLLM(
                model,
                download_dir=os.getenv("HF_HOME"),
                enforce_eager=False,
                tensor_parallel_size=num_gpus,
                max_model_len=108064,
                ignore_patterns=ignore_patterns,
            )
            self._tokenizer = self._llm.get_tokenizer()
        except:
            self._llm, self._tokenizer = load_model(
                model, device=device, num_gpus=num_gpus
            )
        self._post_processor = GPTPostProcessor()
        if num_few_shot_examples > 0:
            # TODO(ronak): Add support for few-shot examples
            pass

    def run_llm_batched(
        self, prompts: List[str], logging: bool = False, vllm: bool = True
    ) -> List[Tuple[str, int]]:
        if logging:
            for i, prompt in enumerate(prompts):
                print(f"Prompt {i}: {prompt}")
        try:
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=self._output_token_estimate,
                min_tokens=200,
            )
            outputs = self._llm.generate(prompts, sampling_params)
            responses = [output.outputs[0].text for output in outputs]
            if logging:
                for response in responses:
                    print(f"Response: {response}")
            answer_rag_exec_info_list = []
            for prompt, response in zip(prompts, responses):
                answer, rag_exec_response = self._post_processor(response)
                rag_exec_info = RAGExecInfo(
                    prompt=prompt,
                    response=rag_exec_response,
                    input_token_count=self.get_num_tokens(prompt),
                    output_token_count=sum([len(ans.text) for ans in answer]),
                    candidates=[],
                )
                answer_rag_exec_info_list.append((answer, rag_exec_info))

            return answer_rag_exec_info_list
        except:
            assert False, "Failed run_llm_batched"

    def run_llm(
        self, prompt: str, logging: bool = False, vllm: bool = True
    ) -> Tuple[str, int]:
        if logging:
            print(f"Prompt: {prompt}")
        try:
            answer_rag_exec_info_list = self.run_llm_batched([prompt], logging, vllm)
            answer, rag_exec_info = answer_rag_exec_info_list[0]
            return answer, rag_exec_info
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
                output_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
            )
            return outputs, output_ids.size(0)

    def create_prompt(self, request: Request, topk: int) -> Tuple[str, int]:
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
            if self._prompt_mode in [
                PromptMode.CHATQA,
                PromptMode.RAGNAROK_V2,
                PromptMode.RAGNAROK_V3,
                PromptMode.RAGNAROK_V4,
                PromptMode.RAGNAROK_V4_BIOGEN,
                PromptMode.RAGNAROK_V5_BIOGEN,
                PromptMode.RAGNAROK_V4_NO_CITE,
                PromptMode.RAGNAROK_V5_BIOGEN_NO_CITE,
            ]:
                ragnarok_template = RagnarokTemplates(self._prompt_mode)
                messages = ragnarok_template(query, context, self._name)
            else:
                raise ValueError(
                    f"unsupported prompt mode for GPT models: {self._prompt_mode}, expected one of {PromptMode.CHATQA}, {PromptMode.RAGNAROK_V2}, {PromptMode.RAGNAROK_V3}, {PromptMode.RAGNAROK_V4}, {PromptMode.RAGNAROK_V4_NO_CITE}."
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

    def create_prompt_batched(
        self,
        requests: List[Request],
        topk: int,
        batch_size: int = 32,
    ) -> List[Tuple[str, int]]:
        """
        Creates prompts in batches for a list of results, processing them in parallel using a thread pool executor.

        Parameters:
        - results (List[Result]): List of results for which prompts are to be generated.
        - topk (int): Number of top candidates to include in each prompt.
        - batch_size (int, optional): Number of prompts to generate in each batch. Defaults to 32.

        Returns:
        - List[Tuple[str, int]]: List of generated prompts and their token counts.
        """

        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        all_completed_prompts = []

        with ThreadPoolExecutor() as executor:
            for batch in chunks(requests, batch_size):
                completed_prompts = list(
                    executor.map(
                        lambda request: self.create_prompt(request, topk),
                        batch,
                    )
                )
                all_completed_prompts.extend(completed_prompts)

        return all_completed_prompts

    def get_num_tokens(self, prompt: str) -> int:
        return len(self._tokenizer.encode(prompt))

    def cost_per_1k_token(self, input_token: bool) -> float:
        return 0
