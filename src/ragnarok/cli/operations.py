from __future__ import annotations

import contextlib
import importlib.util
import io
import logging
import sys
from typing import Any

from tqdm import tqdm

from .io import read_json, read_jsonl


def _disable_progress(args: object) -> bool:
    quiet = getattr(args, "quiet", False)
    output_format = getattr(args, "output", "text")
    return quiet or output_format in ("json", "jsonl") or not sys.stderr.isatty()


def parse_topk(value: str) -> list[int]:
    try:
        return [int(k) for k in value.split(",")]
    except ValueError as exc:
        raise ValueError(f"Invalid comma-separated list of integers: {value}") from exc


def parse_retrieval_methods(value: str) -> list[Any]:
    from ragnarok.retrieve_and_rerank.retriever import RetrievalMethod

    try:
        return [RetrievalMethod(method) for method in value.split(",")]
    except ValueError as exc:
        raise ValueError(
            f"Invalid comma-separated list of retrieval methods: {value}"
        ) from exc


def detect_device() -> str:
    if importlib.util.find_spec("torch") is None:
        return "cpu"
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def create_generation_agent(args: Any) -> Any:
    from ragnarok.generate.llm import PromptMode

    prompt_mode = (
        args.prompt_mode
        if isinstance(args.prompt_mode, PromptMode)
        else PromptMode(args.prompt_mode)
    )
    model_name = args.model_path
    lowered_model_name = model_name.lower()
    if "command-r" in model_name:
        from ragnarok.generate.cohere import Cohere

        return Cohere(
            model=model_name,
            context_size=args.context_size,
            prompt_mode=prompt_mode,
            max_output_tokens=args.max_output_tokens,
            num_few_shot_examples=args.num_few_shot_examples,
        )
    if any(name in lowered_model_name for name in ("llama", "mistral", "qwen")):
        from ragnarok.generate.os_llm import OSLLM

        return OSLLM(
            model=model_name,
            context_size=args.context_size,
            prompt_mode=prompt_mode,
            max_output_tokens=args.max_output_tokens,
            num_few_shot_examples=args.num_few_shot_examples,
            store_reasoning=args.include_reasoning,
            device=detect_device(),
            num_gpus=args.num_gpus,
        )

    # Default to the OpenAI-compatible client for unknown model identifiers.
    # This keeps the backend selection flexible for providers such as OpenRouter.
    from ragnarok.generate.api_keys import get_openai_compatible_args
    from ragnarok.generate.gpt import SafeOpenai

    return SafeOpenai(
        model=model_name,
        context_size=args.context_size,
        prompt_mode=prompt_mode,
        max_output_tokens=args.max_output_tokens,
        num_few_shot_examples=args.num_few_shot_examples,
        store_reasoning=args.include_reasoning,
        reasoning_effort=args.reasoning_effort,
        **get_openai_compatible_args(
            model_name,
            args.use_azure_openai,
            getattr(args, "use_openrouter", False),
        ),
    )


def load_request_records(path: str) -> list[dict[str, Any]]:
    if path.endswith(".json"):
        payload = read_json(path)
        return payload if isinstance(payload, list) else [payload]
    return read_jsonl(path)


def run_request_generation(
    requests: list[Any], args: Any, logger: logging.Logger
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from ragnarok.data import result_to_dict, write_results_jsonl
    from ragnarok.generate.generator import RAG

    agent = create_generation_agent(args)
    rag = RAG(agent=agent, run_id=args.run_id)
    disable = _disable_progress(args)
    logger.info("Generating %d request(s)", len(requests))
    results = rag.answer_batch(
        requests,
        topk=args.topk[-1],
        shuffle_candidates=args.shuffle_candidates,
        logging=args.print_prompts_responses,
        vllm=args.vllm_batched,
    )
    serialized = [
        result_to_dict(
            result,
            args.run_id,
            include_trace=getattr(args, "include_trace", False),
            redact_prompts=getattr(args, "redact_prompts", False),
        )
        for result in tqdm(
            results, desc="Serializing", file=sys.stderr, disable=disable
        )
    ]
    if args.output_file is not None:
        write_results_jsonl(results, args.output_file, args.run_id)
    return serialized, {"generated_records": len(serialized)}


async def async_run_request_generation(
    requests: list[Any], args: Any, logger: logging.Logger
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from ragnarok.data import result_to_dict, write_results_jsonl
    from ragnarok.generate.generator import RAG

    agent = create_generation_agent(args)
    rag = RAG(agent=agent, run_id=args.run_id)
    disable = _disable_progress(args)
    logger.info(
        "Generating %d request(s) with async execution (max_concurrency=%d)",
        len(requests),
        getattr(args, "max_concurrency", 8),
    )
    results = await rag.async_answer_batch(
        requests,
        topk=args.topk[-1],
        shuffle_candidates=args.shuffle_candidates,
        logging=args.print_prompts_responses,
        vllm=args.vllm_batched,
        max_concurrency=getattr(args, "max_concurrency", 8),
    )
    serialized = [
        result_to_dict(
            result,
            args.run_id,
            include_trace=getattr(args, "include_trace", False),
            redact_prompts=getattr(args, "redact_prompts", False),
        )
        for result in tqdm(
            results, desc="Serializing", file=sys.stderr, disable=disable
        )
    ]
    if args.output_file is not None:
        write_results_jsonl(results, args.output_file, args.run_id)
    return serialized, {"generated_records": len(serialized)}


def run_dataset_generation(
    args: Any, logger: logging.Logger
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from ragnarok.data import result_to_dict
    from ragnarok.retrieve_and_generate import retrieve_and_generate

    logger.info("Running dataset-backed generation for %s", args.dataset)
    output_capture = io.StringIO()
    with contextlib.redirect_stdout(output_capture):
        result = retrieve_and_generate(
            args.model_path,
            args.dataset,
            retrieval_method=args.retrieval_method,
            k=args.topk,
            context_size=args.context_size,
            max_output_tokens=args.max_output_tokens,
            device=detect_device(),
            num_gpus=args.num_gpus,
            prompt_mode=args.prompt_mode,
            num_few_shot_examples=args.num_few_shot_examples,
            shuffle_candidates=args.shuffle_candidates,
            print_prompts_responses=args.print_prompts_responses,
            vllm_batched=args.vllm_batched,
            include_reasoning=args.include_reasoning,
            reasoning_effort=args.reasoning_effort,
            use_azure_openai=args.use_azure_openai,
            run_id=args.run_id,
        )
    artifact_data = (
        [
            result_to_dict(
                result,
                args.run_id,
                include_trace=getattr(args, "include_trace", False),
                redact_prompts=getattr(args, "redact_prompts", False),
            )
        ]
        if hasattr(result, "query")
        else []
    )
    return artifact_data, {"stdout": output_capture.getvalue()}


def convert_generate_records_to_requests(records: list[dict[str, Any]]) -> list[Any]:
    from dacite import from_dict

    from ragnarok.data import Request

    requests: list[Request] = []
    for record in records:
        requests.append(from_dict(data_class=Request, data=record))
    return requests


def run_convert_trec25(args: Any) -> tuple[dict[str, Any], str]:
    from ragnarok.scripts.convert_to_trec25_format import convert_jsonl_file

    output_capture = io.StringIO()
    with contextlib.redirect_stdout(output_capture):
        summary = convert_jsonl_file(
            input_file=args.input_file,
            output_file=args.output_file,
            prompt_file=args.prompt_file,
            verbose=args.verbose,
        )
    return summary, output_capture.getvalue()


def run_validate_rag24(args: Any) -> tuple[dict[str, Any], str]:
    from ragnarok.scripts.check_trec_rag24_gen import run_check_rag24_output

    output_capture = io.StringIO()
    with contextlib.redirect_stdout(output_capture):
        summary = run_check_rag24_output(args.topicfile, args.runfile)
    return summary, output_capture.getvalue()


def run_validate_rag25(args: Any) -> tuple[dict[str, Any], str]:
    from ragnarok.scripts.validate_trec_rag25_gen import validate_rag25_file

    output_capture = io.StringIO()
    with contextlib.redirect_stdout(output_capture):
        summary = validate_rag25_file(
            input_path=args.input,
            topics_path=args.topics,
            format_type=args.format,
            fix_length=args.fix_length,
            fix_citations=args.fix_citations,
            verbose=args.verbose,
        )
    return summary, output_capture.getvalue()
