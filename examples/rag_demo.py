#!/usr/bin/env python3
"""Async-first inline-hit RAG demo for Ragnarok."""

import argparse
import asyncio
from textwrap import fill

from ragnarok.data import Candidate, Query, Request, Result
from ragnarok.generate.api_keys import get_openai_compatible_args
from ragnarok.generate.generator import RAG
from ragnarok.generate.llm import LLM, PromptMode


def create_sample_request() -> Request:
    """Create a small in-memory request for manual RAG smoke tests."""
    return Request(
        query=Query(text="how long is life cycle of flea", qid="264014"),
        candidates=[
            Candidate(
                docid="4834547",
                score=14.971799850463867,
                doc={
                    "segment": (
                        "The life cycle of a flea can last anywhere from 20 days "
                        "to an entire year. It depends on how long the flea remains "
                        "in the dormant stage (eggs, larvae, pupa)."
                    )
                },
            ),
            Candidate(
                docid="6641238",
                score=15.090800285339355,
                doc={
                    "segment": (
                        "The flea egg stage is the beginning of the flea cycle. "
                        "Depending on temperature and humidity, eggs can take from "
                        "two to six days to hatch."
                    )
                },
            ),
            Candidate(
                docid="96852",
                score=14.215100288391113,
                doc={
                    "segment": (
                        "Flea larvae spin cocoons around themselves before becoming "
                        "adult fleas. The larvae can remain in the cocoon anywhere "
                        "from one week to one year."
                    )
                },
            ),
            Candidate(
                docid="5611210",
                score=15.780599594116211,
                doc={
                    "segment": (
                        "A flea can live up to a year, but its general lifespan "
                        "depends on living conditions such as temperature and host "
                        "availability."
                    )
                },
            ),
            Candidate(
                docid="5635521",
                score=13.533599853515625,
                doc={
                    "segment": (
                        "If conditions are favorable, flea larvae spin cocoons in "
                        "about 5 to 20 days after hatching. The pupae stage follows "
                        "and accounts for about 10 percent of the flea population "
                        "in a home."
                    )
                },
            ),
        ],
    )


def parse_prompt_mode(value: str) -> PromptMode:
    try:
        return PromptMode(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Unsupported prompt mode: {value}") from exc


def detect_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"

    return "cuda" if torch.cuda.is_available() else "cpu"


def build_agent(args: argparse.Namespace) -> LLM:
    prompt_mode = args.prompt_mode
    if "command-r" in args.model:
        from ragnarok.generate.cohere import Cohere

        return Cohere(
            model=args.model,
            context_size=args.context_size,
            prompt_mode=PromptMode.COHERE,
            max_output_tokens=args.max_output_tokens,
        )

    if any(name in args.model.lower() for name in ("llama", "mistral", "qwen")):
        from ragnarok.generate.os_llm import OSLLM

        return OSLLM(
            model=args.model,
            context_size=args.context_size,
            prompt_mode=prompt_mode,
            max_output_tokens=args.max_output_tokens,
            store_reasoning=args.include_reasoning,
            device=detect_device(),
            num_gpus=args.num_gpus,
        )

    from ragnarok.generate.gpt import SafeOpenai

    openai_compatible_args = get_openai_compatible_args(
        args.model, args.use_azure_openai
    )
    if not openai_compatible_args.get("keys"):
        raise ValueError(
            "No OpenAI-compatible API key found. Set OPENAI_API_KEY or "
            "OPENROUTER_API_KEY before running the demo."
        )
    return SafeOpenai(
        model=args.model,
        context_size=args.context_size,
        prompt_mode=prompt_mode,
        max_output_tokens=args.max_output_tokens,
        store_reasoning=args.include_reasoning,
        reasoning_effort=args.reasoning_effort,
        **openai_compatible_args,
    )


def print_result(
    request: Request,
    result: Result,
    *,
    show_prompt: bool,
    show_raw: bool,
    show_reasoning: bool,
    width: int,
) -> None:
    print("Query")
    print(f"  qid: {request.query.qid}")
    print(f"  text: {request.query.text}")
    print()
    print(f"Top-{len(result.references)} references kept after citation cleanup:")
    for rank, docid in enumerate(result.references, start=1):
        print(f"  {rank}. {docid}")

    print()
    print("Answer")
    for index, sentence in enumerate(result.answer, start=1):
        print(f"  {index}. {fill(sentence.text, width=width)}")
        print(f"     citations: {sentence.citations}")

    if show_prompt:
        print()
        print("Prompt")
        print(result.rag_exec_summary.prompt)

    if show_raw:
        print()
        print("Raw Response")
        print(result.rag_exec_summary.response)

    if show_reasoning and result.rag_exec_summary.reasoning:
        print()
        print("Reasoning")
        print(fill(result.rag_exec_summary.reasoning, width=width))


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the default async inline-hit RAG demo for Ragnarok."
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Generator model to use for the demo.",
    )
    parser.add_argument(
        "--prompt_mode",
        type=parse_prompt_mode,
        default=PromptMode.CHATQA,
        choices=list(PromptMode),
        help="Prompt template family for GPT and open-weight models.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="How many inline hits to include in the prompt.",
    )
    parser.add_argument(
        "--context_size",
        type=int,
        default=8192,
        help="Context size used for prompt construction.",
    )
    parser.add_argument(
        "--max_output_tokens",
        type=int,
        default=512,
        help="Maximum number of generated output tokens.",
    )
    parser.add_argument(
        "--max_concurrency",
        type=int,
        default=8,
        help="Maximum number of concurrent requests for async-capable backends.",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="GPU count for local open-weight models.",
    )
    parser.add_argument(
        "--use_azure_openai",
        action="store_true",
        help="Use Azure OpenAI instead of public OpenAI for GPT models.",
    )
    parser.add_argument(
        "--include_reasoning",
        action="store_true",
        help="Capture model reasoning when the backend supports it.",
    )
    parser.add_argument(
        "--reasoning_effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        help="OpenAI-compatible reasoning effort to request when supported.",
    )
    parser.add_argument(
        "--print_prompt",
        action="store_true",
        help="Print the generated prompt after the answer.",
    )
    parser.add_argument(
        "--print_raw",
        action="store_true",
        help="Print the raw backend response payload after the answer.",
    )
    parser.add_argument(
        "--print_reasoning",
        action="store_true",
        help="Print captured reasoning content when available.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=100,
        help="Wrap width for displayed answer text.",
    )
    args = parser.parse_args()

    request = create_sample_request()
    agent = build_agent(args)
    rag = RAG(agent=agent, run_id="demo-run")

    print(
        f"Running Ragnarok async RAG demo with model={args.model} "
        f"prompt_mode={args.prompt_mode} topk={args.topk} "
        f"max_concurrency={args.max_concurrency}"
    )
    result = await rag.async_answer(
        request,
        topk=min(args.topk, len(request.candidates)),
        logging=args.print_prompt or args.print_raw,
        max_concurrency=args.max_concurrency,
    )
    print()
    print_result(
        request,
        result,
        show_prompt=args.print_prompt,
        show_raw=args.print_raw,
        show_reasoning=args.print_reasoning,
        width=args.width,
    )


if __name__ == "__main__":
    asyncio.run(main())
