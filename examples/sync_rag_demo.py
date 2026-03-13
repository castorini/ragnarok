#!/usr/bin/env python3
"""Synchronous compatibility inline-hit RAG demo for Ragnarok."""

import argparse

from rag_demo import (  # type: ignore[import-not-found]
    build_agent,
    create_sample_request,
    parse_prompt_mode,
    print_result,
)

from ragnarok.generate.generator import RAG
from ragnarok.generate.llm import PromptMode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the synchronous inline-hit RAG demo for Ragnarok."
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
        f"Running Ragnarok sync RAG demo with model={args.model} "
        f"prompt_mode={args.prompt_mode} topk={args.topk}"
    )
    result = rag.answer(
        request,
        topk=min(args.topk, len(request.candidates)),
        logging=args.print_prompt or args.print_raw,
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
    main()
