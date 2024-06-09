import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

import torch

from ragnarok.generate.llm import PromptMode
from ragnarok.retrieve_and_generate import retrieve_and_generate
from ragnarok.retrieve_and_rerank.retriever import RetrievalMethod, RetrievalMode
from ragnarok.retrieve_and_rerank.topics_dict import TOPICS


def parse_topk(value):
    try:
        return [int(k) for k in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid comma-separated list of integers: {value}"
        )


def parse_retrieval_methods(value):
    try:
        # Ensure it is of type RetrievalMethod
        return [RetrievalMethod(e) for e in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid comma-separated list of retrieval methods: {value}"
        )


def main(args):
    model_path = args.model_path
    use_azure_openai = args.use_azure_openai
    context_size = args.context_size
    topk = args.topk
    dataset = args.dataset
    num_gpus = args.num_gpus
    retrieval_method = args.retrieval_method
    prompt_mode = args.prompt_mode
    num_few_shot_examples = args.num_few_shot_examples
    shuffle_candidates = args.shuffle_candidates
    print_prompts_responses = args.print_prompts_responses
    num_few_shot_examples = args.num_few_shot_examples
    max_output_tokens = args.max_output_tokens
    device = "cuda" if torch.cuda.is_available() else "cpu"
    retrieval_mode = RetrievalMode.DATASET

    _ = retrieve_and_generate(
        model_path,
        dataset,
        retrieval_mode,
        retrieval_method,
        topk,
        context_size,
        max_output_tokens,
        device,
        num_gpus,
        prompt_mode,
        num_few_shot_examples,
        shuffle_candidates,
        print_prompts_responses,
        use_azure_openai=use_azure_openai,
    )


""" sample run:
python src/ragnarok/scripts/run_ragnarok.py  --model_path=cohere_command_r_plus  --topk=20 --dataset=researchy-questions  --retrieval_method=bm25,rank_zephyr --prompt_mode=chat_qa  --context_size=8192 --max_output_tokens=1500
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model. If `use_azure_ai`, pass your deployment name.",
    )
    parser.add_argument(
        "--use_azure_openai",
        action="store_true",
        help="If True, use Azure OpenAI instead of regular OpenAI. Requires env var to be set: "
        "`AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_API_BASE`",
    )
    parser.add_argument(
        "--context_size", type=int, default=8192, help="context size used for model"
    )
    parser.add_argument(
        "--topk",
        type=parse_topk,
        default=[100, 20],
        help="Comma-separated list of top k values for each retrieval method. Example: 100,20",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help=f"Should be one of 1- dataset name, must be in {TOPICS.keys()},  2- a list of inline documents  3- a list of inline hits 4- filename containing retrieved results",
    )
    parser.add_argument(
        "--num_gpus", type=int, default=1, help="the number of GPUs to use"
    )
    parser.add_argument(
        "--retrieval_method",
        type=parse_retrieval_methods,
        required=True,
        help="Comma-separated list of retrieval methods. Choices: "
        + ", ".join([e.value for e in RetrievalMethod]),
    )
    parser.add_argument(
        "--prompt_mode",
        type=PromptMode,
        required=True,
        choices=list(PromptMode),
    )
    parser.add_argument(
        "--shuffle_candidates",
        action="store_true",
        help="whether to shuffle the candidates before reranking",
    )
    parser.add_argument(
        "--print_prompts_responses",
        action="store_true",
        help="whether to print promps and responses",
    )
    parser.add_argument(
        "--num_few_shot_examples",
        type=int,
        required=False,
        default=0,
        help="number of in context examples to provide",
    )
    parser.add_argument(
        "--max_output_tokens",
        type=int,
        required=False,
        default=1500,
        help="maximum number of tokens in the output",
    )
    args = parser.parse_args()
    main(args)
