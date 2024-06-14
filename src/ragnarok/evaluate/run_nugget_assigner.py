"""
Use argparse and using gpt_nugget_assigner, load both the scored nuggets and the RAG result and then assign nuggets to the RAG result.

python3 src/ragnarok/evaluate/run_nugget_assigner.py --nuggetized_jsonl_file nuggetized_results/scored_nuggetized_results_rag24.researgy-dev_pool20_s2.jsonl \
    --rag_jsonl_file results/bm25/command-r-plus_8192_20_cohere_msmarco-v2.1-doc-segmented.bm25.rag24.researgy-dev_2024-06-14T10:25:20.546665.jsonl  \
    --output_jsonl_file nuggetized_results/assigned_nuggets_s2_command-r-plus_8192_20_cohere_msmarco-v2.1-doc-segmented.bm25.rag24.researgy-dev_2024-06-14T10:25:20.546665.jsonl --window_size 10 --stride 10 --model gpt-4o
"""

import argparse
import ast
import json

from tqdm import tqdm

from ragnarok.evaluate.gpt_nugget_assigner import NuggetAssignMode, SafeOpenaiNuggetAssigner
from ragnarok.generate.api_keys import get_azure_openai_args, get_openai_api_key
from ragnarok.data import read_results_from_file

def load_nugget_results(nuggetized_jsonl_file):
    with open(nuggetized_jsonl_file, "r") as file:
        return {json.loads(line)["qid"].replace("_0", ""): json.loads(line) for line in file}


def save_nuggetized_results(output_jsonl_file, results):
    with open(output_jsonl_file, "w") as file:
        for (result, nugget_labels, answer_text) in results:
            # Lowercase everything and ensure it is either vital or okay, else shout
            nugget_labels = [nugget_label.lower() for nugget_label in nugget_labels]
            for nugget_label in nugget_labels:
                assert nugget_label in [
                    "support",
                    "not_support",
                ], f"Invalid nugget label: {nugget_label}"
            result["nugget_assignment"] = nugget_labels
            result["answer_text"] = answer_text
            result.pop("nugget_trajectory")
            file.write(json.dumps(result) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Nuggetize assign the retrieve using a sliding window approach."
    )
    parser.add_argument(
        "--nuggetized_jsonl_file",
        required=True,
        help="Path to the pooled results JSONL file.",
    )
    parser.add_argument(
        "--rag_jsonl_file",
        required=True,
        help="Path to the RAG results JSONL file.",
    )
    parser.add_argument(
        "--output_jsonl_file",
        required=True,
        help="Path to save the nuggetized results JSONL file.",
    )
    parser.add_argument(
        "--window_size", type=int, default=10, help="Window size for nuggetizing."
    )
    parser.add_argument(
        "--prompt_mode", type=NuggetAssignMode, default=NuggetAssignMode.SUPPORT_GRADE_2
    )
    parser.add_argument(
        "--stride", type=int, default=10, help="Stride for the sliding window."
    )
    parser.add_argument(
        "--model", required=True, help="OpenAI model to use for nuggetizing."
    )
    parser.add_argument("--logging", action="store_true", help="Log things")

    args = parser.parse_args()

    scored_nuggets = load_nugget_results(args.nuggetized_jsonl_file)
    rag_results = read_results_from_file(args.rag_jsonl_file)
    openai_keys = get_openai_api_key()
    nuggetizer = SafeOpenaiNuggetAssigner(
        model=args.model,
        context_size=4096,  # Assuming 4096 is a typical context size for the models
        prompt_mode=args.prompt_mode,
        window_size=args.window_size,
        keys=openai_keys,
        **(get_azure_openai_args()),
    )

    nugget_assignment_results = []
    fail_count = 0
    fail_examples = []
    for result in tqdm(rag_results):
        request = scored_nuggets[result.query.qid.replace("_0", "")]
        start = 0
        nuggets = request["nuggets"]
        query = request["text"]
        nugget_assignment = []
        while start < len(request["nuggets"]):
            end = min(start + args.window_size, len(request["nuggets"]))
            answer_text_only = " ".join([cited_sentence.text for cited_sentence in result.answer])
            prompt = nuggetizer.create_prompt(query, answer_text_only, start, end, nuggets)
            if args.logging:
                print(prompt)
            response, _ = nuggetizer.run_llm(prompt)
            if args.logging:
                print(response)
            try:
                response = response.replace("```python", "").replace("```", "").strip()
                output = ast.literal_eval(response)
                # Assert for length and only vital and okay
                assert len(output) == len(nuggets[start:end])
                for nugget_label in output:
                    nugget_label = nugget_label.lower()
                    if args.prompt_mode == NuggetAssignMode.SUPPORT_GRADE_2:
                        assert nugget_label in [
                            "support",
                            "not_support",
                        ], f"Invalid nugget label: {nugget_label}"
                    else:
                        assert nugget_label in [
                            "support",
                            "partial_support",
                            "not_support",
                        ], f"Invalid nugget label: {nugget_label}"
            except:
                fail_count += 1
                fail_examples.append(
                    (request.query.qid, request.query.text, answer_text_only, start, end, response)
                )
                output = len(nuggets[start:end]) * ["fail"]

            nugget_assignment.extend(output)
            start += args.stride
        print(f"Query: {request.query}\nNugget Assignment: {nugget_assignment}\nAnswer: {answer_text_only}")
        nugget_assignment_results.append((request, nugget_assignment, answer_text_only))

    save_nuggetized_results(args.output_jsonl_file, nugget_assignment_results)
    print(f"Failed to parse {fail_count} examples.")
    for example in fail_examples:
        print(example)


if __name__ == "__main__":
    main()
