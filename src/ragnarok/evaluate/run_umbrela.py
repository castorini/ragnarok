"""
Read a JSONL file from the pooled results and generated another with UMBRELA scores

Use argparse and read the input file and output file from the command line (along with --fewshot store_true)

Also print "judgment" level stats from output JSON file.

Usage:
    python3 src/ragnarok/evaluate/run_umbrela.py --input_file <input_file> --output_file <output_file> --fewshot

Example:
    python3 src/ragnarok/evaluate/run_umbrela.py --input_file pool_results/pooled_results_rag24.researgy-dev_pool20_s2.jsonl --output_file pool_results/pooled_results_rag24.researgy-dev_pool20_s2_umbrela.jsonl
"""

import argparse
import json

import numpy as np
from tqdm import tqdm
from umbrela.gpt_judge import GPTJudge


def main():
    parser = argparse.ArgumentParser(description="Evaluate with UMBRELA")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input JSONL file with pooled results",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output JSONL file with UMBRELA scores",
    )
    parser.add_argument("--fewshot", action="store_true", help="Use few-shot prompting")

    args = parser.parse_args()

    # Initialize GPTJudge
    if args.fewshot:
        judge_gpt = GPTJudge(qrel="dl19-passage", prompt_type="bing", few_shot_count=4)
    else:
        judge_gpt = GPTJudge(qrel="dl19-passage", prompt_type="bing")

    query_level_results = []
    all_judgments = []

    # Read input JSONL file
    with open(args.input_file, "r") as infile:
        for line in tqdm(infile, desc="Processing"):
            request_dict = json.loads(line)
            judgments = judge_gpt.judge(request_dict=request_dict)
            judgments = [j.pop("judgment", 0) for j in judgments]
            all_judgments.extend(judgments)
            # update the candidates with the judgment mapping
            for i, judgment in enumerate(judgments):
                request_dict["candidates"][i]["judgment"] = judgment
            query_level_results.append(request_dict)

    # Write query-level results to output JSONL file
    with open(args.output_file, "w") as outfile:
        for result in query_level_results:
            json.dump(result, outfile)
            outfile.write("\n")
    print(f"Results written to {args.output_file}")

    # Print judgment level stats
    print("Judgment level stats:")
    unique_judgments = set(all_judgments)
    for uj in unique_judgments:
        count = all_judgments.count(uj)
        print(f"Judgment score {uj}: {count} examples")

    print(f"Mean judgment score: {np.mean(all_judgments)}")
    print(f"Median judgment score: {np.median(all_judgments)}")


if __name__ == "__main__":
    main()
