"""
Use argparse and using gpt_nugget_scorer, load first the pooled results and then nuggetize them using a sliding window approach (stride = window_size). Extract the nugget list and keep updating it.

python3 src/ragnarok/evaluate/run_nugget_scorer.py --nuggetized_jsonl_file nuggetized_results/nuggetized_results_rag24.researgy-dev_pool20_s2.jsonl --output_jsonl_file nuggetized_results/scored_nuggetized_results_rag24.researgy-dev_pool20_s2.jsonl --window_size 10 --stride 10 --model gpt-4o
"""

import argparse
import json
from ragnarok.evaluate.gpt_nugget_scorer import NuggetScoreMode, SafeOpenaiNuggetScorer
from ragnarok.data import read_requests_from_file
from ragnarok.generate.api_keys import get_azure_openai_args, get_openai_api_key
import ast
from tqdm import tqdm
import copy

def load_nugget_results(nuggetized_jsonl_file):
    with open(nuggetized_jsonl_file, 'r') as file:
        return [json.loads(line) for line in file]

def save_nuggetized_results(output_jsonl_file, results):
    with open(output_jsonl_file, 'w') as file:
        for (result, nugget_labels) in results:
            # Lowercase everything and ensure it is either vital or okay, else shout
            nugget_labels = [nugget_label.lower() for nugget_label in nugget_labels]
            for nugget_label in nugget_labels:
                assert nugget_label in ["vital", "okay"], f"Invalid nugget label: {nugget_label}"
            result['nugget_labels'] = nugget_labels
            file.write(json.dumps(result) + '\n')

def main():
    parser = argparse.ArgumentParser(description='Nuggetize pooled results using a sliding window approach.')
    parser.add_argument('--nuggetized_jsonl_file', required=True, help='Path to the pooled results JSONL file.')
    parser.add_argument('--output_jsonl_file', required=True, help='Path to save the nuggetized results JSONL file.')
    parser.add_argument('--window_size', type=int, default=10, help='Window size for nuggetizing.')
    parser.add_argument('--stride', type=int, default=10, help='Stride for the sliding window.')
    parser.add_argument('--model', required=True, help='OpenAI model to use for nuggetizing.')
    parser.add_argument('--logging', action='store_true', help='Log things')

    args = parser.parse_args()

    pooled_results = load_nugget_results(args.nuggetized_jsonl_file)
    openai_keys = get_openai_api_key()
    nuggetizer = SafeOpenaiNuggetScorer(
        model=args.model,
        context_size=4096,  # Assuming 4096 is a typical context size for the models
        prompt_mode=NuggetScoreMode.VITAL_OKAY,
        window_size=args.window_size,
        keys=openai_keys,
        **(get_azure_openai_args()))

    nugget_scored_results = []
    fail_count = 0
    fail_examples = []
    for request in tqdm(pooled_results):
        start = 0
        nuggets = request["nuggets"]
        query = request["text"]
        nugget_scores = []
        while start < len(request["nuggets"]):
            end = min(start + args.window_size, len(request["nuggets"]))
            prompt = nuggetizer.create_prompt(query, start, end, nuggets)
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
                    assert nugget_label in ["vital", "okay"], f"Invalid nugget label: {nugget_label}"
            except:
                fail_count += 1
                fail_examples.append((request.query.qid, request.query.text, start, end, response))
                output = len(nuggets[start:end]) * ["fail"]
                
            nugget_scores.extend(output)
            start += args.stride
        nugget_scored_results.append((request, nugget_scores))

    save_nuggetized_results(args.output_jsonl_file, nugget_scored_results)
    print(f'Failed to parse {fail_count} examples.')
    for example in fail_examples:
        print(example)
if __name__ == '__main__':
    main()