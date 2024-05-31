"""
Use argparse and using gpt_nuggetizer, load first the pooled results and then nuggetize them using a sliding window approach (stride = window_size). Extract the nugget list and keep updating it.

python3 src/ragnarok/evaluate/run_nuggetizer.py --pooled_jsonl_file pool_results/pooled_results_rag24.researchy-dev_tiny_pool20.jsonl --output_jsonl_file nuggetized_results/nuggetized_results_rag24.researchy-dev_tiny_pool20.jsonl --window_size 10 --stride 10 --model gpt-4o
"""

import argparse
import json
from ragnarok.evaluate.gpt_nuggetizer import NuggetMode, SafeOpenaiNuggetizer
from ragnarok.data import Request, read_requests_from_file
from ragnarok.generate.api_keys import get_azure_openai_args, get_openai_api_key
import ast
from tqdm import tqdm

def load_pooled_results(pooled_jsonl_file):
    with open(pooled_jsonl_file, 'r') as file:
        return [json.loads(line) for line in file]

def save_nuggetized_results(output_jsonl_file, results):
    with open(output_jsonl_file, 'w') as file:
        for result, nuggets in results:
            result_dict = result.query.__dict__
            result_dict['nuggets'] = nuggets
            print(result_dict)
            file.write(json.dumps(result_dict) + '\n')

def main():
    parser = argparse.ArgumentParser(description='Nuggetize pooled results using a sliding window approach.')
    parser.add_argument('--pooled_jsonl_file', required=True, help='Path to the pooled results JSONL file.')
    parser.add_argument('--output_jsonl_file', required=True, help='Path to save the nuggetized results JSONL file.')
    parser.add_argument('--window_size', type=int, default=10, help='Window size for nuggetizing.')
    parser.add_argument('--stride', type=int, default=10, help='Stride for the sliding window.')
    parser.add_argument('--model', required=True, help='OpenAI model to use for nuggetizing.')

    args = parser.parse_args()

    pooled_results = read_requests_from_file(args.pooled_jsonl_file)
    openai_keys = get_openai_api_key()
    nuggetizer = SafeOpenaiNuggetizer(
        model=args.model,
        context_size=4096,  # Assuming 4096 is a typical context size for the models
        prompt_mode=NuggetMode.ATOMIC,
        window_size=args.window_size,
        keys=openai_keys,
        **(get_azure_openai_args()))

    nuggetized_results = []
    for request in tqdm(pooled_results):
        start = 0
        nuggets = []
        print(request.query.text)
        while start < len(request.candidates):
            end = min(start + args.window_size, len(request.candidates))
            prompt = nuggetizer.create_prompt(request, start, end, nuggets)
            response, _ = nuggetizer.run_llm(prompt)
            response = response.replace("```python", "").replace("```", "").strip()
            output = ast.literal_eval(response)
            nuggets = output
            print(nuggets)
            start += args.stride
        nuggetized_results.append((request, nuggets))

    save_nuggetized_results(args.output_jsonl_file, nuggetized_results)

if __name__ == '__main__':
    main()