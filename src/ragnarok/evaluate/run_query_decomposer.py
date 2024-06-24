"""
Use argparse and using gpt_nuggetizer, load first the pooled results and then nuggetize them using a sliding window approach (stride = window_size). Extract the nugget list and keep updating it.

python3 src/ragnarok/evaluate/run_query_decomposer.py --query_file ../topics.rag24.researchy-dev-q.txt \
    --decomp_query_file ../topics.rag24.researchy-dev-qd.txt --num 10 --model gpt-4o
"""
import argparse
import json
from tqdm import tqdm
from ragnarok.data import read_requests_from_file
from ragnarok.evaluate.gpt_query_decomposer import QueryDecomposeMode, SafeOpenaiQueryDecomposer
from ragnarok.generate.api_keys import get_azure_openai_args, get_openai_api_key
import ast

def load_topics(topics_tsv_file):
    with open(topics_tsv_file, "r") as file:
        return [line.strip().split("\t") for line in file]


def write_decomp_queries(decomp_query_file, decomp_queries):
    with open(decomp_query_file, "w") as file:
        for qid, query in decomp_queries.items():
            file.write(f"{qid}\t{query}" + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Decompose queries"
    )
    parser.add_argument(
        "--query_file",
        required=True,
        help="Path to the query file.",
    )
    parser.add_argument(
        "--decomp_query_file",
        required=True,
        help="Path to save the decomposed query file.",
    )
    parser.add_argument(
        "--num", type=int, default=10, help="Number of sub-questions."
    )
    parser.add_argument(
        "--model", required=True, help="OpenAI model to use for nuggetizing."
    )
    parser.add_argument("--logging", action="store_true", help="Log things")
    args = parser.parse_args()

    openai_keys = get_openai_api_key()
    decomposer = SafeOpenaiQueryDecomposer(
        model=args.model,
        context_size=4096,  # Assuming 4096 is a typical context size for the models
        prompt_mode=QueryDecomposeMode.RESEARCHY_QUESTIONS,
        window_size=10,
        keys=openai_keys,
        **(get_azure_openai_args()),
    )

    decomposition_results = []
    fail_count = 0
    fail_examples = []

    topics = load_topics(args.query_file)

    for qid, query in tqdm(topics):
        prompt = decomposer.create_prompt(query, num=args.num)
        response, _ = decomposer.run_llm(prompt)
        try:
            response = response.replace("```python", "").replace("```", "").strip()
            output = ast.literal_eval(response)
            output = [(f"{qid}_0", query)] + [(f"{qid}_{i+1}", sub_query) for i, sub_query in enumerate(output)]
            print(output)
        except:
            fail_count += 1
            fail_examples.append(
                (qid, query, response)
            )
            output = []
        decomposition_results.extend(output)
            

    decomp_queries = {qid: query for qid, query in decomposition_results}
    write_decomp_queries(args.decomp_query_file, decomp_queries)

    if args.logging:
        print(f"Failed examples: {fail_examples}")
        print(f"Total failures: {fail_count}")


if __name__ == "__main__":
    main()