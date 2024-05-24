import os
from pathlib import Path
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from ragnarok.data import read_requests_from_file, DataWriter
from ragnarok.generate.os_llm import OSLLM

file_name = (
    "retrieve_results/BM25/retrieve_results_dl23_top20.json"
)
requests = read_requests_from_file(file_name)

generator = OSLLM("meta-llama/Meta-Llama-3-8B-Instruct")
rag_results = OSLLM.answer_batch(requests)
print(rag_results)

# write rerank results
writer = DataWriter(rag_results)
Path(f"demo_outputs/").mkdir(parents=True, exist_ok=True)
writer.write_in_json_format(f"demo_outputs/rag_results.json")
writer.write_in_trec_eval_format(f"demo_outputs/rerank_results.txt")
writer.write_ranking_exec_summary(f"demo_outputs/ranking_execution_summary.json")
