"""
Loads an arbitrary number of TREC run JSONL files (each containing a list of Requests) and pools their retrieval results (selecting a depth of k for each run) using the naive pooling method.
The naive pooling method simply takes the union of the retrieval results from all runs and sorts the documents by their average score.
The union is written into a new JSONL file.

Usage:
    python3 evaluate/pooler.py --run_jsonl_files <path_to_run_jsonl_files> --output_jsonl_file <path_to_output_jsonl_file> --k <depth_of_pooling>
Example:
    python3 src/ragnarok/evaluate/pooler.py --run_jsonl_files retrieve_results/BM25/retrieve_results_rag24.researchy-dev_tiny_top100.jsonl retrieve_results/RANK_ZEPHYR_RHO/retrieve_results_rag24.researchy-dev_tiny_top100.jsonl --output_jsonl_file pool_results/pooled_results_rag24.researchy-dev_tiny_pool2.jsonl --k 2
"""

import argparse
import os
from collections import defaultdict
import numpy as np
import json
from tqdm import tqdm
from ragnarok.data import Candidate, Request, Query

def load_jsonl_run(run_file, topk):
    run = defaultdict(list)
    with open(run_file, "r") as file:
        for line in file:
            request = json.loads(line)
            qid = request['query']['qid']
            query_text = request['query']['text']
            top_candidates = sorted(request['candidates'], key=lambda x: x['score'], reverse=True)[:topk]
            for candidate in top_candidates:
                docid = candidate['docid']
                score = candidate['score']
                doc = candidate['doc']
                run[qid].append((docid, score, query_text, doc))
    return run

def naive_pooling_jsonl(run_files, output_file, k):
    runs = [load_jsonl_run(run_file, k) for run_file in run_files]
    all_qids = set().union(*[run.keys() for run in runs])
    pooled_requests = []
    total_qdoc_pairs = 0
    for qid in tqdm(all_qids, desc="Pooling"):
        doc_scores = defaultdict(list)
        doc_store = {}
        query_text = ""
        for run in runs:
            for docid, score, text, doc in run[qid]:
                doc_scores[docid].append(score)
                doc_store[docid] = doc  # Store the doc information
                query_text = text  # Assume all runs have the same query text for the same qid
        sorted_docs = sorted([(docid, np.mean(scores)) for docid, scores in doc_scores.items()], key=lambda x: x[1], reverse=True)
        total_qdoc_pairs += len(sorted_docs)
        candidates = [Candidate(docid=docid, score=score, doc=doc_store[docid]) for docid, score in sorted_docs]
        request = Request(query=Query(qid=qid, text=query_text), candidates=candidates)
        pooled_requests.append(request)
    print(f"Number of runs: {len(runs)}")
    print(f"Total number of query-document pairs: {total_qdoc_pairs}")
    print(f"Averaged number of query-document pairs per query: {total_qdoc_pairs / len(all_qids)}")
    with open(output_file, "w") as out_file:
        for request in pooled_requests:
            out_file.write(json.dumps(request, default=lambda o: o.__dict__) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_jsonl_files", nargs="+", required=True, help="List of paths to TREC run JSONL files")
    parser.add_argument("--output_jsonl_file", required=True, help="Path to the output JSONL file")
    parser.add_argument("--k", type=int, required=True, help="Depth of pooling")
    args = parser.parse_args()
    naive_pooling_jsonl(args.run_jsonl_files, args.output_jsonl_file, args.k)
