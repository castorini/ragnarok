# Pooling and nuggetization

## Pooling

We can pool sets by running:

```bash
python3 src/ragnarok/evaluate/pooler.py --run_jsonl_files retrieve_results/BM25/retrieve_results_rag24.researchy-dev_tiny_top100.jsonl retrieve_results/RANK_ZEPHYR_RHO/retrieve_results_rag24.researchy-dev_tiny_top100.jsonl --output_jsonl_file pool_results/pooled_results_rag24.researchy-dev_tiny_pool20.jsonl --k 20
# Small
python3 src/ragnarok/evaluate/pooler.py --run_jsonl_files retrieve_results/BM25/retrieve_results_rag24.researchy-dev_small_top100.jsonl retrieve_results/RANK_ZEPHYR_RHO/retrieve_results_rag24.researchy-dev_small_top100.jsonl --output_jsonl_file pool_results/pooled_results_rag24.researchy-dev_small_pool20.jsonl --k 20
# Medium
python3 src/ragnarok/evaluate/pooler.py --run_jsonl_files retrieve_results/BM25/retrieve_results_rag24.researchy-dev_medium_top100.jsonl retrieve_results/RANK_ZEPHYR_RHO/retrieve_results_rag24.researchy-dev_medium_top100.jsonl --output_jsonl_file pool_results/pooled_results_rag24.researchy-dev_medium_pool20.jsonl --k 20

# Regular
python3 src/ragnarok/evaluate/pooler.py --run_jsonl_files retrieve_results/BM25/retrieve_results_rag24.researchy-dev_top100.jsonl retrieve_results/RANK_ZEPHYR_RHO/retrieve_results_rag24.researchy-dev_top100.jsonl --output_jsonl_file pool_results/pooled_results_rag24.researchy-dev_pool20.jsonl --k 20


# rag24.raggy-dev
python3 src/ragnarok/evaluate/pooler.py --run_jsonl_files retrieve_results/BM25/retrieve_results_rag24.raggy-dev_top100.jsonl retrieve_results/RANK_ZEPHYR_RHO/retrieve_results_rag24.raggy-dev_top100.jsonl --output_jsonl_file pool_results/pooled_results_rag24.raggy-dev_pool20.jsonl --k 20
```
