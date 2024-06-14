# Walkthrough for Decomposable Pooling and Nuggetization

# Decomposition Queries

This section is still a TODO, relies on Nandan's code.


# Retrieval

Let's first run Anserini BM25 over the corpora to get the top 100 results for each query.

```bash

export ANSERINI_JAR="anserini-0.36.1-fatjar.jar"
export OUTPUT_DIR="runs"
TOPICS=(rag24.researgy-dev)
for t in ${TOPICS[@]}; do
    bin/run.sh io.anserini.search.SearchCollection \
        -index msmarco-v2.1-doc-segmented \
        -topicReader TsvString \
        -topics /store/scratch/rpradeep/nuggetizer/rag/topics.${t}.tsv \
        -output $OUTPUT_DIR/run.mmsmarco-v2.1-doc-segmented.bm25.${t}.txt \
        -threads 16 \
        -bm25 \
        -hits 100 \
        -outputRerankerRequests $OUTPUT_DIR/retrieve_results_msmarco-v2.1-doc-segmented.bm25.${t}_top100.jsonl &
    done;
```

Now, we can scp this over to watgpu and run reranking with RankLLM (RankZephyr Rho - 3 passes).

```bash
scp $OUTPUT_DIR/retrieve_results_msmarco-v2.1-doc-segmented.bm25.rag24.researgy-dev_top100.jsonl rpradeep@watgpu.cs.uwaterloo.ca:/u3/rpradeep/rank_llm/retrieve_results/BM25/

python src/rank_llm/scripts/run_rank_llm.py  --model_path=castorini/rank_zephyr_7b_v1_full --top_k_candidates=100 --dataset=msmarco-v2.1-doc-segmented.bm25.rag24.researgy-dev --batched --retrieval_method=bm25 --prompt_mode=rank_GPT --context_size=4096 --variable_passages --num_passes=3

scp rerank_results/BM25/retrieve_results_msmarco-v2.1-doc-segmented.bm25.rank_zephyr_rho.rag24.researgy-dev_top100.jsonl rpradeep@basilisk.cs.uwaterloo.ca:/store2/scratch/rpradeep/ragnarok/retrieve_results/RANK_ZEPHYR_RHO/
scp rerank_results/BM25/retrieve_results_msmarco-v2.1-doc-segmented.bm25.rank_zephyr_rho.rag24.researgy-dev_top100.jsonl rpradeep@orca.cs.uwaterloo.ca:/home/rpradeep/anserini/runs/
scp /home/rpradeep/anserini/runs/*researgy* rpradeep@basilisk.cs.uwaterloo.ca:/store2/scratch/rpradeep/ragnarok/retrieve_results/RANK_ZEPHYR_RHO/
mv  /store2/scratch/rpradeep/ragnarok/retrieve_results/RANK_ZEPHYR_RHO/retrieve_results_msmarco-v2.1-doc-segmented.bm25.rag24.researgy-dev_top100.jsonl /store2/scratch/rpradeep/ragnarok/retrieve_results/BM25/
```

Given these files, we can begin with pooling the results. Given we decomposed each query (initial query - q_0) into multiple (q_i where i > 0), we can pool the results in different ways.

Setting 0 - all different with different queries i.e., each q_i is treated as a different query and their retrieved results are individually pooled.

```bash
python3 src/ragnarok/evaluate/decomp_pooler.py --run_jsonl_files retrieve_results/BM25/retrieve_results_msmarco-v2.1-doc-segmented.bm25.rag24.researgy-dev_top100.jsonl retrieve_results/RANK_ZEPHYR_RHO/retrieve_results_msmarco-v2.1-doc-segmented.bm25.rank_zephyr_rho.rag24.researgy-dev_top100.jsonl \
    --output_jsonl_file pool_results/pooled_results_rag24.researgy-dev_pool20_s0.jsonl \
    --k 20 --decomp_queries 0
```

Setting 1 - First setting 0 is run to get a pool for each q_i. Then, all the pools for a query are merged into the pool for q_0.

```bash
python3 src/ragnarok/evaluate/decomp_pooler.py --run_jsonl_files retrieve_results/BM25/retrieve_results_msmarco-v2.1-doc-segmented.bm25.rag24.researgy-dev_top100.jsonl retrieve_results/RANK_ZEPHYR_RHO/retrieve_results_msmarco-v2.1-doc-segmented.bm25.rank_zephyr_rho.rag24.researgy-dev_top100.jsonl \
    --output_jsonl_file pool_results/pooled_results_rag24.researgy-dev_pool20_s1.jsonl \
    --k 20 --decomp_queries 1
```

Of course we can also choose to only pool the results of the q_0 query.

```bash
python3 src/ragnarok/evaluate/decomp_pooler.py --run_jsonl_files retrieve_results/BM25/retrieve_results_msmarco-v2.1-doc-segmented.bm25.rag24.researgy-dev_top100.jsonl retrieve_results/RANK_ZEPHYR_RHO/retrieve_results_msmarco-v2.1-doc-segmented.bm25.rank_zephyr_rho.rag24.researgy-dev_top100.jsonl \
    --output_jsonl_file pool_results/pooled_results_rag24.researgy-dev_pool20_s2.jsonl \
    --k 20 --decomp_queries 2
```

For now we are interested in Setting 1 and Setting 2 as they are the most interesting. We can now run the nuggetizer (with memory) on these pooled results.

```bash
python3 src/ragnarok/evaluate/run_nuggetizer.py --pooled_jsonl_file pool_results/pooled_results_rag24.researgy-dev_pool20_s2.jsonl \
 --output_jsonl_file nuggetized_results/nuggetized_results_rag24.researgy-dev_pool20_s2.jsonl --window_size 10 --stride 10 --model gpt-4o

python3 src/ragnarok/evaluate/run_nuggetizer.py --pooled_jsonl_file pool_results/pooled_results_rag24.researgy-dev_pool20_s1.jsonl \
 --output_jsonl_file nuggetized_results/nuggetized_results_rag24.researgy-dev_pool20_s1.jsonl --window_size 10 --stride 10 --model gpt-4o
```

# Analysis

We can now compare the nuggetized results with the original results to see if the nuggetization process has improved with the larger pool

Given we pooled over two retrieval methods - top 20 from BM25 and top 20 from RankZephyr, the following are some statistics from s1 and s2.

```
Setting 1 (all q_i pooled into q_0)
Decomp Merged - Number of runs: 2
Decomp Merged - Total number of query-document pairs: 1304
Decomp Merged - Averaged number of query-document pairs per query: 217.33333333333334

Setting 2 (only q_0 pooled)
Number of runs: 2
Total number of query-document pairs: 183
Averaged number of query-document pairs per query: 30.5
```

This would mean ~ 22 passes in setting 1 with memory and ~ 3 passes in setting 2 with memory for each query when we run nuggetization.

Finally the results of nuggetization itself are compiled in this [spreadsheet](https://docs.google.com/document/d/1ETBBjEprWspLv6mpS3mC7UeKiz-e_VAIKWJFlM3Ouko/edit?usp=sharing)

The trajectories of nuggets across iterations can be visualized (perhaps useful to share with NIST assessors). This can be done running the following to generate git diff style outputs.

```bash
python3 src/ragnarok/evaluate/trace_nugget_trajectory.py nuggetized_results/nuggetized_results_rag24.researgy-dev_pool20_s2.jsonl 

python3 src/ragnarok/evaluate/trace_nugget_trajectory.py nuggetized_results/nuggetized_results_rag24.researgy-dev_pool20_s1.jsonl 
```
The doc file with the results can be found [here](https://docs.google.com/document/d/1kQ3sXOhaNv3xsQGR_gtatB8TXotaPpLakSLQfPkIku8/edit?usp=sharing).

## UMBRELA

We can also run UMBRELA on the pooled results to get relevance judgments

```bash
 python3 src/ragnarok/evaluate/run_umbrela.py --input_file pool_results/pooled_results_rag24.researgy-dev_pool20_s2.jsonl --output_file pool_results/pooled_results_rag24.researgy-dev_pool20_s2_umbrela.jsonl
 ```


 ```bash
    python3 src/ragnarok/evaluate/run_umbrela.py --input_file pool_results/pooled_results_rag24.researgy-dev_pool20_s1.jsonl --output_file pool_results/pooled_results_rag24.researgy-dev_pool20_s1_umbrela.jsonl
```

The output of these can be summarized here in the following markdown table.

|           | Mean Judgment | Median Judgment | # Judged | % (#) Judged 0 | % (#) Judged 1 | % (#) Judged 2 | % (#) Judged 3 |
|-----------|---------------|-----------------|----------|----------------|----------------|----------------|----------------|
| Setting 2 | 0.90          | 1.0             | 183      | 40.98 (75)     | 33.88 (62)     | 19.67 (36)    | 5.46 (10)      |
| Setting 1 | 0.71          | 1.0             | 1304     | 49.00 (639)    | 34.05 (444)    | 14.19 (185)     | 2.76 (36)      |

The same spreadsheet should have examples with labels for each judgment value as can be found [here](https://docs.google.com/spreadsheets/d/1EZH5oxb4DKdT_5FV4PF8mrLdHUSpQlujxOC3pCOtTWY/edit?usp=sharing)

# Nugget Scorer

Let's label each nugget either an okay or vital based on prior TREC QA tracks:

```bash
python3 src/ragnarok/evaluate/run_nugget_scorer.py --nuggetized_jsonl_file nuggetized_results/nuggetized_results_rag24.researgy-dev_pool20_s2.jsonl --output_jsonl_file nuggetized_results/scored_nuggetized_results_rag24.researgy-dev_pool20_s2.jsonl --window_size 10 --stride 10 --model gpt-4o --logging

python3 src/ragnarok/evaluate/run_nugget_scorer.py --nuggetized_jsonl_file nuggetized_results/nuggetized_results_rag24.researgy-dev_pool20_s1.jsonl --output_jsonl_file nuggetized_results/scored_nuggetized_results_rag24.researgy-dev_pool20_s1.jsonl --window_size 10 --stride 10 --model gpt-4o --logging
```

# RAG

Finally, we can run RAG on the retrieved results to generate the answers.

## Let's first run with BM25

```bash
python src/ragnarok/scripts/run_ragnarok.py  --model_path=command-r-plus  --topk=20 \
  --dataset=msmarco-v2.1-doc-segmented.bm25.rag24.researgy-dev  --retrieval_method=bm25 --prompt_mode=cohere  \
  --context_size=8192 --max_output_tokens=512  --print_prompts_responses
  
python src/ragnarok/scripts/run_ragnarok.py  --model_path=gpt-4o --topk=20   --dataset=msmarco-v2.1-doc-segmented.bm25.rag24.researgy-dev  --retrieval_method=bm25   --context_size=8192 --max_output_tokens=512 --prompt_mode chatqa --use_azure_openai
```

## Next BM25, RANK_ZEPHYR_RHO
```bash  
python src/ragnarok/scripts/run_ragnarok.py  --model_path=command-r-plus  --topk=100,20 \
    --dataset=msmarco-v2.1-doc-segmented.bm25.rank_zephyr_rho.rag24.researgy-dev  --retrieval_method=bm25,rank_zephyr_rho --prompt_mode=cohere  \
    --context_size=8192 --max_output_tokens=512  --print_prompts_responses


python src/ragnarok/scripts/run_ragnarok.py  --model_path=gpt-4o  --topk=100,20 \
    --dataset=msmarco-v2.1-doc-segmented.bm25.rank_zephyr_rho.rag24.researgy-dev  --retrieval_method=bm25,rank_zephyr_rho --prompt_mode=chatqa  \
    --context_size=8192 --max_output_tokens=512  --print_prompts_responses --use_azure_openai
```

## L

```bash

# Nugget Assignment



