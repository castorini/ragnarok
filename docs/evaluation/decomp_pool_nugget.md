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

scp $OUTPUT_DIR/retrieve_results_msmarco-v2.1-doc-segmented.bm25.rag24.researgy-dev_top100.jsonl rpradeep@watgpu.cs.uwaterloo.ca:/u3/rpradeep/rank_llm/retrieve_results/BM25/

python src/rank_llm/scripts/run_rank_llm.py  --model_path=castorini/rank_zephyr_7b_v1_full --top_k_candidates=100 --dataset=msmarco-v2.1-doc-segmented.bm25.rag24.researgy-dev --batched --retrieval_method=bm25 --prompt_mode=rank_GPT --context_size=4096 --variable_passages --num_passes=3

scp rerank_results/BM25/retrieve_results_msmarco-v2.1-doc-segmented.bm25.rank_zephyr_rho.rag24.researgy-dev_top100.jsonl rpradeep@basilisk.cs.uwaterloo.ca:/store2/scratch/rpradeep/ragnarok/retrieve_results/RANK_ZEPHYR_RHO/
scp rerank_results/BM25/retrieve_results_msmarco-v2.1-doc-segmented.bm25.rank_zephyr_rho.rag24.researgy-dev_top100.jsonl rpradeep@orca.cs.uwaterloo.ca:/home/rpradeep/anserini/runs/
scp /home/rpradeep/anserini/runs/*researgy* rpradeep@basilisk.cs.uwaterloo.ca:/store2/scratch/rpradeep/ragnarok/retrieve_results/RANK_ZEPHYR_RHO/
mv  /store2/scratch/rpradeep/ragnarok/retrieve_results/RANK_ZEPHYR_RHO/retrieve_results_msmarco-v2.1-doc-segmented.bm25.rag24.researgy-dev_top100.jsonl /store2/scratch/rpradeep/ragnarok/retrieve_results/BM25/

# Setting 0 - all different with different queries
python3 src/ragnarok/evaluate/decomp_pooler.py --run_jsonl_files retrieve_results/BM25/retrieve_results_msmarco-v2.1-doc-segmented.bm25.rag24.researgy-dev_top100.jsonl retrieve_results/RANK_ZEPHYR_RHO/retrieve_results_msmarco-v2.1-doc-segmented.bm25.rank_zephyr_rho.rag24.researgy-dev_top100.jsonl \
    --output_jsonl_file pool_results/pooled_results_rag24.researgy-dev_pool20_s0.jsonl \
    --k 20 --decomp_queries 0

# Setting 1 - all merged to query 0
python3 src/ragnarok/evaluate/decomp_pooler.py --run_jsonl_files retrieve_results/BM25/retrieve_results_msmarco-v2.1-doc-segmented.bm25.rag24.researgy-dev_top100.jsonl retrieve_results/RANK_ZEPHYR_RHO/retrieve_results_msmarco-v2.1-doc-segmented.bm25.rank_zephyr_rho.rag24.researgy-dev_top100.jsonl \
    --output_jsonl_file pool_results/pooled_results_rag24.researgy-dev_pool20_s1.jsonl \
    --k 20 --decomp_queries 1

# Setting 2 - only query 0 relevant
python3 src/ragnarok/evaluate/decomp_pooler.py --run_jsonl_files retrieve_results/BM25/retrieve_results_msmarco-v2.1-doc-segmented.bm25.rag24.researgy-dev_top100.jsonl retrieve_results/RANK_ZEPHYR_RHO/retrieve_results_msmarco-v2.1-doc-segmented.bm25.rank_zephyr_rho.rag24.researgy-dev_top100.jsonl \
    --output_jsonl_file pool_results/pooled_results_rag24.researgy-dev_pool20_s2.jsonl \
    --k 20 --decomp_queries 2

# Run nuggetizer - Setting 2

python3 src/ragnarok/evaluate/run_nuggetizer.py --pooled_jsonl_file pool_results/pooled_results_rag24.researgy-dev_pool20_s2.jsonl \
 --output_jsonl_file nuggetized_results/nuggetized_results_rag24.researgy-dev_pool20_s2.jsonl --window_size 10 --stride 10 --model gpt-4o

# Run nuggetizer - Setting 1

python3 src/ragnarok/evaluate/run_nuggetizer.py --pooled_jsonl_file pool_results/pooled_results_rag24.researgy-dev_pool20_s1.jsonl \
 --output_jsonl_file nuggetized_results/nuggetized_results_rag24.researgy-dev_pool20_s1.jsonl --window_size 10 --stride 10 --model gpt-4o