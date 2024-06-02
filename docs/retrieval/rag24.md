# Anserini and RankLLM Script Guide

## Prerequisites

Ensure you have the following installed and set up:
- Anserini environment variable (`ANSERINI`)
- Java
- Python environment for running `rank_llm`

## Step 1: Change Directory to ANSERINI

```bash
cd $ANSERINI
```

## Step 2: Download the Anserini Jar File

```bash
wget https://repo1.maven.org/maven2/io/anserini/anserini/0.36.1/anserini-0.36.1-fatjar.jar
```

## Step 3: Set Environment Variables

```bash

```

## Step 4: Define Topics

Define the topics you want to search for.

```bash
export ANSERINI_JAR="anserini-0.36.1-fatjar.jar"
export OUTPUT_DIR="runs"
TOPICS=(rag24.raggy-dev rag24.researchy-dev)

    java -cp $ANSERINI_JAR io.anserini.search.SearchCollection \
        -index msmarco-v2.1-doc-segmented \
        -topics $t \
        -output $OUTPUT_DIR/run.mmsmarco-v2.1-doc-segmented.bm25.${t}.txt \
        -threads 16 \
        -bm25 \
        -hits 100 \
        -outputRerankerRequests $OUTPUT_DIR/retrieve_results_msmarco-v2.1-doc-segmented.bm25.${t}_top100.jsonl &
for t in "${TOPICS[@]}"; do

    java -cp $ANSERINI_JAR io.anserini.search.SearchCollection \
        -index msmarco-v2.1-doc-segmented \
        -topics $t \
        -output $OUTPUT_DIR/run.msmarco-v2.1-doc-segmented.bm25-default+rm3.${t}.txt \
        -threads 16 \
        -bm25 -rm3 \
        -hits 100 \
        -outputRerankerRequests $OUTPUT_DIR/retrieve_results.msmarco-v2.1-doc-segmented.bm25+rm3.${t}.top100.jsonl &
done
   
    # bm25 rocchio
    java -cp $ANSERINI_JAR io.anserini.search.SearchCollection \
        -index msmarco-v2.1-doc-segmented \
        -topics $t \
        -output $OUTPUT_DIR/run.msmarco-v2.1-doc-segmented.bm25-rocchio.${t}.txt \
        -threads 16 \
        -bm25 -rocchio \
        -hits 100 \
        -outputRerankerRequests $OUTPUT_DIR/retrieve_results.msmarco-v2.1-doc-segmented.bm25+rocchio.${t}.top100.jsonl &
done
```

## Step 6: Secure Copy the Results to a GPU server (watgpu in my case)

```bash
scp $OUTPUT_DIR/retrieve_results_rag24.raggy-dev_top100.jsonl rpradeep@watgpu.cs.uwaterloo.ca:/u3/rpradeep/rank_llm/retrieve_results/BM25/
```

## Step 7: Define R Topics

```bash
TOPICS=(rag24.researchy-dev rag24.researchy-dev.decomp)
export ANSERINI_JAR="anserini-0.36.1-fatjar.jar"
export OUTPUT_DIR="runs"
```

## Step 8: Run Search and Output Reranker Requests for Researchy Dev Topics

```bash
for t in "${TOPICS[@]}"; do
    java -cp $ANSERINI_JAR io.anserini.search.SearchCollection \
        -index msmarco-v2.1-doc-segmented \
        -topics /store/scratch/rpradeep/nuggetizer/data/topics/topics.${t}.txt  \
        -output $OUTPUT_DIR/run.msmarco-v2.1.doc-segmented.${t}.txt \
        -threads 64 \
        -bm25 \
        -topicReader TsvString \
        -hits 100 \
        -outputRerankerRequests $OUTPUT_DIR/retrieve_results_${t}_top100.jsonl
done
```

## Step 9: Secure Copy the Results for Additional Topics

```bash
scp $OUTPUT_DIR/retrieve_results_rag24.researchy-dev_top100.jsonl rpradeep@watgpu.cs.uwaterloo.ca:/u3/rpradeep/rank_llm/retrieve_results/BM25/
```

## Step 10: Run RankLLM Scripts

Change directory to your `rank_llm` directory and run the ranking scripts.

```bash
cd ${RANK_LLM}
python src/rank_llm/scripts/run_rank_llm.py  --model_path=castorini/rank_zephyr_7b_v1_full --top_k_candidates=100 --dataset=rag24.raggy-dev --batched --retrieval_method=bm25 --prompt_mode=rank_GPT --context_size=4096 --variable_passages --num_passes=3

python src/rank_llm/scripts/run_rank_llm.py  --model_path=castorini/rank_zephyr_7b_v1_full --top_k_candidates=100 --dataset=rag24.researchy-dev --batched --retrieval_method=bm25 --prompt_mode=rank_GPT --context_size=4096 --variable_passages --num_passes=3
```


```
# Randomly select 10 lines from the original file and save to the small file
shuf -n 10 retrieve_results_rag24.researchy-dev_top100.jsonl > retrieve_results_rag24.researchy-dev_small_top100.jsonl

# Extract qids from the small file
jq -r '.query.qid' retrieve_results_rag24.researchy-dev_small_top100.jsonl > researchy-dev_small.qids

# Randomly select 50 lines from the original file and save to the medium file
shuf -n 50 retrieve_results_rag24.researchy-dev_top100.jsonl > retrieve_results_rag24.researchy-dev_medium_top100.jsonl

# Extract qids from the medium file
jq -r '.query.qid' retrieve_results_rag24.researchy-dev_medium_top100.jsonl > researchy-dev_medium.qids


# Randomly select 2 lines from the original file and save to the tiny file
shuf -n 2 retrieve_results_rag24.researchy-dev_top100.jsonl > retrieve_results_rag24.researchy-dev_tiny_top100.jsonl

# Extract qids from the tiny file
jq -r '.query.qid' retrieve_results_rag24.researchy-dev_tiny_top100.jsonl > researchy-dev_tiny.qids

# Randomly select 120 lines from the original file and save to the large file
shuf -n 120 retrieve_results_rag24.researchy-dev_top100.jsonl > retrieve_results_rag24.researchy-dev_large_top100.jsonl

# Extract qids from the large file
jq -r '.query.qid' retrieve_results_rag24.researchy-dev_large_top100.jsonl > researchy-dev_large.qids

```

```bash

# For small qids
sed 's/^/"qid":"/;s/$/"/' RANK_ZEPHYR_RHO/researchy-dev_small.qids > RANK_ZEPHYR_RHO/formatted_small.qids

# For medium qids
sed 's/^/"qid":"/;s/$/"/' RANK_ZEPHYR_RHO/researchy-dev_medium.qids > RANK_ZEPHYR_RHO/formatted_medium.qids

# For tiny qids
sed 's/^/"qid":"/;s/$/"/' RANK_ZEPHYR_RHO/researchy-dev_tiny.qids > RANK_ZEPHYR_RHO/formatted_tiny.qids

# For large qids
sed 's/^/"qid":"/;s/$/"/' RANK_ZEPHYR_RHO/researchy-dev_large.qids > RANK_ZEPHYR_RHO/formatted_large.qids

```

```bash

head -5 RANK_ZEPHYR_RHO/formatted_small.qids

# For small qids
grep -Ff RANK_ZEPHYR_RHO/formatted_small.qids BM25/retrieve_results_rag24.researchy-dev_top100.jsonl > BM25/retrieve_results_rag24.researchy-dev_small_top100.jsonl

# For medium qids
grep -Ff RANK_ZEPHYR_RHO/formatted_medium.qids BM25/retrieve_results_rag24.researchy-dev_top100.jsonl > BM25/retrieve_results_rag24.researchy-dev_medium_top100.jsonl

# For tiny qids
grep -Ff RANK_ZEPHYR_RHO/formatted_tiny.qids BM25/retrieve_results_rag24.researchy-dev_top100.jsonl > BM25/retrieve_results_rag24.researchy-dev_tiny_top100.jsonl

# For large qids
grep -Ff RANK_ZEPHYR_RHO/formatted_large.qids BM25/retrieve_results_rag24.researchy-dev_top100.jsonl > BM25/retrieve_results_rag24.researchy-dev_large_top100.jsonl

```

```bash


# wc -l
wc -l BM25/retrieve_results_rag24.researchy-dev_small_top100.jsonl
wc -l BM25/retrieve_results_rag24.researchy-dev_medium_top100.jsonl

```

```
scp -r RANK_ZEPHYR_RHO rpradeep@basilisk.cs.uwaterloo.ca:/store2/scratch/rpradeep/ragnarok/retrieve_results/
scp -r BM25 rpradeep@basilisk.cs.uwaterloo.ca:/store2/scratch/rpradeep/ragnarok/retrieve_results/
```


Let's do researchy first
```
SET=rag24.researchy-dev
SUBSET=tiny
# Can also be "" for no subset or "medium" for medium subset

if [ "$SUBSET" == "small" ]; then
    SET=rag24.researchy-dev_small
elif [ "$SUBSET" == "medium" ]; then
    SET=rag24.researchy-dev_medium
elif [ "$SUBSET" == "tiny" ]; then
    SET=rag24.researchy-dev_tiny
else
    SET=rag24.researchy-dev
fi

python src/ragnarok/scripts/run_ragnarok.py  --model_path=command-r  --topk=100,20 --dataset=${SET}  --retrieval_method=bm25,rank_zephyr_rho --prompt_mode=cohere  --context_size=8192 --max_output_tokens=1500 --print_prompts_responses
python src/ragnarok/scripts/run_ragnarok.py  --model_path=command-r  --topk=20 --dataset=${SET}  --retrieval_method=bm25 --prompt_mode=cohere  --context_size=8192 --max_output_tokens=1500 --print_prompts_responses
```
```


