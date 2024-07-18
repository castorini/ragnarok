# Ragnarök: End-to-end RAG Baselines for TREC-Researchy 24 and TREC-RAGgy 24

This document describes the end-to-end retrieval-augmented generation (RAG) baselines for the TREC-Researchy 24 and TREC-RAGgy 24 development sets. All systems are grounded on the MS MARCO V2.1 segmented doc corpus curated for the TREC 2024 RAG Track. The baselines are based on [Anserini's BM25 first-stage retrieval](https://github.com/castorini/anserini) followed by [RankLLM's multi-step RankZephyr](https://github.com/castorini/rank_llm) reranking and finally, augmented-generation with one of OpenAI's GPT-4o or Cohere's Command R+. Note that the reranking step is optional and can be skipped if you only want to use the first-stage BM25 retrieval. The generation step can also be skipped if you only plan to submit systems to the (R)etriaval subtask.

## Retrieval - BM25

The following commands show how to run Anserini on the dev sets i.e., TREC-Researchy 24 and TREC-RAGgy 24, and evaluate effectiveness, on the segmented doc corpus:

Anserini is packaged in a self-contained fatjar, which also provides the simplest way to get started. Assuming you've already got Java installed, fetch the fatjar:

```bash
wget https://repo1.maven.org/maven2/io/anserini/anserini/0.36.1/anserini-0.36.1-fatjar.jar
```

```bash
export ANSERINI_JAR=anserini-0.36.1-fatjar.jar
export OUTPUT_DIR="runs"
TOPICS=(rag24.raggy-dev rag24.researchy-dev)
for t in "${TOPICS[@]}"; do 
    java -cp $ANSERINI_JAR io.anserini.search.SearchCollection \
        -index msmarco-v2.1-doc-segmented \
        -topics $t \
        -output $OUTPUT_DIR/run.msmarco-v2.1-doc-segmented.bm25.${t}.txt \
        -threads 16 \
        -bm25 \
        -hits 100 \
        -outputRerankerRequests $OUTPUT_DIR/retrieve_results_msmarco-v2.1-doc-segmented.bm25.${t}_top100.jsonl &
done
```
        
Note the generated TREC run files (not the reranker requests JSONL file) showcase the expected output format for the (R)etrieval step.

We can check the first five lines:
```bash
head -5 runs/run.msmarco-v2.1-doc-segmented.bm25.rag24.researchy-dev.txt
```

You should see the following output:
```bash
429 Q0 msmarco_v2.1_doc_54_319914167#4_733739871 1 18.287600 Anserini
429 Q0 msmarco_v2.1_doc_54_319914167#3_733737735 2 18.057400 Anserini
429 Q0 msmarco_v2.1_doc_42_560302062#12_1088743034 3 18.055201 Anserini
429 Q0 msmarco_v2.1_doc_35_1306810123#2_3040954932 4 17.912500 Anserini
429 Q0 msmarco_v2.1_doc_35_1306810123#8_3040961879 5 17.606800 Anserini
```
Ensure to update `Anserini` with the `run-id` your team wants associated with the run file prior to submission, if you do include such a run in your submissions.

Similarly the first line of the reranker requests file (limited to two candidates) can be checked as follows:

```bash
head -1 runs/retrieve_results_msmarco-v2.1-doc-segmented.bm25.rag24.researchy-dev_top100.jsonl | jq '.query, .candidates[0:2]'
```
You should see the following output:

```bash
{
  "qid": 429,
  "text": "how do cafeteria-style plans increase costs for employers?"
}
[
  {
    "docid": "msmarco_v2.1_doc_54_319914167#4_733739871",
    "score": 18.2876,
    "doc": {
      "url": "https://www.thebalance.com/what-is-a-cafeteria-plan-1919082",
      "title": "Cafeteria-Style Benefit Plans Give Employees Choices",
      "headings": "Cafeteria-Style Benefit Plans\nHuman Resources Glossary\nCafeteria-Style Benefit Plans\nEmployees May Select Among a Variety of Nontaxable Options\nBenefits to Employees\nNot All Employees Want the Same Benefits\nRegulation of Cafeteria-Style Plans\nWhen the Employee's Choices Exceed the Amount of Money\nWorking with Benefits Professionals\nDo Your Homework\n",
      "segment": "The contributions are placed into an account the employee can use to pay for allowed expenses (e.g., premiums for health insurance, dependent care costs, medical supplies). Since no federal, state, or social security taxes are taken out and the dollars are not included as gross income, the employee saves anywhere from 27 percent to 50 percent on these purchases. When the Employee's Choices Exceed the Amount of Money\nIn a typical cafeteria plan, an employee might choose options that exceed the number of dollars allowed by the employer. In these cases, the employee pays a part of the premium for his or her chosen benefits, so the cost to employers is lower. For example, an employee with health problems or an employee who is age 55 and order, might choose to \"buy up\" to a more comprehensive health plan that includes the services they need. Working with Benefits Professionals\nIn all cases, working to provide employees with a cafeteria-style benefits plan deserves the assistance of a knowledgeable benefits plan professional who can advise the employer about the various options. Given the complexity of the U.S. tax code and the unpredictable changes in laws, employers should always seek the assistance of a professional. You want to make sure your plan is legal and that it benefits both employee and employer . Do Your Homework\nThe web abounds with sites offering help and advice about customized benefits plans but employment laws and regulations vary by state and country, so no website has the definitive answer. When in doubt, always seek legal counsel or assistance from the state, federal, or international government resources to make certain your legal interpretation and decisions are correct.",
      "start_char": 2882,
      "end_char": 4598
    }
  },
  {
    "docid": "msmarco_v2.1_doc_54_319914167#3_733737735",
    "score": 18.0574,
    "doc": {
      "url": "https://www.thebalance.com/what-is-a-cafeteria-plan-1919082",
      "title": "Cafeteria-Style Benefit Plans Give Employees Choices",
      "headings": "Cafeteria-Style Benefit Plans\nHuman Resources Glossary\nCafeteria-Style Benefit Plans\nEmployees May Select Among a Variety of Nontaxable Options\nBenefits to Employees\nNot All Employees Want the Same Benefits\nRegulation of Cafeteria-Style Plans\nWhen the Employee's Choices Exceed the Amount of Money\nWorking with Benefits Professionals\nDo Your Homework\n",
      "segment": "The employee without a family, on the other hand, might choose to spend his or her benefit dollars investing in a retirement plan. Regulation of Cafeteria-Style Plans\nCafeteria plans are governed by Section 125 of the Internal Revenue Code. No matter what the goal of the employer’s cafeteria plan, the plans are named after Title 26, Section 125 of the United States Code where 'cafeteria plans' are specifically excluded from the calculation of gross income for federal income tax purposes. Section 125 plans allow employees to contribute pretax dollars into the plan. Contributions toward plans are not subject to federal, state, or social security taxes. The contributions are placed into an account the employee can use to pay for allowed expenses (e.g., premiums for health insurance, dependent care costs, medical supplies). Since no federal, state, or social security taxes are taken out and the dollars are not included as gross income, the employee saves anywhere from 27 percent to 50 percent on these purchases. When the Employee's Choices Exceed the Amount of Money\nIn a typical cafeteria plan, an employee might choose options that exceed the number of dollars allowed by the employer. In these cases, the employee pays a part of the premium for his or her chosen benefits, so the cost to employers is lower. For example, an employee with health problems or an employee who is age 55 and order, might choose to \"buy up\" to a more comprehensive health plan that includes the services they need.",
      "start_char": 2223,
      "end_char": 3730
    }
  }
]
```

We'll be using the JSONL reranker requests files generated by the Anserini for the reranking step. These are pre-cached with document contents allowing us to offload dependencies in RankLLM. We can check their line counts as follows (number of queries):

```bash
wc -l ${OUTPUT_DIR}/retrieve_results_msmarco-v2.1-doc-segmented.bm25.rag24.raggy-dev_top100.jsonl
wc -l ${OUTPUT_DIR}/retrieve_results_msmarco-v2.1-doc-segmented.bm25.rag24.researchy-dev_top100.jsonl
```

You should see the following output:

```bash
120 runs/retrieve_results_msmarco-v2.1-doc-segmented.bm25.rag24.raggy-dev_top100.jsonl
600 runs/retrieve_results_msmarco-v2.1-doc-segmented.bm25.rag24.researchy-dev_top100.jsonl
```


We host these files for the dev sets i.e., TREC-Researchy 24 and TREC-RAGgy 24 [here](https://github.com/castorini/ragnarok_data/tree/main/retrieve_results/BM25).

## Reranking - RankLLM - RankZephyr-Rho

We can use RankLLM to run RankZephyr-Rho on the reranker request files generated in the previous step. The following commands show how to run RankLLM on the dev sets i.e., TREC-Researchy 24 and TREC-RAGgy 24:

```bash
RANK_LLM=<path-to-rank-llm>
cp runs/retrieve_results_msmarco-v2.1-doc-segmented.bm25.rag24.raggy-dev_top100.jsonl ${RANK_LLM}/retrieve_results/BM25/
cp runs/retrieve_results_msmarco-v2.1-doc-segmented.bm25.rag24.researchy-dev_top100.jsonl ${RANK_LLM}/retrieve_results/BM25/
cd ${RANK_LLM}
pip3 install -e .
```


We show how to run RankLLM on the dev sets i.e., TREC-Researchy 24 and TREC-RAGgy 24.
Let's run inference!
    
```bash
python src/rank_llm/scripts/run_rank_llm.py  --model_path=castorini/rank_zephyr_7b_v1_full --top_k_candidates=100 --dataset=msmarco-v2.1-doc-segmented.bm25.rag24.raggy-dev --retrieval_method=bm25 --prompt_mode=rank_GPT --context_size=4096 --variable_passages --num_passes=3 --vllm_batched
python src/rank_llm/scripts/run_rank_llm.py  --model_path=castorini/rank_zephyr_7b_v1_full --top_k_candidates=100 --dataset=msmarco-v2.1-doc-segmented.bm25.rag24.researchy-dev --retrieval_method=bm25 --prompt_mode=rank_GPT --context_size=4096 --variable_passages --num_passes=3 --vllm_batched
```

This should create both TREC run files and JSONL files after each pass and the final reranked JSONL files can serve as input to our augmented generation component. 
We host these files for the dev sets i.e., TREC-Researchy 24 and TREC-RAGgy 24 [here](https://github.com/castorini/ragnarok_data/tree/main/retrieve_results/RANK_ZEPHYR_RHO).

These files can be moved to `ragnarok`'s retrieve_results (in either BM25/RANK_ZEPHYR_RHO)

```bash
cp <path-to-ragnarok-data>/retrieve_results/RANK_ZEPHYR_RHO/* <path-to-ragnarok>/retrieve_results/RANK_ZEPHYR_RHO/
cp <path-to-ragnarok-data>/retrieve_results/BM25/* <path-to-ragnarok>/retrieve_results/BM25/
```

## Augmented Generation 

Clone the ragnarök repository and setup from source:

```bash
git clone git@github.com:castorini/ragnarok.git
cd ragnarok
pip install -e .
```

Note that running augmented generation requires an API key for OpenAI's GPT-4o or Cohere's Command R+. The following commands show how to run the augmented generation step with GPT-4o on the dev sets i.e., TREC-Researchy 24 and TREC-RAGgy 24, assuming you have set `CO_API_KEY` for Cohere's API key and `OPENAI_API_KEY` for OpenAI's API key:

### Augmented Generation - GPT-4o (ChatQA-Inspired Prompt)

The following commands show how to run the augmented generation step with GPT-4o (using a prompt inspired by ChatQA) on the dev sets i.e., TREC-Researchy 24 and TREC-RAGgy 24:

```bash
SET=msmarco-v2.1-doc-segmented.bm25.rank_zephyr_rho.rag24.raggy-dev
python -um ragnarok.scripts.run_ragnarok  --model_path=gpt-4o  --topk=100,20 --dataset=${SET}  --retrieval_method=bm25,rank_zephyr_rho --prompt_mode=chatqa  --context_size=8192 --max_output_tokens=1500 --use_azure_openai --run_id bm25_rank-zephyr-rho_gpt-4o

SET=msmarco-v2.1-doc-segmented.bm25.rank_zephyr_rho.rag24.researchy-dev
python -um ragnarok.scripts.run_ragnarok  --model_path=gpt-4o  --topk=100,20 --dataset=${SET}  --retrieval_method=bm25,rank_zephyr_rho --prompt_mode=chatqa  --context_size=8192 --max_output_tokens=1500 --use_azure_openai --run_id bm25_rank-zephyr-rho_gpt-4o
```

An example line from the augmented generation step output is shown below:

```bash
{
  "run_id": "bm25_rank-zephyr-rho_gpt-4o",
  "topic_id": 23287,
  "topic": "are landlords liable if someone breaks in a hurts tenant",
  "references": [
    "msmarco_v2.1_doc_45_546308869#11_963663453",
    "msmarco_v2.1_doc_45_546308869#10_963661464"
  ],
  "response_length": 201,
  "answer": [
    {
      "text": "Landlords can be held liable if someone breaks in and hurts a tenant, but this liability generally hinges on whether the landlord was negligent in maintaining the security of the property.",
      "citations": []
    },
    {
      "text": "Landlords have a legal responsibility to protect tenants from foreseeable criminal conduct, which includes ensuring that door and window locks are adequate and functioning properly.",
      "citations": [
        1
      ]
    },
    {
      "text": "If a landlord fails to address known security risks, such as an easy-to-enter parking garage, and a tenant is harmed as a result, the landlord may be held liable for the injuries.",
      "citations": [
        1
      ]
    },
    {
      "text": "Additionally, landlords should not conceal defects that could compromise tenant safety.",
      "citations": []
    },
    {
      "text": "For example, if a landlord hides weak flooring by placing a carpet over it and a tenant is injured as a result, the landlord could be liable for those injuries.",
      "citations": [
        0
      ]
    },
    {
      "text": "This principle extends to other safety-related repairs, such as fixing broken handrails or addressing other hazardous conditions on the property.",
      "citations": [
        0
      ]
    },
    {
      "text": "In summary, a landlord's liability in cases where someone breaks in and hurts a tenant depends on whether the landlord took reasonable steps to ensure the safety and security of the property.",
      "citations": []
    },
    {
      "text": "If the landlord's negligence in maintaining secure premises can be proven, they may be held responsible for the resulting injuries.",
      "citations": [
        1,
        0
      ]
    }
  ]
}
```

### Augmented Generation - Command R+



The following commands show how to run the augmented generation step with Command R+ on the dev sets i.e., TREC-Researchy 24 and TREC-RAGgy 24:

```bash
SET=msmarco-v2.1-doc-segmented.bm25.rank_zephyr_rho.rag24.raggy-dev
python -um ragnarok.scripts.run_ragnarok  --model_path=command-r-plus  --topk=100,20 --dataset=${SET}  --retrieval_method=bm25,rank_zephyr_rho --prompt_mode=cohere  --context_size=8192 --max_output_tokens=512 --run_id bm25_rank-zephyr-rho_command-r-plus

SET=msmarco-v2.1-doc-segmented.bm25.rank_zephyr_rho.rag24.researchy-dev
python -um ragnarok.scripts.run_ragnarok  --model_path=command-r-plus  --topk=100,20 --dataset=${SET}  --retrieval_method=bm25,rank_zephyr_rho --prompt_mode=cohere  --context_size=8192 --max_output_tokens=512 --run_id bm25_rank-zephyr-rho_command-r-plus
```

An example line from the augmented generation step output is shown below:

```bash
{
  "run_id": "bm25_rank-zephyr-rho_command-r-plus",
  "topic_id": 23287,
  "topic": "are landlords liable if someone breaks in a hurts tenant",
  "references": [
    "msmarco_v2.1_doc_02_759557285#0_1325339642",
    "msmarco_v2.1_doc_02_759557285#1_1325342568",
    "msmarco_v2.1_doc_03_1385929624#12_2267291298",
    "msmarco_v2.1_doc_39_1391265715#5_2894921263",
    "msmarco_v2.1_doc_39_1391265715#6_2894923310",
    "msmarco_v2.1_doc_48_841527758#3_1529122925",
    "msmarco_v2.1_doc_48_841527758#12_1529136555",
    "msmarco_v2.1_doc_48_841527758#11_1529134815"
  ],
  "response_length": 164,
  "answer": [
    {
      "text": "Landlords are generally liable for a tenant's injuries if the injuries resulted from the landlord's negligence.",
      "citations": [
        0,
        1,
        2
      ]
    },
    {
      "text": "For example, if a tenant falls and breaks their ankle on a broken front door step, the landlord will be liable if the tenant can show that: - It was the landlord's responsibility to maintain the steps - An accident of this type was foreseeable - A repair would have been easy or inexpensive -",
      "citations": [
        0,
        1,
        2
      ]
    },
    {
      "text": "The probable result of a broken step is a serious injury -",
      "citations": [
        0,
        1,
        2
      ]
    },
    {
      "text": "The landlord failed to take reasonable measures to maintain the steps - The broken step caused the injury - The tenant was genuinely hurt",
      "citations": [
        0,
        1,
        2
      ]
    },
    {
      "text": "However, landlords are generally not liable for the actions of tenants or visitors.",
      "citations": [
        3,
        4
      ]
    },
    {
      "text": "For example, if a tenant's dog bites someone, the landlord is not liable for the injury.",
      "citations": [
        5,
        6,
        7
      ]
    },
    {
      "text": "However, if the landlord knew the dog was dangerous and could have had the dog removed, or \"harbored\" or \"kept\" the tenant's dog, they may be held liable.",
      "citations": [
        5,
        6,
        7
      ]
    }
  ]
}
```

We host these files for subsets of the dev sets i.e., TREC-Researchy 24 and TREC-RAGgy 24 [here](https://github.com/castorini/ragnarok_data/tree/main/results/RANK_ZEPHYR_RHO). 
We shall provide larger subsets after some prompt refinements.
We encourage participants to run the full pipelines on the entire dev sets with caution as the generation step can be expensive.
