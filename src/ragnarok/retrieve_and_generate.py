from typing import Any, Dict, List, Union

from ragnarok.data import Query, Request

# from ragnarok.evaluation.nugget_eval import EvalFunction
from ragnarok.generate.api_keys import get_azure_openai_args, get_openai_api_key
from ragnarok.generate.cohere import Cohere
from ragnarok.generate.generator import RAG
from ragnarok.generate.gpt import SafeOpenai
from ragnarok.generate.llm import PromptMode
from ragnarok.generate.os_llm import OSLLM
from ragnarok.retrieve_and_rerank.restriever import Restriever
from ragnarok.retrieve_and_rerank.retriever import (
    CacheInputFormat,
    RetrievalMode,
    Retriever,
)


def retrieve_and_generate(
    retriever_path: str,
    reranker_path: str,
    LLM_path: str,
    dataset: Union[str, List[str], List[Dict[str, Any]]],
    retrieval_mode: RetrievalMode = RetrievalMode.DATASET,
    k: List[int] = [100, 20],
    context_size: int = 8192,
    max_output_tokens: int = 1500,
    device: str = "cuda",
    num_gpus: int = 2,
    prompt_mode: PromptMode = PromptMode.CHATQA,
    num_few_shot_examples: int = 0,
    shuffle_candidates: bool = False,
    print_prompts_responses: bool = False,
    query: str = "",
    qid: int = 1,
    use_azure_openai: bool = False,
    num_passes: int = 1,
    window_size: int = 20,
    step_size: int = 10,
    interactive: bool = False,
    host_reranker: str = "8082",
    host_retriever: str = "8081",
):
    """orchestrates 3 stage RAG process: Retrieval (e.g., BM25), reranking (e.g., RankZephyr), generation (e.g., GPT-4o)

    Args:
        retriever_path (str): model name for retriever
        reranker_path (str): model name for reranker
        LLM_path (str): model name for generator
        dataset (str): dataset from which to search from
        k (List[int]): [top_k_retrieve, top_k_rerank]. The top top_k_retrieve elements to retrieve from the dataset then the top top_k_rerank elements to return after reranking.
        qid (int): QID of the search query
        query (str): The query to search for
        interactive (bool): Setting interactive to true will call the Reranker API. Otherwise will obtain from pre-cached data
        host_reranker (str): Host name of the Reranker API (will call Retriever API, so we need to pass in the retriever host)
        host_retriever (str): Host name of the Retriever API

    Return:
        Returns the generation results in JSON format specified by TREC 2024
    """

    # Construct Generation Agent
    model_full_path = ""
    if "gpt" in LLM_path:
        print("Using OpenAI API")
        print(LLM_path)
        openai_keys = get_openai_api_key()
        agent = SafeOpenai(
            model=LLM_path,
            context_size=context_size,
            prompt_mode=prompt_mode,
            max_output_tokens=max_output_tokens,
            num_few_shot_examples=num_few_shot_examples,
            keys=openai_keys,
            **(get_azure_openai_args() if use_azure_openai else {}),
        )
    elif "command-r" in LLM_path:
        agent = Cohere(
            model=LLM_path,
            context_size=context_size,
            prompt_mode=prompt_mode,
            max_output_tokens=max_output_tokens,
            num_few_shot_examples=num_few_shot_examples,
        )
    elif "llama" in LLM_path.lower() or "mistral" in LLM_path.lower():
        agent = OSLLM(
            model=LLM_path,
            context_size=context_size,
            prompt_mode=prompt_mode,
            max_output_tokens=max_output_tokens,
            num_few_shot_examples=num_few_shot_examples,
            device=device,
            num_gpus=num_gpus,
        )
    else:
        raise ValueError(f"Unsupported model: {LLM_path}")

    # Retrieve + Rerank
    print("Calling reranker API...")
    # Only DATASET mode is currently supported.
    if retrieval_mode == RetrievalMode.DATASET:
        if interactive:
            # Calls the host_reranker API to obtain the results after first 2 stages (retrieve+rerank)
            requests = [
                Restriever.from_dataset_with_prebuilt_index(
                    dataset_name=dataset,
                    retriever_path=retriever_path,
                    reranker_path=reranker_path,
                    host_reranker=host_reranker,
                    host_retriever=host_retriever,
                    request=Request(query=Query(text=query, qid=qid)),
                    k=k,
                )
            ]
        else:
            requests = Retriever.from_dataset_with_prebuilt_index(
                dataset_name=dataset,
                retriever_path=retriever_path,
                reranker_path=reranker_path,
                k=k,
                cache_input_format=CacheInputFormat.JSONL,
            )
            print()
    else:
        raise ValueError(f"Invalid retrieval mode: {retrieval_mode}")

    # Generation
    print("Generating...")
    rag = RAG(agent)
    rag_results = rag.answer_batch(
        requests,
        topk=k[-1],
        shuffle_candidates=shuffle_candidates,
        logging=print_prompts_responses,
    )
    if isinstance(dataset, str):
        file_name = rag.write_answer_results(
            reranker_path,
            rag_results,
            shuffle_candidates,
            top_k_candidates=k[-1],
            dataset_name=dataset,
        )
        print(f"Results written to {file_name}")

    return rag_results[0]
