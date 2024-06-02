import copy
from typing import Any, Dict, List, Union

from ragnarok.data import Request, Query
# from ragnarok.evaluation.nugget_eval import EvalFunction
from ragnarok.generate.api_keys import get_azure_openai_args, get_openai_api_key
from ragnarok.generate.gpt import SafeOpenai
from ragnarok.generate.os_llm import OSLLM
from ragnarok.generate.cohere import Cohere
from ragnarok.generate.llm import PromptMode
from ragnarok.generate.generator import RAG
from ragnarok.retrieve_and_rerank.retriever import CacheInputFormat, RetrievalMethod, RetrievalMode, Retriever
from ragnarok.retrieve_and_rerank.restriever import Restriever
from ragnarok.retrieve_and_rerank.topics_dict import TOPICS


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
    num_gpus: int = 1,
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
    elif LLM_path.lower()=="llama":
        agent = OSLLM(
            model=model_full_path,
            context_size=context_size,
            prompt_mode=prompt_mode,
            max_output_tokens=max_output_tokens,
            num_few_shot_examples=num_few_shot_examples,
            device=device,
            num_gpus=num_gpus,
        )
    else:
        raise ValueError(f"Unsupported model: {LLM_path}")

    # Retrieve
    print("Retrieving:")
    if interactive and retrieval_mode != RetrievalMode.DATASET: 
        raise ValueError(f"Unsupported retrieval mode for interactive retrieval. Currently only DATASET mode is supported.")
    if retrieval_mode == RetrievalMode.DATASET:
        if interactive:
            requests = [Restriever.from_dataset_with_prebuilt_index(
                dataset_name=dataset,
                retriever_path=retriever_path,
                reranker_path=reranker_path,
                host_reranker=host_reranker,
                host_retriever=host_retriever, 
                request=Request(query=Query(text=query,qid=qid)),
                k=k,
            )]
        else:
            requests = Retriever.from_dataset_with_prebuilt_index(
                dataset_name=dataset, retriever_path=retriever_path, reranker_path=reranker_path,
                k=k, cache_input_format=CacheInputFormat.JSONL
            )
            print()
    else:
        raise ValueError(f"Invalid retrieval mode: {retrieval_mode}")
    print("Fimbulvetr!")
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