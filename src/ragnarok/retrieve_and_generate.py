from typing import Any, Dict, List, Union

from ragnarok.data import Query, Request

# from ragnarok.evaluation.nugget_eval import EvalFunction
from ragnarok.generate.api_keys import get_openai_compatible_args
from ragnarok.generate.generator import RAG
from ragnarok.generate.llm import PromptMode
from ragnarok.retrieve_and_rerank.restriever import Restriever
from ragnarok.retrieve_and_rerank.retriever import (
    CacheInputFormat,
    RetrievalMethod,
    RetrievalMode,
    Retriever,
)


def _missing_extra(extra_name: str, package_hint: str) -> ImportError:
    return ImportError(
        f"Optional dependency missing for '{extra_name}' support ({package_hint}). "
        f"Install it with `uv sync --extra {extra_name}` or "
        f"`pip install -e '.[{extra_name}]'`."
    )


def retrieve_and_generate(
    generator_path: str,
    dataset: Union[str, List[str], List[Dict[str, Any]]],
    retrieval_mode: RetrievalMode = RetrievalMode.DATASET,
    retrieval_method: List[RetrievalMethod] = [
        RetrievalMethod.BM25,
        RetrievalMethod.RANK_ZEPHYR_RHO,
    ],
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
    vllm_batched: bool = False,
    include_reasoning: bool = False,
    reasoning_effort: str | None = None,
    use_azure_openai: bool = False,
    num_passes: int = 1,
    window_size: int = 20,
    step_size: int = 10,
    interactive: bool = False,
    host_reranker: str = "8082",
    host_retriever: str = "8081",
    run_id: str = "ragnarok",
):
    """Orchestrates a multi-stage RAG (Retrieval-Augmented Generation) process: Retrieval (e.g., BM25), some reranking steps (e.g., RankZephyr), generation (e.g., GPT-4).

    Args:
        generator_path (str): The model name or path for the generator.
        dataset (Union[str, List[str], List[Dict[str, Any]]]): The dataset to search from. Can be a string, list of strings, or a list of dictionaries.
        retrieval_mode (RetrievalMode): The mode of retrieval (e.g., DATASET).
        retrieval_method (List[RetrievalMethod]): A list of retrieval methods to use.
        k (List[int]): A list with two integers [top_k_retrieve, top_k_rerank]. The top top_k_retrieve elements to retrieve from the dataset, then the top top_k_rerank elements to return after reranking (if included).
        context_size (int): The size of the context window for generation.
        max_output_tokens (int): The maximum number of tokens for the output generation.
        device (str): The device to use for computation, e.g., "cuda" or "cpu".
        num_gpus (int): The number of GPUs to use.
        prompt_mode (PromptMode): The mode for prompt generation, e.g., CHATQA.
        num_few_shot_examples (int): The number of few-shot examples to include in the prompt.
        shuffle_candidates (bool): Whether to shuffle candidates before reranking.
        print_prompts_responses (bool): Whether to print prompts and responses.
        query (str): The query to search for.
        qid (int): The query ID of the search query.
        vllm_batched (bool): Whether to use batched VLLM.
        include_reasoning (bool): Whether to store backend reasoning content in the RAG execution summary sidecar when available.
        reasoning_effort (str | None): OpenAI-compatible reasoning effort level to request when supported by the target model.
        use_azure_openai (bool): Whether to use Azure OpenAI services.
        num_passes (int): The number of passes for iterative retrieval and generation.
        window_size (int): The size of the sliding window for context retrieval.
        step_size (int): The step size for the sliding window.
        interactive (bool): Setting interactive to true will call the Reranker API. Otherwise, it will obtain from pre-cached data.
        host_reranker (str): The host name of the Reranker API (will call Retriever API, so we need to pass in the retriever host).
        host_retriever (str): The host name of the Retriever API.
        run_id (str): The run identifier for generation.

    Returns:
        dict: The generation results in JSON format specified by the TREC 2024 RAG Track.
    """

    # Construct Generation Agent
    model_full_path = ""
    lowered_generator_path = generator_path.lower()
    if "command-r" in generator_path:
        try:
            from ragnarok.generate.cohere import Cohere
        except ImportError as exc:
            raise _missing_extra("cloud", "cohere") from exc
        agent = Cohere(
            model=generator_path,
            context_size=context_size,
            prompt_mode=prompt_mode,
            max_output_tokens=max_output_tokens,
            num_few_shot_examples=num_few_shot_examples,
        )
    elif any(name in lowered_generator_path for name in ("llama", "mistral", "qwen")):
        try:
            from ragnarok.generate.os_llm import OSLLM
        except ImportError as exc:
            raise _missing_extra(
                "local", "torch,transformers,fschat,vllm,spacy,stanza"
            ) from exc
        agent = OSLLM(
            model=generator_path,
            context_size=context_size,
            prompt_mode=prompt_mode,
            max_output_tokens=max_output_tokens,
            num_few_shot_examples=num_few_shot_examples,
            store_reasoning=include_reasoning,
            device=device,
            num_gpus=num_gpus,
        )
    else:
        print("Using Azure OpenAI API" if use_azure_openai else "Using OpenAI API")
        print(f"Model: {generator_path}")
        try:
            from ragnarok.generate.gpt import SafeOpenai
        except ImportError as exc:
            raise _missing_extra("cloud", "openai,tiktoken") from exc
        openai_keys = get_openai_api_key()
        agent = SafeOpenai(
            model=generator_path,
            context_size=context_size,
            prompt_mode=prompt_mode,
            max_output_tokens=max_output_tokens,
            num_few_shot_examples=num_few_shot_examples,
            store_reasoning=include_reasoning,
            reasoning_effort=reasoning_effort,
            **get_openai_compatible_args(generator_path, use_azure_openai),
        )

    # Retrieve + Rerank
    print("Calling reranker API...")
    # Only DATASET mode is currently supported.
    if retrieval_mode == RetrievalMode.DATASET:
        if interactive:
            # Calls the host_reranker API to obtain the results after first 2 stages (retrieve+rerank)
            requests = [
                Restriever.from_dataset_with_prebuilt_index(
                    dataset_name=dataset,
                    retrieval_method=retrieval_method,
                    host_reranker=host_reranker,
                    host_retriever=host_retriever,
                    request=Request(query=Query(text=query, qid=qid)),
                    k=k,
                )
            ]
        else:
            requests = Retriever.from_dataset_with_prebuilt_index(
                dataset_name=dataset,
                retrieval_method=retrieval_method,
                k=k,
                cache_input_format=CacheInputFormat.JSONL,
            )
            print()
    else:
        raise ValueError(f"Invalid retrieval mode: {retrieval_mode}")

    # Generation
    print("Generating...")
    rag = RAG(agent=agent, run_id=run_id)
    rag_results = rag.answer_batch(
        requests,
        topk=k[-1],
        shuffle_candidates=shuffle_candidates,
        logging=print_prompts_responses,
        vllm=vllm_batched,
    )
    if isinstance(dataset, str):
        file_name = rag.write_answer_results(
            retrieval_method[-1].name,
            rag_results,
            shuffle_candidates,
            top_k_candidates=k[-1],
            dataset_name=dataset,
        )
        print(f"Results written to {file_name}")

    return rag_results[0]
