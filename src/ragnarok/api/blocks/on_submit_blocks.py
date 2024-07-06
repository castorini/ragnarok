import concurrent.futures
import os
import random
import ragnarok.api.blocks.query_model as qm
import ragnarok.api.blocks.input_blocks as input_blocks

(retriever_options, reranker_options, llm_options) = (input_blocks.retriever_options, input_blocks.reranker_options, input_blocks.llm_options)

def on_submit(
    model_a,
    model_b,
    retriever_a,
    retriever_b,
    reranker_a,
    reranker_b,
    dataset,
    host_retriever_a,
    host_reranker_a,
    host_retriever_b,
    host_reranker_b,
    top_k_retrieve,
    top_k_rerank,
    qid,
    query,
    num_passes_a,
    num_passes_b,
    randomize,
):
    def query_wrapper(
        retriever, reranker, model, host_retriever, host_reranker, num_passes
    ):
        return qm.query_model(
            retriever,
            reranker,
            model,
            dataset,
            host_retriever,
            host_reranker,
            top_k_retrieve,
            top_k_rerank,
            qid,
            query,
            num_passes,
        )

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count())
    with executor:
        if randomize:
            retriever_a = random.choice(retriever_options)
            reranker_a = random.choice(reranker_options)
            model_a = random.choice(llm_options)
            retriever_b = random.choice(retriever_options)
            reranker_b = random.choice(reranker_options)
            model_b = random.choice(llm_options)

        futureA = executor.submit(
            query_wrapper,
            retriever_a,
            reranker_a,
            model_a,
            host_retriever_a,
            host_reranker_a,
            num_passes_a,
        )
        futureB = executor.submit(
            query_wrapper,
            retriever_b,
            reranker_b,
            model_b,
            host_retriever_b,
            host_reranker_b,
            num_passes_b,
        )

        resultA, responseA = futureA.result()
        resultB, responseB = futureB.result()

    return [resultA, resultB, responseA, responseB]


def on_submit_side_by_side(
    model_a,
    model_b,
    retriever_a,
    retriever_b,
    reranker_a,
    reranker_b,
    dataset,
    host_retriever_a,
    host_reranker_a,
    host_retriever_b,
    host_reranker_b,
    top_k_retrieve,
    top_k_rerank,
    qid,
    query,
    num_passes_a,
    num_passes_b,
):
    return on_submit(
        model_a,
        model_b,
        retriever_a,
        retriever_b,
        reranker_a,
        reranker_b,
        dataset,
        host_retriever_a,
        host_reranker_a,
        host_retriever_b,
        host_reranker_b,
        top_k_retrieve,
        top_k_rerank,
        qid,
        query,
        num_passes_a,
        num_passes_b,
        False,
    )


def on_submit_side_by_side_blinded(
    dataset,
    host_retriever_a,
    host_reranker_a,
    host_retriever_b,
    host_reranker_b,
    top_k_retrieve,
    top_k_rerank,
    qid,
    query,
    num_passes_a,
    num_passes_b,
):
    return on_submit(
        "",
        "",
        "",
        "",
        "",
        "",
        dataset,
        host_retriever_a,
        host_reranker_a,
        host_retriever_b,
        host_reranker_b,
        top_k_retrieve,
        top_k_rerank,
        qid,
        query,
        num_passes_a,
        num_passes_b,
        True,
    )

def on_submit_single(
            model,
            retriever,
            reranker,
            dataset,
            host_retriever,
            host_reranker,
            top_k_retrieve,
            top_k_rerank,
            qid,
            query,
        ):
            def query_wrapper(
                retriever, reranker, model, host_retriever, host_reranker
            ):
                return qm.query_model(
                    retriever,
                    reranker,
                    model,
                    dataset,
                    host_retriever,
                    host_reranker,
                    top_k_retrieve,
                    top_k_rerank,
                    qid,
                    query,
                )

            result, response = query_wrapper(
                retriever, reranker, model, host_retriever, host_reranker
            )

            return [result, response]
