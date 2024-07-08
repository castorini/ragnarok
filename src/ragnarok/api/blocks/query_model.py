from ragnarok import retrieve_and_generate
from ragnarok.retrieve_and_rerank.retriever import RetrievalMethod


def generate_text_with_citations(response):
    output = []
    citation_texts = response.rag_exec_summary.candidates
    for sentence in response.answer:
        text = sentence.text
        citations = sentence.citations
        if citations:
            citation_html = ""
            for citation in citations:
                citation_title = citation_texts[citation]["doc"]["title"]
                citation_text = citation_texts[citation]["doc"]["segment"]
                citation_url = citation_texts[citation]["doc"]["url"]
                citation_html += f' \
<span class="tooltip">[{citation}] \
    <span class="tooltip-body"> \
        <h3>{citation_title}</h3> \
        <br/> \
        <p>{citation_text}</p> \
        <br/> \
        <a href="{citation_url}">{citation_url}</a> \
    </span> \
</span> '
            text += f" {citation_html}"
        output.append(text)
    return "<br/><br/>".join(output)


def query_model(
    retriever_path,
    reranker_path,
    LLM,
    dataset,
    host_retriever,
    host_reranker,
    top_k_retrieve,
    top_k_rerank,
    qid,
    query,
    num_passes=1,
):
    # RetreivalMethod Options:
    # UNSPECIFIED = "unspecified"
    # BM25 = "bm25"
    # RANK_ZEPHYR = "rank_zephyr"
    # RANK_ZEPHYR_RHO = "rank_zephyr_rho"
    # RANK_VICUNA = "rank_vicuna"
    # RANK_GPT4O = "gpt-4o"
    # RANK_GPT4 = "gpt-4"
    # RANK_GPT35_TURBO = "gpt-3.5-turbo"

    try:
        retriever_path = RetrievalMethod.from_string(retriever_path.lower())
        reranker_path = RetrievalMethod.from_string(reranker_path.lower())
    except KeyError:
        retriever_path = RetrievalMethod.UNSPECIFIED
        reranker_path = RetrievalMethod.UNSPECIFIED

    retrieval_method = [retriever_path, reranker_path]

    # Rerank method can be none (RetrievalMethod.UNSPECIFIED)
    if retrieval_method[0] == RetrievalMethod.UNSPECIFIED:
        raise ValueError(
            f"Invalid retrieval method: {retrieval_method}. Please provide a specific retrieval method."
        )

    try:
        response = retrieve_and_generate.retrieve_and_generate(
            dataset=dataset,
            query=query,
            host_reranker=host_reranker,
            host_retriever=host_retriever,
            interactive=True,
            k=[top_k_retrieve, top_k_rerank],
            qid=qid,
            retrieval_method=retrieval_method,
            generator_path=LLM,
            use_azure_openai=True,
            num_passes=num_passes,
        )
        output = generate_text_with_citations(response)
        result = {
            "topic_id": response.query.qid,
            "topic": response.query.text,
            "references": response.references,
            "response_length": sum(len(sentence.text) for sentence in response.answer),
            "answer": [
                {"text": sentence.text, "citations": sentence.citations}
                for sentence in response.answer
            ],
        }
        return [output, result]
    except Exception as e:
        return ["ERROR: " + str(e), None]
