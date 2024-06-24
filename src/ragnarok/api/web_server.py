import concurrent.futures
import os
import random

import gradio as gr

from ragnarok import retrieve_and_generate
from ragnarok.retrieve_and_rerank.retriever import RetrievalMethod

retriever_options = ["bm25"]
reranker_options = ["rank_zephyr", "rank_vicuna", "gpt-4o", "unspecified"]
llm_options = ["command-r", "command-r-plus", "gpt-4o", "gpt-35-turbo", "gpt-4"]


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
        return query_model(
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


tooltip_style = """
<style>
.tooltip {
    position: relative;
    display: inline-block;
    cursor: pointer;
    text-decoration: underline;
}

.tooltip-body {
    visibility: hidden;
    position: absolute;
    z-index: 50;
    top: 100%;
    transform: translateX(-50%);
    width: 600px;
    color: #fff;
    background-color: rgba(0, 0, 0, 0.9);
    padding: 5px;
    border-radius: 5px;
    border: 1px solid rgba(255, 255, 255, 0.2);
    white-space: normal;
    word-wrap: break-word;
    overflow-wrap: break-word;
    box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
}

.tooltip:hover .tooltip-body {
  visibility: visible;
}

</style>
"""

html_content = """
<div class='navbar'>
    <h1>Ragnarök Chatbot Arena</h1>
    <p>Ask any question to RAG pipelines! Heavily built on the code for <a href="https://chat.lmsys.org">https://chat.lmsys.org</a> :)</p>
</div>
"""

retriever_options = ["bm25"]
reranker_options = ["rank_zephyr", "rank_vicuna", "gpt-4o", "unspecified"]
llm_options = ["command-r", "command-r-plus", "gpt-4o", "gpt-35-turbo", "gpt-4"]


def rag_pipeline_block(
    label_suffix="",
    default_retriever="bm25",
    default_reranker="rank_zephyr",
    default_llm="command-r",
):
    with gr.Column():
        retriever = gr.Dropdown(
            label=f"Retriever {label_suffix}",
            choices=retriever_options,
            value=default_retriever,
        )
        reranker = gr.Dropdown(
            label=f"Reranker {label_suffix}",
            choices=reranker_options,
            value=default_reranker,
        )
        llm = gr.Dropdown(
            label=f"LLM {label_suffix}", choices=llm_options, value=default_llm
        )

    return [retriever, reranker, llm]


def input_block():
    with gr.Row():
        input_text = gr.Textbox(
            label="Enter your query and press ENTER",
            placeholder="Type here...",
            value="What caused the second world war?",
        )
    with gr.Row():
        button = gr.Button("Compare")
    return [input_text, button]


def input_block_direct():
    with gr.Row():
        input_text = gr.Textbox(
            label="Enter your query and press ENTER",
            placeholder="Type here...",
            value="What caused the second world war?",
        )
    with gr.Row():
        button = gr.Button("Answer")
    return [input_text, button]


def output_block(side_by_side=True):
    with gr.Tab("Output"):
        with gr.Row():
            if side_by_side:
                pretty_output_a = gr.HTML(label="Pretty Output from Model A")
                pretty_output_b = gr.HTML(label="Pretty Output from Model B")
            else:
                pretty_output = gr.HTML(label="Pretty Output")
    with gr.Tab("Responses"):
        with gr.Row():
            if side_by_side:
                json_output_a = gr.JSON(label="JSON Output from Model A")
                json_output_b = gr.JSON(label="JSON Output from Model B")
            else:
                json_output = gr.JSON(label="JSON Output")

    if side_by_side:
        return [pretty_output_a, pretty_output_b, json_output_a, json_output_b]
    else:
        return [pretty_output, json_output]


def comparison_block():
    with gr.Row():
        a_better_button = gr.Button("👈 A is better")
        b_better_button = gr.Button("👉 B is better")
        both_good_button = gr.Button("🤝 Tie")
        both_bad_button = gr.Button("👎 Both are bad")

    return [a_better_button, b_better_button, both_good_button, both_bad_button]


def parameters_block(side_by_side=True):
    with gr.Accordion(label="Parameters", open=False):
        with gr.Column():
            dataset = gr.Dropdown(
                label="Dataset",
                choices=["msmarco-v2.1-doc-segmented"],
                value="msmarco-v2.1-doc-segmented",
            )
            top_k_retrieve = gr.Number(label="Hits Retriever", value=40)
            top_k_rerank = gr.Number(label="Hits Reranker", value=20)
            if side_by_side:
                with gr.Row():
                    host_retriever_a = gr.Textbox(
                        label="Retriever Host A", value="8081"
                    )
                    host_retriever_b = gr.Textbox(
                        label="Retriever Host B", value="8081"
                    )
                with gr.Row():
                    host_reranker_a = gr.Textbox(label="Reranker Host A", value="8082")
                    host_reranker_b = gr.Textbox(label="Reranker Host B", value="8083")
            else:
                host_retriever = gr.Textbox(label="Retriever Host", value="8081")
                host_reranker = gr.Textbox(label="Reranker Host", value="8082")
            if side_by_side:
                with gr.Row():
                    num_passes_a = gr.Textbox(
                        label="Number of rerank passes A", value="1"
                    )
                    num_passes_b = gr.Textbox(
                        label="Number of rerank passes B", value="1"
                    )
            else:
                num_passes = gr.Textbox(label="Number of rerank passes", value="1")
            qid = gr.Number(label="QID", value=1)

    if side_by_side:
        return [
            dataset,
            top_k_retrieve,
            top_k_rerank,
            host_retriever_a,
            host_retriever_b,
            host_reranker_a,
            host_reranker_b,
            num_passes_a,
            num_passes_b,
            qid,
        ]
    else:
        return [
            dataset,
            top_k_retrieve,
            top_k_rerank,
            host_retriever,
            host_reranker,
            num_passes,
            qid,
        ]


with gr.Blocks() as demo:
    gr.HTML(tooltip_style)
    gr.HTML(html_content)

    with gr.Tab("⚔️ ragnarok (side-by-side unblinded)"):
        with gr.Row():
            retriever_a, reranker_a, llm_a = rag_pipeline_block(label_suffix="A")
            retriever_b, reranker_b, llm_b = rag_pipeline_block(label_suffix="B")

        input_text, button = input_block()
        pretty_output_a, pretty_output_b, json_output_a, json_output_b = output_block(
            side_by_side=True
        )
        (
            a_better_button,
            b_better_button,
            both_good_button,
            both_bad_button,
        ) = comparison_block()

        (
            dataset,
            top_k_retrieve,
            top_k_rerank,
            host_retriever_a,
            host_retriever_b,
            host_reranker_a,
            host_reranker_b,
            num_passes_a,
            num_passes_b,
            qid,
        ) = parameters_block(side_by_side=True)

        button.click(
            on_submit_side_by_side,
            inputs=[
                llm_a,
                llm_b,
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
                input_text,
                num_passes_a,
                num_passes_b,
            ],
            outputs=[pretty_output_a, pretty_output_b, json_output_a, json_output_b],
        )

    with gr.Tab("⚔️ ragnarok (side-by-side blinded)"):
        with gr.Row():
            gr.Textbox(
                value="Unknown retriever + reranker + generation model",
                label="System A",
            )
            gr.Textbox(
                value="Unknown retriever + reranker + generation model",
                label="System B",
            )

        input_text, button = input_block()

        pretty_output_a, pretty_output_b, json_output_a, json_output_b = output_block(
            side_by_side=True
        )
        (
            a_better_button,
            b_better_button,
            both_good_button,
            both_bad_button,
        ) = comparison_block()

        (
            dataset,
            top_k_retrieve,
            top_k_rerank,
            host_retriever_a,
            host_retriever_b,
            host_reranker_a,
            host_reranker_b,
            num_passes_a,
            num_passes_b,
            qid,
        ) = parameters_block(side_by_side=True)

        button.click(
            on_submit_side_by_side_blinded,
            inputs=[
                dataset,
                host_retriever_a,
                host_reranker_a,
                host_retriever_b,
                host_reranker_b,
                top_k_retrieve,
                top_k_rerank,
                qid,
                input_text,
            ],
            outputs=[pretty_output_a, pretty_output_b, json_output_a, json_output_b],
        )

    with gr.Tab("💬 Direct Chat"):
        retriever, reranker, llm = rag_pipeline_block()

        input_text, button = input_block_direct()
        pretty_output, json_output = output_block(side_by_side=False)

        (
            dataset,
            top_k_retrieve,
            top_k_rerank,
            host_retriever,
            host_reranker,
            num_passes,
            qid,
        ) = parameters_block(side_by_side=False)

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
                return query_model(
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

        button.click(
            on_submit_single,
            inputs=[
                llm,
                retriever,
                reranker,
                dataset,
                host_retriever,
                host_reranker,
                top_k_retrieve,
                top_k_rerank,
                qid,
                input_text,
            ],
            outputs=[pretty_output, json_output],
        )

    with gr.Tab("🏆 Leaderboard"):
        html_content = """
        <div class='navbar'>
            <h2>Ragnarök Chatbot Arena Leaderboard</h2>
        </div>
        """
        gr.HTML(html_content)

demo.launch()
