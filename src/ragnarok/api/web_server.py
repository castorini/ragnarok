import gradio as gr
import concurrent.futures
import os
import pandas as pd
from ragnarok import retrieve_and_generate

def generate_text_with_citations(response):
    output = []
    citation_texts = response.rag_exec_summary.candidates
    for sentence in response.answer:
        text = sentence.text
        citations = sentence.citations
        if citations:
            citation_html = ''
            for citation in citations:
                citation_title = citation_texts[citation]['doc']['title']
                citation_text = citation_texts[citation]['doc']['segment']
                citation_url = citation_texts[citation]['doc']['url']
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
            text += f' {citation_html}'
        output.append(text)
    return '<br/><br/>'.join(output)

def query_model(retriever_path,reranker_path, LLM, dataset, host_retriever, host_reranker, top_k_retrieve, top_k_rerank, qid, query):
    try:
        response = retrieve_and_generate.retrieve_and_generate(
            dataset=dataset,
            query=query,
            host_reranker=host_reranker,
            host_retriever=host_retriever,
            interactive=True, 
            k=[top_k_retrieve, top_k_rerank],
            qid=qid,
            reranker_path=reranker_path,
            retriever_path=retriever_path,
            LLM_path=LLM,
            use_azure_openai=True
        )
        output = generate_text_with_citations(response)
        result = {
            "topic_id": response.query.qid,
            "topic": response.query.text,
            "references": response.references,
            "response_length": sum(len(sentence.text) for sentence in response.answer),
            "answer": [{"text": sentence.text, "citations": sentence.citations} for sentence in response.answer]
        }
        return [output, result]
    except Exception as e:
        return ["ERROR: " + str(e), None]

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
    <h1>Ragnarok Chatbot Arena</h1>
    <p>Ask any question to two chosen RAG pipelines and compare the results</p>
</div>
"""

retriever_options = ["bm25"]
reranker_options = ["rank_zephyr", "rank_vicuna", "gpt-4o", "unspecified"]
llm_options = ["command-r", "command-r-plus", "gpt-4o", "gpt-35-turbo", "gpt-4"]

with gr.Blocks() as demo:
    gr.HTML(tooltip_style)
    gr.HTML(html_content)

    with gr.Tab("⚔️ ragnarok (side-by-side unblinded)"):
        with gr.Row():
            with gr.Column():
                Retriever_A = gr.Dropdown(label="Retriever A", choices=retriever_options, value="bm25")
                Reranker_A = gr.Dropdown(label="Reranker A", choices=reranker_options, value="rank_zephyr")
                LLM_A = gr.Dropdown(label="LLM A", choices=llm_options, value="command-r")
            with gr.Column():
                Retriever_B = gr.Dropdown(label="Retriever B", choices=retriever_options, value="bm25")
                Reranker_B = gr.Dropdown(label="Reranker B", choices=reranker_options, value="rank_vicuna")
                LLM_B = gr.Dropdown(label="LLM B", choices=llm_options, value="command-r-plus")

        with gr.Row():
            input_text = gr.Textbox(label="Enter your query and press ENTER", placeholder="Type here...", value="What caused the second world war?")
        with gr.Row():
            button = gr.Button("Compare")
        with gr.Tab("Output"):
            with gr.Row():
                output_a = gr.HTML(label="Output from Model A")
                output_b = gr.HTML(label="Output from Model B")
        with gr.Tab("Responses"):
            with gr.Row():
                response_a = gr.JSON(label="Response A")
                response_b = gr.JSON(label="Response B")
        with gr.Row():
            a_better_button = gr.Button("👈 A is better")
            b_better_button = gr.Button("👉 B is better")
            both_good_button = gr.Button("🤝 Tie")
            both_bad_button = gr.Button("👎 Both are bad")

        with gr.Accordion(label="Parameters", open=False):
            with gr.Column():
                dataset = gr.Dropdown(label="Dataset", choices=["msmarco-v2.1-doc-segmented"], value="msmarco-v2.1-doc-segmented")
                top_k_retrieve = gr.Number(label="Hits Retriever", value=40)
                top_k_rerank = gr.Number(label="Hits Reranker", value=20)
                with gr.Row():
                    host_retriever_a = gr.Textbox(label="Retriever Host A", value="8081")
                    host_retriever_b = gr.Textbox(label="Retriever Host B", value="8081")
                with gr.Row():
                    host_reranker_a = gr.Textbox(label="Reranker Host A", value="8082")
                    host_reranker_b = gr.Textbox(label="Reranker Host B", value="8083")
                qid = gr.Number(label="QID", value=1)

        def on_submit(model_a, model_b, retriever_a, retriever_b, reranker_a, reranker_b, dataset, host_retriever_a, host_reranker_a, host_retriever_b, host_reranker_b, top_k_retrieve, top_k_rerank, qid, query):
            def query_wrapper(retriever, reranker, model, host_retriever, host_reranker):
                return query_model(retriever, reranker, model, dataset, host_retriever, host_reranker, top_k_retrieve, top_k_rerank, qid, query)
            
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count())
            with executor:
                futureA = executor.submit(query_wrapper, retriever_a, reranker_a, model_a, host_retriever_a, host_reranker_a)
                futureB = executor.submit(query_wrapper, retriever_b, reranker_b, model_b, host_retriever_b, host_reranker_b)

                resultA, responseA = futureA.result()
                resultB, responseB = futureB.result()

            return [resultA, resultB, responseA, responseB]

        button.click(
            on_submit, 
            inputs=[LLM_A, LLM_B, Retriever_A, Retriever_B, Reranker_A, Reranker_B, dataset, host_retriever_a, host_reranker_a, host_retriever_b, host_reranker_b, top_k_retrieve, top_k_rerank, qid, input_text],
            outputs=[output_a, output_b, response_a, response_b]
        )

    with gr.Tab("⚔️ ragnarok (side-by-side blind)"):
        with gr.Row():
            gr.Textbox(value="Unknown retriever + reranker + generation model", label="System A")
            gr.Textbox(value="Unknown retriever + reranker + generation model", label="System B")
        with gr.Row():
            input_text = gr.Textbox(label="Enter your query and press ENTER", placeholder="Type here...", value="What caused the second world war?")
        with gr.Row():
            button = gr.Button("Compare")
        with gr.Tab("Output"):
            with gr.Row():
                output_a = gr.HTML(label="Output from Model A")
                output_b = gr.HTML(label="Output from Model B")
        with gr.Tab("Responses"):
            with gr.Row():
                response_a = gr.JSON(label="Response A")
                response_b = gr.JSON(label="Response B")
        with gr.Row():
            a_better_button = gr.Button("👈 A is better")
            b_better_button = gr.Button("👉 B is better")
            both_good_button = gr.Button("🤝 Tie")
            both_bad_button = gr.Button("👎 Both are bad")

        with gr.Accordion(label="Parameters", open=False):
            with gr.Column():
                dataset = gr.Dropdown(label="Dataset", choices=["msmarco-v2.1-doc-segmented"], value="msmarco-v2.1-doc-segmented")
                top_k_retrieve = gr.Number(label="Hits Retriever", value=40)
                top_k_rerank = gr.Number(label="Hits Reranker", value=20)
                with gr.Row():
                    host_retriever_a = gr.Textbox(label="Retriever Host A", value="8081")
                    host_retriever_b = gr.Textbox(label="Retriever Host B", value="8081")
                with gr.Row():
                    host_reranker_a = gr.Textbox(label="Reranker Host A", value="8082")
                    host_reranker_b = gr.Textbox(label="Reranker Host B", value="8083")
                qid = gr.Number(label="QID", value=1)

        def on_submit(dataset, host_retriever_a, host_reranker_a, host_retriever_b, host_reranker_b, top_k_retrieve, top_k_rerank, qid, query):
            def query_wrapper(retriever, reranker, model, host_retriever, host_reranker):
                return query_model(retriever, reranker, model, dataset, host_retriever, host_reranker, top_k_retrieve, top_k_rerank, qid, query)
            
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count())
            with executor:
                import random
                retriever_a = random.choice(retriever_options)
                reranker_a = random.choice(reranker_options)
                model_a = random.choice(llm_options)
                retriever_b = random.choice(retriever_options)
                reranker_b = random.choice(reranker_options)
                model_b = random.choice(llm_options)

                futureA = executor.submit(query_wrapper, retriever_a, reranker_a, model_a, host_retriever_a, host_reranker_a)
                futureB = executor.submit(query_wrapper, retriever_b, reranker_b, model_b, host_retriever_b, host_reranker_b)

                resultA, responseA = futureA.result()
                resultB, responseB = futureB.result()

            return [resultA, resultB, responseA, responseB]

        button.click(
            on_submit, 
            inputs=[dataset, host_retriever_a, host_reranker_a, host_retriever_b, host_reranker_b, top_k_retrieve, top_k_rerank, qid, input_text],
            outputs=[output_a, output_b, response_a, response_b]
        )

    with gr.Tab("💬 Direct Chat"):
        with gr.Column():
            Retriever = gr.Dropdown(label="Retriever", choices=retriever_options, value="bm25")
            Reranker = gr.Dropdown(label="Reranker", choices=reranker_options, value="rank_zephyr")
            LLM = gr.Dropdown(label="LLM", choices=llm_options, value="command-r")

        with gr.Row():
            input_text = gr.Textbox(label="Enter your query and press ENTER", placeholder="Type here...", value="What caused the second world war?")
        with gr.Row():
            button = gr.Button("Compare")
        with gr.Tab("Output"):
            output = gr.HTML(label="Output")
        with gr.Tab("Responses"):
            response = gr.JSON(label="Response")

        with gr.Accordion(label="Parameters", open=False):
            with gr.Column():
                dataset = gr.Dropdown(label="Dataset", choices=["msmarco-v2.1-doc-segmented"], value="msmarco-v2.1-doc-segmented")
                top_k_retrieve = gr.Number(label="Hits Retriever", value=40)
                top_k_rerank = gr.Number(label="Hits Reranker", value=20)
                host_retriever = gr.Textbox(label="Retriever Host A", value="8081")
                host_reranker = gr.Textbox(label="Reranker Host A", value="8082")
                qid = gr.Number(label="QID", value=1)

        def on_submit(model, retriever, reranker, dataset, host_retriever, host_reranker, top_k_retrieve, top_k_rerank, qid, query):
            def query_wrapper(retriever, reranker, model, host_retriever, host_reranker):
                return query_model(retriever, reranker, model, dataset, host_retriever, host_reranker, top_k_retrieve, top_k_rerank, qid, query)
            
            result, response = query_wrapper(retriever, reranker, model, host_retriever, host_reranker)

            return [result, response]

        button.click(
            on_submit,
            inputs=[LLM, Retriever, Reranker, dataset, host_retriever, host_reranker, top_k_retrieve, top_k_rerank, qid, input_text],
            outputs=[output, response]
        )

    with gr.Tab("🏆 Leaderboard"):
        html_content = """
        <div class='navbar'>
            <h2>Ragnarok Chatbot Arena Leaderboard</h2>
        </div>
        """
        gr.HTML(html_content)

demo.launch()