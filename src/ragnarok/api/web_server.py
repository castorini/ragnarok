import gradio as gr
import concurrent.futures
import os
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
            LLM_path=LLM
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
    <h2>Ragnarok Chatbot Arena</h2>
    <p>Ask any question to two chosen RAG pipelines and compare the results</p>
</div>
"""

with gr.Blocks() as demo:
    gr.HTML(tooltip_style)
    gr.HTML(html_content)
    with gr.Row():
        with gr.Column():
            Retriever_A = gr.Dropdown(label="Retriever A", choices=["bm25"], value="bm25")
            Reranker_A = gr.Dropdown(label="Reranker A", choices=["rank_zephyr", "rank_vicuna", "gpt-4o", "unspecified"], value="rank_zephyr")
            LLM_A = gr.Dropdown(label="LLM A", choices=["command-r", "command-r-plus"], value="command-r")
        with gr.Column():
            Retriever_B = gr.Dropdown(label="Retriever B", choices=["bm25"], value="bm25")
            Reranker_B = gr.Dropdown(label="Reranker B", choices=["rank_zephyr", "rank_vicuna", "gpt-4o", "unspecified"], value="rank_vicuna")
            LLM_B = gr.Dropdown(label="LLM B", choices=["command-r", "command-r-plus"], value="command-r")

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
                host_reranker_b = gr.Textbox(label="Reranker Host B", value="8082")
            qid = gr.Number(label="QID", value=1)

    def on_submit(model_a, model_b, retriever_a, retriever_b, reranker_a, reranker_b, dataset, host_retriever_a, host_reranker_a, host_retriever_b, host_reranker_b, top_k_retrieve, top_k_rerank, qid, query):
        def query_wrapper(retriever, reranker, model, host_retriever, host_reranker):
            return query_model(retriever, reranker, model, dataset, host_retriever, host_reranker, top_k_retrieve, top_k_rerank, qid, query)
        
        # results = []
        # executor = concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count())
        # with executor:
        #     future_a = executor.submit(query_wrapper, retriever_a, reranker_a, model_a)
        #     future_b = executor.submit(query_wrapper, retriever_b, reranker_b, model_b)
            
        #     for future in concurrent.futures.as_completed([future_a, future_b]):
        #         results.append(future.result())
        
        # return results

        [resultA, responseA] = query_wrapper(retriever_a, reranker_a, model_a, host_retriever_a, host_reranker_a)
        [resultB, responseB] = query_wrapper(retriever_b, reranker_b, model_b, host_retriever_b, host_reranker_b)

        return [resultA, resultB, responseA, responseB]

    button.click(
        on_submit, 
        inputs=[LLM_A, LLM_B, Retriever_A, Retriever_B, Reranker_A, Reranker_B, dataset, host_retriever_a, host_reranker_a, host_retriever_b, host_reranker_b, top_k_retrieve, top_k_rerank, qid, input_text],
        outputs=[output_a, output_b, response_a, response_b]
    )

demo.launch()