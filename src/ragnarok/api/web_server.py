import gradio as gr
import concurrent.futures
import os
from ragnarok import retrieve_and_generate

def generate_text_with_citations(response):
    output = []
    citation_texts = response['rag_exec_summary']['candidates']    
    for sentence in response['answer']:
        text = sentence['text']
        
        citations = sentence['citations']
        if citations:
            citation_html = ''
            for citation in citations:
                citation_title = citation_texts[citation]['doc']['title']
                citation_text = citation_texts[citation]['doc']['segment']
                citation_html += f'<span class="citation" text="{citation_title}: {citation_text}">[{citation}]</span>' + ' '
            text += f' {citation_html}'
        output.append('<p>'+text+'</p>')
    return '<br/>'.join(output)

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
        result = generate_text_with_citations(response)
        return result
    except Exception as e:
        return f"ERROR: {str(e)}"

tooltip_style = """
<style>
.citation {
    position: relative;
    display: inline-block;
    cursor: pointer;
}

.citation:hover::after {
    content: attr(text);
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    width: 300px;
    color: #fff;
    background-color: rgba(0, 0, 0, 0.9);
    padding: 5px;
    border-radius: 5px;
    border: 1px solid rgba(255, 255, 255, 0.2);
    white-space: normal;
    word-wrap: break-word;
    overflow-wrap: break-word;
    z-index: 50;
    box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
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
            Reranker_A = gr.Dropdown(label="Reranker A", choices=["rank_zephyr", "rank_vicuna", "gpt_4o", "none"], value="rank_zephyr")
            LLM_A = gr.Dropdown(label="LLM A", choices=["command-r", "command-r-plus"], value="command-r")
        with gr.Column():
            Retriever_B = gr.Dropdown(label="Retriever B", choices=["bm25"], value="bm25")
            Reranker_B = gr.Dropdown(label="Reranker B", choices=["rank_zephyr", "rank_vicuna", "gpt_4o", "none"], value="rank_vicuna")
            LLM_B = gr.Dropdown(label="LLM B", choices=["command-r", "command-r-plus"], value="command-r")

    with gr.Row():
        input_text = gr.Textbox(label="Enter your query and press ENTER", placeholder="Type here...", value="What caused the second world war?")
    with gr.Row():
        button = gr.Button("Compare")
    with gr.Row():
        output_a = gr.HTML(label="Output from Model A")
        output_b = gr.HTML(label="Output from Model B")

    with gr.Accordion(label="Parameters", open=False):
        with gr.Row():
            with gr.Column():
                dataset = gr.Dropdown(label="Dataset", choices=["msmarco-v2.1-doc-segmented"], value="msmarco-v2.1-doc-segmented")
                host = gr.Textbox(label="Retriever Host", value="8081")
                host_reranker = gr.Textbox(label="Reranker Host", value="8082")
                top_k_retrieve = gr.Number(label="Hits Retriever", value=10)
                top_k_rerank = gr.Number(label="Hits Reranker", value=5)
                qid = gr.Number(label="QID", value=1)

    def on_submit(model_a, model_b, retriever_a, retriever_b, reranker_a, reranker_b, dataset, host, host_reranker, top_k_retrieve, top_k_rerank, qid, query):
        def query_wrapper(retriever, reranker, model):
            return query_model(retriever, reranker, model, dataset, host, host_reranker, top_k_retrieve, top_k_rerank, qid, query)
        
        # results = []
        # executor = concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count())
        # with executor:
        #     future_a = executor.submit(query_wrapper, retriever_a, reranker_a, model_a)
        #     future_b = executor.submit(query_wrapper, retriever_b, reranker_b, model_b)
            
        #     for future in concurrent.futures.as_completed([future_a, future_b]):
        #         results.append(future.result())
        
        # return results

        resultA = query_wrapper(retriever_a, reranker_a, model_a)
        resultB = query_wrapper(retriever_b, reranker_b, model_b)

        return [resultA,resultB]
    
    button.click(
        on_submit, 
        inputs=[LLM_A, LLM_B, Retriever_A, Retriever_B, Reranker_A, Reranker_B, dataset, host, host_reranker, top_k_retrieve, top_k_rerank, qid, input_text],
        outputs=[output_a, output_b]
    )

demo.launch()