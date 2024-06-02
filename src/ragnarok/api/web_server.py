import gradio as gr
from ragnarok import retrieve_and_generate

citation_texts = {
    0: "Source 0 text: Details about the Second World War.",
    1: "Source 1 text: Discussion on the Treaty of Versailles.",
    2: "Source 2 text: Analysis of expansionism in the 20th century.",
    3: "Source 3 text: Germany's imperial ambitions.",
    4: "Source 4 text: Economic impacts of the Treaty of Versailles."
}

def generate_text_with_citations(response):
    output = []
    for sentence in response['answer']:
        text = sentence['text']
        citations = sentence['citations']
        if citations:
            citation_html = ' '.join([
                f'<span class="citation" title="{citation_texts[citation]}">[{citation}]</span>'
                for citation in citations
            ])
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
        return {"error": str(e)}

tooltip_style = """
<style>
.citation {
    position: relative;
    display: inline-block;
    cursor: pointer;
}

.citation:hover::after {
    content: attr(title);
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    background-color: #333;
    color: #fff;
    padding: 5px;
    border-radius: 5px;
    white-space: nowrap;
    z-index: 1;
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
            Reranker_A = gr.Dropdown(label="Reranker A", choices=["rank_zephyr", "rank_vicuna", "gpt_4o"], value="rank_zephyr")
            LLM_A = gr.Dropdown(label="LLM A", choices=["command-r", "command-r-plus"], value="command-r")
        with gr.Column():
            Retriever_B = gr.Dropdown(label="Retriever B", choices=["bm25"], value="bm25")
            Reranker_B = gr.Dropdown(label="Reranker B", choices=["rank_zephyr", "rank_vicuna", "gpt_4o"], value="rank_vicuna")
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
        resultA = query_model(retriever_a, reranker_a, model_a, dataset, host, host_reranker, top_k_retrieve, top_k_rerank, qid, query)
        # resultB = query_model(retriever_b, reranker_b, model_b, dataset, host, host_reranker, top_k_retrieve, top_k_rerank, qid, query)
        resultB = resultA
        return [resultA, resultB]
    button.click(
        on_submit, 
        inputs=[LLM_A, LLM_B, Retriever_A, Retriever_B, Reranker_A, Reranker_B, dataset, host, host_reranker, top_k_retrieve, top_k_rerank, qid, input_text],
        outputs=[output_a, output_b]
    )

demo.launch()