import gradio as gr
from ragnarok import retrieve_and_generate

def query_model(model_path, dataset, host, host_reranker, top_k_retrieve, top_k_rerank, qid, query):
    try:
        response = retrieve_and_generate.retrieve_and_generate(
            dataset=dataset,
            query=query,
            model_path=model_path,
            host="http://localhost:" + host_reranker,
            interactive=True, 
            k=[top_k_retrieve, top_k_rerank],
            qid=qid,
        )
        result = {
            "topic_id": response.query.qid,
            "topic": response.query.text,
            "references": response.references,
            "response_length": sum(len(sentence["text"]) for sentence in response.answer),
            "answer": [{"text": sentence["text"], "citations": sentence["citations"]} for sentence in response.answer]
        }
        return result
    except Exception as e:
        return {"error": str(e)}

def highlight_json(json_data):
    import json
    from pygments import highlight, lexers, formatters
    formatter = formatters.HtmlFormatter(style="colorful", full=True)
    json_str = json.dumps(json_data, indent=2)
    result = highlight(json_str, lexers.JsonLexer(), formatter)
    return result


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            Retriever_A = gr.Dropdown(label="Retriever A", choices=["BM25"])
            Reranker_A = gr.Dropdown(label="Reranker A", choices=["RankZephyr", "RankVicuna", "RankGPT4o"])
            LLM_A = gr.Dropdown(label="LLM A", choices=["commandR", "commandRPlus"])
        with gr.Column():
            Retriever_B = gr.Dropdown(label="Retriever B", choices=["BM25"])
            Reranker_B = gr.Dropdown(label="Reranker B", choices=["RankZephyr", "RankVicuna", "RankGPT4o"])
            LLM_B = gr.Dropdown(label="LLM B", choices=["commandR", "commandRPlus"])

    with gr.Row():
        input_text = gr.Textbox(label="Enter your prompt and press ENTER", placeholder="Type here...")
    with gr.Row():
        button = gr.Button("Compare")
    with gr.Row():
        output_a = gr.Textbox(label="Output from Model A")
        output_b = gr.Textbox(label="Output from Model B")

    button.click(inputs=[LLM_A, LLM_B, input_text], outputs=[output_a, output_b])

    with gr.Accordion(label="Parameters", open=False):
        with gr.Row():
            with gr.Column():
                dataset_a = gr.Textbox(label="Dataset", value="msmarco-v2.1-doc-segmented")
                host_a = gr.Textbox(label="Retriever Host", value="8081")
                host_reranker_a = gr.Textbox(label="Reranker Host", value="8082")
                top_k_retrieve_a = gr.Number(label="Hits Retriever", value=40)
                top_k_rerank_a = gr.Number(label="Hits Reranker", value=40)
                qid_a = gr.Number(label="QID", value=1)
            with gr.Column():
                dataset_b = gr.Textbox(label="Dataset", value="msmarco-v2.1-doc-segmented")
                host_b = gr.Textbox(label="Retriever Host", value="8081")
                host_reranker_b = gr.Textbox(label="Reranker Host", value="8082")
                top_k_retrieve_b = gr.Number(label="Hits Retriever", value=40)
                top_k_rerank_b = gr.Number(label="Hits Reranker", value=40)
                qid_b = gr.Number(label="QID", value=1)

    def on_submit(model_path, dataset, host, host_reranker, top_k_retrieve, top_k_rerank, qid, query):
        result = query_model(model_path, dataset, host, host_reranker, top_k_retrieve, top_k_rerank, qid, query)
        if "error" in result:
            return f"<pre>{result['error']}</pre>"
        else:
            highlighted_result = highlight_json(result)
            query_response_html = f"""
            <div style='width: 100%; padding: 10px;'>
                <h3>Response</h3>
                {highlighted_result}
            </div>
            """
            return query_response_html

    submit_btn.click(
        on_submit, 
        inputs=[model_path, dataset, host, host_reranker, top_k_retrieve, top_k_rerank, qid, query], 
        outputs=result_output
    )

demo.launch()