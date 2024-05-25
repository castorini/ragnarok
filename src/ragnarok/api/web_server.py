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
            query = gr.Textbox(label="Query", value="What caused the second world war")
            submit_btn = gr.Button("Submit")
        with gr.Column():
            result_output = gr.HTML()

    with gr.Accordion(label="Parameters", open=False):
        with gr.Row():
            with gr.Column():
                model_path = gr.Textbox(label="Model Path", value="command-r-plus")
                dataset = gr.Textbox(label="Dataset", value="msmarco-v2.1-doc-segmented")
                host = gr.Textbox(label="Retriever Host", value="8081")
                host_reranker = gr.Textbox(label="Reranker Host", value="8082")
                top_k_retrieve = gr.Number(label="Hits Retriever", value=40)
                top_k_rerank = gr.Number(label="Hits Reranker", value=40)
                qid = gr.Number(label="QID", value=1)

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