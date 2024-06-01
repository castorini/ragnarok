import gradio as gr
from ragnarok import retrieve_and_generate

def query_model(retriever_path,reranker_path, LLM, dataset, host_retriever, host_reranker, top_k_retrieve, top_k_rerank, qid, query):
    try:
        response = retrieve_and_generate.retrieve_and_generate(
            dataset=dataset,
            query=query,
            model_path=LLM,
            host_reranker="http://localhost:" + host_reranker,
            host_retriever="http://localhost:" + host_retriever,
            interactive=True, 
            k=[top_k_retrieve, top_k_rerank],
            qid=qid,
            reranker_path=reranker_path,
            retriever_path=retriever_path,
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

html_content = """
<style>
    .navbar {
        background-color: #333; /* Dark background color */
        color: white;
        padding: 10px 20px;
        font-family: Arial, sans-serif;
    }
    .navbar h2 {
        margin: 0;
        padding-bottom: 8px;
    }
    .navbar a {
        color: #f0db4f; /* Gold color for links */
        text-decoration: none;
        margin-right: 20px;
        font-weight: bold;
    }
    .navbar a:hover {
        text-decoration: underline;
    }
    .navbar ul {
        list-style-type: none;
        padding: 0;
    }
    .navbar li {
        display: inline;
    }
</style>

<div class='navbar'>
    <h2>LMSYS Chatbot Arena: Benchmarking LLMs in the Wild</h2>

    <p>Rules</p>
    <ul>
        <li>Ask any question to two chosen models (e.g., ChatGPT, Claude, Llama) and vote for the better one!</li>
        <li>You can chat for multiple turns until you identify a winner.</li>
    </ul>
</div>
"""


with gr.Blocks() as demo:
    gr.HTML(html_content)
    with gr.Row():
        with gr.Column():
            Retriever_A = gr.Dropdown(label="Retriever A", choices=["bm25"])
            Reranker_A = gr.Dropdown(label="Reranker A", choices=["RankZephyr", "RankVicuna", "RankGPT4o"])
            LLM_A = gr.Dropdown(label="LLM A", choices=["commandR", "commandRPlus"])
        with gr.Column():
            Retriever_B = gr.Dropdown(label="Retriever B", choices=["bm25"])
            Reranker_B = gr.Dropdown(label="Reranker B", choices=["RankZephyr", "RankVicuna", "RankGPT4o"])
            LLM_B = gr.Dropdown(label="LLM B", choices=["commandR", "commandRPlus"])

    with gr.Row():
        query = gr.Textbox(label="Enter your prompt and press ENTER", placeholder="Type here...")
    with gr.Row():
        button = gr.Button("Compare")
    with gr.Row():
        output_a = gr.Textbox(label="Output from Model A")
        output_b = gr.Textbox(label="Output from Model B")

    with gr.Accordion(label="Parameters", open=False):
        with gr.Row():
            with gr.Column():
                dataset = gr.Textbox(label="Dataset", value="msmarco-v2.1-doc-segmented")
                host_retriever = gr.Textbox(label="Retriever Host", value="8081")
                host_reranker = gr.Textbox(label="Reranker Host", value="8082")
                top_k_retrieve = gr.Number(label="Hits Retriever", value=40)
                top_k_rerank = gr.Number(label="Hits Reranker", value=40)
                qid = gr.Number(label="QID", value=1)

    def on_submit(Retriever_A, Reranker_A, LLM_A, Retriever_B, Reranker_B, LLM_B, dataset, host, host_reranker, top_k_retrieve, top_k_rerank, qid, query):
        

        resultA = query_model(Retriever_A,Reranker_A, LLM_A, dataset, host, host_reranker, top_k_retrieve, top_k_rerank, qid, query)
        resultB = query_model(Retriever_B, Reranker_B, LLM_B, dataset, host, host_reranker, top_k_retrieve, top_k_rerank, qid, query)

        if "error" in resultA:
            return f"<pre>{resultA['error']}</pre>"
        else:
            highlighted_result = highlight_json(resultA)
            query_response_html = f"""
            <div style='width: 100%; padding: 10px;'>
                <h3>Response</h3>
                {highlighted_result}
            </div>
            """
            return query_response_html
        
    button.click(
        on_submit, 
        inputs=[Retriever_A, Reranker_A, LLM_A, Retriever_B, Reranker_B, LLM_B, dataset, host_retriever, host_reranker, top_k_retrieve, top_k_rerank, qid, query],
        outputs=result_output
    )

demo.launch()