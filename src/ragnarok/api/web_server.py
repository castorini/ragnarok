import gradio as gr
from ragnarok import retrieve_and_generate

def query_model(retriever_path,reranker_path, LLM, dataset, host_retriever, host_reranker, top_k_retrieve, top_k_rerank, qid, query):
    try:
        response = retrieve_and_generate.retrieve_and_generate(
            dataset=dataset,
            query=query,
            model_path=LLM,
            host_reranker=host_reranker,
            host_retriever=host_retriever,
            interactive=True, 
            k=[top_k_retrieve, top_k_rerank],
            qid=qid,
            reranker_path=reranker_path,
            retriever_path=retriever_path,
        )
        return response
    except Exception as e:
        return {"error": str(e)}

def format_json(json_data):
    import json
    return json.dumps(json_data, indent=2)

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
            Retriever_A = gr.Dropdown(label="Retriever A", choices=["BM25"], value="BM25")
            Reranker_A = gr.Dropdown(label="Reranker A", choices=["RankZephyr", "RankVicuna", "RankGPT4o"], value="RankZephyr")
            LLM_A = gr.Dropdown(label="LLM A", choices=["command-r", "command-r-plus"], value="command-r")
        with gr.Column():
            Retriever_B = gr.Dropdown(label="Retriever B", choices=["BM25"], value="BM25")
            Reranker_B = gr.Dropdown(label="Reranker B", choices=["RankZephyr", "RankVicuna", "RankGPT4o"], value="RankVicuna")
            LLM_B = gr.Dropdown(label="LLM B", choices=["command-r", "command-r-plus"], value="command-r")

    with gr.Row():
        input_text = gr.Textbox(label="Enter your query and press ENTER", placeholder="Type here...", value="What caused the second world war?")
    with gr.Row():
        button = gr.Button("Compare")
    with gr.Row():
        output_a = gr.JSON(label="Output from Model A")
        output_b = gr.JSON(label="Output from Model B")
        # output_a = gr.Textbox(label="Output from Model A")
        # output_b = gr.Textbox(label="Output from Model B")

    # button.click(inputs=[LLM_A, LLM_B, input_text], outputs=[output_a, output_b])
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
        result = query_model(model_a, dataset, host, host_reranker, top_k_retrieve, top_k_rerank, qid, query)
        result = query_model(model_b, dataset, host, host_reranker, top_k_retrieve, top_k_rerank, qid, query)
        return [result, result]
        
    button.click(
        on_submit, 
        inputs=[LLM_A, LLM_B, Retriever_A, Retriever_B, Reranker_A, Reranker_B, dataset, host, host_reranker, top_k_retrieve, top_k_rerank, qid, input_text],
        outputs=[output_a, output_b]
    )

demo.launch()