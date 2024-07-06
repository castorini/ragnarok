import gradio as gr

# RAG pipeline options
retriever_options = ["bm25"]
reranker_options = ["rank_zephyr", "rank_vicuna", "gpt-4o", "unspecified"]
llm_options = ["command-r", "command-r-plus", "gpt-4o", "gpt-35-turbo", "gpt-4"]


def comparison_block():
    gr.Markdown("## Answer")
    with gr.Row():
        answer_a = gr.Button("👈 A is better")
        answer_tie = gr.Button("🤝 Tie")
        answer_b = gr.Button("👉 B is better")
    gr.Markdown("## Evidence")
    with gr.Row():
        evidence_a = gr.Button("👈 A is better")
        evidence_tie = gr.Button("🤝 Tie")
        evidence_b = gr.Button("👉 B is better")

    return [answer_a, answer_tie, answer_b, evidence_a, evidence_tie, evidence_b]


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
