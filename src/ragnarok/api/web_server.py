import gradio as gr

import ragnarok.api.blocks.html_blocks as html_blocks
import ragnarok.api.blocks.input_blocks as input_blocks
import ragnarok.api.blocks.on_submit_blocks as on_submit_blocks
import ragnarok.api.blocks.output_blocks as output_blocks
import ragnarok.api.elo as elo

(retriever_options, reranker_options, llm_options) = (
    input_blocks.retriever_options,
    input_blocks.reranker_options,
    input_blocks.llm_options
    + ["meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-70B-Instruct"],
)

with gr.Blocks() as demo:
    gr.HTML(html_blocks.tooltip_style)
    gr.HTML(html_blocks.html_content)

    with gr.Tab("⚔️ ragnarok (side-by-side unblinded)"):
        with gr.Row():
            retriever_a, reranker_a, llm_a = input_blocks.rag_pipeline_block(
                label_suffix="A"
            )
            retriever_b, reranker_b, llm_b = input_blocks.rag_pipeline_block(
                label_suffix="B"
            )

        input_text, button = input_blocks.input_block()
        (
            pretty_output_a,
            pretty_output_b,
            json_output_a,
            json_output_b,
        ) = output_blocks.output_block(side_by_side=True)
        (
            answer_a,
            answer_tie,
            answer_b,
            evidence_a,
            evidence_tie,
            evidence_b,
        ) = input_blocks.comparison_block()

        answer_a.click(
            elo.handle_battle_answer_a,
            inputs=[llm_a, llm_b, retriever_a, retriever_b, reranker_a, reranker_b],
            outputs=[],
        )
        answer_tie.click(
            elo.handle_battle_answer_tie,
            inputs=[llm_a, llm_b, retriever_a, retriever_b, reranker_a, reranker_b],
            outputs=[],
        )
        answer_b.click(
            elo.handle_battle_answer_b,
            inputs=[llm_a, llm_b, retriever_a, retriever_b, reranker_a, reranker_b],
        )
        evidence_a.click(
            elo.handle_battle_evidence_a,
            inputs=[llm_a, llm_b, retriever_a, retriever_b, reranker_a, reranker_b],
            outputs=[],
        )
        evidence_tie.click(
            elo.handle_battle_evidence_tie,
            inputs=[llm_a, llm_b, retriever_a, retriever_b, reranker_a, reranker_b],
            outputs=[],
        )
        evidence_b.click(
            elo.handle_battle_evidence_b,
            inputs=[llm_a, llm_b, retriever_a, retriever_b, reranker_a, reranker_b],
            outputs=[],
        )

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
        ) = input_blocks.parameters_block(side_by_side=True)

        button.click(
            on_submit_blocks.on_submit_side_by_side,
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

        input_text, button = input_blocks.input_block()

        (
            pretty_output_a,
            pretty_output_b,
            json_output_a,
            json_output_b,
        ) = output_blocks.output_block(side_by_side=True)

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
        ) = input_blocks.parameters_block(side_by_side=True)

        button.click(
            on_submit_blocks.on_submit_side_by_side_blinded,
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
                num_passes_a,
                num_passes_b,
            ],
            outputs=[pretty_output_a, pretty_output_b, json_output_a, json_output_b],
        )

    with gr.Tab("💬 Direct Chat"):
        retriever, reranker, llm = input_blocks.rag_pipeline_block()

        input_text, button = input_blocks.input_block_direct()
        pretty_output, json_output = output_blocks.output_block(side_by_side=False)

        (
            dataset,
            top_k_retrieve,
            top_k_rerank,
            host_retriever,
            host_reranker,
            num_passes,
            qid,
        ) = input_blocks.parameters_block(side_by_side=False)

        button.click(
            on_submit_blocks.on_submit_single,
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
        gr.Markdown("## Ragnarök Chatbot Arena Leaderboard")

        llm_scoreboard, retrieve_scoreboard, rag_scoreboard = elo.elo_table_block()

        refresh_btn = gr.Button("Refresh Scoreboards", size="sm")
        refresh_btn.click(
            elo.elo_table_block,
            inputs=[],
            outputs=[llm_scoreboard, retrieve_scoreboard, rag_scoreboard],
        )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
