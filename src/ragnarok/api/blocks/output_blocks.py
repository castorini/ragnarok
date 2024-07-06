import gradio as gr


def output_block(side_by_side=True):
    with gr.Tab("Output"):
        with gr.Row():
            if side_by_side:
                pretty_output_a = gr.HTML(label="Pretty Output from Model A")
                pretty_output_b = gr.HTML(label="Pretty Output from Model B")
            else:
                pretty_output = gr.HTML(label="Pretty Output")
    with gr.Tab("Responses"):
        with gr.Row():
            if side_by_side:
                json_output_a = gr.JSON(label="JSON Output from Model A")
                json_output_b = gr.JSON(label="JSON Output from Model B")
            else:
                json_output = gr.JSON(label="JSON Output")

    if side_by_side:
        return [pretty_output_a, pretty_output_b, json_output_a, json_output_b]
    else:
        return [pretty_output, json_output]
