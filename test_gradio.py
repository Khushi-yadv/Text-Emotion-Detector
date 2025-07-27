import gradio as gr

def greet(name):
    return f"Hello, {name}!"

demo = gr.Interface(
    fn=greet,
    inputs="text",
    outputs="text",
    title="Text emotion detection",
    description="Created by Khushi"
)

demo.launch(share=True)
