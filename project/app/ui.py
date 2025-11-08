import gradio as gr
from typing import List
from typing import Callable

def build_ui(respond_fn: Callable,title: str = "Student's Companion (History 6–12)"):
    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}")        
        mode = gr.Radio(["Summarize", "Bullet Notes", "Quiz", "Free Chat"], value="Summarize", label="Task")
        grade = gr.Slider(6, 12, value=9, step=1, label="Grade")
        files = gr.Files(label="Upload extra material (PDFs)")
        chat = gr.Chatbot(height=400)
        txt = gr.Textbox(label="Ask a question or enter a topic")
        clear = gr.Button("Clear Chat")
        txt.submit(respond_fn, [txt, chat, mode, grade, files], [chat, txt])

        def _clear():
            return [], ""
        clear.click(_clear, outputs=[chat, txt])

        return demo
