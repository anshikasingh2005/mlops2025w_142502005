import gradio as gr
from typing import Callable
from quiz.qn_gen import run_tutor_with_rag   # ✅ adjust import if needed
from quiz.evaluation import evaluate           # ✅ adjust import if needed


def build_ui(respond_fn: Callable, title: str = "Student's Companion (History 6–12)"):
    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}")

        with gr.Tabs():

            # ================= ORIGINAL CHAT TAB (UNCHANGED) =================
            with gr.Tab("Chat / Study Assistant"):
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

            # ================= ✅ NEW QUIZ TAB =================
            with gr.Tab("Practice Q/A"):
                gr.Markdown("Question and Answer")

                topic = gr.Textbox(label="Enter Topic / Chapter Name", placeholder="e.g., Machine Learning")
                question_box = gr.Textbox(label="Generated Question", interactive=False)

                user_answer_box = gr.Textbox(label="Your Answer")
                feedback_box = gr.Markdown()

                score_state = gr.State(value=0)
                total_state = gr.State(value=0)
                score_display = gr.Markdown("**Score: 0 / 0**")

                generate_btn = gr.Button("Generate Question")
                submit_btn = gr.Button("Submit Answer")

                # --- Generate Question from topic ---
                def handle_generate(topic, score, total):
                    if not topic.strip():
                        return "⚠️ Enter a topic first.", score, total
                    correct_ans, question = run_tutor_with_rag(topic)
                    # Store hidden correct answer + expected student answer
                    return question, score, total, correct_ans

                correct_answer_state = gr.State("")
                expected_user_state = gr.State("")

                generate_btn.click(
                    handle_generate,
                    [topic, score_state, total_state],
                    [question_box, score_state, total_state, correct_answer_state]
                )

                # --- Submit Answer ---
                def handle_submit(user_answer, correct_answer, score, total):
                    total += 1
                    similarity, feedback = evaluate(user_answer, correct_answer)
                    score=similarity*100
                    feedback_msg = f"### Feedback:\n{feedback}\n\n**Correct Answer:**\n{correct_answer}"
                    score_msg = f"**Score: {score} **"
                    return feedback_msg, score, total, score_msg

                submit_btn.click(
                    handle_submit,
                    [user_answer_box, correct_answer_state, score_state, total_state],
                    [feedback_box, score_state, total_state, score_display]
                )

    return demo