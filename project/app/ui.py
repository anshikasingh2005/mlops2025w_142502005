import gradio as gr
from typing import Callable, List, Dict

import sys
import os



# Get the absolute path of the directory where this file (ui.py) is located
app_dir = os.path.dirname(os.path.abspath(__file__))

# Get the path of the parent directory of 'app', which is 'project'
project_dir = os.path.dirname(app_dir)

# Add the 'project' directory to the Python path so it can find the 'quiz' module
sys.path.append(project_dir)

# Now you can import your functions
from quiz.evaluation import evaluate
from quiz.qn_gen import run_tutor_with_rag

# # Mock functions for demonstration since the actual implementations are not provided.
# # You should replace these with your actual 'quiz' module imports.
# def evaluate(user_answer, correct_answer):
#     """Mock evaluation function."""
#     # In a real scenario, this would compute a similarity score.
#     similarity = len(set(user_answer.lower().split()) & set(correct_answer.lower().split())) / len(set(correct_answer.lower().split()))
#     return f"{similarity:.2f}"

# def run_tutor_with_rag(topic: str):
#     """Mock RAG function."""
#     # In a real scenario, this would generate a question and answer based on the topic.
#     refined_question = f"What is the significance of the element with the symbol '{topic.upper()}' in organic chemistry?"
#     refined_correct_answer = f"{topic.capitalize()} is the fundamental element for life and the basis of organic chemistry, forming stable bonds with many other elements."
#     user_answer = "" # This would be captured from the user.
#     return user_answer, refined_correct_answer, refined_question

# Get the absolute path of the directory where this file (ui.py) is located
# app_dir = os.path.dirname(os.path.abspath(__file__))

# # Get the path of the parent directory of 'app', which is 'project'
# project_dir = os.path.dirname(app_dir)

# # Add the 'project' directory to the Python path so it can find the 'quiz' module
# sys.path.append(project_dir)



def generate_single_question(topic: str) -> (str, str):
    """
    Generates a single question and its correct answer based on a topic.
    """
    _, refined_correct_answer, refined_question = run_tutor_with_rag(topic)
    return refined_question, refined_correct_answer


def build_ui(respond_fn: Callable = None, title: str = "Student's Companion (History 6–12)"):
    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}")

        mode = gr.Radio(["Summarize", "Bullet Notes", "Quiz", "Free Chat"], value="Summarize", label="Task")
        grade = gr.Slider(6, 12, value=9, step=1, label="Grade")
        files = gr.Files(label="Upload extra material (PDFs)")

        # State to hold the correct answer for the current quiz question
        correct_answer_state = gr.State("")

        chat = gr.Chatbot(height=350)
        txt = gr.Textbox(label="Ask a question or enter a topic for the quiz")
        clear = gr.Button("Clear Chat")

        # --- Quiz UI Section ---
        quiz_box = gr.Group(visible=False)
        with quiz_box:
            gr.Markdown("### Quiz Time!")
            quiz_question_display = gr.Markdown("")
            user_answer_input = gr.Textbox(label="Your Answer", lines=4)
            submit_answer_button = gr.Button("Submit Answer")
            quiz_feedback_display = gr.Markdown("")
        # --- End of Quiz UI Section ---

        def handle_main_submit(text, chat_history, mode_selected, grade_value, uploaded_files):
            """
            Handles submission from the main textbox.
            If mode is 'Quiz', it generates a question.
            Otherwise, it calls the original respond_fn for chat/summarization.
            """
            if mode_selected != "Quiz":
                if respond_fn: # Check if a chat function was provided
                    response = respond_fn(text, chat_history, mode_selected, grade_value, uploaded_files)
                    return response, "", gr.update(visible=False), "", "", ""
                else: # Fallback if no chat function is available
                    chat_history.append((text, "Chat function not implemented."))
                    return chat_history, "", gr.update(visible=False), "", "", ""

            # --- Quiz Mode Logic ---
            # Generate a new question and store its correct answer
            question, correct_answer = generate_single_question(text)
            
            # Update the UI to show the question and hide previous feedback
            return (
                chat_history,                           # Keep chat history
                "",                                     # Clear the main textbox
                gr.update(visible=True),                # Make the quiz box visible
                question,                               # Display the new question
                correct_answer,                         # Store the correct answer in state
                gr.update(value=""),                    # Clear previous user answer
                gr.update(value="")                     # Clear previous feedback
            )

        txt.submit(
            handle_main_submit,
            [txt, chat, mode, grade, files],
            [chat, txt, quiz_box, quiz_question_display, correct_answer_state, user_answer_input, quiz_feedback_display]
        )

        def handle_quiz_submission(user_answer, correct_answer):
            """
            Evaluates the user's answer and provides feedback.
            """
            if not user_answer.strip():
                return "Please enter an answer before submitting."
                
            similarity_score = evaluate(user_answer, correct_answer)
            feedback_text = (
                f"**Correct Answer:** {correct_answer}\n\n"
                f"### Your Similarity Score: **{similarity_score}**"
            )
            return feedback_text

        submit_answer_button.click(
            handle_quiz_submission,
            [user_answer_input, correct_answer_state],
            quiz_feedback_display
        )

        def _clear():
            """Clears all inputs and outputs."""
            return (
                [],                               # Clear chatbot
                "",                               # Clear textbox
                gr.update(visible=False),         # Hide quiz box
                "",                               # Clear quiz question
                "",                               # Clear user answer input
                "",                               # Clear quiz feedback
                ""                                # Clear correct answer state
            )
        clear.click(
            _clear,
            outputs=[chat, txt, quiz_box, quiz_question_display, user_answer_input, quiz_feedback_display, correct_answer_state]
        )

        return demo

# To run this UI, you would typically have a main execution block like this:
if __name__ == "__main__":
    # You can pass a chat/summarization function to build_ui if you have one.
    # For demonstration, we'll pass None.
    # def dummy_respond_fn(text, history, *args):
    #     history.append((text, f"This is a dummy response for '{text}' in {args[0]} mode."))
    #     return history
    
    # demo_ui = build_ui(dummy_respond_fn)
    demo_ui = build_ui() # Running without a chat function
    demo_ui.launch()