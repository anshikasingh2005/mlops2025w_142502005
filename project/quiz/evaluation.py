import os
from dotenv import load_dotenv
load_dotenv()
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from .qn_gen import run_tutor_with_rag

# --- 2. Define the new AnswerEvaluator class ---
# This class handles the logic for comparing the meaning of two sentences.



def evaluate(user_answer: str, correct_answer: str):
    """Compares the semantic similarity between the user's answer and the correct answer."""
    model_name=os.getenv("EMBEDDING_MODEL")
    model = SentenceTransformer(model_name)
    # Convert both answers into vector embeddings
    embeddings = model.encode([user_answer, correct_answer])

    # Calculate the cosine similarity score between the two vectors
    similarity_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    # Provide feedback based on the similarity score
    if similarity_score > 0.85:
        feedback = "Excellent! Your answer is semantically correct."
    elif similarity_score > 0.65:
        feedback = "You're on the right track! Your answer is close to the correct one."
    elif similarity_score > 0.40:
        feedback = "Your answer has some of the right ideas, but the meaning is different."
    else:
        feedback = "It looks like your answer is on a different topic."

    return similarity_score, feedback

#refined_correct_answer, refined_question = run_tutor_with_rag("machine learning")

#evaluate(user_answer, refined_correct_answer)
