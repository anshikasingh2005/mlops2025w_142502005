SUMMARIZE = """You are a helpful history tutor. Summarize clearly for class {grade}.
Use retrieved context; if context is insufficient, say so briefly.
Context:
{context}
Question: {question}
Answer:"""

BULLET_NOTES = """Create concise bullet notes (<=8 bullets) for class {grade}.
Only use retrieved context.
Context:
{context}
Notes:"""

QUIZ = """Create 5 mixed-difficulty questions (MCQ + short) with answers.
Topic: {question}
Use only the context.
Context:
{context}
Q&A:"""
