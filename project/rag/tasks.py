
'''
from .prompts import SUMMARIZE, BULLET_NOTES, QUIZ
from .chain import make_chain, ask

MODES = {
    "Summarize": SUMMARIZE,
    "Bullet Notes": BULLET_NOTES,
    "Quiz": QUIZ,
    "Free Chat": "Use context to answer clearly.\nContext:\n{context}\nQuestion: {question}\nAnswer:"
}

def run_task(llm, retriever, mode: str, grade: int, question: str, history):
    prompt = MODES.get(mode, MODES["Free Chat"])
    chain = make_chain(llm, retriever, prompt)
    # Inject grade into prompt at runtime (LangChain Template partial)
    chain.combine_docs_chain.prompt = chain.combine_docs_chain.prompt.partial(grade=grade)
    answer, sources = ask(chain, question, history or [])
    srcs = sorted({d.metadata.get("source", "") for d in sources if d is not None})
    if srcs:
        answer += "\n\nSources:\n" + "\n".join(srcs)
    return answer
'''
# rag/tasks.py
from .prompts import SUMMARIZE, BULLET_NOTES, QUIZ
from .chain import make_chain, ask

MODES = {
    "Summarize": SUMMARIZE,
    "Bullet Notes": BULLET_NOTES,
    "Quiz": QUIZ,
    "Free Chat": "Use context to answer clearly.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
}


# rag/tasks.py
from .prompts import SUMMARIZE, BULLET_NOTES, QUIZ
from .chain import make_chain, ask

MODES = {
    "Summarize": SUMMARIZE,
    "Bullet Notes": BULLET_NOTES,
    "Quiz": QUIZ,
    "Free Chat": "Use context to answer clearly.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
}


def run_task(llm, retriever, mode: str, grade: int, question: str, history):
    """
    Executes a mode-specific task using the RAG pipeline.
    Compatible with LangChain 0.3+ (Runnable-based chains).
    """

    # 1️. Select the appropriate prompt template
    prompt_template = MODES.get(mode, MODES["Free Chat"])

    # 2️. Replace only {grade}, leave {context}/{question} for LangChain
    prompt_text = prompt_template.replace("{grade}", str(grade))

    # 3️.Build the chain dynamically
    chain = make_chain(llm, retriever, prompt_text)

    # 4️. Run the query
    answer, sources = ask(chain, question, history or [])

    # 5.Collect source metadata (if retriever provides docs)
    srcs = sorted({
        d.metadata.get("source", "")
        for d in sources
        if getattr(d, "metadata", None) is not None
    })

    if srcs:
        answer += "\n\nSources:\n" + "\n".join(srcs)

    return answer

