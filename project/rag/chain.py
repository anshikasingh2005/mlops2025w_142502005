
"""
from langchain.prompts import PromptTemplate

def make_chain(llm, retriever, prompt_text: str):
    prompt = PromptTemplate.from_template(prompt_text)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
    )
    return chain

def ask(chain, question, chat_history):
    out = chain({"question": question, "chat_history": chat_history})
    return out["answer"], out.get("source_documents", [])


# rag/chain.py

# rag/chain.py

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_core.output_parsers import StrOutputParser


def make_chain(llm, retriever, prompt_text=None):
    """'''Modern RAG chain for LangChain 0.3.x+ (fully supported, no deprecated modules).'''"""
    if prompt_text is None:
        prompt_text = (
            "Use the following context to answer the question.\n\n{context}\n\nQuestion: {question}"
        )
    # Define how to format the prompt
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Combine retriever, prompt, llm, and parser explicitly
    chain = (
        RunnableMap({
            "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
            "question": RunnablePassthrough(),
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def ask(chain, query):
    """'''Run a user query through the chain and return the LLM’s answer.'''"""
    response = chain.invoke(query)
    return response

"""
# rag/chain.py
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_core.output_parsers import StrOutputParser


def make_chain(llm, retriever, prompt_text=None):
    """
    Modern RAG chain for LangChain 0.3.x+ (no deprecated modules).
    Supports dynamic prompt templates.
    """
    if prompt_text is None:
        prompt_text = (
            "Use the following context to answer the question.\n\n"
            "{context}\n\nQuestion: {question}"
        )

    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Build explicit Runnable chain (retrieval → prompt → llm → parser)
    chain = (
        RunnableMap({
            "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
            "question": RunnablePassthrough(),
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def ask(chain, query, history=None):
    """Run a query through the RAG chain."""
    if history is None:
        history = []
    response = chain.invoke(query)
    # The new chain only returns text, not structured output
    return response, []

