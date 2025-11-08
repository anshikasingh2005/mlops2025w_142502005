from langchain_text_splitters import RecursiveCharacterTextSplitter


def make_splitter(chunk_size=1000, chunk_overlap=200):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

def split_docs(docs, splitter=None):
    splitter = splitter or make_splitter()
    return splitter.split_documents(docs)
