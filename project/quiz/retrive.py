import chromadb
from .db_init import get_data



def search(query: str) -> str | None:
    """
    Searches the collection and retrieves the most relevant text chunk.

    Args:
        collection: The ChromaDB collection object returned by `load_chroma_collection`.
        query (str): The query text.
        k (int): Number of results to retrieve (default 1).

    Returns:
        str | None: Best matching document text or None if not found.
    """
    collection=get_data()
    

    results = collection.query(
        query_texts=[query],
        n_results=1
    )
    

    docs = results.get("documents", [])
    if docs and docs[0]:
        return docs[0][0]

    return None
#c=search("carbon atoms")