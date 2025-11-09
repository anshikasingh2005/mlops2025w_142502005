def build_retriever(store, k: int = 4, search_type: str = "mmr"):
    return store.as_retriever(search_type=search_type, search_kwargs={"k": k})
