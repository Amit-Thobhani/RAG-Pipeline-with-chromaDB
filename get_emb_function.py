"""Embedding function defined"""

from langchain_community.embeddings import HuggingFaceEmbeddings

MODAL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def embedding_function(model_name=MODAL_NAME):
    """
    Loads embedding function.
    """
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    return embeddings

