"""Reads the document provided and adds to vector database with embeddings"""

import argparse
import os
import shutil
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_emb_function import embedding_function
from langchain_community.vectorstores.chroma import Chroma


DATA_DIR = "./data"
CHROMA_DIR = "./langchain_chroma_db"

def load_documents(data_dir: str = DATA_DIR) -> list:
    """
    Loads the documents from input data_dir directory
    """
    
    document_loader = PyPDFDirectoryLoader(data_dir)
    return document_loader.load()


def split_documents(documents: list) -> list:
    """
    Splits documents into smaller chunks with overlap
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80, 
        length_function=len, 
        is_separator_regex=False
    )
    return text_splitter.split_documents(documents)


def create_chunk_ids(split_chunks: list):
    """
    Create unique chunks id for database
    """
    last_page_id = None
    current_chunk_idx = 0
    
    for chunk in split_chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        
        current_page_id = f"{source}-{page}"
        current_chunk_idx = (current_chunk_idx + 1) if (current_page_id == last_page_id) else 0
        chunk_id = f"{current_page_id}-{current_chunk_idx}"
        
        chunk.metadata["id"] = chunk_id
        
        last_page_id = current_page_id
        
    return split_chunks


def add_to_db(chunks_with_ids: list, config: dict, db_path: str = CHROMA_DIR):
    """
    Add chunks to DB.
    """
    # # >> load embedding model into memory
    embeddings = config['embeddings']
    
    # # >> load existing DB
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    
    # # >> get present document ids in DB
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")
    
    # # >> Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("No new documents to add")
    

def update_database(config: dict):
    """
    Populate database with available docs.
    """
    # # >> load documents 
    docs = load_documents()
    print(f"Total {len(docs)} document loaded.")
    
    # # >> split into chunks
    split_docs = split_documents(docs)
    print(f"Document splits into {len(split_docs)}.")
    # # >> create unique chunk ids
    chunks_with_ids = create_chunk_ids(split_docs)
    
    add_to_db(chunks_with_ids, config)

    
def clear_database():
    """
    Remove entire DB.
    """
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

if __name__ == "__main__":
    update_database()
