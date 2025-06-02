# Retrieval-Augmented Generation (RAG) Pipeline

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline using:

- **ChromaDB** for storing and retrieving document embeddings
- **all-MiniLM-L6-v2** as the embedding model for document chunks
- **google/flan-t5-xl** as the language model for generating answers based on retrieved context

## 🔧 Components

- **Embedding Model**: [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)  
- **Retriever**: [ChromaDB](https://www.trychroma.com/) for vector similarity search  
- **Generator**: [`google/flan-t5-xl`](https://huggingface.co/google/flan-t5-xl) via Hugging Face Transformers

## 🚀 Usage

Follow the steps in [`RAG_Pipeline.ipynb`](RAG_Pipeline.ipynb) to run the pipeline locally or in Google Colab.

The notebook demonstrates:
- Loading documents
- Creating embeddings
- Indexing with ChromaDB
- Accepting user queries
- Retrieving relevant context
- Generating responses using `flan-t5-xl`

## 📦 Requirements

Install dependencies with:
```
pip install transformers chromadb sentence-transformers langchain langchain-community pypdf
```

## 📝 Example Query
```
query_text = "Rules for defining python variables"

run_rag_pipeline(query_text, config)
```
### 💬 Output
```
"Variable names may contain upper-case and lowercase letters (A–Z, a–z), digits (0–9), and underscores (_), but they cannot begin with a digit"
```
