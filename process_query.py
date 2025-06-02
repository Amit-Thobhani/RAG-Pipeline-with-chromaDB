"""Runs the inference pipeline"""

from langchain.vectorstores.chroma import Chroma
from get_emb_function import embedding_function
from text2text_model import load_qa_model
from langchain.prompts import ChatPromptTemplate
import warnings
warnings.filterwarnings('ignore')


DEVICE = "CPU"
TEXT_GENERATION_MODEL = "google/flan-t5-xl"
CHROMA_DIR = "./langchain_chroma_db"

PROMPT_TEMPLATE = """
Learn following context for Quenstion Answer:

{context}

Using knowledge of above context answer: {question}
"""


def load_config(device: str = DEVICE, model_name: str = TEXT_GENERATION_MODEL):
    config = {}
    config['device'] = device
    # # >> load embedings, tokenizer and model
    config['embeddings'] = embedding_function()
    tokenizer, model = load_qa_model(model_name=TEXT_GENERATION_MODEL, device=device)
    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=config['embeddings'])
    
    config['tokenizer'] = tokenizer
    config['model'] = model
    config['db'] = db
    
    return config

    
def run_rag_pipeline(query_text: str, config: dict):
    
    # # >> get relevant documents
    results = config['db'].similarity_search_with_score(query_text, k=1)
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    # # >> tokenize the input and generate output
    inputs = config['tokenizer'](prompt, return_tensors="pt", max_length=512, truncation=True).to(config['device'])
    outputs = config['model'].generate(**inputs, max_new_tokens=150)
    
    return config['tokenizer'].decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    config = load_config(device='cpu')
    query_text = input("Ask question related documents")
    print(run_rag_pipeline(query_text))
