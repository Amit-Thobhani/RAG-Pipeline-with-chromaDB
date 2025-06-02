"""Loads the QA model for question answers"""

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


TEXT_GENERATION_MODEL = "google/flan-t5-xl"


def load_qa_model(model_name: str = TEXT_GENERATION_MODEL, device: str = 'cpu'):
    """
    Load model and tokenizer for question answering.
    """
    tokenizer = AutoTokenizer.from_pretrained(TEXT_GENERATION_MODEL, 
                                              device_map=device)

    model = AutoModelForSeq2SeqLM.from_pretrained(TEXT_GENERATION_MODEL,
                                 device_map=device,
                                 torch_dtype='auto',
                                 trust_remote_code=False)

    # rag_pipline = pipeline("text2text-generation", 
    #                        model=model, 
    #                        tokenizer=tokenizer,
    #                        return_full_text=False,
    #                        do_sample=False
    #                       )
    
    return tokenizer, model

