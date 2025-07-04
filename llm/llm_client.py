from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import logging

# Suppress tokenizer warnings
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

MODEL_NAME = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

def generate_response(prompt: str) -> str:
    # Avoid unsupported kwargs like 'return_full_text'
    result = pipe(prompt, max_new_tokens=256)  
    return result[0]['generated_text'].replace(prompt, "").strip()
