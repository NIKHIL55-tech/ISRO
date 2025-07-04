# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# MODEL_NAME = "google/flan-t5-base"

# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# # Build pipeline (disable internal max_length conflict)
# pipe = pipeline(
#     "text2text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     return_full_text=False  # Prevents repeating input in output
# )

# # Generate response
# def generate_response(prompt: str) -> str:
#     result = pipe(prompt)
#     return result[0]['generated_text'].strip()

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

MODEL_NAME = "google/flan-t5-base"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Build pipeline (❌ remove return_full_text)
pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer
)

# Generate response
def generate_response(prompt: str) -> str:
    result = pipe(prompt)
    return result[0]['generated_text'].strip()
