import re
import spacy

nlp = spacy.load("en_core_web_sm")

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)  # normalize whitespace
    text = re.sub(r"<[^>]+>", "", text)  # remove HTML tags
    return text.strip()

def chunk_text(text: str, chunk_size: int = 500) -> list:
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks
