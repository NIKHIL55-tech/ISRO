import os
import json
from sentence_transformers import SentenceTransformer
import chromadb

import sys

# Add the parent directory (project root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.text_utils import clean_text, chunk_text

model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("mosdac_docs")

processed_dir = "data/processed/"

for file in os.listdir(processed_dir):
    if file.endswith(".json"):
        with open(os.path.join(processed_dir, file)) as f:
            doc = json.load(f)
            text = clean_text(doc.get("text", ""))
            chunks = chunk_text(text)

            for i, chunk in enumerate(chunks):
                embedding = model.encode(chunk).tolist()
                collection.add(
                    documents=[chunk],
                    embeddings=[embedding],
                    metadatas=[{"source": file, "chunk_id": i}],
                    ids=[f"{file}_{i}"]
                )
print("✅ Embeddings created and stored.")
