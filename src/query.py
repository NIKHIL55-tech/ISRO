import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sentence_transformers import SentenceTransformer
import chromadb
from llm.llm_client import generate_response
from llm.prompt_templates import RAG_TEMPLATE
from utils.kg_utils import query_kg



model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("mosdac_docs")

def vector_search(query: str, top_k=5):
    embedding = model.encode(query).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=top_k)
    docs = results.get("documents", [[]])[0]
    if not docs:
        return "No relevant documents found."
    return "\n".join(docs)

def kg_search(query: str):
    return query_kg(query)  # Load triples from kg.json and run simple match


def compile_prompt(query, context):
    return RAG_TEMPLATE.format(context=context, question=query)

def detect_intent(query: str) -> str:
    query = query.lower()
    if any(kw in query for kw in ["list", "name of", "show all", "provide entities"]):
        return "kg"
    return "rag"
def preprocess_query(query: str) -> str:
    return query.strip().lower()



def process_query(query: str) -> str:
    query = preprocess_query(query)
    intent = detect_intent(query)

    if intent == "kg":
        return kg_search(query)

    context = vector_search(query)
    prompt = compile_prompt(query, context)
    return generate_response(prompt)

