# src/vector_search.py

# Replace this import with actual function from Member B’s code
from src.query import generate_response as get_response
  # <- Check with Member B


def vector_search(query: str) -> str:
    try:
        return get_response(query)  # Member B's main pipeline
    except Exception as e:
        return f"Vector Search Failed: {str(e)}"
