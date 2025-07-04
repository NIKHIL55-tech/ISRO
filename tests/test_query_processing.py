import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.query import preprocess_query, detect_intent

def test_preprocess_query():
    query = "   What satellites are used for rainfall data?   "
    assert preprocess_query(query) == "what satellites are used for rainfall data?"

def test_detect_intent_rag():
    assert detect_intent("How do I download SST data?") == "rag"

def test_detect_intent_kg():
    assert detect_intent("List all data products") == "kg"
