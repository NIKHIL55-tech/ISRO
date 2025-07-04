import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.query import vector_search

def test_vector_search_basic():
    result = vector_search("What is SST data?")
    assert isinstance(result, str)
    assert len(result) > 0
