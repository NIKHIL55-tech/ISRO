import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from llm.llm_client import generate_response

def test_generate_response_structure():
    prompt = "What is the role of INSAT in weather monitoring?"
    response = generate_response(prompt)
    assert isinstance(response, str)
    assert len(response) > 0
