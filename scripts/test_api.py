"""
Script to test the MOSDAC AI Help Bot API endpoints.
"""
import requests
import json
from typing import Dict, Any, List
import time

def test_health_endpoint(base_url: str = "http://localhost:8000"):
    """Test the health check endpoint."""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        response.raise_for_status()
        print(f"✅ Health check passed: {response.json()}")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {str(e)}")
        return False

def test_search_endpoint(
    query: str,
    top_k: int = 3,
    filters: Dict[str, Any] = None,
    base_url: str = "http://localhost:8000"
):
    """Test the search endpoint with a query."""
    print(f"\nTesting search with query: '{query}'")
    if filters:
        print(f"Using filters: {json.dumps(filters, indent=2)}")
    
    try:
        payload = {
            "query": query,
            "top_k": top_k
        }
        
        if filters:
            payload["filters"] = filters
        
        start_time = time.time()
        response = requests.post(
            f"{base_url}/search",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response_time = (time.time() - start_time) * 1000  # in milliseconds
        
        response.raise_for_status()
        result = response.json()
        
        print(f"✅ Search successful (took {response_time:.2f}ms)")
        print(f"Processed query: {result.get('processed_query', 'N/A')}")
        print(f"Detected intent: {result.get('intent', 'N/A')} (confidence: {result.get('confidence', 0):.2f})")
        
        print("\nResults:")
        for i, res in enumerate(result.get('results', []), 1):
            print(f"\n--- Result {i} (score: {res.get('score', 0):.4f}) ---")
            print(f"Text: {res.get('text', '')}")
            print(f"Metadata: {json.dumps(res.get('metadata', {}), indent=2)}")
        
        return result
        
    except requests.exceptions.RequestException as e:
        error_msg = str(e)
        try:
            error_msg += f"\nResponse: {e.response.text}"
        except:
            pass
        print(f"❌ Search failed: {error_msg}")
        return None

def run_tests(base_url: str = "http://localhost:8000"):
    """Run a series of test queries to verify the API functionality."""
    print("=== MOSDAC AI Help Bot API Tester ===\n")
    
    # Test health endpoint
    if not test_health_endpoint(base_url):
        print("\n❌ Health check failed. Is the API server running?")
        return
    
    # Test search with different queries
    test_queries = [
        "ocean temperature data",
        "weather satellite information",
        "vegetation index data",
        "aerosol data from satellites"
    ]
    
    for query in test_queries:
        test_search_endpoint(query, top_k=2)
    
    # Test with filters
    print("\n=== Testing with Filters ===")
    test_search_endpoint(
        "satellite data",
        top_k=2,
        filters={"satellite": "Oceansat-2"}
    )
    
    # Test with no results
    print("\n=== Testing No Results Case ===")
    test_search_endpoint("nonexistent query that should return no results", top_k=2)
    
    print("\n=== All tests completed ===")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the MOSDAC AI Help Bot API")
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the API (default: http://localhost:8000)"
    )
    
    args = parser.parse_args()
    run_tests(args.base_url)
