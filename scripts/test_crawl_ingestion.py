#!/usr/bin/env python3
"""
Test script for ingesting and searching MOSDAC crawl results.
"""
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.retrieval.document_processor import DocumentProcessor, load_crawl_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_limited_crawl_data(file_path: str, max_pages: int = 1) -> Dict:
    """Load a limited number of pages from crawl results to reduce memory usage."""
    try:
        logger.info(f"Loading up to {max_pages} pages from: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if not isinstance(data, dict) or 'pages' not in data:
            logger.error("Invalid crawl results format")
            return {}
            
        # Take only the first max_pages
        limited_pages = dict(list(data['pages'].items())[:max_pages])
        logger.info(f"Loaded {len(limited_pages)} out of {len(data['pages'])} pages")
        
        return {'pages': limited_pages}
        
    except Exception as e:
        logger.error(f"Error loading crawl data: {str(e)}")
        return {}

def test_ingestion(crawl_file: str, persist_dir: str = 'data/vector_store', max_pages: int = 2) -> DocumentProcessor:
    """Test ingestion of crawl results into the vector store with memory optimizations."""
    logger.info("=== Starting Test Ingestion (Limited Mode) ===")
    
    # Initialize the document processor with smaller chunks
    processor = DocumentProcessor(
        model_name='all-MiniLM-L6-v2',
        chunk_size=512,  # Smaller chunks for memory efficiency
        chunk_overlap=50,
        persist_dir=persist_dir,
        collection_name='mosdac_test',  # Different collection for testing
        min_chunk_length=50  # Lower minimum chunk length
    )
    
    # Load limited crawl results
    logger.info(f"Loading up to {max_pages} pages from: {crawl_file}")
    crawl_data = load_limited_crawl_data(crawl_file, max_pages)
    
    if not crawl_data or 'pages' not in crawl_data or not crawl_data['pages']:
        logger.error("No valid crawl data found")
        return None
        
    logger.info(f"Processing {len(crawl_data['pages'])} pages")
    
    # Process with a smaller batch size
    success = processor.add_crawl_results(crawl_data, batch_size=5)  # Smaller batch size
    if not success:
        logger.error("Failed to add crawl results to vector store")
        return None
        
    logger.info("Successfully processed all pages")
    return processor

def test_search(processor: DocumentProcessor, query: str, top_k: int = 3) -> None:
    """Test searching the vector store."""
    if not processor:
        logger.error("No processor provided")
        return
        
    logger.info(f"\n=== Testing Search ===")
    logger.info(f"Query: {query}")
    
    # Basic search
    logger.info("\nBasic search results:")
    results = processor.search(query, top_k=top_k)
    for i, result in enumerate(results, 1):
        print(f"\nResult {i} (Score: {result['score']:.3f})")
        print(f"Title: {result['metadata'].get('title', 'No title')}")
        print(f"URL: {result['metadata'].get('url', 'No URL')}")
        print(f"Content: {result['text'][:200]}...")
    
    # Search with filters
    logger.info("\nSearch with filters (content_categories=weather):")
    filtered_results = processor.search(
        query,
        top_k=top_k,
        content_categories=["weather"]
    )
    
    if not filtered_results:
        logger.info("No results with filters, trying without filters...")
        filtered_results = results
    
    for i, result in enumerate(filtered_results, 1):
        print(f"\nFiltered Result {i} (Score: {result['score']:.3f})")
        print(f"Categories: {result['metadata'].get('content_categories', 'N/A')}")
        print(f"Title: {result['metadata'].get('title', 'No title')}")
        print(f"Content: {result['text'][:200]}...")

def clear_vector_store(persist_dir: str = 'data/vector_store'):
    """Clear the test vector store to free up memory."""
    try:
        import shutil
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)
            logger.info(f"Cleared vector store at {persist_dir}")
    except Exception as e:
        logger.error(f"Error clearing vector store: {str(e)}")

def main():
    # Path to your crawl results file
    crawl_file = os.path.join('data', 'processed', 'crawl_results_www_mosdac_gov_in.json')
    
    # Clear previous vector store to ensure clean state
    clear_vector_store()
    
    try:
        # Test with just 2 pages
        logger.info("=== Starting Test with 2 Pages ===")
        processor = test_ingestion(crawl_file, max_pages=2)
        
        if not processor:
            logger.error("Failed to initialize processor")
            return
        
        # Test with simpler queries
        test_queries = [
            "weather",
            "satellite",
            "ocean"
        ]
        
        for query in test_queries:
            test_search(processor, query, top_k=2)  # Limit to top 2 results
            print("\n" + "="*80 + "\n")
            
    except MemoryError:
        logger.error("Memory error occurred. Try with even fewer pages or smaller chunks.")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
    finally:
        # Clean up
        if 'processor' in locals():
            del processor

if __name__ == "__main__":
    main()
