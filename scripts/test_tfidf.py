"""
Test script for TfidfRetriever with ChromaDB integration.
"""
import logging
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import chromadb
from src.retrieval.tfidf_retriever import TfidfRetriever

def setup_logging():
    """Configure logging for the test script."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('test_tfidf.log')
        ]
    )
    return logging.getLogger(__name__)

def test_tfidf_retriever():
    """Test the TF-IDF retriever with ChromaDB."""
    logger = setup_logging()
    logger.info("Starting TF-IDF retriever test...")
    
    # Initialize ChromaDB client
    try:
        chroma_client = chromadb.PersistentClient(path='data/vector_store')
        collection_name = 'mosdac_documents'
        logger.info(f"Using ChromaDB collection: {collection_name}")
        
        # List available collections for debugging (ChromaDB v0.6.0+ compatibility)
        collections = chroma_client.list_collections()
        logger.info(f"Available collections: {collections}")
        
        # Check if our target collection exists
        if collection_name not in collections:
            logger.error(f"Collection '{collection_name}' not found in ChromaDB")
            return False
        
        # Initialize TF-IDF retriever
        logger.info("Initializing TF-IDF retriever...")
        tfidf = TfidfRetriever(
            chroma_client=chroma_client,
            collection_name=collection_name,
            content_key='text'
        )
        
        # Check if documents were loaded
        if not hasattr(tfidf, 'documents') or not tfidf.documents:
            logger.error("No documents were loaded!")
            return False
        
        logger.info(f"Successfully loaded {len(tfidf.documents)} documents")
        
        # Print sample document info
        logger.info("Sample documents:")
        for i, doc in enumerate(tfidf.documents[:3]):  # Show first 3 documents
            doc_id = doc.get('id', 'N/A')
            text_preview = (doc.get('text', '')[:100] + '...') if doc.get('text') else 'EMPTY'
            metadata = doc.get('metadata', {})
            logger.info(f"Document {i+1}:")
            logger.info(f"  ID: {doc_id}")
            logger.info(f"  Preview: {text_preview}")
            logger.info(f"  Metadata: {metadata}")
        
        # Test search functionality if documents exist
        if tfidf.documents:
            test_query = "what is SCATSAT-1"
            logger.info(f"\nTesting search with query: '{test_query}'")
            results = tfidf.search(test_query, top_k=3)
            
            if not results:
                logger.warning("No results returned from search!")
            else:
                logger.info(f"Found {len(results)} results:")
                for i, result in enumerate(results, 1):
                    doc = result['document']
                    score = result['score']
                    metadata = result.get('metadata', {})
                    doc_id = metadata.get('id', 'N/A')
                    text_preview = (doc[:100] + '...') if doc else 'EMPTY'
                    logger.info(f"  {i}. [Score: {score:.4f}] ID: {doc_id}")
                    logger.info(f"     Text: {text_preview}")
                    logger.info(f"     Metadata: {metadata}")
        
        return True
        
    except Exception as e:
        logger.exception("Error testing TF-IDF retriever:")
        return False

if __name__ == "__main__":
    success = test_tfidf_retriever()
    sys.exit(0 if success else 1)
