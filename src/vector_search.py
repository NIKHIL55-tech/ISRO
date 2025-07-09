# src/vector_search.py

"""Vector search module for the MOSDAC AI Help Bot."""
from typing import List, Dict, Any, Optional, Tuple
import logging
import os
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.tfidf_retriever import TfidfRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generate_embeddings import DocumentEmbeddings
from src.nlp.enhanced_query_processor import EnhancedQueryProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
hybrid_retriever = None
query_processor = None
vector_retriever = None
tfidf_retriever = None

# Default paths
import os
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = os.path.join(os.getcwd(), "data")  # Use current working directory
VECTOR_STORE_PATH = os.path.join(DATA_DIR, "vector_store")
TFIDF_MODEL_PATH = os.path.join(DATA_DIR, "models", "tfidf.pkl")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Lightweight model for local use

def initialize_retrieval():
    """Initialize the retrieval components."""
    global hybrid_retriever, query_processor, vector_retriever, tfidf_retriever
    
    try:
        # Create necessary directories
        os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
        os.makedirs(os.path.dirname(TFIDF_MODEL_PATH), exist_ok=True)
        
        logger.info("Initializing vector retriever...")
        # Initialize ChromaDB client
        chroma_client = chromadb.PersistentClient(
            path=str(VECTOR_STORE_PATH),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize sentence transformer model
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        # Initialize vector retriever with automatic collection creation
        collection_name = "mosdac_documents"
        logger.info(f"Initializing vector retriever with collection: {collection_name}")
        
        try:
            
            # Now create a new collection
            vector_retriever = VectorRetriever(
                embedding_model=embedding_model,
                chroma_client=chroma_client,
                collection_name=collection_name
            )
            logger.info("Vector retriever initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector retriever: {str(e)}", exc_info=True)
            raise
        
        logger.info("Initializing TF-IDF retriever with ChromaDB integration...")
        
        # Initialize TF-IDF retriever with ChromaDB integration
        tfidf_retriever = TfidfRetriever(
            chroma_client=chroma_client,
            collection_name=collection_name,
            content_key='text'
        )
        
        logger.info(f"TF-IDF retriever initialized with {len(tfidf_retriever.documents)} documents")
        
        # Add a method to refresh documents from ChromaDB
        def refresh_tfidf_documents():
            """Refresh documents in the TF-IDF retriever from ChromaDB."""
            try:
                tfidf_retriever._load_from_chroma(chroma_client, collection_name)
                tfidf_retriever._fit_vectorizer()
                logger.info(f"TF-IDF retriever refreshed with {len(tfidf_retriever.documents)} documents")
                return True
            except Exception as e:
                logger.error(f"Error refreshing TF-IDF documents: {str(e)}")
                return False
        
        # Attach the refresh method to the retriever
        tfidf_retriever.refresh_documents = refresh_tfidf_documents
        
        logger.info("Initializing hybrid retriever...")
        # Initialize hybrid retriever with both retrievers
        try:
            # Create an adapter for VectorRetriever to work with HybridRetriever
            class VectorRetrieverAdapter:
                def __init__(self, vector_retriever):
                    self.vector_retriever = vector_retriever
                    self.model = vector_retriever.embedding_model  # Reuse the embedding model
                    self.collection = vector_retriever.collection  # Reference to the Chroma collection
                
                def search(self, query, top_k=5, **kwargs):
                    # Extract filters from kwargs if provided
                    filters = kwargs.get('filter_metadata')
                    if filters and 'filter_metadata' in filters:
                        filters = filters['filter_metadata']
                    
                    # Call the vector retriever's search method
                    results = self.vector_retriever.search(
                        query=query,
                        top_k=top_k,
                        filters=filters
                    )
                    
                    # The results are already in the correct format:
                    # [{'text': ..., 'score': ..., 'metadata': ..., 'id': ...}, ...]
                    return results
                
                # Required by DocumentEmbeddings interface
                def process_documents(self, *args, **kwargs):
                    # Not implemented as we're using VectorRetriever's own document storage
                    pass
                
                def get_embedding(self, text):
                    # Get the embedding for a single text using the vector retriever's model
                    return self.vector_retriever.embedding_model.encode(
                        text, 
                        convert_to_numpy=True
                    ).tolist()
            
            # Create the adapter
            vector_retriever_adapter = VectorRetrieverAdapter(vector_retriever)
            
            # Initialize the hybrid retriever
            hybrid_retriever = HybridRetriever(
                tfidf_retriever=tfidf_retriever,
                vector_retriever=vector_retriever_adapter,
                tfidf_weight=0.4,  # Adjust weights as needed
                vector_weight=0.6
            )
            logger.info("Hybrid retriever initialized successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import required modules: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize hybrid retriever: {str(e)}", exc_info=True)
            raise
        
        logger.info("Initializing query processor...")
        query_processor = EnhancedQueryProcessor()
        
        logger.info("Retrieval components initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize retrieval components: {str(e)}", exc_info=True)
        raise

def vector_search(query: str) -> str:
    try:
        results = search(query)
        return "\n".join([f"{result['text']} (Score: {result['score']})" for result in results])
    except Exception as e:
        return f"Vector Search Failed: {str(e)}"

def search(
    query: str, 
    top_k: int = 5, 
    filters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Perform a hybrid search using both TF-IDF and vector retrieval.
    
    Args:
        query: The search query string
        top_k: Number of results to return
        filters: Optional filters to apply to the search
        
    Returns:
        List of search results with scores and metadata
    """
    if hybrid_retriever is None or query_processor is None:
        initialize_retrieval()
    
    try:
        # Process the query
        processed_query = query_processor.clean_query(query)
        
        # Get search results
        results = hybrid_retriever.search(
            query=processed_query,
            top_k=top_k,
            filter_metadata=filters or {}
        )
        
        # Format results
        formatted_results = []
        for result in results:
            # Get the document text, trying both 'document' and 'text' keys
            document_text = result.get('document') or result.get('text', '')
            
            # Get the metadata, handling nested structure
            metadata = result.get('metadata', {})
            if isinstance(metadata, dict) and 'metadata' in metadata:
                metadata = metadata['metadata']
            
            formatted_results.append({
                'text': document_text,  # Use the actual document text
                'score': float(result.get('score', 0.0)),
                'metadata': {
                    'metadata': metadata,  # Keep the nested structure for compatibility
                    'id': result.get('id', '')
                },
                'explanation': result.get('explanation', '')
            })
        
        return formatted_results
    
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise ValueError(f"Search failed: {str(e)}")

# Initialize components when module is imported
initialize_retrieval()
