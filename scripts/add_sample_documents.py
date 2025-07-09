"""
Script to add document chunks to the vector store.

Usage:
    python scripts/add_sample_documents.py --input path/to/documents.json
    
The input JSON file should be an array of document objects, where each object has:
    - text: The main content of the document
    - metadata: A dictionary containing metadata fields
"""
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add the project root to the Python path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.retrieval.vector_retriever import VectorRetriever
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

def load_documents(file_path: str) -> List[Dict]:
    """Load documents from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)
            
        if not isinstance(documents, list):
            raise ValueError("Input file should contain a JSON array of documents")
            
        return documents
        
    except Exception as e:
        print(f"Error loading documents: {str(e)}")
        return []

def enrich_document(doc: Dict) -> Dict:
    """Enhance document text with metadata for better searchability."""
    if not isinstance(doc, dict) or 'text' not in doc:
        return doc
        
    # Ensure metadata exists
    if 'metadata' not in doc:
        doc['metadata'] = {}
    
    metadata = doc["metadata"]
    base_text = doc["text"]
    
    # Add default values for required metadata fields if missing
    for field in ["satellite", "parameter", "source", "year", "last_updated"]:
        if field not in metadata:
            metadata[field] = "Unknown"
    
    # Create a string of metadata values
    metadata_text = " ".join([
        str(metadata["satellite"]),
        str(metadata["parameter"]),
        str(metadata["source"]),
        str(metadata["year"]),
        str(metadata["last_updated"])
    ])
    
    # Combine base text with metadata
    doc["text"] = f"{base_text} {metadata_text}"
    return doc

def initialize_retrieval(documents: List[Dict]):
    """Initialize the vector retriever with sample data."""
    try:
        # Create data directory if it doesn't exist
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True, parents=True)
        vector_store_path = data_dir / "vector_store"
        
        print(f"Using vector store path: {vector_store_path.absolute()}")
        
        # Initialize ChromaDB client
        chroma_client = chromadb.PersistentClient(
            path=str(vector_store_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        print("ChromaDB client initialized successfully")
        
        # Initialize embedding model
        print("Loading sentence transformer model...")
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Sentence transformer model loaded")
        
        # First try to get the collection to see if it exists
        collection_name = "mosdac_documents"
        print(f"Checking for collection: {collection_name}")
        
        # Get list of existing collections
        try:
            collections = chroma_client.list_collections()
            collection_names = [c.name for c in collections]
            print(f"Existing collections: {collection_names}")
            
            if collection_name in collection_names:
                print(f"Collection '{collection_name}' exists, deleting it...")
                chroma_client.delete_collection(collection_name)
                print(f"Collection '{collection_name}' deleted")
                
        except Exception as e:
            print(f"Error checking collections: {str(e)}")
        
        print(f"Creating new collection: {collection_name}")
        
        # Initialize vector retriever with the collection name
        vector_retriever = VectorRetriever(
            embedding_model=embedding_model,
            chroma_client=chroma_client,
            collection_name=collection_name
        )
        
        print("Vector retriever initialized successfully")
        
        # Prepare documents with unique IDs and metadata
        print(f"Preparing {len(documents)} documents...")
        documents_to_add = []
        for i, doc in enumerate(documents):
            if not isinstance(doc, dict):
                print(f"Skipping invalid document at index {i}: not a dictionary")
                continue
                
            doc_id = doc.get('id') or f"doc_{i}_{int(datetime.now().timestamp())}"
            
            # Enrich the document text with metadata
            try:
                enriched_doc = enrich_document(doc.copy())  # Make a copy to avoid modifying the original
                doc_with_id = {
                    "id": doc_id,
                    "text": enriched_doc["text"],  # Use the enriched text
                    "metadata": enriched_doc.get("metadata", {})
                }
                documents_to_add.append(doc_with_id)
            except Exception as e:
                print(f"Error processing document {i}: {str(e)}")
                continue
        
        print(f"Adding {len(documents_to_add)} documents to the vector store...")
        try:
            vector_retriever.add_documents(documents_to_add)
            print("Documents added successfully!")
            
            # Verify the documents were added
            print("\nVerifying document addition with a sample search...")
            results = vector_retriever.search("ocean temperature", top_k=min(5, len(documents_to_add)))
            
            if not results:
                print("Warning: No results found in the search. The documents may not have been added correctly.")
                return False
                
            print("\nSample search results for 'ocean temperature':")
            for i, result in enumerate(results, 1):
                print(f"\n--- Result {i} (score: {result.get('score', 0):.4f}) ---")
                print(f"Text: {result.get('text', 'No text')}")
                print(f"Metadata: {json.dumps(result.get('metadata', {}), indent=2)}")
                
            return True
            
        except Exception as e:
            print(f"Error adding documents: {str(e)}")
            print("Please check if the ChromaDB server is running and accessible.")
            return False
        
        print("\nSample documents added successfully!")
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Load documents into the vector store.')
    parser.add_argument('--input', required=True, help='Path to the JSON file containing documents')
    args = parser.parse_args()
    
    print("=== MOSDAC AI Help Bot - Document Loader ===\n")
    
    # Load documents from the input file
    print(f"Loading documents from: {args.input}")
    documents = load_documents(args.input)
    
    if not documents:
        print("No documents found or error loading documents.")
        sys.exit(1)
        
    print(f"Successfully loaded {len(documents)} documents")
    
    # Initialize retrieval and add documents
    if initialize_retrieval(documents):
        print("\nDocuments loaded successfully!")
        print("You can now start the API server using: python run_api.py")
    else:
        print("\nFailed to load documents. Please check the error messages above.")
        sys.exit(1)
