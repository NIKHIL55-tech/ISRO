"""
Vector retriever implementation using ChromaDB for document similarity search.
"""
from typing import List, Dict, Any, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    """Custom embedding function using SentenceTransformer models."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        logger.info(f"Initialized SentenceTransformer with model: {model_name}")
    
    def __call__(self, input: Documents) -> Embeddings:
        """Convert documents to embeddings."""
        # The model already handles batching and normalization
        embeddings = self.model.encode(
            input,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embeddings.tolist()

class VectorRetriever:
    """Vector-based retriever using ChromaDB for efficient similarity search."""
    
    def __init__(
        self,
        embedding_model: Optional[SentenceTransformer] = None,
        chroma_client: Optional[chromadb.Client] = None,
        collection_name: str = "mosdac_documents",
        embedding_model_name: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the vector retriever.
        
        Args:
            embedding_model: Pre-initialized SentenceTransformer model.
                            If None, a new one will be created.
            chroma_client: ChromaDB client instance. If None, a new one will be created.
            collection_name: Name of the Chroma collection to use.
            embedding_model_name: Name of the SentenceTransformer model to use if no model is provided.
        """
        logger.info(f"Initializing VectorRetriever with collection: {collection_name}")
        
        # Initialize embedding model if not provided
        if embedding_model is None:
            logger.info(f"Loading sentence transformer model: {embedding_model_name}")
            self.embedding_model = SentenceTransformer(embedding_model_name)
        else:
            self.embedding_model = embedding_model
        
        # Initialize ChromaDB client if not provided
        self.chroma_client = chroma_client or chromadb.PersistentClient()
        self.collection_name = collection_name
        
        # Initialize embedding function
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=embedding_model_name
        )
        
        # Get or create the collection
        try:
            logger.info(f"Attempting to get collection: {collection_name}")
            self.collection = self.chroma_client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Successfully loaded existing collection: {collection_name}")
            
        except Exception as e:
            logger.warning(f"Could not get collection '{collection_name}'. Creating new one. Error: {str(e)}")
            try:
                self.collection = self.chroma_client.create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
                logger.info(f"Successfully created new collection: {collection_name}")
            except Exception as create_error:
                logger.error(f"Failed to create collection '{collection_name}': {str(create_error)}")
                raise
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 100,
        **kwargs
    ) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document dictionaries with 'text' and 'metadata' keys.
            batch_size: Number of documents to process in each batch.
            **kwargs: Additional arguments to pass to the collection.add() method.
        """
        if not documents:
            logger.warning("No documents provided to add")
            return
            
        # Prepare data for batch processing
        ids = []
        texts = []
        metadatas = []
        
        logger.info(f"Preparing to add {len(documents)} documents to collection '{self.collection_name}'")
        
        for i, doc in enumerate(documents):
            if not isinstance(doc, dict):
                logger.warning(f"Document at index {i} is not a dictionary, skipping")
                continue
                
            if 'text' not in doc:
                logger.warning(f"Document at index {i} is missing required 'text' field, skipping")
                continue
                
            # Generate a unique ID if not provided
            doc_id = doc.get('id')
            if not doc_id:
                doc_id = f"doc_{i}_{int(datetime.now().timestamp())}"
            
            # Ensure the ID is unique within this batch
            if doc_id in ids:
                doc_id = f"{doc_id}_{i}"
            
            metadata = doc.get('metadata', {})
            
            # Ensure metadata is a dictionary
            if not isinstance(metadata, dict):
                logger.warning(f"Document {doc_id} has non-dictionary metadata, converting to empty dict")
                metadata = {}
            
            ids.append(doc_id)
            texts.append(doc['text'])
            metadatas.append(metadata)
        
        if not ids:
            logger.warning("No valid documents to add after validation")
            return
        
        # Add documents in batches
        total_added = 0
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_texts = texts[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            
            try:
                logger.debug(f"Adding batch of {len(batch_ids)} documents (batch {i//batch_size + 1})")
                
                # Check if any of these IDs already exist
                existing = set()
                try:
                    existing_docs = self.collection.get(ids=batch_ids)
                    existing = set(existing_docs['ids'])
                except Exception as e:
                    # If the collection is empty or IDs don't exist, this is fine
                    pass
                
                # Filter out existing documents
                new_batch_ids = []
                new_batch_texts = []
                new_batch_metadatas = []
                
                for idx, doc_id in enumerate(batch_ids):
                    if doc_id in existing:
                        logger.debug(f"Document {doc_id} already exists, skipping")
                        continue
                    new_batch_ids.append(doc_id)
                    new_batch_texts.append(batch_texts[idx])
                    new_batch_metadatas.append(batch_metadatas[idx])
                
                if not new_batch_ids:
                    logger.debug("No new documents in this batch, skipping")
                    continue
                
                # Add the new documents
                self.collection.add(
                    documents=new_batch_texts,
                    metadatas=new_batch_metadatas,
                    ids=new_batch_ids,
                    **kwargs
                )
                
                total_added += len(new_batch_ids)
                logger.debug(f"Successfully added {len(new_batch_ids)} documents to the vector store")
                
            except Exception as e:
                logger.error(f"Error adding batch {i//batch_size + 1}: {str(e)}")
                # Try to add documents one by one to identify the problematic one
                for j in range(len(batch_ids)):
                    try:
                        self.collection.add(
                            documents=[batch_texts[j]],
                            metadatas=[batch_metadatas[j]],
                            ids=[batch_ids[j]],
                            **kwargs
                        )
                        total_added += 1
                    except Exception as single_error:
                        logger.error(f"Failed to add document {batch_ids[j]}: {str(single_error)}")
        
        logger.info(f"Successfully added {total_added} out of {len(ids)} documents to the vector store")
        return total_added
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: The search query string.
            top_k: Number of results to return.
            filters: Optional filters to apply to the search.
            **kwargs: Additional arguments to pass to the collection.query() method.
            
        Returns:
            List of result dictionaries with 'text', 'score', and 'metadata' keys.
        """
        if not query or not query.strip():
            return []
        
        # Convert filters to Chroma's where clause format
        where_clause = self._convert_filters(filters) if filters else None
        
        try:
            # Perform the search
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_clause,
                **kwargs
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'text': results['documents'][0][i],
                    'score': float(1.0 - results['distances'][0][i]),  # Convert distance to similarity
                    'metadata': results['metadatas'][0][i],
                    'id': results['ids'][0][i]
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error during vector search: {str(e)}")
            return []
    
    def _convert_filters(self, filters: Dict[str, Any]) -> Dict:
        """
        Convert filter dictionary to Chroma's where clause format.
        
        Args:
            filters: Dictionary of filters (e.g., {"category": "satellite", "year": 2023})
            
        Returns:
            Chroma-compatible where clause dictionary.
        """
        if not filters:
            return {}
            
        # Handle simple equality filters
        if len(filters) == 1:
            key, value = next(iter(filters.items()))
            return {"$eq": [f"metadata.{key}", value]}
            
        # Handle multiple filters with AND condition
        return {"$and": [{"$eq": [f"metadata.{key}", value]} for key, value in filters.items()]}
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document by its ID.
        
        Args:
            doc_id: The document ID to retrieve.
            
        Returns:
            Document dictionary or None if not found.
        """
        try:
            result = self.collection.get(ids=[doc_id])
            if not result['documents']:
                return None
                
            return {
                'id': result['ids'][0],
                'text': result['documents'][0],
                'metadata': result['metadatas'][0],
                'embedding': result['embeddings'][0] if result['embeddings'] else None
            }
        except Exception as e:
            logger.error(f"Error retrieving document {doc_id}: {str(e)}")
            return None
    
    def delete_documents(self, doc_ids: List[str]) -> bool:
        """
        Delete documents by their IDs.
        
        Args:
            doc_ids: List of document IDs to delete.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            self.collection.delete(ids=doc_ids)
            logger.info(f"Deleted {len(doc_ids)} documents")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            return False
