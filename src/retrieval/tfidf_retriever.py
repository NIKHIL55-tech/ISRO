"""
TF-IDF Retriever for keyword-based document retrieval with ChromaDB integration.
"""
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

class TfidfRetriever:
    """
    A retriever that uses TF-IDF for keyword-based document retrieval.
    Can load documents from ChromaDB or use provided documents.
    """
    
    def __init__(self, 
                documents: List[Dict] = None,
                chroma_client: chromadb.Client = None,
                collection_name: str = 'mosdac_documents',
                content_key: str = 'text'):
        """
        Initialize the TF-IDF retriever.
        
        Args:
            documents: Optional list of document dictionaries with text content and metadata
            chroma_client: Optional ChromaDB client instance
            collection_name: Name of the Chroma collection (default: 'mosdac_documents')
            content_key: Key in the document dict that contains the text content
        """
        self.documents = documents or []
        self.content_key = content_key
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),  # Include unigrams and bigrams
            min_df=1,  # Accept terms that appear in at least 1 document
            max_df=1.0,  # Allow terms that appear in up to 100% of documents
            max_features=5000,  # Increased vocabulary size
            token_pattern=r'(?u)\b\w+\b',  # Simple token pattern
            analyzer='word',  # Split into words
            lowercase=True,  # Convert to lowercase
            strip_accents='unicode'  # Remove accents
        )
        
        # If ChromaDB client is provided, try to load documents from it
        if chroma_client is not None:
            self._load_from_chroma(chroma_client, collection_name)
        
        if not self.documents:
            logger.warning("No documents available. Please add documents before performing searches.")
            # Initialize with empty matrix
            self.tfidf_matrix = None
            self.feature_names = np.array([])
        else:
            self._fit_vectorizer()
    
    def _load_from_chroma(self, chroma_client: chromadb.Client, collection_name: str):
        """Load documents from ChromaDB collection."""
        try:
            logger.info(f"Loading documents from ChromaDB collection: {collection_name}")
            
            # List all available collections for debugging
            try:
                collections = chroma_client.list_collections()
                logger.debug(f"Available collections: {collections}")
            except Exception as e:
                logger.warning(f"Error listing collections: {e}")
            
            try:
                # Get the collection without embedding function since we don't need it for TF-IDF
                collection = chroma_client.get_collection(
                    name=collection_name,
                    embedding_function=None
                )
                logger.info(f"Successfully loaded collection: {collection_name}")
                
                # Get document count
                count = collection.count()
                logger.info(f"Collection {collection_name} has {count} documents")
                
                if count == 0:
                    logger.warning(f"Collection {collection_name} is empty")
                    return
                
                # Get all documents
                results = collection.get(
                    include=["documents", "metadatas"],
                    limit=count
                )
                
                if not results or 'documents' not in results or not results['documents']:
                    logger.warning(f"No documents found in ChromaDB collection: {collection_name}")
                    logger.debug(f"Results structure: {results}")
                    return
                
                # Process documents
                documents = []
                for doc_text, metadata, doc_id in zip(
                    results['documents'],
                    results['metadatas'] or [{}] * len(results['documents']),
                    results.get('ids', [f'doc_{i}' for i in range(len(results['documents']))])
                ):
                    if not doc_text or not isinstance(doc_text, str) or not doc_text.strip():
                        logger.warning(f"Skipping empty document with ID: {doc_id}")
                        continue
                        
                    doc_data = {
                        self.content_key: doc_text,
                        'metadata': metadata or {},
                        'id': doc_id
                    }
                    documents.append(doc_data)
                
                if not documents:
                    logger.warning("No valid documents found after processing")
                    return
                    
                self.documents = documents
                logger.info(f"Successfully loaded {len(self.documents)} documents from ChromaDB")
                
            except Exception as e:
                logger.error(f"Error accessing ChromaDB collection: {str(e)}", exc_info=True)
                return
                
            # Log some debug info
            logger.debug(f"Found {len(results['documents'])} documents in collection")
            
            # Process documents
            documents = []
            for i, (doc_id, doc_text, metadata) in enumerate(zip(
                results['ids'],
                results['documents'],
                results['metadatas'] or [{}] * len(results['documents'])
            )):
                if not doc_text or not isinstance(doc_text, str) or not doc_text.strip():
                    logger.warning(f"Skipping empty document with ID: {doc_id}")
                    continue
                    
                doc_data = {
                    self.content_key: doc_text,
                    'metadata': metadata or {},
                    'id': doc_id
                }
                documents.append(doc_data)
                
                # Log first few documents for debugging
                if i < 3:  # Log first 3 documents
                    logger.debug(f"Document {i+1} sample: {doc_text[:100]}...")
            
            if not documents:
                logger.warning("No valid documents found after processing")
                return
                
            self.documents = documents
            logger.info(f"Successfully loaded {len(self.documents)} documents from ChromaDB")
            
        except Exception as e:
            logger.error(f"Error loading from ChromaDB: {str(e)}")
            if not self.documents:  # If we have no documents, add a minimal one
                self.documents = [{
                    'text': 'mosdac satellite data ocean atmosphere land surface temperature',
                    'metadata': {'source': 'dummy', 'type': 'placeholder'}
                }]
                logger.warning("Using placeholder document due to ChromaDB load error")
    
    def _fit_vectorizer(self):
        """Fit the TF-IDF vectorizer on the document collection."""
        try:
            if not self.documents:
                logger.warning("No documents available to fit TF-IDF vectorizer")
                return
                
            texts = [doc[self.content_key] for doc in self.documents if self.content_key in doc]
            
            if not texts:
                logger.error(f"No valid text content found in documents (content_key: {self.content_key})")
                logger.debug(f"Document keys: {list(self.documents[0].keys()) if self.documents else 'No documents'}")
                return
                
            logger.info(f"Fitting TF-IDF vectorizer on {len(texts)} documents")
            logger.debug(f"First document sample: {texts[0][:100]}...")
            
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)
            self.feature_names = self.vectorizer.get_feature_names_out()
            
            logger.info(f"TF-IDF vectorizer fitted with {len(self.feature_names)} features")
            
        except Exception as e:
            logger.error(f"Error fitting TF-IDF vectorizer: {str(e)}")
            logger.debug(f"Document count: {len(self.documents) if self.documents else 0}")
            if self.documents:
                logger.debug(f"First document structure: {self.documents[0]}")
            raise
    
    def get_top_terms(self, doc_idx: int, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Get the top N most important terms for a document.
        
        Args:
            doc_idx: Index of the document
            top_n: Number of top terms to return
            
        Returns:
            List of (term, tfidf_score) tuples
        """
        feature_index = self.tfidf_matrix[doc_idx, :].nonzero()[1]
        tfidf_scores = zip(
            feature_index,
            [self.tfidf_matrix[doc_idx, x] for x in feature_index]
        )
        sorted_terms = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
        return [(self.feature_names[i], score) for i, score in sorted_terms[:top_n]]
    
    def search(self, query: str, top_k: int = 5, min_score: float = 0.1) -> List[Dict]:
        """
        Search for documents most relevant to the query.
        
        Args:
            query: Search query string
            top_k: Maximum number of results to return
            min_score: Minimum similarity score (0-1) for results
            
        Returns:
            List of result dictionaries with 'document', 'score', and 'metadata' keys
        """
        # Transform query to TF-IDF vector
        query_vec = self.vectorizer.transform([query])
        
        # Calculate cosine similarity between query and documents
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top K results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Format results
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score < min_score:
                continue
                
            doc = self.documents[idx]
            results.append({
                'document': doc[self.content_key],
                'score': score,
                'metadata': {k: v for k, v in doc.items() if k != self.content_key}
            })
        
        return results
    
    def add_documents(self, new_documents: List[Dict]):
        """
        Add new documents to the retriever and update the TF-IDF matrix.
        
        Args:
            new_documents: List of new document dictionaries
        """
        self.documents.extend(new_documents)
        self._fit_vectorizer()
        
    def update_documents(self, new_documents: List[Dict], replace: bool = True):
        """
        Update the documents in the retriever, optionally replacing all existing documents.
        
        Args:
            new_documents: List of new document dictionaries
            replace: If True, replace all existing documents. If False, append to existing documents.
        """
        if replace:
            self.documents = new_documents
        else:
            self.documents.extend(new_documents)
        self._fit_vectorizer()

    def get_related_terms(self, term: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Find terms that are most related to the given term based on co-occurrence.
        
        Args:
            term: The term to find related terms for
            top_n: Number of related terms to return
            
        Returns:
            List of (related_term, correlation_score) tuples
        """
        if term not in self.vectorizer.vocabulary_:
            return []
            
        term_idx = self.vectorizer.vocabulary_[term]
        term_vec = self.tfidf_matrix[:, term_idx].toarray().flatten()
        
        # Calculate correlation with all other terms
        correlations = []
        for other_idx, other_term in enumerate(self.vectorizer.get_feature_names_out()):
            if other_idx == term_idx:
                continue
                
            other_vec = self.tfidf_matrix[:, other_idx].toarray().flatten()
            corr = np.corrcoef(term_vec, other_vec)[0, 1]
            if not np.isnan(corr):
                correlations.append((other_term, corr))
        
        # Return top N correlated terms
        return sorted(correlations, key=lambda x: abs(x[1]), reverse=True)[:top_n]
