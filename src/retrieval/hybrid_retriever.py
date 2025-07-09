"""
Hybrid Retriever that combines TF-IDF and vector search results.
"""
from typing import List, Dict, Optional, Union
import numpy as np
from .tfidf_retriever import TfidfRetriever
from src.generate_embeddings import DocumentEmbeddings

class HybridRetriever:
    """
    A retriever that combines TF-IDF and vector search results.
    """
    
    def __init__(
        self,
        tfidf_retriever: TfidfRetriever,
        vector_retriever: DocumentEmbeddings,
        tfidf_weight: float = 0.4,
        vector_weight: float = 0.6
    ):
        """
        Initialize the hybrid retriever.
        
        Args:
            tfidf_retriever: Initialized TF-IDF retriever
            vector_retriever: Initialized vector retriever (DocumentEmbeddings)
            tfidf_weight: Weight for TF-IDF scores (0-1)
            vector_weight: Weight for vector search scores (0-1)
        """
        self.tfidf = tfidf_retriever
        self.vector = vector_retriever
        
        # Normalize weights
        total = tfidf_weight + vector_weight
        self.tfidf_weight = tfidf_weight / total
        self.vector_weight = vector_weight / total
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.1,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search using both TF-IDF and vector search, then combine results.
        
        Args:
            query: Search query
            top_k: Number of results to return
            min_score: Minimum combined score for results
            filter_metadata: Optional metadata filters to apply
            
        Returns:
            List of result dictionaries with 'document', 'score', and 'metadata' keys
        """
        # Get results from both retrievers
        tfidf_results = self.tfidf.search(query, top_k=top_k * 2)
        vector_results = self.vector.search(query, top_k=top_k * 2)
        
        # Create a mapping of document content to combined scores
        scored_docs = {}
        
        def get_document_text(result):
            # Handle both 'document' and 'text' keys for backward compatibility
            return result.get('document') or result.get('text', '')
            
        def get_document_metadata(result):
            # Extract metadata, handling different possible locations
            if 'metadata' in result:
                return result['metadata']
            # If no metadata key, return all non-standard fields as metadata
            return {k: v for k, v in result.items() if k not in ['document', 'text', 'score']}
        
        # Process TF-IDF results
        for result in tfidf_results:
            doc_text = get_document_text(result)
            doc_key = doc_text[:500]  # Use first 500 chars as key
            if doc_key and doc_key not in scored_docs:
                scored_docs[doc_key] = {
                    'document': doc_text,
                    'metadata': get_document_metadata(result),
                    'tfidf_score': result.get('score', 0.0),
                    'vector_score': 0.0,
                    'combined_score': result.get('score', 0.0) * self.tfidf_weight,
                    'id': result.get('id', f'tfidf_{len(scored_docs)}')
                }
        
        # Process vector results and combine scores
        for result in vector_results:
            doc_text = get_document_text(result)
            doc_key = doc_text[:500]  # Same key generation as above
            vector_score = result.get('score', 0.0)
            
            if doc_key in scored_docs:
                # Document exists in both retrievers, combine scores
                scored_docs[doc_key]['vector_score'] = vector_score
                scored_docs[doc_key]['combined_score'] += vector_score * self.vector_weight
            elif doc_key:  # Only add if we have a valid document key
                # Document only in vector results
                scored_docs[doc_key] = {
                    'document': doc_text,
                    'metadata': get_document_metadata(result),
                    'tfidf_score': 0.0,
                    'vector_score': vector_score,
                    'combined_score': vector_score * self.vector_weight,
                    'id': result.get('id', f'vector_{len(scored_docs)}')
                }
        
        # Convert to list and sort by combined score
        all_results = list(scored_docs.values())
        all_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Apply filters if provided
        if filter_metadata:
            all_results = [
                r for r in all_results
                if self._matches_filters(r['metadata'], filter_metadata)
            ]
        
        # Return top K results above min_score
        return [
            {
                'document': r['document'],
                'score': r['combined_score'],
                'metadata': r['metadata'],
                'scores': {
                    'tfidf': r['tfidf_score'],
                    'vector': r['vector_score'],
                    'combined': r['combined_score']
                }
            }
            for r in all_results
            if r['combined_score'] >= min_score
        ][:top_k]
    
    def _matches_filters(self, metadata: Dict, filters: Dict) -> bool:
        """Check if document metadata matches all provided filters."""
        for key, value in filters.items():
            if key not in metadata:
                return False
            if isinstance(value, (list, set, tuple)):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False
        return True
    
    def explain_scores(self, query: str, top_k: int = 3) -> Dict:
        """
        Provide an explanation of how the scores were calculated for a query.
        
        Args:
            query: The search query
            top_k: Number of top terms to show
            
        Returns:
            Dictionary with scoring explanation
        """
        # Get TF-IDF vector for the query
        query_vec = self.tfidf.vectorizer.transform([query])
        
        # Get top terms in the query
        feature_index = query_vec.nonzero()[1]
        query_terms = [
            (self.tfidf.feature_names[i], query_vec[0, i])
            for i in feature_index
        ]
        query_terms.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'query_terms': query_terms[:top_k],
            'weights': {
                'tfidf': self.tfidf_weight,
                'vector': self.vector_weight
            },
            'explanation': (
                f"Scores are combined with {self.tfidf_weight*100:.0f}% weight "
                f"for TF-IDF and {self.vector_weight*100:.0f}% for vector similarity."
            )
        }
