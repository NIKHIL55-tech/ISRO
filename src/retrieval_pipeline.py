# src/rag/retrieval_pipeline.py

from typing import Dict, List, Optional
from src.query import QueryProcessor
from src.generate_embeddings import DocumentEmbeddings
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

class ResultType(Enum):
    DIRECT_ANSWER = "direct_answer"
    RELATED_INFO = "related_info"
    TECHNICAL_SPEC = "technical_spec"
    DATA_ACCESS = "data_access"
    NOT_FOUND = "not_found"

@dataclass
class SearchResult:
    text: str
    score: float
    metadata: Dict
    result_type: ResultType
    confidence: float

class RAGPipeline:
    def __init__(self):
        """Initialize the RAG pipeline components"""
        self.query_processor = QueryProcessor()
        self.embeddings = DocumentEmbeddings()
        
        # Configure result scoring weights
        self.weights = {
            'embedding_score': 0.6,
            'metadata_score': 0.2,
            'recency_score': 0.1,
            'query_match_score': 0.1
        }

    def process_query(self, user_query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Process a user query through the complete RAG pipeline
        
        Args:
            user_query: Raw user query
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        # Step 1: Process and analyze query
        print(f"\nProcessing query: {user_query}")
        query_info = self.query_processor.preprocess_query(user_query)
        
        # Step 2: Get enhanced search query
        enhanced_query = self.query_processor.enhance_search_query(query_info)
        print(f"Enhanced query: {enhanced_query}")
        
        # Step 3: Perform embedding search
        raw_results = self.embeddings.search(enhanced_query, top_k=top_k * 2)  # Get extra results for filtering
        
        # Step 4: Score and rank results
        scored_results = self._score_results(raw_results, query_info)
        
        # Step 5: Filter and classify results
        final_results = self._process_results(scored_results, query_info, top_k)
        
        return final_results

    def _score_results(self, raw_results: List[Dict], query_info: Dict) -> List[Dict]:
        """Score results based on multiple factors"""
        scored_results = []
        
        for result in raw_results:
            # Get base similarity score from embedding search
            embedding_score = result['similarity_score']
            
            # Calculate metadata matching score
            metadata_score = self._calculate_metadata_score(result['metadata'], query_info)
            
            # Calculate recency score if last_updated is available
            recency_score = self._calculate_recency_score(result['metadata'].get('last_updated'))
            
            # Calculate query term matching score
            query_match_score = self._calculate_query_match_score(result['text'], query_info)
            
            # Calculate final weighted score
            final_score = (
                self.weights['embedding_score'] * embedding_score +
                self.weights['metadata_score'] * metadata_score +
                self.weights['recency_score'] * recency_score +
                self.weights['query_match_score'] * query_match_score
            )
            
            scored_results.append({
                **result,
                'final_score': final_score,
                'score_components': {
                    'embedding_score': embedding_score,
                    'metadata_score': metadata_score,
                    'recency_score': recency_score,
                    'query_match_score': query_match_score
                }
            })
        
        # Sort by final score
        scored_results.sort(key=lambda x: x['final_score'], reverse=True)
        return scored_results

    def _calculate_metadata_score(self, metadata: Dict, query_info: Dict) -> float:
        """Calculate how well metadata matches query intent"""
        score = 0.0
        
        # Check category match
        if query_info['query_type']:
            if any(qtype in metadata.get('category', '').lower() for qtype in query_info['query_type']):
                score += 0.3
        
        # Check entity matches
        for entity_type, entities in query_info['entities'].items():
            if entities:
                metadata_entities = metadata.get(entity_type, [])
                if isinstance(metadata_entities, str):
                    metadata_entities = [metadata_entities]
                
                matches = sum(1 for e in entities if any(me.lower() == e.lower() for me in metadata_entities))
                score += 0.2 * (matches / len(entities))
        
        # Check temporal/spatial matches if present in query
        if query_info['temporal_info'] and metadata.get('last_updated'):
            score += 0.2
            
        if query_info['spatial_info'] and metadata.get('spatial_coverage'):
            score += 0.2
            
        return min(score, 1.0)

    def _calculate_recency_score(self, last_updated: Optional[str]) -> float:
        """Calculate recency score based on last update date"""
        if not last_updated:
            return 0.5  # Neutral score if no date
            
        try:
            update_date = datetime.strptime(last_updated, "%Y-%m-%d")
            days_old = (datetime.now() - update_date).days
            
            # Exponential decay score based on age
            score = np.exp(-days_old / 365)  # 1 year half-life
            return score
        except:
            return 0.5

    def _calculate_query_match_score(self, text: str, query_info: Dict) -> float:
        """Calculate direct term matching score"""
        score = 0.0
        text_lower = text.lower()
        
        # Check for keyword matches
        if query_info['keywords']:
            matches = sum(1 for kw in query_info['keywords'] if kw.lower() in text_lower)
            score += 0.5 * (matches / len(query_info['keywords']))
        
        # Check for exact phrase matches
        if query_info['cleaned_query'] in text_lower:
            score += 0.5
            
        return min(score, 1.0)

    def _process_results(self, scored_results: List[Dict], query_info: Dict, top_k: int) -> List[SearchResult]:
        """Process and classify top results"""
        final_results = []
        
        for result in scored_results[:top_k]:
            # Determine result type based on query and content
            result_type = self._classify_result_type(result, query_info)
            
            # Calculate confidence based on scores and classification
            confidence = self._calculate_confidence(result, query_info)
            
            final_results.append(SearchResult(
                text=result['text'],
                score=result['final_score'],
                metadata=result['metadata'],
                result_type=result_type,
                confidence=confidence
            ))
            
        return final_results

    def _classify_result_type(self, result: Dict, query_info: Dict) -> ResultType:
        """Classify the type of result based on content and query"""
        text_lower = result['text'].lower()
        
        # Check for direct answers
        if any(qt in ['product_info', 'technical'] for qt in query_info['query_type']):
            if any(pattern in text_lower for pattern in ['is a', 'refers to', 'defined as']):
                return ResultType.DIRECT_ANSWER
                
        # Check for data access instructions
        if any(qt == 'data_access' for qt in query_info['query_type']):
            if any(pattern in text_lower for pattern in ['download', 'access', 'available at']):
                return ResultType.DATA_ACCESS
                
        # Check for technical specifications
        if any(pattern in text_lower for pattern in ['specification', 'resolution', 'bandwidth']):
            return ResultType.TECHNICAL_SPEC
            
        return ResultType.RELATED_INFO

    def _calculate_confidence(self, result: Dict, query_info: Dict) -> float:
        """Calculate confidence score for the result"""
        # Base confidence on final score
        confidence = result['final_score']
        
        # Adjust based on score components
        components = result['score_components']
        
        # Boost confidence if multiple scoring components are high
        high_scores = sum(1 for score in components.values() if score > 0.8)
        if high_scores >= 2:
            confidence = min(confidence * 1.2, 1.0)
            
        # Reduce confidence if scores are inconsistent
        score_std = np.std(list(components.values()))
        if score_std > 0.3:
            confidence *= 0.8
            
        return confidence

def main():
    """Test the RAG pipeline"""
    pipeline = RAGPipeline()
    
    # Test queries
    test_queries = [
        "How do I download SST data?",
        "What is the resolution of INSAT images?",
        "Show me rainfall data for Mumbai from last month",
        "What are the different types of data products available?"
    ]
    
    for query in test_queries:
        print("\n" + "="*50)
        print(f"Testing query: {query}")
        
        results = pipeline.process_query(query)
        
        print("\nResults:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Type: {result.result_type.value}")
            print(f"Confidence: {result.confidence:.2f}")
            print(f"Score: {result.score:.2f}")
            print(f"Text: {result.text[:200]}...")
            print(f"Source: {result.metadata.get('source_url', 'N/A')}")

if __name__ == "__main__":
    main()