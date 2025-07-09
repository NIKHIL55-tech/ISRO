"""
Tests for the retrieval components.
"""
import unittest
import numpy as np
from src.retrieval.tfidf_retriever import TfidfRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generate_embeddings import DocumentEmbeddings

class TestTfidfRetriever(unittest.TestCase):
    """Test cases for TfidfRetriever."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test documents."""
        cls.documents = [
            {
                'text': 'The quick brown fox jumps over the lazy dog',
                'title': 'Fox and Dog',
                'category': 'animals'
            },
            {
                'text': 'The quick brown fox is very fast',
                'title': 'Fast Fox',
                'category': 'animals'
            },
            {
                'text': 'The lazy dog is sleeping',
                'title': 'Sleeping Dog',
                'category': 'animals'
            },
            {
                'text': 'The sun is shining bright today',
                'title': 'Weather',
                'category': 'weather'
            }
        ]
        
        # Initialize retrievers
        cls.tfidf_retriever = TfidfRetriever(cls.documents)
        
        # For hybrid testing, we'll mock the vector retriever
        class MockVectorRetriever:
            def search(self, query, top_k=5):
                # Simple mock that returns documents containing query terms
                query_terms = set(query.lower().split())
                results = []
                for doc in TestTfidfRetriever.documents:
                    doc_terms = set(doc['text'].lower().split())
                    common_terms = query_terms.intersection(doc_terms)
                    if common_terms:
                        score = len(common_terms) / len(query_terms)
                        results.append({
                            'document': doc['text'],
                            'score': score,
                            'metadata': {'title': doc['title']}
                        })
                return sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]
        
        cls.vector_retriever = MockVectorRetriever()
        cls.hybrid_retriever = HybridRetriever(
            cls.tfidf_retriever,
            cls.vector_retriever,
            tfidf_weight=0.5,
            vector_weight=0.5
        )
    
    def test_tfidf_search(self):
        """Test basic TF-IDF search functionality."""
        results = self.tfidf_retriever.search('quick fox', top_k=2)
        self.assertEqual(len(results), 2)
        self.assertIn('fox', results[0]['document'].lower())
        self.assertGreaterEqual(results[0]['score'], 0.1)
    
    def test_hybrid_search(self):
        """Test hybrid search combining TF-IDF and vector results."""
        results = self.hybrid_retriever.search('quick fox', top_k=2)
        self.assertEqual(len(results), 2)
        self.assertIn('fox', results[0]['document'].lower())
        self.assertIn('scores', results[0])
        self.assertIn('tfidf', results[0]['scores'])
        self.assertIn('vector', results[0]['scores'])
    
    def test_metadata_filtering(self):
        """Test metadata filtering in hybrid search."""
        results = self.hybrid_retriever.search(
            'fox',
            top_k=5,
            filter_metadata={'category': 'animals'}
        )
        self.assertTrue(all(
            r['metadata'].get('category') == 'animals'
            for r in results
        ))
    
    def test_explain_scores(self):
        """Test the explanation of scoring."""
        explanation = self.hybrid_retriever.explain_scores('quick fox')
        self.assertIn('query_terms', explanation)
        self.assertIn('weights', explanation)
        self.assertIn('explanation', explanation)
        self.assertEqual(len(explanation['query_terms']), 2)  # 'quick' and 'fox'

if __name__ == '__main__':
    unittest.main()
