from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import spacy
from difflib import get_close_matches
from scipy.sparse import csr_matrix

nlp = spacy.load("en_core_web_sm")

class MOSDACEntityExtractor:
    def __init__(self):
        self.vectorizer = None
        self.vocab_terms = []

    def build_entity_vocab(self, text_corpus: list[str], min_tfidf: float = 0.1):
        """Build vocabulary from corpus using TF-IDF weights"""
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_df=0.95,
            min_df=1,
            lowercase=True
        )
        tfidf_matrix = self.vectorizer.fit_transform(text_corpus)

        if not isinstance(tfidf_matrix, csr_matrix):
            raise TypeError("Expected sparse CSR matrix from TfidfVectorizer")

        # ✅ SAFELY convert sparse matrix to dense
        dense_matrix = tfidf_matrix.toarray()  # <-- fix for Pyright/Pylance
        mean_scores = np.mean(dense_matrix, axis=0)

        terms = np.array(self.vectorizer.get_feature_names_out())
        self.vocab_terms = terms[mean_scores > min_tfidf].tolist()
    def extract_entities(self, query: str) -> dict:
        """Extract tokens from query that match high-TFIDF vocab terms"""
        if not self.vocab_terms:
            return {"matched_entities": []}

        tokens = [token.text.lower() for token in nlp(query) if token.is_alpha]
        matched = [token for token in tokens if token in self.vocab_terms]
        return {"matched_entities": list(set(matched))}
