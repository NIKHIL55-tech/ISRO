import re
import numpy as np
import spacy
from difflib import get_close_matches

nlp = spacy.load("en_core_web_sm")

class QueryPreprocessor:
    def __init__(self):
        # Dynamic technical term vocabulary (to be derived later if needed)
        self.terms = np.array(["sst", "insat", "irs", "mosdac"])
        
  # These will be extended automatically if needed

    def clean_query(self, query: str) -> str:
        query = re.sub(r"\s+", " ", query.strip().lower())
        return query

    def spell_check_technical(self, query: str) -> str:
        words = query.split()
        corrected = [self._fuzzy_correct(w) for w in words]
        return ' '.join(corrected)

    def _fuzzy_correct(self, word: str) -> str:
        matches = get_close_matches(word, self.terms.tolist(), n=1, cutoff=0.8)
        return matches[0] if matches else word

    def tokenize_domain_specific(self, query: str) -> list:
        doc = nlp(query)
        return [token.text for token in doc if not token.is_punct and not token.is_stop]