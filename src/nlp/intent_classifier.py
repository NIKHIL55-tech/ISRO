from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import numpy as np

class MOSDACIntentClassifier:
    def __init__(self):
        self.pipeline = None
        self.intent_labels = []  # Will be dynamically learned
    
    def train_classifier(self, data: list[tuple[str, str]]):
        # Expecting list of (query, label)
        queries, labels = zip(*data)
        self.intent_labels = sorted(set(labels))
        
        self.pipeline = make_pipeline(
            TfidfVectorizer(),
            LogisticRegression()
        )
        self.pipeline.fit(queries, labels)

    def classify_intent(self, query: str) :
        if self.pipeline is None:
            raise ValueError("Intent classifier not trained")

        predicted = self.pipeline.predict([query])[0]
        confidence = max(self.pipeline.predict_proba([query])[0])
        return {
            "intent": predicted,
            "confidence": round(confidence, 3)
        }
