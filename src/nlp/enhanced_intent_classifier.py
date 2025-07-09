"""
Enhanced intent classification for MOSDAC queries.
"""
import re
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import spacy

class MOSDACIntentClassifier:
    """Enhanced intent classifier for MOSDAC queries with domain-specific support."""
    
    # Define domain-specific intents
    DOMAIN_INTENTS = [
        # Data retrieval intents
        "data_availability",      # Check if data exists
        "data_download",         # How to download data
        "data_access",           # API/data access methods
        "data_quality",          # Data quality information
        
        # Informational intents
        "parameter_info",        # Information about parameters
        "satellite_info",        # Information about satellites
        "instrument_info",       # Information about instruments
        "product_info",          # Information about data products
        
        # Technical intents
        "technical_specs",       # Technical specifications
        "data_format",           # Data format details
        "processing_level",      # Information about processing levels
        
        # Support intents
        "help",                  # General help
        "contact_support",       # Contact information
        "documentation",         # Link to documentation
        "tutorial"               # Tutorials/how-tos
    ]
    
    # Intent patterns for rule-based classification
    INTENT_PATTERNS = {
        "data_availability": [
            r"(is|are).*data available",
            r"(can|could).*get data",
            r"(do you have|is there).*data"
        ],
        "data_download": [
            r"how to download",
            r"get.*data",
            r"download.*instructions"
        ],
        "parameter_info": [
            r"what is (the )?[\w\s]+\?*$",
            r"explain.*parameter",
            r"information about"
        ]
        # Add more patterns as needed
    }
    
    def __init__(self, use_advanced_nlp: bool = False, model_type: str = 'tfidf'):
        """Initialize the intent classifier.
        
        Args:
            use_advanced_nlp: Whether to use spaCy for advanced NLP features
            model_type: Type of model to use ('tfidf' or 'bert')
        """
        self.use_advanced_nlp = use_advanced_nlp
        self.model_type = model_type
        self.nlp = spacy.load("en_core_web_sm") if use_advanced_nlp else None
        self.model = None
        self.vectorizer = None
        self.classes_ = None
        
        # Initialize with default training data if needed
        self._initialize_default_training_data()
    
    def _initialize_default_training_data(self):
        """Initialize with some default training examples."""
        self.training_examples = [
            ("Is INSAT-3D data available for 2022?", "data_availability"),
            ("How to download OCEANSAT-2 SST data?", "data_download"),
            ("What is the resolution of SCATSAT-1?", "technical_specs"),
            ("Show me documentation for MOSDAC API", "documentation"),
            ("What parameters does INSAT-3D measure?", "parameter_info"),
            ("Contact support for data access", "contact_support")
        ]
    
    def train(self, examples: List[Tuple[str, str]] = None):
        """Train the intent classifier.
        
        Args:
            examples: List of (text, intent_label) tuples for training.
                     If None, uses default training examples.
        """
        if examples is None:
            examples = self.training_examples
            
        texts, labels = zip(*examples)
        self.classes_ = sorted(set(labels))
        
        if self.model_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                stop_words='english',
                max_features=1000
            )
            X = self.vectorizer.fit_transform(texts)
            self.model = LogisticRegression(
                multi_class='multinomial',
                max_iter=1000
            )
            self.model.fit(X, labels)
        
        # Add BERT-based model support here if needed
        elif self.model_type == 'bert':
            raise NotImplementedError("BERT model training not implemented yet")
    
    def predict_intent(self, text: str) -> Dict[str, Union[str, float]]:
        """Predict the intent of a query.
        
        Args:
            text: Input query text
            
        Returns:
            Dictionary with 'intent' and 'confidence' keys
        """
        # First try rule-based matching
        rule_based = self._rule_based_classification(text)
        if rule_based['confidence'] > 0.9:  # High confidence in rule-based match
            return rule_based
            
        # Fall back to model-based classification
        if self.model is None:
            self.train()  # Train with default examples if not already trained
            
        if self.model_type == 'tfidf':
            X = self.vectorizer.transform([text])
            probas = self.model.predict_proba(X)[0]
            max_idx = np.argmax(probas)
            return {
                'intent': self.model.classes_[max_idx],
                'confidence': float(probas[max_idx]),
                'all_intents': dict(zip(self.model.classes_, probas))
            }
            
        # Fallback to rule-based if model not available
        return rule_based
    
    def _rule_based_classification(self, text: str) -> Dict[str, Union[str, float]]:
        """Perform rule-based intent classification."""
        text_lower = text.lower()
        
        # Check against patterns
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return {
                        'intent': intent,
                        'confidence': 0.95,  # High confidence for rule-based
                        'method': 'rule_based'
                    }
        
        # Default to general help if no pattern matches
        return {
            'intent': 'help',
            'confidence': 0.5,
            'method': 'default'
        }
    
    def add_training_example(self, text: str, intent: str):
        """Add a new training example and retrain the model."""
        self.training_examples.append((text, intent))
        self.train()  # Retrain with the new example
    
    def get_supported_intents(self) -> List[str]:
        """Get the list of supported intents."""
        return self.DOMAIN_INTENTS
    
    def explain_intent(self, text: str) -> Dict:
        """Provide an explanation of the intent classification."""
        result = self.predict_intent(text)
        explanation = {
            'query': text,
            'predicted_intent': result['intent'],
            'confidence': result['confidence'],
            'method': result.get('method', 'model'),
            'suggested_actions': self._get_suggested_actions(result['intent'])
        }
        
        if 'all_intents' in result:
            explanation['all_intents'] = result['all_intents']
            
        return explanation
    
    def _get_suggested_actions(self, intent: str) -> List[str]:
        """Get suggested actions based on the intent."""
        actions = {
            'data_availability': [
                "Check data availability",
                "View temporal coverage",
                "Check spatial coverage"
            ],
            'data_download': [
                "Show download options",
                "View data access methods",
                "Check access permissions"
            ],
            'parameter_info': [
                "Show parameter details",
                "View related parameters",
                "Show data visualization"
            ]
            # Add more as needed
        }
        
        return actions.get(intent, ["View help documentation"])
