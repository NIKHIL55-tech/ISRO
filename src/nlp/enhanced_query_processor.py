"""
Enhanced query processing with domain-specific knowledge for MOSDAC.
"""
import re
import spacy
from typing import List, Dict, Tuple, Set, Optional
from difflib import get_close_matches
import numpy as np

class EnhancedQueryProcessor:
    """Enhanced query processing with domain-specific knowledge."""
    
    def __init__(self, use_advanced_nlp: bool = True):
        """Initialize the enhanced query processor."""
        self.nlp = spacy.load("en_core_web_sm")
        self.use_advanced_nlp = use_advanced_nlp
        
        # Domain-specific knowledge
        self.synonyms = {
            "sst": ["sea surface temperature", "sst"],
            "ndvi": ["vegetation index", "ndvi", "normalized difference vegetation index"],
            "aod": ["aerosol optical depth", "aod", "aerosol"],
            "insat": ["insat", "indian national satellite"],
            "oceansat": ["oceansat", "ocean satellite"],
            "scatsat": ["scatsat", "scatterometer satellite"],
            "rainfall": ["precipitation", "rain", "rainfall"],
            "windspeed": ["wind speed", "wind velocity", "winds"],
            "humidity": ["moisture", "relative humidity", "humidity"],
            "temperature": ["temp", "temperature", "thermal"]
        }
        
        self.acronyms = {
            "sst": "Sea Surface Temperature",
            "ndvi": "Normalized Difference Vegetation Index",
            "aod": "Aerosol Optical Depth",
            "insat": "Indian National Satellite",
            "oceansat": "OceanSat",
            "scatsat": "SCATterometer SATellite",
            "vhrr": "Very High Resolution Radiometer",
            "dwr": "Doppler Weather Radar",
            "saphir": "Sondeur Atmosphérique du Profil d'Humidité Intertropicale par Radiométrie"
        }
        
        # Initialize technical terms from synonyms and acronyms
        self.technical_terms = set(self.synonyms.keys()) | set(self.acronyms.keys())
        
        # Common temporal patterns
        self.temporal_patterns = [
            (r'(from\s+)?(\d{4})(?:\s*[-/]\s*(\d{4}))?', 'year_range'),
            (r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*(?:\s+\d{4})?', 'month'),
            (r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', 'date'),
            (r'(last|past|previous)\s+(\d+)\s+(day|week|month|year)s?', 'relative_time')
        ]
        
        # Common spatial patterns
        self.spatial_patterns = [
            (r'lat(?:itude)?\s*[:=]?\s*([-+]?\d*\.?\d+)(?:\s*,\s*long(?:itude)?\s*[:=]?\s*([-+]?\d*\.?\d+))?', 'coordinates'),
            (r'bounding box\s*[:=]?\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)', 'bbox'),
            (r'(near|around|in|over|for)\s+(?:the\s+)?([\w\s]+(?:region|area|country|ocean|sea))', 'location')
        ]
    
    def clean_query(self, query: str) -> str:
        """Basic query cleaning and normalization."""
        if not query:
            return ""
        # Convert to lowercase and normalize whitespace
        query = ' '.join(query.strip().lower().split())
        # Remove special characters except those used in technical terms
        query = re.sub(r'[^\w\s-]', ' ', query)
        # Remove any remaining extra whitespace and strip
        return ' '.join(query.split())
    
    def expand_query(self, query: str) -> str:
        """Expand query with synonyms and acronyms."""
        terms = query.split()
        expanded = set(terms)  # Start with original terms
        
        # Add synonyms and acronyms
        for term in terms:
            term_lower = term.lower()
            # Add synonyms
            if term_lower in self.synonyms:
                expanded.update(self.synonyms[term_lower])
            # Add acronym expansions
            if term_lower in self.acronyms:
                expanded.update(self.acronyms[term_lower].lower().split())
        
        return ' '.join(expanded)
    
    def extract_temporal_info(self, query: str) -> List[Dict]:
        """Extract temporal information from query."""
        temporal_info = []
        
        for pattern, pattern_type in self.temporal_patterns:
            for match in re.finditer(pattern, query, re.IGNORECASE):
                if pattern_type == 'year_range':
                    start_year = match.group(2)
                    end_year = match.group(3) or start_year
                    temporal_info.append({
                        'type': 'year_range',
                        'start': start_year,
                        'end': end_year,
                        'original': match.group(0)
                    })
                # Add more pattern types as needed
                
        return temporal_info
    
    def extract_spatial_info(self, query: str) -> List[Dict]:
        """Extract spatial information from query."""
        spatial_info = []
        
        for pattern, pattern_type in self.spatial_patterns:
            for match in re.finditer(pattern, query, re.IGNORECASE):
                if pattern_type == 'coordinates':
                    lat = match.group(1)
                    lon = match.group(2) if len(match.groups()) > 1 else None
                    spatial_info.append({
                        'type': 'coordinates',
                        'latitude': lat,
                        'longitude': lon,
                        'original': match.group(0)
                    })
                # Add more pattern types as needed
                
        return spatial_info
    
    def process_query(self, query: str) -> Dict:
        """Process a query with all enhancements."""
        # Clean the query
        cleaned = self.clean_query(query)
        
        # Expand with synonyms and acronyms
        expanded = self.expand_query(cleaned)
        
        # Extract structured information
        temporal = self.extract_temporal_info(cleaned)
        spatial = self.extract_spatial_info(cleaned)
        
        # Tokenize and remove stopwords
        doc = self.nlp(cleaned)
        tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
        
        # Extract named entities if using advanced NLP
        entities = []
        if self.use_advanced_nlp:
            entities = [{'text': ent.text, 'label': ent.label_} for ent in doc.ents]
        
        return {
            'original_query': query,
            'cleaned_query': cleaned,
            'expanded_query': expanded,
            'tokens': tokens,
            'temporal_info': temporal,
            'spatial_info': spatial,
            'entities': entities,
            'processed_at': str(datetime.now())
        }
    
    def _fuzzy_correct(self, word: str) -> str:
        """Correct typos in technical terms using fuzzy matching."""
        if not word:
            return word
            
        word_lower = word.lower()
        
        # Check if it's already a known term
        if word_lower in self.technical_terms:
            return word
            
        # Try to find close matches
        matches = get_close_matches(
            word_lower, 
            self.technical_terms, 
            n=1, 
            cutoff=0.7
        )
        
        return matches[0] if matches else word
