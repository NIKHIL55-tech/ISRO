"""
Knowledge Graph Configuration

This file contains the configuration for building a knowledge graph from website data.
Website-specific configurations should inherit from BaseKGConfig.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Pattern, Set, Optional
import re

@dataclass
class EntityPatterns:
    """Patterns for extracting entities from URLs and content."""
    # Regex patterns for identifying entities in URLs
    url_patterns: Dict[str, Pattern] = field(default_factory=dict)
    
    # Keywords to identify entity types in page content
    content_keywords: Dict[str, Set[str]] = field(default_factory=dict)
    
    # HTML/CSS selectors for extracting entities from pages
    selectors: Dict[str, str] = field(default_factory=dict)

@dataclass
class RelationshipRules:
    """Rules for establishing relationships between entities."""
    # Parent-child relationships (e.g., category -> subcategory)
    hierarchy: Dict[str, List[str]] = field(default_factory=dict)
    
    # Entity-attribute relationships (e.g., product -> price)
    attributes: Dict[str, List[str]] = field(default_factory=dict)
    
    # Custom relationship patterns
    custom: List[Dict] = field(default_factory=list)

@dataclass
class BaseKGConfig:
    """Base configuration for building a knowledge graph."""
    # Website domain (e.g., 'mosdac.gov.in')
    domain: str = ""
    
    # Entity types to extract (e.g., 'product', 'category', 'article')
    entity_types: List[str] = field(default_factory=list)
    
    # Patterns for entity extraction
    patterns: EntityPatterns = field(default_factory=EntityPatterns)
    
    # Relationship rules
    relationships: RelationshipRules = field(default_factory=RelationshipRules)
    
    # File paths
    processed_dir: str = "../data/processed"
    output_dir: str = "../data/knowledge_graph"
    
    def get_entity_type(self, url: str, content: Optional[Dict] = None) -> Optional[str]:
        """Determine entity type from URL or content."""
        for entity_type, pattern in self.patterns.url_patterns.items():
            if pattern.search(url):
                return entity_type
        return None
    
    def extract_entities(self, text: str, entity_type: str) -> List[str]:
        """Extract entities of a specific type from text."""
        if entity_type in self.patterns.content_keywords:
            return [
                entity for entity in self.patterns.content_keywords[entity_type]
                if entity.lower() in text.lower()
            ]
        return []

# Example configuration for MOSDAC
class MOSDACConfig(BaseKGConfig):
    def __init__(self):
        super().__init__()
        self.domain = "mosdac.gov.in"
        self.entity_types = ["mission", "product", "category", "service"]
        
        # URL patterns for entity detection
        self.patterns.url_patterns = {
            "mission": re.compile(r'mosdac\.gov\.in/([^/]+-\d+[A-Z]*)', re.IGNORECASE),
            "product": re.compile(r'mosdac\.gov\.in/([^/]+-product)', re.IGNORECASE),
            "category": re.compile(r'mosdac\.gov\.in/catalog/([^/]+)', re.IGNORECASE)
        }
        
        # Content keywords for entity extraction
        self.patterns.content_keywords = {
            "mission": {"mission", "satellite", "insat", "oceansat", "scatsat"},
            "product": {"data", "product", "service", "tool"},
            "category": {"atmosphere", "ocean", "land", "weather", "climate"}
        }
        
        # HTML/CSS selectors
        self.patterns.selectors = {
            "mission": ".mission-details",
            "product": ".product-info",
            "category": ".category-list"
        }
        
        # Define relationship hierarchy
        self.relationships.hierarchy = {
            "category": ["subcategory", "product"],
            "mission": ["instrument", "product"]
        }
        
        # Define entity attributes
        self.relationships.attributes = {
            "mission": ["launch_date", "status", "agency"],
            "product": ["format", "resolution", "temporal_coverage"]
        }

# Factory function to get configuration
def get_config(domain: str) -> BaseKGConfig:
    """Get the appropriate configuration for a domain."""
    configs = {
        "mosdac.gov.in": MOSDACConfig()
    }
    return configs.get(domain, BaseKGConfig())
