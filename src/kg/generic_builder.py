"""
Generic Knowledge Graph Builder

Builds a knowledge graph from crawled website data in a site-agnostic way.
"""
import re
import logging
from typing import Dict, List, Optional, Set, Any, Tuple
from urllib.parse import urlparse

import spacy
from spacy.tokens import Doc, Span
from spacy.language import Language
import networkx as nx

from models.graph_models import (
    Entity, EntityType, Relationship, RelationshipType, KnowledgeGraph
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenericKGConfig:
    """Configuration for the generic knowledge graph builder."""
    
    # Entity extraction patterns
    ENTITY_PATTERNS = {
        "url": r'https?://[^\s]+',
        "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        "phone": r'\+?[\d\s-]{10,}',
    }
    
    # Default entity types for common terms
    ENTITY_TYPES = {
        "contact": EntityType.NAMED_ENTITY,
        "about": EntityType.CONCEPT,
        "data": EntityType.DATA_PRODUCT,
        "download": EntityType.DATA_PRODUCT,
        "documentation": EntityType.WEB_PAGE,
        "api": EntityType.WEB_PAGE,
    }
    
    # Relationship extraction rules
    RELATIONSHIP_RULES = [
        # URL-based relationships
        (r'https?://[^\s/]+/downloads?/', RelationshipType.PROVIDES),
        (r'https?://[^\s/]+/documentation/', RelationshipType.DESCRIBES),
        
        # Content-based relationships
        (r'(?i)see also', RelationshipType.RELATED_TO),
        (r'(?i)related to', RelationshipType.RELATED_TO),
        (r'(?i)part of', RelationshipType.PART_OF),
    ]

class GenericKnowledgeGraphBuilder:
    """Builds a knowledge graph from crawled website data."""
    
    def __init__(self, config: Optional[GenericKGConfig] = None):
        self.config = config or GenericKGConfig()
        self.nlp = spacy.load("en_core_web_sm")
        self.kg = KnowledgeGraph()
        
        # Add custom pipeline components
        self._setup_pipeline()
    
    def _setup_pipeline(self) -> None:
        """Set up the NLP pipeline."""
        if not self.nlp.has_pipe("entity_ruler"):
            ruler = self.nlp.add_pipe("entity_ruler")
            
            # Add patterns for common entities
            patterns = [
                {"label": "URL", "pattern": [{"TEXT": {"REGEX": "https?://[^\s]+"}}]},
                {"label": "EMAIL", "pattern": [{"TEXT": {"REGEX": "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"}}]},
                {"label": "PHONE", "pattern": [{"TEXT": {"REGEX": "\\+?[\\d\\s-]{10,}"}}]},
            ]
            ruler.add_patterns(patterns)
    
    def process_page(self, url: str, content: Dict[str, Any]) -> None:
        """Process a single web page and extract entities/relationships."""
        # Create page entity
        page_entity = self._create_page_entity(url, content)
        self.kg.add_entity(page_entity)
        
        # Extract text content
        text = self._extract_text(content)
        doc = self.nlp(text)
        
        # Extract and link entities
        self._extract_entities(doc, page_entity)
        
        # Extract relationships
        self._extract_relationships(doc, page_entity)
        
        # Process links
        self._process_links(content.get('links', []), page_entity)
    
    def _create_page_entity(self, url: str, content: Dict[str, Any]) -> Entity:
        """Create a WebPage entity from URL and content."""
        return Entity(
            id=f"page:{url}",
            type=EntityType.WEB_PAGE,
            label=content.get('title', url),
            properties={
                'url': url,
                'title': content.get('title', ''),
                'description': content.get('description', ''),
                'content_type': content.get('content_type', ''),
                'word_count': len(content.get('text', '').split()),
                'last_modified': content.get('last_modified', ''),
            }
        )
    
    def _extract_text(self, content: Dict[str, Any]) -> str:
        """Extract and combine relevant text from page content."""
        text_parts = []
        
        # Add main content
        if 'text' in content:
            text_parts.append(content['text'])
            
        # Add headings
        for heading in content.get('headings', []):
            text_parts.append(heading['text'])
            
        # Add link text
        for link in content.get('links', []):
            if 'text' in link:
                text_parts.append(link['text'])
                
        return '\n'.join(text_parts)
    
    def _extract_entities(self, doc: Doc, source_entity: Entity) -> None:
        """Extract entities from text and link them to the source page."""
        for ent in doc.ents:
            # Skip very short entities
            if len(ent.text.strip()) < 2:
                continue
                
            # Determine entity type
            ent_type = self._determine_entity_type(ent, ent.label_)
            
            # Create entity
            entity = Entity(
                id=f"{ent_type.lower()}:{ent.text.lower()}",
                type=ent_type,
                label=ent.text,
                properties={'source': 'ner'}
            )
            
            # Add to graph
            self.kg.add_entity(entity)
            
            # Link to source page
            self.kg.add_relationship(
                Relationship(
                    source_id=source_entity.id,
                    target_id=entity.id,
                    type=RelationshipType.MENTIONS,
                    properties={'context': ent.sent.text if ent.sent else ''}
                )
            )
    
    def _determine_entity_type(self, ent: Span, label: str) -> EntityType:
        """Map spaCy entity types to our entity types."""
        # Default mapping from spaCy labels
        label_mapping = {
            'PERSON': EntityType.PERSON,
            'ORG': EntityType.ORGANIZATION,
            'GPE': EntityType.LOCATION,
            'LOC': EntityType.LOCATION,
            'PRODUCT': EntityType.DATA_PRODUCT,
            'WORK_OF_ART': EntityType.CONCEPT,
            'LAW': EntityType.CONCEPT,
            'DATE': None,  # Skip dates
            'TIME': None,  # Skip times
            'PERCENT': None,  # Skip percentages
            'MONEY': None,  # Skip money
            'QUANTITY': None,  # Skip quantities
            'ORDINAL': None,  # Skip ordinals
            'CARDINAL': None,  # Skip cardinals
        }
        
        # Check if we have a specific type for this text
        text = ent.text.lower()
        for key, ent_type in self.config.ENTITY_TYPES.items():
            if key in text:
                return ent_type
        
        # Use the spaCy label mapping
        return label_mapping.get(label, EntityType.CONCEPT)
    
    def _extract_relationships(self, doc: Doc, source_entity: Entity) -> None:
        """Extract relationships between entities in the text."""
        # Simple implementation: look for verb phrases between entities
        for sent in doc.sents:
            entities = list(sent.ents)
            
            # Look for relationships between consecutive entities
            for i in range(len(entities) - 1):
                ent1 = entities[i]
                ent2 = entities[i + 1]
                
                # Skip if either entity is not in our graph
                ent1_id = f"{self._determine_entity_type(ent1, ent1.label_).lower()}:{ent1.text.lower()}"
                ent2_id = f"{self._determine_entity_type(ent2, ent2.label_).lower()}:{ent2.text.lower()}"
                
                if ent1_id not in self.kg.entities or ent2_id not in self.kg.entities:
                    continue
                
                # Add a generic relationship
                self.kg.add_relationship(
                    Relationship(
                        source_id=ent1_id,
                        target_id=ent2_id,
                        type=RelationshipType.RELATED_TO,
                        properties={
                            'context': sent.text,
                            'source': 'sentence'
                        }
                    )
                )
    
    def _process_links(self, links: List[Dict[str, str]], source_entity: Entity) -> None:
        """Process links and create relationships."""
        for link in links:
            if 'url' not in link:
                continue
                
            # Create target page entity
            target_entity = Entity(
                id=f"page:{link['url']}",
                type=EntityType.WEB_PAGE,
                label=link.get('text', link['url']),
                properties={'url': link['url']}
            )
            self.kg.add_entity(target_entity)
            
            # Add relationship
            rel_type = self._determine_link_relationship(link)
            self.kg.add_relationship(
                Relationship(
                    source_id=source_entity.id,
                    target_id=target_entity.id,
                    type=rel_type,
                    properties={
                        'anchor_text': link.get('text', ''),
                        'link_type': link.get('type', 'link')
                    }
                )
            )
    
    def _determine_link_relationship(self, link: Dict[str, str]) -> RelationshipType:
        """Determine the type of relationship based on link properties."""
        url = link.get('url', '').lower()
        text = link.get('text', '').lower()
        
        # Check URL patterns
        for pattern, rel_type in self.config.RELATIONSHIP_RULES:
            if re.search(pattern, url):
                return rel_type
        
        # Check link text
        if 'download' in text or 'download' in url:
            return RelationshipType.PROVIDES
        if 'documentation' in text or 'docs' in text or 'documentation' in url:
            return RelationshipType.DESCRIBES
        if 'related' in text or 'see also' in text:
            return RelationshipType.RELATED_TO
        
        # Default to generic link
        return RelationshipType.LINKS_TO
    
    def to_networkx(self) -> nx.DiGraph:
        """Convert the knowledge graph to a NetworkX DiGraph."""
        return self.kg.to_networkx()
    
    def save(self, path: str, format: str = 'graphml') -> None:
        """Save the knowledge graph to a file."""
        G = self.to_networkx()
        
        if format.lower() == 'graphml':
            nx.write_graphml(G, path)
        elif format.lower() == 'gexf':
            nx.write_gexf(G, path)
        elif format.lower() == 'json':
            import json
            data = nx.node_link_data(G)
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Knowledge graph saved to {path} ({format})")
