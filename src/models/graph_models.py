"""
Core data models for the site-agnostic knowledge graph.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
from enum import Enum
from datetime import datetime

class EntityType(str, Enum):
    """Types of entities in the knowledge graph."""
    WEB_PAGE = "WebPage"
    CONCEPT = "Concept"
    MEDIA = "Media"
    NAMED_ENTITY = "NamedEntity"
    DATA_PRODUCT = "DataProduct"
    DATASET = "Dataset"
    ORGANIZATION = "Organization"
    PERSON = "Person"
    LOCATION = "Location"

class RelationshipType(str, Enum):
    """Types of relationships between entities."""
    # Structural relationships
    LINKS_TO = "linksTo"
    CONTAINS = "contains"
    # Content relationships
    MENTIONS = "mentions"
    REFERENCES = "references"
    # Semantic relationships
    SIMILAR_TO = "similarTo"
    RELATED_TO = "relatedTo"
    # Domain-specific
    PROVIDES = "provides"
    DESCRIBES = "describes"
    PART_OF = "partOf"
    USES = "uses"
    CREATED_BY = "createdBy"

@dataclass
class Entity:
    """Base entity in the knowledge graph."""
    id: str
    type: EntityType
    label: str
    properties: Dict[str, Any] = field(default_factory=dict)
    source_url: Optional[str] = None
    last_updated: datetime = field(default_factory=datetime.utcnow)

@dataclass
class Relationship:
    """Relationship between two entities."""
    source_id: str
    target_id: str
    type: RelationshipType
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

class KnowledgeGraph:
    """In-memory knowledge graph storage."""
    
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []
        
    def add_entity(self, entity: Entity) -> None:
        """Add or update an entity in the graph."""
        self.entities[entity.id] = entity
        
    def add_relationship(self, relationship: Relationship) -> None:
        """Add a relationship between entities."""
        if (relationship.source_id in self.entities and 
            relationship.target_id in self.entities):
            self.relationships.append(relationship)
            return True
        return False
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Retrieve an entity by ID."""
        return self.entities.get(entity_id)
    
    def find_entities(self, **properties) -> List[Entity]:
        """Find entities matching the given properties."""
        results = []
        for entity in self.entities.values():
            match = all(
                entity.properties.get(k) == v 
                for k, v in properties.items()
            )
            if match:
                results.append(entity)
        return results
    
    def to_networkx(self) -> 'nx.DiGraph':
        """Convert to NetworkX DiGraph for analysis/visualization."""
        G = nx.DiGraph()
        
        # Add nodes
        for entity_id, entity in self.entities.items():
            G.add_node(
                entity_id,
                label=entity.label,
                type=entity.type,
                **entity.properties
            )
            
        # Add edges
        for rel in self.relationships:
            G.add_edge(
                rel.source_id,
                rel.target_id,
                type=rel.type,
                **rel.properties
            )
            
        return G
