from typing import Dict, List, Optional, Set, Any, Union
import json
import networkx as nx
from pathlib import Path
from networkx.readwrite import json_graph
from .entities import Entity, EntityType, Relationship

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.entity_index: Dict[str, Entity] = {}
    
    # === Core Methods ===
    def add_entity(self, entity: Entity) -> None:
        """Add or update an entity in the graph."""
        if not isinstance(entity, Entity):
            raise ValueError("Entity must be an instance of Entity class")
            
        if entity.id not in self.entity_index:
            self.entity_index[entity.id] = entity
            self.graph.add_node(entity.id, **{
                'type': entity.type.value,
                'name': entity.name,
                **entity.attributes
            })
        else:
            # Update existing entity
            existing = self.entity_index[entity.id]
            existing.attributes.update(entity.attributes)
            existing.source_urls.update(entity.source_urls)
            
            # Update node attributes
            for key, value in {
                'type': entity.type.value,
                'name': entity.name,
                **entity.attributes
            }.items():
                self.graph.nodes[entity.id][key] = value
    
    def add_relationship(self, relationship: Relationship) -> bool:
        """Add a relationship between entities. Returns True if successful."""
        if not (relationship.source_id in self.entity_index and 
               relationship.target_id in self.entity_index):
            return False
        
        # Get existing edges between these nodes
        existing_edges = self.graph.get_edge_data(relationship.source_id, relationship.target_id)
        edge_count = len(existing_edges) if existing_edges else 0
        
        self.graph.add_edge(
            relationship.source_id,
            relationship.target_id,
            key=f"{relationship.type}_{edge_count}",
            type=relationship.type,
            **relationship.attributes
        )
        return True
    
    # === Query Methods ===
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        return self.entity_index.get(entity_id)
    
    def find_entities(self, 
                     entity_type: Optional[Union[EntityType, str]] = None,
                     **attributes) -> List[Entity]:
        """
        Find entities by type and attributes.
        
        Args:
            entity_type: Optional entity type to filter by
            **attributes: Attribute key-value pairs to filter by
            
        Returns:
            List of matching entities
        """
        if isinstance(entity_type, str):
            try:
                entity_type = EntityType(entity_type.lower())
            except ValueError:
                return []
                
        results = []
        for entity in self.entity_index.values():
            if entity_type is not None and entity.type != entity_type:
                continue
            if all(entity.attributes.get(k) == v for k, v in attributes.items()):
                results.append(entity)
        return results
    
    def find_related_entities(self, 
                            entity_id: str,
                            relationship_type: Optional[str] = None,
                            max_hops: int = 1) -> List[Dict[str, Any]]:
        """
        Find entities related to the given entity.
        
        Args:
            entity_id: ID of the source entity
            relationship_type: Optional relationship type to filter by
            max_hops: Maximum number of hops for traversal (default: 1)
            
        Returns:
            List of dictionaries containing related entities and relationship info
        """
        if entity_id not in self.graph:
            return []
            
        results = []
        visited = set()
        
        # BFS queue: (node_id, distance, path)
        from collections import deque
        queue = deque([(entity_id, 0, [entity_id])])
        
        while queue:
            current_id, distance, path = queue.popleft()
            
            # Skip if we've already visited this node or exceeded max_hops
            if current_id in visited or distance > max_hops:
                continue
                
            visited.add(current_id)
            
            # Process neighbors
            for source, target, data in self.graph.out_edges(current_id, data=True):
                if target == entity_id:  # Avoid cycles
                    continue
                    
                # Check relationship type if specified
                if relationship_type is not None and data.get('type') != relationship_type:
                    continue
                    
                target_entity = self.entity_index.get(target)
                if not target_entity:
                    continue
                    
                # Add to results if this is a new relationship
                if distance + 1 <= max_hops:
                    results.append({
                        'entity': target_entity,
                        'relationship': data.get('type'),
                        'distance': distance + 1,
                        'path': path + [target],
                        **{k: v for k, v in data.items() if k != 'type'}
                    })
                    
                    # Continue BFS if we haven't reached max_hops
                    if distance + 1 < max_hops:
                        queue.append((target, distance + 1, path + [target]))
        
        return results
    
    # === Persistence Methods ===
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save the knowledge graph to a file.
        
        Args:
            filepath: Path to save the graph to (JSON format)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for serialization
        graph_data = {
            'nodes': [],
            'edges': [],
            'entity_index': {}
        }
        
        # Add nodes
        for node_id, node_data in self.graph.nodes(data=True):
            node_data = node_data.copy()
            node_data['id'] = node_id
            graph_data['nodes'].append(node_data)
        
        # Add edges
        for source, target, key, data in self.graph.edges(data=True, keys=True):
            edge_data = data.copy()
            edge_data.update({
                'source': source,
                'target': target,
                'key': key
            })
            graph_data['edges'].append(edge_data)
        
        # Add entity index
        for entity_id, entity in self.entity_index.items():
            graph_data['entity_index'][entity_id] = {
                'id': entity.id,
                'type': entity.type.value,
                'name': entity.name,
                'description': entity.description,
                'attributes': entity.attributes,
                'source_urls': list(entity.source_urls)
            }
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'KnowledgeGraph':
        """
        Load a knowledge graph from a file.
        
        Args:
            filepath: Path to the graph file (JSON format)
            
        Returns:
            Loaded KnowledgeGraph instance
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        kg = cls()
        
        # Rebuild entity index first
        for entity_id, entity_data in graph_data.get('entity_index', {}).items():
            entity = Entity(
                id=entity_data['id'],
                type=EntityType(entity_data['type']),
                name=entity_data['name'],
                description=entity_data.get('description', ''),
                attributes=entity_data.get('attributes', {}),
                source_urls=set(entity_data.get('source_urls', []))
            )
            kg.entity_index[entity_id] = entity
        
        # Rebuild graph
        for node_data in graph_data.get('nodes', []):
            node_id = node_data.pop('id')
            kg.graph.add_node(node_id, **node_data)
        
        for edge_data in graph_data.get('edges', []):
            source = edge_data.pop('source')
            target = edge_data.pop('target')
            key = edge_data.pop('key', None)
            kg.graph.add_edge(source, target, key=key, **edge_data)
        
        return kg
    
    # === Utility Methods ===
    def to_networkx(self) -> nx.MultiDiGraph:
        """Convert to NetworkX graph for visualization or analysis."""
        return self.graph
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        return {
            'num_entities': len(self.entity_index),
            'num_relationships': self.graph.number_of_edges(),
            'entity_types': {
                et.name: sum(1 for e in self.entity_index.values() if e.type == et)
                for et in EntityType
            },
            'relationship_types': {
                data['type']: sum(1 for _, _, d in self.graph.edges(data=True) 
                               if d.get('type') == data['type'])
                for _, _, data in self.graph.edges(data=True)
            }
        }
