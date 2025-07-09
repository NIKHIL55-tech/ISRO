import sys
import os
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.nlp.entity_extractor import MOSDACEntityExtractor
from src.models.knowledge_graph import KnowledgeGraph
from src.models.entities import EntityType

def test_entity_extraction():
    # Sample MOSDAC document text
    sample_text = """
    The INSAT-3D satellite, launched by ISRO, carries the Very High Resolution Radiometer (VHRR) 
    and Sounder instruments. It provides data on Sea Surface Temperature (SST) and Atmospheric 
    profiles. The satellite monitors the Indian Ocean region, including the Bay of Bengal.
    """
    
    # Initialize extractor and knowledge graph
    extractor = MOSDACEntityExtractor()
    kg = KnowledgeGraph()
    
    # Extract entities and relationships
    entities, relationships = extractor.process_document(sample_text, "http://example.com/mosdac")
    
    # Add entities to knowledge graph
    for entity in entities:
        kg.add_entity(entity)
    
    # Add relationships to knowledge graph
    for rel in relationships:
        kg.add_relationship(rel)
    
    # Print results
    print("\n=== Extracted Entities ===")
    for entity in entities:
        print(f"- {entity.type.value}: {entity.name} (ID: {entity.id})")
    
    print("\n=== Extracted Relationships ===")
    for rel in relationships:
        source = kg.get_entity(rel.source_id)
        target = kg.get_entity(rel.target_id)
        if source and target:
            print(f"- {source.name} --[{rel.type}]--> {target.name}")
    
    # Query the knowledge graph
    print("\n=== Query Results ===")
    satellite = next((e for e in entities if e.type == EntityType.SATELLITE), None)
    if satellite:
        print(f"\nEntities related to {satellite.name}:")
        related = kg.find_related_entities(satellite.id)
        for item in related:
            target = kg.get_entity(item['entity'].id)
            if target:
                print(f"- {item['relationship']}: {target.name} (Confidence: {item['attributes'].get('confidence', 1.0)})")

if __name__ == "__main__":
    test_entity_extraction()
