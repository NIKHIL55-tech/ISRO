import os
import json
import tempfile
import pytest
from pathlib import Path

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.entities import Entity, EntityType, Relationship
from src.models.knowledge_graph import KnowledgeGraph

def test_knowledge_graph_basic_operations():
    """Test basic operations of the KnowledgeGraph class."""
    # Initialize graph
    kg = KnowledgeGraph()
    
    # Create test entities
    satellite = Entity(
        id="sat_insat3d",
        type=EntityType.SATELLITE,
        name="INSAT-3D",
        description="Indian geostationary weather satellite",
        attributes={"launch_year": 2013, "status": "active"}
    )
    
    instrument = Entity(
        id="inst_vhrr",
        type=EntityType.INSTRUMENT,
        name="VHRR",
        description="Very High Resolution Radiometer",
        attributes={"resolution": "1km"}
    )
    
    # Test adding entities
    kg.add_entity(satellite)
    kg.add_entity(instrument)
    
    # Verify entities were added
    assert kg.get_entity("sat_insat3d") == satellite
    assert kg.get_entity("inst_vhrr") == instrument
    
    # Test adding a relationship
    rel = Relationship(
        source_id="sat_insat3d",
        target_id="inst_vhrr",
        type="carries",
        attributes={"since": 2013}
    )
    assert kg.add_relationship(rel) is True
    
    # Test finding related entities
    related = kg.find_related_entities("sat_insat3d")
    assert len(related) == 1
    assert related[0]["entity"].id == "inst_vhrr"
    assert related[0]["relationship"] == "carries"

def test_knowledge_graph_persistence():
    """Test saving and loading the knowledge graph."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        try:
            # Initialize and populate graph
            kg = KnowledgeGraph()
            
            # Add a test entity
            entity = Entity(
                id="test_entity",
                type=EntityType.SATELLITE,
                name="Test Satellite",
                description="Test description",
                attributes={"test": "value"},
                source_urls={"http://example.com"}
            )
            kg.add_entity(entity)
            
            # Save and reload
            kg.save(tmp.name)
            
            # Load into a new graph
            loaded_kg = KnowledgeGraph.load(tmp.name)
            
            # Verify the loaded graph
            loaded_entity = loaded_kg.get_entity("test_entity")
            assert loaded_entity is not None
            assert loaded_entity.name == "Test Satellite"
            assert loaded_entity.type == EntityType.SATELLITE
            assert loaded_entity.attributes["test"] == "value"
            assert "http://example.com" in loaded_entity.source_urls
            
        finally:
            # Clean up
            try:
                os.unlink(tmp.name)
            except:
                pass

def test_knowledge_graph_queries():
    """Test various query capabilities."""
    kg = KnowledgeGraph()
    
    # Add test data
    entities = [
        Entity("s1", EntityType.SATELLITE, "INSAT-3D", "", {"status": "active"}),
        Entity("s2", EntityType.SATELLITE, "SCATSAT-1", "", {"status": "active"}),
        Entity("i1", EntityType.INSTRUMENT, "VHRR", ""),
        Entity("i2", EntityType.INSTRUMENT, "SCAT", ""),
        Entity("p1", EntityType.PARAMETER, "SST", ""),
    ]
    
    for entity in entities:
        kg.add_entity(entity)
    
    # Add relationships
    relationships = [
        ("s1", "i1", "carries"),
        ("s2", "i2", "carries"),
        ("i1", "p1", "measures"),
    ]
    
    for src, tgt, rel_type in relationships:
        kg.add_relationship(Relationship(src, tgt, rel_type))
    
    # Test entity type filtering
    satellites = kg.find_entities(EntityType.SATELLITE)
    assert len(satellites) == 2
    assert {s.name for s in satellites} == {"INSAT-3D", "SCATSAT-1"}
    
    # Test attribute filtering
    active_sats = kg.find_entities(EntityType.SATELLITE, status="active")
    assert len(active_sats) == 2
    
    # Test relationship queries
    insat_instruments = kg.find_related_entities("s1", "carries")
    assert len(insat_instruments) == 1
    assert insat_instruments[0]["entity"].name == "VHRR"
    
    # Test multi-hop query
    sst_related = kg.find_related_entities("s1", max_hops=2)
    assert len(sst_related) == 2  # Should find both direct and 2-hop relationships
    assert any(r["entity"].name == "SST" for r in sst_related)
    
    # Test graph statistics
    stats = kg.get_stats()
    assert stats["num_entities"] == 5
    assert stats["num_relationships"] == 3
    assert stats["entity_types"]["SATELLITE"] == 2
    assert stats["relationship_types"]["carries"] == 2
