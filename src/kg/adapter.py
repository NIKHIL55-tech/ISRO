"""
Adapter to connect the website crawler with the knowledge graph builder.
"""
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from models.graph_models import EntityType, RelationshipType
from kg.generic_builder import GenericKnowledgeGraphBuilder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrawlResultAdapter:
    """Adapts crawler output to knowledge graph builder input."""
    
    def __init__(self, output_dir: str = "data/processed"):
        """Initialize the adapter with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.kg_builder = GenericKnowledgeGraphBuilder()
    
    def process_crawl_results(self, crawl_results: Dict[str, Any]) -> None:
        """Process crawl results and build knowledge graph."""
        if not crawl_results or 'pages' not in crawl_results:
            logger.warning("No pages found in crawl results")
            return
        
        logger.info(f"Processing {len(crawl_results['pages'])} pages...")
        
        for url, page_data in crawl_results['pages'].items():
            try:
                self.kg_builder.process_page(url, page_data)
                logger.debug(f"Processed page: {url}")
            except Exception as e:
                logger.error(f"Error processing page {url}: {str(e)}")
        
        logger.info("Finished processing crawl results")
    
    def save_knowledge_graph(self, filename: str = "knowledge_graph", 
                           format: str = "graphml") -> str:
        """Save the knowledge graph to a file.
        
        Args:
            filename: Base name of the output file (without extension)
            format: Output format ('graphml', 'gexf', or 'json')
            
        Returns:
            Path to the saved file
        """
        output_path = self.output_dir / f"{filename}.{format.lower()}"
        self.kg_builder.save(str(output_path), format)
        return str(output_path)
    
    def get_entity_count(self) -> int:
        """Get the number of entities in the knowledge graph."""
        return len(self.kg_builder.kg.entities)
    
    def get_relationship_count(self) -> int:
        """Get the number of relationships in the knowledge graph."""
        return len(self.kg_builder.kg.relationships)
    
    def get_entity_types(self) -> Dict[str, int]:
        """Get counts of each entity type in the knowledge graph."""
        type_counts = {}
        for entity in self.kg_builder.kg.entities.values():
            type_name = entity.type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        return type_counts
    
    def get_relationship_types(self) -> Dict[str, int]:
        """Get counts of each relationship type in the knowledge graph."""
        type_counts = {}
        for rel in self.kg_builder.kg.relationships:
            type_name = rel.type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        return type_counts
