"""
Modular Knowledge Graph Builder

This script builds a knowledge graph from crawled website data using a configuration-based approach.
It's designed to be website-agnostic and easily extensible.
"""
import os
import re
import json
import logging
import networkx as nx
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple, Union
from urllib.parse import urlparse, urljoin
from dataclasses import asdict

from config.kg_config import get_config, BaseKGConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KnowledgeGraphBuilder:
    """Builds a knowledge graph from crawled website data."""
    
    def __init__(self, domain: str, config: Optional[BaseKGConfig] = None):
        """Initialize the knowledge graph builder.
        
        Args:
            domain: The website domain (e.g., 'mosdac.gov.in')
            config: Optional custom configuration. If None, will try to load based on domain.
        """
        self.domain = domain
        self.config = config or get_config(domain)
        self.graph = nx.MultiDiGraph(name=f"{self.domain} Knowledge Graph")
        
        # Create output directories
        os.makedirs(self.config.processed_dir, exist_ok=True)
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Define output paths
        self.graphml_path = os.path.join(self.config.output_dir, f"kg_{self.domain}.graphml")
        self.json_path = os.path.join(self.config.output_dir, f"kg_{self.domain}.json")
    
    def load_crawl_data(self) -> pd.DataFrame:
        """Load and preprocess crawled data."""
        csv_path = os.path.join(
            self.config.processed_dir, 
            f"crawl_results_{self.domain}.csv"
        )
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Crawl data not found at {csv_path}")
        
        logger.info(f"Loading crawl data from {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Basic preprocessing
        df = df.drop_duplicates(subset=['url'])
        df = df[df['url'].notna()]
        
        return df
    
    def extract_entities_from_page(self, url: str, content: str) -> Dict[str, List[Dict]]:
        """Extract entities from a single page."""
        entities = {}
        
        # Extract entities based on URL patterns
        for entity_type, pattern in self.config.patterns.url_patterns.items():
            matches = pattern.findall(url)
            if matches:
                if entity_type not in entities:
                    entities[entity_type] = []
                for match in matches:
                    entities[entity_type].append({
                        'name': match,
                        'source': 'url',
                        'url': url
                    })
        
        # Extract entities from content using keywords
        if content and isinstance(content, str):
            for entity_type, keywords in self.config.patterns.content_keywords.items():
                found = [kw for kw in keywords if kw.lower() in content.lower()]
                if found:
                    if entity_type not in entities:
                        entities[entity_type] = []
                    entities[entity_type].extend([{
                        'name': kw,
                        'source': 'content',
                        'url': url
                    } for kw in found])
        
        return entities
    
    def add_entities_to_graph(self, entities: Dict[str, List[Dict]], url: str) -> None:
        """Add extracted entities to the knowledge graph."""
        # Add page node if it doesn't exist
        if not self.graph.has_node(url):
            self.graph.add_node(url, type='page')
        
        # Add entities and relationships
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                entity_id = f"{entity_type}:{entity['name']}"
                
                # Add entity node if it doesn't exist
                if not self.graph.has_node(entity_id):
                    self.graph.add_node(entity_id, 
                                     type=entity_type,
                                     name=entity['name'])
                
                # Add relationship: entity -> page
                self.graph.add_edge(
                    entity_id, url,
                    relationship='appears_on',
                    source=entity.get('source', 'unknown')
                )
                
                # Add hierarchical relationships if defined
                if entity_type in self.config.relationships.hierarchy:
                    for child_type in self.config.relationships.hierarchy[entity_type]:
                        # This is a simplified example - in practice, you'd need
                        # additional logic to determine parent-child relationships
                        pass
    
    def build(self) -> None:
        """Build the knowledge graph from crawled data."""
        try:
            # Load crawl data
            df = self.load_crawl_data()
            logger.info(f"Processing {len(df)} pages from {self.domain}")
            
            # Process each page
            for _, row in df.iterrows():
                url = row.get('url', '')
                content = row.get('content', '')
                
                # Extract entities
                entities = self.extract_entities_from_page(url, content)
                
                # Add to graph
                self.add_entities_to_graph(entities, url)
            
            logger.info(f"Knowledge graph built with {len(self.graph.nodes())} nodes and {len(self.graph.edges())} edges")
            
        except Exception as e:
            logger.error(f"Error building knowledge graph: {str(e)}")
            raise
    
    def save(self, format: str = 'both') -> None:
        """Save the knowledge graph to disk."""
        try:
            if format in ['both', 'graphml']:
                nx.write_graphml(self.graph, self.graphml_path)
                logger.info(f"Graph saved to {self.graphml_path}")
            
            if format in ['both', 'json']:
                # Convert to node-link format for JSON serialization
                data = nx.node_link_data(self.graph)
                with open(self.json_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                logger.info(f"Graph saved to {self.json_path}")
                
        except Exception as e:
            logger.error(f"Error saving knowledge graph: {str(e)}")
            raise

def main():
    """Main function to build the knowledge graph."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build a knowledge graph from crawled website data.')
    parser.add_argument('--domain', type=str, required=True,
                       help='Website domain (e.g., mosdac.gov.in)')
    parser.add_argument('--format', type=str, default='both',
                       choices=['graphml', 'json', 'both'],
                       help='Output format(s)')
    
    args = parser.parse_args()
    
    try:
        # Initialize and build the knowledge graph
        kg_builder = KnowledgeGraphBuilder(domain=args.domain)
        kg_builder.build()
        kg_builder.save(format=args.format)
        
        logger.info("Knowledge graph construction completed successfully!")
        
    except Exception as e:
        logger.error(f"Failed to build knowledge graph: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())

# Regular expression patterns for extracting mission names
MISSION_PATTERNS = [
    r'INSAT-\d+[A-Z]*',  # INSAT-3D, INSAT-3DR, etc.
    r'KALPANA-\d+',     # KALPANA-1
    r'MeghaTropiques',
    r'SARAL-AltiKa',
    r'OCEANSAT-\d+',     # OCEANSAT-2, OCEANSAT-3
    r'SCATSAT-\d+',      # SCATSAT-1
    r'INSAT-\d+[A-Z]+'  # INSAT-3DS, etc.
]

# Keywords for identifying products
PRODUCT_KEYWORDS = {
    'rainfall', 'temperature', 'humidity', 'wind', 'cloud', 'aerosol',
    'soil moisture', 'snow', 'ice', 'ocean', 'atmosphere', 'precipitation',
    'radiation', 'water vapor', 'vegetation', 'land surface', 'sea surface',
    'current', 'wave', 'cyclone', 'typhoon', 'hurricane', 'flood', 'drought'
}

# Output paths
GRAPHML_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/processed/kg_entitycentric.graphml'))
JSON_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/processed/kg_entitycentric.json'))

# MOSDAC Sitemap as structured hierarchy
MOSDAC_SITE = {
    "Home": {
        "Missions": [
            "INSAT-3DR", "INSAT-3D", "KALPANA-1", "INSAT-3A", "MeghaTropiques", "SARAL-AltiKa", "OCEANSAT-2", "OCEANSAT-3", "INSAT-3DS", "SCATSAT-1"
        ],
        "Catalog": ["Satellite", "Insitu (AWS)", "RADAR"],
        "Galleries": [
            "Satellite Products", "Weather Forecast", "Ocean Forecast", "RADAR (DWR)", "Global Ocean Current"
        ],
        "Data Access": {
            "Order Data": {},
            "Open Data": {
                "Atmosphere": [
                    "Bayesian based MT-SAPHIR rainfall",
                    "GPS derived Integrated water vapour",
                    "GSMap ISRO Rain",
                    "METEOSAT8 Cloud Properties"
                ],
                "Land": [
                    "3D Volumetric TERLS DWRproduct",
                    "Inland Water Height",
                    "River Discharge",
                    "Soil Moisture"
                ],
                "Ocean": [
                    "Global Ocean Surface Current",
                    "High Resolution Sea Surface Salinity",
                    "Indian Mainland Coastal Product",
                    "Ocean Subsurface",
                    "Oceanic Eddies Detection",
                    "Sea Ice Occurrence Probability",
                    "Wave based Renewable Energy"
                ]
            }
        },
        "Cal-Val": {},
        "Forecast": {},
        "RSS Feeds": {},
        "Reports": {
            "Calibration": ["Insitu", "Relative"],
            "Validation": [],
            "Data Quality": [],
            "Weather": []
        },
        "Atlases": {},
        "Tools": {},
        "Sitemap": {},
        "Help": {},
        "Feedback": {},
        "About Us": {},
        "Contact Us": {},
        "Copyright Policy": {},
        "Data Access Policy": {},
        "Hyperlink Policy": {},
        "Privacy Policy": {},
        "Website Policies": {},
        "Terms & Conditions": {},
        "FAQs": {},
        "SignUp": {},
        "Login": {},
        "Logout": {}
    }
}

# Helper to recursively add nodes and edges to the graph
def add_hierarchy(G, parent, structure):
    if isinstance(structure, dict):
        for k, v in structure.items():
            G.add_node(k, type='category')
            if parent:
                G.add_edge(parent, k, relationship='has_section')
            add_hierarchy(G, k, v)
    elif isinstance(structure, list):
        for item in structure:
            # Missions and products are treated as entities
            if parent == 'Missions':
                G.add_node(item, type='mission')
                G.add_edge(parent, item, relationship='has_mission')
            elif parent in ['Atmosphere', 'Land', 'Ocean']:
                G.add_node(item, type='product', product_category=parent)
                G.add_edge(parent, item, relationship='has_product')
            elif parent == 'Catalog':
                G.add_node(item, type='catalog')
                G.add_edge(parent, item, relationship='has_catalog')
            elif parent == 'Galleries':
                G.add_node(item, type='gallery')
                G.add_edge(parent, item, relationship='has_gallery')
            elif parent == 'Calibration':
                G.add_node(item, type='calibration')
                G.add_edge(parent, item, relationship='has_calibration')
            else:
                G.add_node(item, type='item')
                G.add_edge(parent, item, relationship='has_item')

# Build the graph
G = nx.DiGraph()
add_hierarchy(G, None, MOSDAC_SITE)

# Optionally, link products to missions if naming matches
mission_names = {n for n, d in G.nodes(data=True) if d.get('type') == 'mission'}
product_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'product']
for prod in product_nodes:
    for mission in mission_names:
        if mission.replace('-', '').lower() in prod.replace(' ', '').lower():
            G.add_edge(mission, prod, relationship='provides_product')

# Save outputs
nx.write_graphml(G, GRAPHML_PATH)
data = nx.readwrite.json_graph.node_link_data(G)
with open(JSON_PATH, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)

print(f"Knowledge graph built from sitemap and saved to:\n  - {GRAPHML_PATH}\n  - {JSON_PATH}")

def extract_missions(title: str) -> Set[str]:
    """Extract mission names from a title using predefined patterns."""
    missions = set()
    if not isinstance(title, str):
        return missions
        
    for pat in MISSION_PATTERNS:
        found = re.findall(pat, title, flags=re.IGNORECASE)
        for m in found:
            missions.add(m.upper())
    return missions

def extract_products(title: str, key_phrases: List[str]) -> Set[str]:
    products = set()
    title_lower = title.lower()
    for prod in PRODUCT_KEYWORDS:
        if prod in title_lower:
            products.add(prod)
    for phrase in key_phrases:
        if phrase in PRODUCT_KEYWORDS:
            products.add(phrase)
    return products

def build_entity_kg() -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    page_to_missions = {}
    page_to_products = {}
    mission_nodes = set()
    product_nodes = set()
    url_to_title = {}
    
    # Load the crawl results using pandas
    df = load_crawl_results()
    
    for _, row in df.iterrows():
        url = row['url']
        title = row['title']
        key_phrases = []
        if pd.notna(row.get('key_phrases')):
            key_phrases = [k.strip() for k in str(row['key_phrases']).split(',') if k.strip()]
        url_to_title[url] = title
        
        missions = extract_missions(title)
        products = extract_products(title, key_phrases)
        page_to_missions[url] = missions
        page_to_products[url] = products
        
        for m in missions:
            mission_nodes.add(m)
        for p in products:
            product_nodes.add(p)
    # Add mission nodes
    for m in mission_nodes:
        G.add_node(m, type='mission')
    # Add product nodes
    for p in product_nodes:
        G.add_node(p, type='product')
    # Add relationships: mission -> product (if both found on same page)
    for url, missions in page_to_missions.items():
        products = page_to_products.get(url, set())
        for m in missions:
            for p in products:
                G.add_edge(m, p, relationship='provides', source_url=url)
    # Optionally, relate products to each other if found on same page
    for url, products in page_to_products.items():
        products = list(products)
        for i in range(len(products)):
            for j in range(i+1, len(products)):
                G.add_edge(products[i], products[j], relationship='related_product', source_url=url)
    # Optionally, add mapping from mission/product to page
    for url, missions in page_to_missions.items():
        for m in missions:
            G.add_edge(m, url, relationship='described_in', node_type='page')
    for url, products in page_to_products.items():
        for p in products:
            G.add_edge(p, url, relationship='described_in', node_type='page')
    return G

def save_kg(G):
    # Create a clean DiGraph for Gephi compatibility
    G_gephi = nx.DiGraph()
    
    # Add nodes with attributes
    for node, attrs in G.nodes(data=True):
        G_gephi.add_node(node, **attrs)
    
    # Add edges with attributes
    for src, tgt, attrs in G.edges(data=True):
        G_gephi.add_edge(src, tgt, **attrs)
    
    # Save as GraphML with explicit attribute types for Gephi
    for _, _, data in G_gephi.edges(data=True):
        for key in data:
            if isinstance(data[key], (list, dict, set)):
                data[key] = str(data[key])
    
    # Write the GraphML file
    nx.write_graphml(G_gephi, GRAPHML_PATH, named_key_ids=True)
    
    # Also save as JSON (node-link format)
    data = nx.readwrite.json_graph.node_link_data(G_gephi)
    with open(JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def load_crawl_results() -> pd.DataFrame:
    """Load the crawl results CSV into a pandas DataFrame."""
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 
                                         '../data/processed/crawl_results_www_mosdac_gov_in.csv'))
    return pd.read_csv(csv_path)

def get_entity_url_mapping(df: pd.DataFrame) -> Dict[str, str]:
    """Create a mapping from entity names to their URLs."""
    entity_to_url = {}
    
    for _, row in df.iterrows():
        url = row['url']
        title = row['title']
        
        # Extract the base title (before the | separator)
        base_title = title.split('|')[0].strip() if '|' in title else title.strip()
        
        # Skip if this is a generic page
        if base_title in ['Meteorological & Oceanographic Satellite Data Archival Centre', 
                         'Welcome to MOSDAC', 'Sitemap', 'Help']:
            continue
            
        # Add mapping if not already present
        if base_title and base_title not in entity_to_url:
            entity_to_url[base_title] = url
    
    return entity_to_url

def add_entity_page_links(G: nx.DiGraph, entity_to_url: Dict[str, str]) -> None:
    """Add 'described_in' edges from entities to their source pages."""
    for node, attrs in G.nodes(data=True):
        if node in entity_to_url:
            url = entity_to_url[node]
            # Add page node if it doesn't exist
            if not G.has_node(url):
                G.add_node(url, type='page', title=f"Page: {node}")
            # Add edge from entity to page
            if not G.has_edge(node, url):
                G.add_edge(node, url, relationship='described_in')

def main():
    # Build the basic entity graph
    G = build_entity_kg()
    
    # Load crawl results and add page links
    try:
        df = load_crawl_results()
        entity_to_url = get_entity_url_mapping(df)
        add_entity_page_links(G, entity_to_url)
        print(f"Added links for {len(entity_to_url)} entities to their source pages")
    except Exception as e:
        print(f"Warning: Could not add page links: {str(e)}")
    
    # Save the enhanced knowledge graph
    save_kg(G)
    print("Knowledge graph construction complete!")

if __name__ == "__main__":
    main()
