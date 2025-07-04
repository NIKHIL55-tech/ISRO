#!/usr/bin/env python3
"""
Generic Knowledge Graph Builder for Websites

This script demonstrates how to build a knowledge graph from crawled website data
using the generic knowledge graph builder.
"""
import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import after path modification
from src.ingest import WebsiteCrawler
from src.kg.adapter import CrawlResultAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('knowledge_graph_builder.log')
    ]
)
logger = logging.getLogger(__name__)

def load_crawl_results(file_path: str) -> Optional[Dict[str, Any]]:
    """Load crawl results from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading crawl results: {str(e)}")
        return None

def main():
    """Main function to build the knowledge graph."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build a knowledge graph from crawled website data.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--crawl', action='store_true', help='Crawl a website first')
    group.add_argument('--input', type=str, help='Path to existing crawl results (JSON)')
    
    # Crawl options
    parser.add_argument('--url', type=str, help='URL to crawl (required with --crawl)')
    parser.add_argument('--max-pages', type=int, default=50, help='Maximum number of pages to crawl')
    parser.add_argument('--max-depth', type=int, default=3, help='Maximum depth to crawl')
    
    # Output options
    parser.add_argument('--output', type=str, default='knowledge_graph', 
                       help='Output filename (without extension)')
    parser.add_argument('--format', type=str, default='graphml', 
                       choices=['graphml', 'gexf', 'json'], 
                       help='Output format')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.crawl and not args.url:
        parser.error("--url is required with --crawl")
    
    # Step 1: Crawl the website or load existing results
    if args.crawl:
        logger.info(f"Crawling {args.url} (max pages: {args.max_pages}, depth: {args.max_depth})...")
        crawler = WebsiteCrawler(
            start_url=args.url,
            max_pages=args.max_pages,
            max_depth=args.max_depth,
            respect_robots=True,
            download_files=False
        )
        
        # Run the crawler
        site_structure = crawler.crawl()
        
        # Export results
        output_file = f"crawl_results_{urlparse(args.url).netloc}.json"
        crawler.export_results('json', output_file)
        logger.info(f"Crawl results saved to {output_file}")
        
        # Get the crawl results
        crawl_results = {
            'pages': crawler.page_contents,
            'site_structure': asdict(site_structure) if site_structure else {}
        }
    else:
        # Load existing crawl results
        logger.info(f"Loading crawl results from {args.input}...")
        crawl_results = load_crawl_results(args.input)
        if not crawl_results:
            logger.error("Failed to load crawl results")
            return 1
    
    # Step 2: Process crawl results and build knowledge graph
    logger.info("Building knowledge graph...")
    adapter = CrawlResultAdapter()
    adapter.process_crawl_results(crawl_results)
    
    # Step 3: Save the knowledge graph
    output_path = adapter.save_knowledge_graph(args.output, args.format)
    
    # Print summary
    print("\nKnowledge Graph Summary:")
    print(f"- Entities: {adapter.get_entity_count()}")
    print(f"- Relationships: {adapter.get_relationship_count()}")
    
    print("\nEntity Types:")
    for etype, count in adapter.get_entity_types().items():
        print(f"- {etype}: {count}")
    
    print("\nRelationship Types:")
    for rtype, count in adapter.get_relationship_types().items():
        print(f"- {rtype}: {count}")
    
    print(f"\nKnowledge graph saved to: {output_path}")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception("An error occurred")
        sys.exit(1)
