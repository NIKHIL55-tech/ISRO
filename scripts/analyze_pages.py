#!/usr/bin/env python3
"""
Analyze the structure of the crawl results with a focus on page content.
"""
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from pprint import pprint

def load_json_file(file_path: str) -> Optional[Dict]:
    """Load and parse a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return None

def analyze_pages(data: Dict) -> Dict:
    """Analyze the pages in the crawl results."""
    if 'pages' not in data or not isinstance(data['pages'], dict):
        return {"error": "No 'pages' dictionary found in the data"}
    
    pages = data['pages']
    non_null_pages = {k: v for k, v in pages.items() if v is not None}
    
    if not non_null_pages:
        return {"error": "No non-null pages found"}
    
    # Get the first non-null page
    first_page_url = next(iter(non_null_pages.keys()))
    first_page = non_null_pages[first_page_url]
    
    # Count pages with different content types
    content_types = {}
    for url, page in non_null_pages.items():
        if not isinstance(page, dict):
            content_type = type(page).__name__
        else:
            content_type = 'unknown'
            if 'content' in page and page['content']:
                content = page['content']
                if isinstance(content, dict):
                    content_type = 'dict_content'
                elif isinstance(content, str):
                    content_type = 'text_content'
                elif content is None:
                    content_type = 'null_content'
                else:
                    content_type = f'other_{type(content).__name__}'
            else:
                content_type = 'no_content'
        
        content_types[content_type] = content_types.get(content_type, 0) + 1
    
    # Analyze the first page's structure
    page_structure = {}
    if isinstance(first_page, dict):
        for key, value in first_page.items():
            if value is None:
                page_structure[key] = None
            elif isinstance(value, (str, int, float, bool)):
                page_structure[key] = f"{type(value).__name__}: {str(value)[:100]}"
            elif isinstance(value, (list, dict)):
                page_structure[key] = {
                    'type': type(value).__name__,
                    'length': len(value) if hasattr(value, '__len__') else None,
                    'sample': value[0] if isinstance(value, list) and value else None
                }
            else:
                page_structure[key] = f"{type(value).__name__} (unhandled)"
    
    return {
        'total_pages': len(pages),
        'non_null_pages': len(non_null_pages),
        'content_types': content_types,
        'first_page_url': first_page_url,
        'first_page_structure': page_structure,
        'first_page_sample': first_page
    }

def main():
    parser = argparse.ArgumentParser(description='Analyze crawl results.')
    parser.add_argument('--input', required=True, help='Path to the crawl results JSON file')
    parser.add_argument('--output', help='Path to save the analysis results (JSON)')
    args = parser.parse_args()
    
    # Load the data
    print(f"Loading data from: {args.input}")
    data = load_json_file(args.input)
    if data is None:
        return
    
    # Analyze the data
    print("Analyzing data...")
    analysis = {
        'root_keys': list(data.keys()),
        'pages_analysis': analyze_pages(data)
    }
    
    # Print the analysis
    print("\n=== Analysis Results ===\n")
    print(f"Root keys in JSON: {', '.join(analysis['root_keys'])}\n")
    
    pages_analysis = analysis['pages_analysis']
    if 'error' in pages_analysis:
        print(f"Error: {pages_analysis['error']}")
        return
    
    print(f"Total pages: {pages_analysis['total_pages']}")
    print(f"Non-null pages: {pages_analysis['non_null_pages']}\n")
    
    print("Content types found:")
    for content_type, count in pages_analysis['content_types'].items():
        print(f"- {content_type}: {count} pages")
    
    print("\nFirst page URL:", pages_analysis['first_page_url'])
    
    print("\nFirst page structure:")
    pprint(pages_analysis['first_page_structure'], indent=2, width=100, sort_dicts=False)
    
    print("\nFirst page sample content:")
    pprint(pages_analysis['first_page_sample'], indent=2, width=100, sort_dicts=False)
    
    # Save results if output path is provided
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"\nAnalysis saved to: {output_path}")

if __name__ == "__main__":
    main()
