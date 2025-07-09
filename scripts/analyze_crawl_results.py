#!/usr/bin/env python3
"""
Analyze the structure of the crawl results JSON file.

Usage:
    python scripts/analyze_crawl_results.py --input data/processed/crawl_results_www_mosdac_gov_in.json
"""
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from pprint import pprint

def analyze_json_structure(data: Any, path: str = "") -> Dict:
    """Recursively analyze the structure of a JSON object."""
    if isinstance(data, dict):
        result = {
            "type": "object",
            "keys": {},
            "total_items": len(data)
        }
        for key, value in data.items():
            result["keys"][key] = analyze_json_structure(value, f"{path}.{key}" if path else key)
        return result
    elif isinstance(data, list):
        if not data:
            return {"type": "array", "item_type": "empty", "total_items": 0}
        
        # Check if all items have the same type
        item_types = {type(item).__name__ for item in data}
        item_type = f"union[{', '.join(sorted(item_types))}]".replace('NoneType', 'null')
        
        # If it's a list of objects, check if they have similar structure
        if all(isinstance(item, dict) for item in data if item is not None):
            sample_item = next((item for item in data if item is not None), {})
            return {
                "type": "array",
                "item_type": item_type,
                "sample_item": analyze_json_structure(sample_item, f"{path}[]") if sample_item else "empty",
                "total_items": len(data),
                "unique_items": len({json.dumps(item, sort_keys=True) for item in data if item is not None})
            }
        return {
            "type": "array",
            "item_type": item_type,
            "sample_item": analyze_json_structure(data[0], f"{path}[0]") if data and data[0] is not None else None,
            "total_items": len(data)
        }
    else:
        return {
            "type": type(data).__name__,
            "value": str(data)[:100] + ("..." if len(str(data)) > 100 else ""),
            "length": len(data) if hasattr(data, '__len__') and not isinstance(data, (str, bytes)) else None
        }

def analyze_pages_structure(pages: Dict[str, Any]) -> Dict:
    """Analyze the structure of the pages dictionary."""
    if not pages:
        return {"error": "No pages found"}
    
    # Get sample page
    first_page_url = next(iter(pages.keys()))
    first_page = pages[first_page_url]
    
    # Count non-null pages
    non_null_pages = {k: v for k, v in pages.items() if v is not None}
    
    return {
        "total_pages": len(pages),
        "non_null_pages": len(non_null_pages),
        "sample_page_url": first_page_url,
        "sample_page_structure": analyze_json_structure(first_page) if first_page else None
    }

def main():
    parser = argparse.ArgumentParser(description='Analyze the structure of crawl results JSON file.')
    parser.add_argument('--input', required=True, help='Path to the crawl results JSON file')
    parser.add_argument('--output', help='Path to save the analysis results (JSON)')
    args = parser.parse_args()
    
    # Load the JSON file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found")
        return
    
    print(f"Loading crawl results from: {input_path}")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return
    
    # Analyze the structure
    print("\n=== Analyzing JSON Structure ===")
    structure = {
        "root_keys": list(data.keys()),
        "structure": {}
    }
    
    for key, value in data.items():
        print(f"\nAnalyzing key: {key}")
        if key == 'pages' and isinstance(value, dict):
            structure["structure"][key] = analyze_pages_structure(value)
        else:
            structure["structure"][key] = analyze_json_structure(value, key)
    
    # Print summary
    print("\n=== Analysis Summary ===")
    print(f"Root keys: {', '.join(structure['root_keys'])}")
    
    if 'pages' in structure['structure']:
        pages_info = structure['structure']['pages']
        print(f"\nPages Analysis:")
        print(f"- Total pages: {pages_info['total_pages']}")
        print(f"- Non-null pages: {pages_info['non_null_pages']}")
        print(f"- Sample page URL: {pages_info['sample_page_url']}")
        
        if 'sample_page_structure' in pages_info and pages_info['sample_page_structure']:
            print("\nSample Page Structure:")
            pprint(pages_info['sample_page_structure'], indent=2, width=120, sort_dicts=False)
    
    # Save results if output path is provided
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(structure, f, indent=2, default=str)
        print(f"\nAnalysis saved to: {output_path}")

if __name__ == "__main__":
    main()
