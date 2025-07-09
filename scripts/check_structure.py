#!/usr/bin/env python3
"""
Check the structure of a JSON file.
"""
import json
import sys
from pathlib import Path

def print_structure(obj, indent=0):
    """Recursively print the structure of a JSON object."""
    if isinstance(obj, dict):
        print('  ' * indent + 'Object with keys:')
        for key, value in obj.items():
            print('  ' * (indent + 1) + f'{key}: ', end='')
            if isinstance(value, (dict, list)):
                print()
                print_structure(value, indent + 2)
            else:
                value_str = str(value)
                if len(value_str) > 50:
                    value_str = value_str[:50] + '...'
                print(f"{type(value).__name__} = {value_str}")
    elif isinstance(obj, list):
        print('  ' * indent + f'List of {len(obj)} items')
        if obj:
            print('  ' * (indent + 1) + 'First item:')
            print_structure(obj[0], indent + 2)
    else:
        value_str = str(obj)
        if len(value_str) > 50:
            value_str = value_str[:50] + '...'
        print('  ' * indent + f"{type(obj).__name__} = {value_str}")

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <json_file>")
        sys.exit(1)
    
    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"Error: File '{file_path}' not found")
        sys.exit(1)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        sys.exit(1)
    
    print(f"File: {file_path}")
    print("=" * 80)
    print_structure(data)

if __name__ == "__main__":
    main()
