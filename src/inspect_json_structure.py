import json
import os
from pprint import pprint

def inspect_json(file_path: str, depth: int = 1):
    if not os.path.exists(file_path):
        print("❌ File not found:", file_path)
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print("❌ Failed to decode JSON:", e)
            return

    print(f"\n✅ JSON loaded successfully: {file_path}")
    print(f"Top-level data type: {type(data)}")

    if isinstance(data, dict):
        print("📚 Top-level keys:", list(data.keys()))

        for key, value in data.items():
            print(f"\n🔍 Key: '{key}' → type: {type(value)}")
            if isinstance(value, list):
                print(f"   📏 List length: {len(value)}")
                if value:
                    print(f"   🔑 First item keys (if dict): {list(value[0].keys()) if isinstance(value[0], dict) else 'Not a dict'}")
            elif isinstance(value, dict) and depth > 0:
                print(f"   🔑 Nested dict keys: {list(value.keys())}")
            else:
                print("   📄 Value preview:", str(value)[:100])

    elif isinstance(data, list):
        print(f"📏 Top-level list length: {len(data)}")
        if data:
            print(f"🔍 First item type: {type(data[0])}")
            if isinstance(data[0], dict):
                print(f"   🔑 Keys in first item: {list(data[0].keys())}")
            else:
                print(f"   📄 First item preview: {str(data[0])[:100]}")

    else:
        print("⚠️ Unexpected JSON structure (not dict or list)")

# Example usage
if __name__ == "__main__":
    inspect_json("data/processed/crawl_results_www_mosdac_gov_in.json")
