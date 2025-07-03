import json
import os

KG_PATH = os.path.join("data", "kg.json")

def load_kg():
    with open(KG_PATH, "r") as f:
        return json.load(f)

def query_kg(user_query):
    triples = load_kg()
    user_query = user_query.lower()

    matches = []
    for triple in triples:
        if any(term in user_query for term in [triple["subject"].lower(), triple["object"].lower()]):
            matches.append(f"{triple['subject']} {triple['predicate']} {triple['object']}")

    if matches:
        return "\n".join(matches)
    else:
        return "No matching entities found in the knowledge graph."
