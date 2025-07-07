import re
import spacy
import os
import json
nlp = spacy.load("en_core_web_sm")

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)  # normalize whitespace
    text = re.sub(r"<[^>]+>", "", text)  # remove HTML tags
    return text.strip()

def chunk_text(text: str, chunk_size: int = 500) -> list:
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks
import json
import os
from bs4 import BeautifulSoup
from .text_utils import clean_text  # Your own cleaning function

# utils/text_utils.py

def load_documents(json_path: str):
    """Load and clean text from crawled pages"""
    print(f"\nAttempting to load documents from: {json_path}")
    
    if not os.path.exists(json_path):
        print(f"❌ File not found: {json_path}")
        return []

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"✅ Successfully loaded JSON file")
            print(f"JSON structure: {list(data.keys())}")
    except Exception as e:
        print(f"❌ Error loading JSON: {str(e)}")
        return []

    pages = data.get("pages", {})
    print(f"Found {len(pages)} pages in JSON")
    print(pages.keys())
    

    documents = []
    for url, content in pages.items():
        print(content.keys())
        
        
        print(f"\nProcessing URL: {url}")
        raw_text = ""

        # Try plain text field
        if isinstance(content, dict):
            raw_text = content.get("text") or content.get("content_text") or ""
            #print(raw_text)
            # Fallback to parsing HTML if present
            
        
        cleaned = clean_text(raw_text)
        print(cleaned)
        if len(cleaned) > 100:  # Only keep substantial content
            metadata = {
                # Basic metadata
                "description": content.get("description", ""),
                "keywords": content.get("keywords", []),
                
                # Content analysis
                "word_count": content.get("word_count", 0),
                "reading_difficulty": content.get("reading_difficulty", ""),
                "key_phrases": content.get("key_phrases", []),
                
                # Document info
                "file_type": content.get("file_type", ""),
                "file_size": content.get("file_size", ""),
                "content_hash": content.get("content_hash", ""),
                "page_type": content.get("page_type", ""),
                
                # Structured elements
                "headings": content.get("headings", []),
                "images": content.get("images", []),
                "links": content.get("links", []),
                "forms": content.get("forms", []),
                "tables": content.get("tables", []),
                
                # Additional metadata
                "meta_tags": content.get("meta_tags", {}),
                "structured_data": content.get("structured_data", {}),
                "document_metadata": content.get("document_metadata", {})
            }
            doc = {
                "text": cleaned,
                "title": content.get("title", ""),
                "source_url": url,
                "category": content.get("content_categories", ""),
                "last_updated": content.get("last_updated", ""),
                "metadata": metadata
            }
            print(doc)
            print("\nDocument Details:")
            print(f"Title: {doc['title']}")
            print(f"URL: {doc['source_url']}")
            print(f"Categories: {doc['category']}")
            print(f"Word Count: {metadata['word_count']}")
            print(f"Key Phrases: {metadata['key_phrases'][:5]}...")  # Show first 5 key phrases
            print(f"File Type: {metadata['file_type']}")
            print(f"Number of Images: {len(metadata['images'])}")
            print(f"Number of Links: {len(metadata['links'])}")
            print(f"Number of Tables: {len(metadata['tables'])}")
            print("-------------------------------------------")
            
            documents.append(doc)
            
            print(f"✅ Added document: {doc['title'][:50]}...")
        else:
            print(f"⚠️ Skipped document (too short): {url}")

    print(f"\n✅ Successfully processed {len(documents)} documents")
    return documents