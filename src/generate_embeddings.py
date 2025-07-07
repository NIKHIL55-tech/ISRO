# src/generate_embeddings.py
from chromadb import PersistentClient
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from utils.text_utils import chunk_text, load_documents
import numpy as np
import os
from tqdm import tqdm

class DocumentEmbeddings:
    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        collection_name: str = 'mosdac_docs',
        persist_dir: str = 'data/embeddings'
    ):
        print(f"Initializing DocumentEmbeddings with model: {model_name}")
        self.model = SentenceTransformer(model_name)

        # Create persist directory if it doesn't exist
        os.makedirs(persist_dir, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = PersistentClient(path=persist_dir)
        print(f"ChromaDB client initialized at: {persist_dir}")

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Collection '{collection_name}' ready")

    def process_documents(self, documents: List[Dict], chunk_size: int = 500):
        """Process documents, chunk them, generate embeddings, and store in ChromaDB"""
        print(f"\nProcessing {len(documents)} documents...")
        all_chunks = []
        
        for doc_id, doc in enumerate(tqdm(documents, desc="Processing documents")):
            # Print document info
            print(f"\nDocument {doc_id + 1}:")
            print(f"Title: {doc.get('title', 'No title')}")
            print(f"URL: {doc.get('source_url', 'No URL')}")
            
            # Chunk the document
            chunks = chunk_text(doc['text'], chunk_size)
            print(f"Generated {len(chunks)} chunks")
            
            all_texts = []
            all_ids = []
            all_metadatas = []

            # Process each chunk
            for chunk_id, chunk in enumerate(chunks):
                chunk_key = f"doc_{doc_id}_chunk_{chunk_id}"
                
                # Print chunk info
                print(f"\nChunk {chunk_id + 1}:")
                print(f"Text: {chunk[:200]}...")  # Print first 200 characters
                
                # Convert lists to strings in metadata
                keywords = doc['metadata'].get('keywords', [])
                key_phrases = doc['metadata'].get('key_phrases', [])
                
                # Prepare enhanced metadata with list conversions
                metadata = {
                    'doc_id': str(doc_id),
                    'chunk_id': str(chunk_id),
                    'title': doc.get('title', ''),
                    'source_url': doc.get('source_url', ''),
                    'category': str(doc.get('category', '')),  # Convert list to string
                    'last_updated': doc.get('last_updated', ''),
                    'total_chunks': len(chunks),
                    
                    # Include enhanced metadata from text_utils
                    'description': doc['metadata'].get('description', ''),
                    'keywords': ', '.join(str(k) for k in keywords),  # Convert list to string
                    'word_count': doc['metadata'].get('word_count', 0),
                    'reading_difficulty': doc['metadata'].get('reading_difficulty', ''),
                    'key_phrases': ', '.join(str(p) for p in key_phrases),  # Convert list to string
                    'file_type': doc['metadata'].get('file_type', ''),
                    'file_size': str(doc['metadata'].get('file_size', '')),
                    'page_type': doc['metadata'].get('page_type', ''),
                    'content_hash': doc['metadata'].get('content_hash', ''),
                    
                    # Count of structured elements (already integers)
                    'num_images': len(doc['metadata'].get('images', [])),
                    'num_links': len(doc['metadata'].get('links', [])),
                    'num_tables': len(doc['metadata'].get('tables', [])),
                    'num_headings': len(doc['metadata'].get('headings', []))
                }

                all_texts.append(chunk)
                all_ids.append(chunk_key)
                all_metadatas.append(metadata)
                
                all_chunks.append({
                    "id": chunk_key,
                    "text": chunk,
                    "metadata": metadata
                })

            # Generate embeddings
            print(f"\nGenerating embeddings for {len(chunks)} chunks...")
            embeddings = np.array(self.model.encode(all_texts)).tolist()

            # Store in ChromaDB
            self.collection.add(
                documents=all_texts,
                embeddings=embeddings,
                metadatas=all_metadatas,
                ids=all_ids
            )
            print(f"Stored {len(chunks)} chunks in ChromaDB")

        return all_chunks


    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for top_k most relevant chunks to a query"""
        query_embedding = np.array(self.model.encode(query)).tolist()

        # Perform similarity search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        if not results or not results['ids'] or not results['documents'] or not results['metadatas'] or not results['distances']:
            return []

        # Return results with enhanced metadata
        hits = []
        for i in range(top_k):
            if i < len(results['ids'][0]):
                hit = {
                    "id": results['ids'][0][i],
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i],
                    "similarity_score": 1 - (results['distances'][0][i] / 2)
                }
                hits.append(hit)
        return hits

def main():
    """Main function to process crawl results and store in ChromaDB"""
    try:
        # Step 1: Load crawl results with enhanced metadata
        crawl_file = "data/processed/crawl_results_www_mosdac_gov_in.json"
        print(f"\nLoading documents from: {crawl_file}")
        documents = load_documents(crawl_file)
        
        if not documents:
            print("❌ No documents loaded!")
            return
        
        print(f"✅ Loaded {len(documents)} documents")

        # Step 2: Initialize embeddings
        print("\nInitializing DocumentEmbeddings...")
        embedder = DocumentEmbeddings()

        # Step 3: Process documents and get chunks
        print("\nProcessing documents and generating chunks...")
        chunks = embedder.process_documents(documents)

        # Step 4: Print chunk information with enhanced metadata
        print("\n================ CHUNK SUMMARY ================")
        print(f"Total chunks generated: {len(chunks)}")
        
        # Print detailed chunk information
        print("\n================ CHUNK DETAILS ================")
        for chunk in chunks[:5]:  # Show first 5 chunks
            print(f"\n🔹 Chunk ID: {chunk['id']}")
            print(f"📝 Text Preview: {chunk['text'][:150]}...")
            print(f"📎 Metadata:")
            print(f"   - Title: {chunk['metadata']['title']}")
            print(f"   - Category: {chunk['metadata']['category']}")
            print(f"   - Word Count: {chunk['metadata']['word_count']}")
            print(f"   - Key Phrases: {chunk['metadata']['key_phrases'][:3]}")
            print(f"   - File Type: {chunk['metadata']['file_type']}")
            print(f"   - Page Type: {chunk['metadata']['page_type']}")
            print(f"   - Structured Elements:")
            print(f"     • Images: {chunk['metadata']['num_images']}")
            print(f"     • Links: {chunk['metadata']['num_links']}")
            print(f"     • Tables: {chunk['metadata']['num_tables']}")
            print(f"     • Headings: {chunk['metadata']['num_headings']}")
            print("-------------------------------------------")

        # Step 5: Test search functionality
        print("\n================ TESTING SEARCH ================")
        test_query = "What is SST data?"
        results = embedder.search(test_query, top_k=3)
        
        print(f"\nSearch Results for: '{test_query}'")
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"🎯 Similarity Score: {result['similarity_score']:.3f}")
            print(f"📄 Text: {result['text'][:200]}...")
            print(f"📎 Metadata:")
            print(f"   - Title: {result['metadata']['title']}")
            print(f"   - Category: {result['metadata']['category']}")
            print(f"   - Key Phrases: {result['metadata']['key_phrases'][:3]}")
            print(f"🔗 Source: {result['metadata']['source_url']}")

    except Exception as e:
        print(f"❌ Error in main: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()