"""
Document Processor for MOSDAC AI Help Bot

Handles document chunking, metadata enrichment, and vector store integration.
"""
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import re
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Processes documents for vector storage with chunking and metadata handling.
    
    This processor is specifically designed to handle the MOSDAC website crawl results,
    which include rich metadata and content from web pages.
    """
    
    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        chunk_size: int = 800,  # Increased for better context
        chunk_overlap: int = 100,  # Increased for better context
        persist_dir: str = 'data/vector_store',
        collection_name: str = 'mosdac_documents',
        min_chunk_length: int = 100  # Minimum length for a chunk to be included
    ):
        """Initialize the document processor.
        
        Args:
            model_name: Name of the sentence transformer model
            chunk_size: Size of each text chunk in characters
            chunk_overlap: Overlap between chunks in characters
            persist_dir: Directory to store the vector database
            collection_name: Name of the Chroma collection
            min_chunk_length: Minimum length for a chunk to be included
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        
        # Initialize ChromaDB
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None  # We'll handle embeddings ourselves
        )
        logger.info(f"Using collection: {collection_name} at {persist_dir}")
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        if not text:
            return []
            
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            
            # Try to end at a sentence boundary if possible
            if end < text_length:
                # Look for sentence endings near the chunk end
                sentence_end = text.rfind('. ', start, end)
                if sentence_end > start and (end - sentence_end) < 100:  # Don't go too far back
                    end = sentence_end + 1  # Include the period
            
            chunks.append(text[start:end].strip())
            
            # Move start position, accounting for overlap
            start = end - min(self.chunk_overlap, end - start)
            
            # Prevent infinite loops with very small chunks
            if start >= text_length - 1:
                break
                
        return chunks
    
    def enrich_metadata(self, doc: Dict, chunk_id: int = 0, total_chunks: int = 1) -> Dict:
        """Enhance document metadata with additional fields."""
        metadata = doc.get('metadata', {})
        
        # Ensure required fields exist
        for field in ["satellite", "parameter", "source", "year", "last_updated"]:
            if field not in metadata:
                metadata[field] = "Unknown"
        
        # Add chunk information
        metadata.update({
            'chunk_id': chunk_id,
            'total_chunks': total_chunks,
            'processed_at': datetime.utcnow().isoformat()
        })
        
        # Add content statistics
        text = doc.get('text', '')
        word_count = len(text.split())
        metadata['word_count'] = word_count
        
        return metadata
    
    def process_crawl_results(self, crawl_data: Dict) -> List[Dict]:
        """Process crawl results into document chunks with metadata.
        
        Args:
            crawl_data: Dictionary containing 'pages' key with crawled pages
            
        Returns:
            List of processed document chunks
        """
        if 'pages' not in crawl_data or not isinstance(crawl_data['pages'], dict):
            logger.error("Invalid crawl data: 'pages' key not found or not a dictionary")
            return []
            
        all_chunks = []
        
        for url, page in crawl_data['pages'].items():
            if not page or not isinstance(page, dict):
                continue
                
            try:
                # Extract main content and metadata
                content = page.get('content_text', '')
                if not content or not isinstance(content, str):
                    logger.debug(f"Skipping page with no content: {url}")
                    continue
                    
                # Generate chunks from the content
                chunks = self.chunk_text(content)
                if not chunks:
                    continue
                    
                # Process each chunk
                for i, chunk in enumerate(chunks):
                    # Skip chunks that are too short
                    if len(chunk) < self.min_chunk_length:
                        continue
                        
                    # Create chunk metadata
                    chunk_metadata = self.enrich_page_metadata(page, chunk_id=i, total_chunks=len(chunks))
                    
                    # Create chunk document
                    chunk_doc = {
                        'id': f"{url.replace('/', '_').replace(':', '')}_chunk{i}",
                        'text': chunk,
                        'metadata': chunk_metadata
                    }
                    
                    all_chunks.append(chunk_doc)
                    
            except Exception as e:
                logger.error(f"Error processing page {url}: {str(e)}")
                continue
                
        return all_chunks
        
    def enrich_page_metadata(self, page: Dict, chunk_id: int = 0, total_chunks: int = 1) -> Dict:
        """Enhance page metadata with additional fields and cleanup.
        
        Args:
            page: Raw page data from crawl
            chunk_id: Current chunk ID
            total_chunks: Total number of chunks for this page
            
        Returns:
            Enhanced metadata dictionary
        """
        # Extract basic metadata
        metadata = {
            'url': page.get('url', ''),
            'title': page.get('title', 'Untitled'),
            'description': page.get('description', ''),
            'chunk_id': chunk_id,
            'total_chunks': total_chunks,
            'processed_at': datetime.utcnow().isoformat(),
            'source': 'mosdac_website',
            'content_type': 'webpage',
            'language': 'en',
            'reading_difficulty': page.get('reading_difficulty'),
            'word_count': page.get('word_count'),
            'page_type': page.get('page_type', 'unknown'),
            'file_type': page.get('file_type', 'html'),
            'content_categories': page.get('content_categories', []),
            'key_phrases': page.get('key_phrases', []),
            'headings': page.get('headings', {})
        }
        
        # Add counts of structural elements
        for field in ['images', 'links', 'tables', 'forms']:
            if field in page and isinstance(page[field], list):
                metadata[f'num_{field}'] = len(page[field])
                
        # Clean and normalize metadata values
        for key, value in list(metadata.items()):
            # Remove None values
            if value is None:
                metadata.pop(key, None)
            # Convert lists to strings if needed
            elif isinstance(value, list):
                metadata[key] = ', '.join(str(v) for v in value if v is not None)
            # Ensure all values are JSON serializable
            elif not isinstance(value, (str, int, float, bool)):
                metadata[key] = str(value)
                
        return metadata
            
        return processed_chunks
    
    def add_crawl_results(self, crawl_data: Dict, batch_size: int = 50) -> bool:
        """Add crawled documents to the vector store.
        
        Args:
            crawl_data: Dictionary containing crawl results with 'pages' key
            batch_size: Number of chunks to process in each batch
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not crawl_data or 'pages' not in crawl_data:
            logger.error("Invalid crawl data: 'pages' key not found")
            return False
            
        # Process all pages into chunks
        logger.info(f"Processing {len(crawl_data['pages'])} pages from crawl results...")
        all_chunks = self.process_crawl_results(crawl_data)
        
        if not all_chunks:
            logger.warning("No valid chunks generated from crawl results")
            return False
            
        logger.info(f"Generated {len(all_chunks)} chunks from {len(crawl_data['pages'])} pages")
        
        # Add chunks to vector store in batches
        total_chunks = len(all_chunks)
        for i in range(0, total_chunks, batch_size):
            batch = all_chunks[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_chunks + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} with {len(batch)} chunks")
            
            try:
                # Extract batch data
                texts = [chunk['text'] for chunk in batch]
                metadatas = [chunk['metadata'] for chunk in batch]
                ids = [chunk['id'] for chunk in batch]
                
                # Generate embeddings
                logger.debug(f"Generating embeddings for batch {batch_num}...")
                embeddings = self.embedding_model.encode(
                    texts,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                ).tolist()
                
                # Add to collection
                logger.debug(f"Adding batch {batch_num} to vector store...")
                self.collection.add(
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
                
                logger.info(f"Successfully added batch {batch_num}/{total_batches}: {len(batch)} chunks")
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {str(e)}")
                # Continue with next batch even if one fails
                continue
        
        logger.info(f"Finished processing all {total_chunks} chunks from {len(crawl_data['pages'])} pages")
        return True
        
        logger.info(f"Successfully added {len(all_chunks)} chunks to the vector store")
        return True
    
    def search(
        self, 
        query: str, 
        top_k: int = 5, 
        min_score: float = 0.5,
        **filters
    ) -> List[Dict]:
        """Search the vector store with enhanced filtering and scoring.
        
        Args:
            query: Search query string
            top_k: Maximum number of results to return
            min_score: Minimum similarity score (0-1) for results
            **filters: Additional metadata filters (e.g., page_type='document')
            
        Returns:
            List of search results with metadata and scores
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(
                query,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            ).tolist()
            
            # Prepare filter if any
            filter_dict = None
            if filters:
                filter_conditions = []
                for key, value in filters.items():
                    if isinstance(value, (list, tuple)):
                        # Handle list filters (e.g., content_categories=['weather', 'ocean'])
                        filter_conditions.append({
                            "$or": [{"metadata": {key: v}} for v in value]
                        })
                    else:
                        filter_conditions.append({"metadata": {key: value}})
                
                if len(filter_conditions) > 1:
                    filter_dict = {"$and": filter_conditions}
                elif filter_conditions:
                    filter_dict = filter_conditions[0]
            
            # Query the collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k * 2, 20),  # Get more results to filter by score
                where=filter_dict,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format and filter results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                score = 1.0 - (results['distances'][0][i] / 2.0)  # Convert to [0,1] range
                
                # Skip results below minimum score
                if score < min_score:
                    continue
                    
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'score': score,
                    'url': results['metadatas'][0][i].get('url', ''),
                    'title': results['metadatas'][0][i].get('title', 'Untitled')
                })
            
            # Sort by score (descending) and limit to top_k
            formatted_results.sort(key=lambda x: x['score'], reverse=True)
            return formatted_results[:top_k]
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []


def load_crawl_results(file_path: str) -> Dict:
    """Load crawl results from a JSON file.
    
    Args:
        file_path: Path to the crawl results JSON file
        
    Returns:
        Dictionary containing crawl results with 'pages' key
    """
    try:
        logger.info(f"Loading crawl results from: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if not isinstance(data, dict) or 'pages' not in data:
            logger.error("Invalid crawl results: 'pages' key not found or not a dictionary")
            return {}
            
        # Count valid pages
        valid_pages = {k: v for k, v in data['pages'].items() if v and isinstance(v, dict)}
        logger.info(f"Loaded {len(valid_pages)} valid pages from {len(data['pages'])} total pages")
        
        # Replace pages with only valid ones
        data['pages'] = valid_pages
        return data
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {file_path}: {str(e)}")
        return {}
    except Exception as e:
        logger.error(f"Error loading crawl results from {file_path}: {str(e)}")
        return {}


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process documents for vector search.')
    parser.add_argument('--input', required=True, help='Path to JSON file containing documents')
    parser.add_argument('--collection', default='mosdac_documents', help='Chroma collection name')
    parser.add_argument('--chunk-size', type=int, default=500, help='Chunk size in characters')
    parser.add_argument('--overlap', type=int, default=50, help='Overlap between chunks')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize processor
    processor = DocumentProcessor(
        collection_name=args.collection,
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap
    )
    
    # Load and process documents
    documents = load_documents_from_file(args.input)
    if not documents:
        logger.error("No documents loaded. Exiting.")
        return
        
    success = processor.add_documents(documents)
    if success:
        logger.info("Document processing completed successfully")
        
        # Test search
        test_query = "What is SST data?"
        logger.info(f"Testing search with query: {test_query}")
        results = processor.search(test_query, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"\nResult {i} (Score: {result['score']:.3f}):")
            print(f"Text: {result['text'][:200]}...")
            print(f"Metadata: {json.dumps(result['metadata'], indent=2, default=str)}")
    
    else:
        logger.error("Document processing failed")


if __name__ == "__main__":
    main()
