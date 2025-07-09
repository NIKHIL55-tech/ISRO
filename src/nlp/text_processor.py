"""
Enhanced text processing and chunking utilities for the MOSDAC AI Help Bot.

This module provides advanced text cleaning, normalization, and chunking capabilities
with support for different content types and semantic boundaries.
"""
import re
import unicodedata
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import spacy
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path

# Load spaCy model with sentencizer for sentence boundary detection
nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "tagger", "parser"])
nlp.add_pipe('sentencizer')  # Add sentencizer for sentence boundary detection

@dataclass
class TextChunk:
    """A class to represent a chunk of text with metadata."""
    text: str
    chunk_type: str = 'text'  # 'text', 'table', 'code', 'list', 'heading'
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_section: Optional[str] = None

class TextProcessor:
    """
    A class for processing and chunking text with support for different content types.
    
    Features:
    - Advanced text cleaning and normalization
    - Smart chunking with overlap
    - HTML/XML content handling
    - Table extraction and processing
    - Semantic boundary awareness
    """
    
    def __init__(self, min_chunk_size: int = 100, max_chunk_size: int = 1000):
        """
        Initialize the TextProcessor.
        
        Args:
            min_chunk_size: Minimum size of a chunk in characters
            max_chunk_size: Maximum size of a chunk in characters
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_size = 100  # Number of characters to overlap between chunks
        
        # Compiled regex patterns for better performance
        self.whitespace_re = re.compile(r'\s+')
        self.html_tag_re = re.compile(r'<[^>]+>')
        self.url_re = re.compile(r'https?://\S+|www\.\S+')
        self.email_re = re.compile(r'\S+@\S+\.\S+')
        self.special_chars_re = re.compile(r'[^\w\s\-.,;:!?()\[\]\{\}]')
        self.multiple_newlines = re.compile(r'\n{3,}')
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned and normalized text
        """
        if not text:
            return ""
            
        # Handle None or non-string input
        if not isinstance(text, str):
            text = str(text)
            
        # Remove HTML/XML tags
        text = self.html_tag_re.sub(' ', text)
        
        # Remove URLs and emails
        text = self.url_re.sub(' [URL] ', text)
        text = self.email_re.sub(' [EMAIL] ', text)
        
        # Normalize unicode characters and handle accented characters
        text = unicodedata.normalize('NFKD', text)
        text = ''.join([c for c in text if not unicodedata.combining(c)])
        
        # Replace special characters with spaces
        text = self.special_chars_re.sub(' ', text)
        
        # Normalize whitespace and newlines
        text = self.whitespace_re.sub(' ', text)
        text = self.multiple_newlines.sub('\n\n', text)
        
        return text.strip()
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs while preserving structure."""
        if not text.strip():
            return []
            
        # Split by double newlines first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # Further split long paragraphs
        result = []
        for para in paragraphs:
            if len(para) > self.max_chunk_size * 1.5:
                # Try to split at sentence boundaries for long paragraphs
                doc = nlp(para)
                sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
                current_chunk = []
                current_length = 0
                
                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                        
                    sent_len = len(sent)
                    if current_length + sent_len > self.max_chunk_size and current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        if chunk_text.strip():
                            result.append(chunk_text)
                        
                        # Keep last 2 sentences for overlap if possible
                        overlap_sents = min(2, len(current_chunk))
                        current_chunk = current_chunk[-overlap_sents:]
                        current_length = sum(len(s) for s in current_chunk) + len(current_chunk) - 1
                    
                    current_chunk.append(sent)
                    current_length += sent_len + 1  # +1 for space
                
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    if chunk_text.strip():
                        result.append(chunk_text)
            elif para.strip():
                result.append(para)
                
        return result
    
    def _process_html_content(self, html: str) -> List[TextChunk]:
        """Process HTML content and extract structured chunks."""
        from bs4 import BeautifulSoup, Tag
        
        soup = BeautifulSoup(html, 'html.parser')
        chunks = []
        
        # Process tables first
        for table in soup.find_all('table'):
            try:
                # Extract table content
                df = pd.read_html(str(table))[0]
                table_text = df.to_string()
                if table_text.strip():
                    chunks.append(TextChunk(
                        text=table_text,
                        chunk_type='table',
                        metadata={'columns': list(df.columns) if hasattr(df, 'columns') else []}
                    ))
                # Remove the table from the document to avoid duplication
                table.decompose()
            except Exception as e:
                # If table parsing fails, fall back to text extraction
                table_text = table.get_text(separator=' ', strip=True)
                if table_text:
                    chunks.append(TextChunk(
                        text=table_text,
                        chunk_type='text',
                        metadata={'original_type': 'table'}
                    ))
                table.decompose()
        
        # Process the remaining content
        def process_element(element, parent_heading=None):
            if not isinstance(element, Tag):
                return
                
            # Process headings and paragraphs
            if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p']:
                text = element.get_text(separator=' ', strip=True)
                if text:
                    chunk_type = 'heading' if element.name.startswith('h') else 'text'
                    chunks.append(TextChunk(
                        text=text,
                        chunk_type=chunk_type,
                        parent_section=parent_heading
                    ))
                    # If this is a heading, update the parent heading for nested content
                    if chunk_type == 'heading':
                        parent_heading = text
            
            # Recursively process child elements
            for child in element.children:
                process_element(child, parent_heading)
        
        # Start processing from the body or the root if no body
        root = soup.body if soup.body else soup
        process_element(root)
        
        # If we still don't have any chunks, try to extract text from the whole document
        if not chunks:
            text_content = soup.get_text(separator='\n', strip=True)
            if text_content.strip():
                # Split into paragraphs and add as separate chunks
                paragraphs = [p.strip() for p in text_content.split('\n\n') if p.strip()]
                for para in paragraphs:
                    chunks.append(TextChunk(
                        text=para,
                        chunk_type='text'
                    ))
        
        return chunks
    
    def _merge_small_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """Merge small chunks with their neighbors."""
        if not chunks:
            return []
            
        merged_chunks = []
        current_chunk = chunks[0]
        
        for chunk in chunks[1:]:
            # If current chunk is too small and compatible with next chunk, merge them
            if (len(current_chunk.text) < self.min_chunk_size and 
                chunk.chunk_type == current_chunk.chunk_type and
                (not current_chunk.metadata or not chunk.metadata or 
                 current_chunk.metadata == chunk.metadata)):
                current_chunk.text += "\n\n" + chunk.text
            else:
                merged_chunks.append(current_chunk)
                current_chunk = chunk
        
        # Add the last chunk
        merged_chunks.append(current_chunk)
        return merged_chunks
    
    def chunk_text(self, 
                  text: str, 
                  chunk_size: int = 500,
                  chunk_overlap: int = 100,
                  content_type: str = 'text/plain') -> List[Dict[str, Any]]:
        """
        Split text into chunks with overlap, respecting semantic boundaries.
        
        Args:
            text: Input text to chunk
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            content_type: MIME type of the content (e.g., 'text/html', 'text/plain')
            
        Returns:
            List of dictionaries with chunk information
        """
        if not text or not isinstance(text, str) or not text.strip():
            return []
            
        self.max_chunk_size = chunk_size
        self.overlap_size = min(chunk_overlap, chunk_size // 2)  # Ensure overlap is reasonable
        
        # Clean the input text
        text = self.clean_text(text)
        
        # Process based on content type
        if content_type == 'text/html':
            chunks = self._process_html_content(text)
        else:
            # For plain text, split into paragraphs first
            paragraphs = self._split_into_paragraphs(text)
            chunks = [TextChunk(text=p, chunk_type='text') for p in paragraphs if p]
        
        # Merge small chunks
        chunks = self._merge_small_chunks(chunks)
        
        # Final pass to ensure chunk size limits
        final_chunks = []
        for chunk in chunks:
            if len(chunk.text) <= self.max_chunk_size:
                final_chunks.append(chunk)
            else:
                # Split large chunks while trying to preserve sentences
                words = chunk.text.split()
                current_chunk = []
                current_length = 0
                
                for word in words:
                    word_len = len(word) + 1  # +1 for space
                    if current_length + word_len > self.max_chunk_size and current_chunk:
                        final_chunks.append(TextChunk(
                            text=' '.join(current_chunk),
                            chunk_type=chunk.chunk_type,
                            metadata=chunk.metadata,
                            parent_section=chunk.parent_section
                        ))
                        # Keep overlap
                        current_chunk = current_chunk[-int(self.overlap_size/5):]  # Approximate word count for overlap
                        current_length = sum(len(w) + 1 for w in current_chunk)
                    
                    current_chunk.append(word)
                    current_length += word_len
                
                if current_chunk:
                    final_chunks.append(TextChunk(
                        text=' '.join(current_chunk),
                        chunk_type=chunk.chunk_type,
                        metadata=chunk.metadata,
                        parent_section=chunk.parent_section
                    ))
        
        # Convert to list of dictionaries for better serialization
        return [{
            'text': chunk.text,
            'chunk_type': chunk.chunk_type,
            'metadata': chunk.metadata,
            'parent_section': chunk.parent_section,
            'length': len(chunk.text)
        } for chunk in final_chunks]

# Default instance for convenience
text_processor = TextProcessor()

def clean_text(text: str) -> str:
    """Clean text using the default text processor."""
    return text_processor.clean_text(text)

def chunk_text(text: str, 
              chunk_size: int = 500, 
              chunk_overlap: int = 100,
              content_type: str = 'text/plain') -> List[Dict[str, Any]]:
    """
    Split text into chunks with overlap, respecting semantic boundaries.
    
    This is a convenience function that uses the default TextProcessor instance.
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
        content_type: MIME type of the content (e.g., 'text/html', 'text/plain')
        
    Returns:
        List of dictionaries with chunk information
    """
    return text_processor.chunk_text(
        text=text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        content_type=content_type
    )
