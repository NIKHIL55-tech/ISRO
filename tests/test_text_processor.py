"""
Tests for the text processing and chunking functionality.
"""
import unittest
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.nlp.text_processor import TextProcessor, TextChunk, clean_text, chunk_text

class TestTextProcessor(unittest.TestCase):
    """Test cases for the TextProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = TextProcessor(min_chunk_size=50, max_chunk_size=500)
        
    def test_clean_text(self):
        """Test text cleaning functionality."""
        # Test HTML tag removal
        self.assertEqual(
            self.processor.clean_text("<p>Hello</p> <b>World</b>"),
            "Hello World"
        )
        
        # Test URL and email removal
        self.assertEqual(
            self.processor.clean_text("Visit https://example.com or email test@example.com"),
            "Visit [URL] or email [EMAIL]"
        )
        
        # Test unicode normalization
        self.assertEqual(
            self.processor.clean_text("café"),
            "cafe"
        )
        
        # Test whitespace normalization
        self.assertEqual(
            self.processor.clean_text("  too    much\n\nwhitespace  "),
            "too much whitespace"
        )
    
    def test_chunk_text_basic(self):
        """Test basic text chunking."""
        text = """
        This is a test document. It has multiple sentences.
        
        This is a new paragraph with more text. It should be in a separate chunk.
        
        This is another paragraph that is quite long and might need to be split into multiple chunks if it exceeds the maximum chunk size, which it probably will because this is a very long sentence that just keeps going and going without any regard for proper paragraph structure or readability.
        """
        
        chunks = self.processor.chunk_text(text, chunk_size=100, chunk_overlap=20)
        
        # Should have at least 3 chunks (one per paragraph, with the last one split)
        self.assertGreaterEqual(len(chunks), 3)
        
        # Check chunk sizes
        for chunk in chunks:
            self.assertLessEqual(len(chunk['text']), 100)
            self.assertGreaterEqual(len(chunk['text']), 20)
    
    def test_small_chunk_merging(self):
        """Test that small chunks are merged with neighbors."""
        chunks = [
            TextChunk("This is a small chunk.", "text"),
            TextChunk("This is another small chunk.", "text"),
            TextChunk("This is a very long chunk that should be split " + 
                     "into multiple parts because it exceeds the maximum chunk size " +
                     "by a significant margin and needs to be properly handled " +
                     "by the chunking algorithm.", "text")
        ]
        
        merged = self.processor._merge_small_chunks(chunks)
        
        # First two small chunks should be merged
        self.assertEqual(len(merged), 2)
        self.assertIn("This is a small chunk.\n\nThis is another small chunk.", 
                     merged[0].text)
    
    def test_convenience_functions(self):
        """Test the module-level convenience functions."""
        # Test clean_text
        self.assertEqual(
            clean_text("<p>Clean me!</p>"),
            "Clean me!"
        )
        
        # Test chunk_text
        chunks = chunk_text("Short text.")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]['text'], "Short text.")

if __name__ == "__main__":
    unittest.main()
