"""
Document Chunking Module

WHY CHUNKING IS NECESSARY:
1. LLMs have token limits (GPT-4o: ~128K tokens, but cost increases with size)
2. Embeddings work better on focused, coherent text
3. Retrieval is more precise with smaller, specific chunks
4. Large documents would overwhelm the context window

CHUNKING STRATEGIES EXPLAINED:

1. Fixed Size Chunking:
   - Split every N characters
   - Pros: Simple, predictable
   - Cons: May cut mid-sentence, loses context
   
2. Sentence-Based Chunking:
   - Split on sentence boundaries
   - Pros: Coherent units
   - Cons: Sentences vary wildly in length

3. Recursive Chunking (WHAT WE USE):
   - Try to split on paragraphs first
   - If chunk too big, try sentences
   - If still too big, try words
   - Pros: Respects document structure
   - Cons: More complex

4. Semantic Chunking:
   - Use embeddings to find topic boundaries
   - Pros: Most coherent chunks
   - Cons: Slow, expensive (needs embeddings)

WHY OVERLAP:
When we split "The policy is 30 days. Contact support for help."
into two chunks, the second chunk loses context about "the policy."
Overlap keeps some shared text so context isn't completely lost.
"""

import re
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

# Document loaders
import PyPDF2


@dataclass
class Chunk:
    """
    A piece of a document.
    
    WHY TRACK METADATA:
    - source: Know which document this came from (for citations)
    - chunk_index: Order within document
    - page_number: For PDFs, which page (user can verify)
    - total_chunks: Context for how much of document this represents
    """
    text: str
    source: str
    chunk_index: int
    page_number: Optional[int] = None
    total_chunks: Optional[int] = None
    
    def __repr__(self):
        preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"Chunk({self.source}, idx={self.chunk_index}, text='{preview}')"


class DocumentLoader:
    """
    Load documents from various file formats.
    
    WHY A SEPARATE CLASS:
    - Single responsibility: Only handles file I/O
    - Easy to add new formats (Word, HTML, etc.)
    - Testable in isolation
    """
    
    @staticmethod
    def load(file_path: str) -> tuple[str, dict]:
        """
        Load a document and return (text, metadata).
        
        Returns:
            tuple: (document_text, metadata_dict)
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        suffix = path.suffix.lower()
        
        if suffix == ".txt":
            return DocumentLoader._load_txt(path)
        elif suffix == ".pdf":
            return DocumentLoader._load_pdf(path)
        elif suffix == ".md":
            return DocumentLoader._load_txt(path)  # Markdown is just text
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    @staticmethod
    def _load_txt(path: Path) -> tuple[str, dict]:
        """Load a text file."""
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        return text, {"source": str(path), "format": "txt"}
    
    @staticmethod
    def _load_pdf(path: Path) -> tuple[str, dict]:
        """
        Load a PDF file.
        
        WHY PyPDF2:
        - Pure Python, no system dependencies
        - Handles most PDFs well
        - Alternative: pdfplumber for complex layouts
        """
        text_parts = []
        page_count = 0
        
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            page_count = len(reader.pages)
            
            for page in reader.pages:
                text_parts.append(page.extract_text() or "")
        
        return "\n\n".join(text_parts), {
            "source": str(path),
            "format": "pdf",
            "page_count": page_count
        }


class RecursiveChunker:
    """
    Split documents into chunks using recursive strategy.
    
    HOW IT WORKS:
    1. Try to split on double newlines (paragraphs)
    2. If a piece is still too big, split on single newlines
    3. If still too big, split on sentences
    4. If still too big, split on words
    5. Add overlap between chunks
    
    WHY RECURSIVE:
    - Respects natural document boundaries
    - Paragraphs > Sentences > Words (in terms of coherence)
    - Falls back gracefully when needed
    """
    
    # Separators in order of preference (try first one first)
    SEPARATORS = [
        "\n\n",     # Paragraphs
        "\n",       # Lines
        ". ",       # Sentences
        "? ",       # Questions
        "! ",       # Exclamations
        " ",        # Words
        ""          # Characters (last resort)
    ]
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100
    ):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Target size for each chunk in characters
            chunk_overlap: How many characters to overlap between chunks
            min_chunk_size: Minimum size to keep a chunk (filters tiny pieces)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
    
    def chunk_text(self, text: str, source: str = "unknown") -> List[Chunk]:
        """
        Split text into chunks.
        
        Args:
            text: The full document text
            source: Source identifier (filename, URL, etc.)
            
        Returns:
            List of Chunk objects
        """
        # Clean the text
        text = self._clean_text(text)
        
        # Split recursively
        raw_chunks = self._split_recursive(text, self.SEPARATORS)
        
        # Merge small chunks and add overlap
        merged_chunks = self._merge_chunks(raw_chunks)
        
        # Create Chunk objects with metadata
        chunks = []
        for i, chunk_text in enumerate(merged_chunks):
            if len(chunk_text.strip()) >= self.min_chunk_size:
                chunks.append(Chunk(
                    text=chunk_text.strip(),
                    source=source,
                    chunk_index=i,
                    total_chunks=len(merged_chunks)
                ))
        
        # Update total_chunks now that we know final count
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text before chunking.
        
        WHY CLEAN:
        - Remove excessive whitespace (wastes tokens)
        - Normalize line endings
        - Remove control characters
        """
        # Replace multiple spaces with single space
        text = re.sub(r" +", " ", text)
        # Replace multiple newlines with double newline
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Strip whitespace from each line
        lines = [line.strip() for line in text.split("\n")]
        return "\n".join(lines)
    
    def _split_recursive(
        self,
        text: str,
        separators: List[str]
    ) -> List[str]:
        """
        Recursively split text using separators.
        
        HOW IT WORKS:
        1. If text is small enough, return it
        2. Try first separator
        3. If pieces are still too big, recurse with next separator
        """
        # Base case: text is small enough
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        
        # Try each separator in order
        for i, separator in enumerate(separators):
            if separator == "":
                # Last resort: split by characters
                return self._split_by_size(text)
            
            if separator in text:
                # Split by this separator
                parts = text.split(separator)
                
                # Recursively process each part
                result = []
                for part in parts:
                    if len(part) <= self.chunk_size:
                        if part.strip():
                            result.append(part)
                    else:
                        # Part is too big, recurse with remaining separators
                        result.extend(
                            self._split_recursive(part, separators[i+1:])
                        )
                
                return result
        
        # No separator worked, split by size
        return self._split_by_size(text)
    
    def _split_by_size(self, text: str) -> List[str]:
        """Split text into fixed-size chunks (last resort)."""
        chunks = []
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks
    
    def _merge_chunks(self, chunks: List[str]) -> List[str]:
        """
        Merge small chunks and add overlap.
        
        WHY MERGE:
        - After splitting, some chunks may be too small
        - Better to have fewer, more substantial chunks
        
        WHY OVERLAP:
        - Preserves context at chunk boundaries
        - A sentence that ends one chunk and continues context
          into the next won't be completely lost
        """
        if not chunks:
            return []
        
        merged = []
        current = chunks[0]
        
        for i in range(1, len(chunks)):
            next_chunk = chunks[i]
            
            # If adding next chunk keeps us under limit, merge
            if len(current) + len(next_chunk) <= self.chunk_size:
                current = current + " " + next_chunk
            else:
                # Save current chunk
                merged.append(current)
                
                # Start new chunk with overlap from previous
                overlap_start = max(0, len(current) - self.chunk_overlap)
                overlap_text = current[overlap_start:]
                current = overlap_text + " " + next_chunk
        
        # Don't forget the last chunk
        if current.strip():
            merged.append(current)
        
        return merged


def chunk_document(
    file_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Chunk]:
    """
    Convenience function to load and chunk a document.
    
    Args:
        file_path: Path to the document
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of Chunk objects
        
    Example:
        chunks = chunk_document("policy.pdf", chunk_size=1000)
        for chunk in chunks:
            print(f"Chunk {chunk.chunk_index}: {chunk.text[:100]}...")
    """
    # Load document
    text, metadata = DocumentLoader.load(file_path)
    
    # Chunk it
    chunker = RecursiveChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    return chunker.chunk_text(text, source=metadata["source"])


# Example usage
if __name__ == "__main__":
    # Demo with sample text
    sample_text = """
    Introduction to Machine Learning
    
    Machine learning is a subset of artificial intelligence that enables 
    systems to learn and improve from experience without being explicitly 
    programmed. It focuses on developing algorithms that can access data 
    and use it to learn for themselves.
    
    Types of Machine Learning
    
    There are three main types of machine learning:
    
    1. Supervised Learning: The algorithm learns from labeled training data, 
    making predictions based on that data. Examples include classification 
    and regression tasks.
    
    2. Unsupervised Learning: The algorithm finds patterns in unlabeled data. 
    Common techniques include clustering and dimensionality reduction.
    
    3. Reinforcement Learning: The algorithm learns by interacting with an 
    environment, receiving rewards or penalties for actions taken.
    """
    
    chunker = RecursiveChunker(chunk_size=300, chunk_overlap=50)
    chunks = chunker.chunk_text(sample_text, source="sample.txt")
    
    print(f"Created {len(chunks)} chunks:\n")
    for chunk in chunks:
        print(f"--- Chunk {chunk.chunk_index} ({len(chunk.text)} chars) ---")
        print(chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text)
        print()
