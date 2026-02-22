"""
Vector Store Module

WHAT IS A VECTOR STORE:
A database optimized for storing and searching vectors (embeddings).
Instead of searching by keywords, we search by meaning similarity.

WHY WE NEED IT:
1. Embeddings are just lists of numbers - we need somewhere to store them
2. Need to associate embeddings with their source text/metadata
3. Need efficient similarity search (especially at scale)

TYPES OF VECTOR STORES:

1. In-Memory (WHAT WE USE FOR LEARNING):
   - Store embeddings in a Python list
   - Search by comparing against all vectors
   - Pros: Simple, no setup
   - Cons: Slow for large datasets, lost when program ends
   
2. Local Vector Databases:
   - ChromaDB, FAISS, Qdrant (local)
   - Persist to disk
   - Faster search with indexing
   
3. Cloud Vector Databases:
   - Azure AI Search (WHAT YOU'D USE IN PRODUCTION)
   - Pinecone, Weaviate
   - Managed, scalable, production-ready

SEARCH ALGORITHMS:

1. Brute Force (Exact):
   - Compare query to every vector
   - O(n) - linear in number of documents
   - Accurate but slow for large collections

2. Approximate Nearest Neighbor (ANN):
   - Build index for fast approximate search
   - O(log n) or O(1) depending on algorithm
   - Slightly less accurate but much faster
   - Examples: HNSW, IVF, LSH

For learning, we use brute force. In production, use Azure AI Search.
"""

import json
import pickle
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import numpy as np

from src.chunking import Chunk
from src.embeddings import cosine_similarity


@dataclass
class VectorDocument:
    """
    A document stored in the vector store.
    
    WHAT IT CONTAINS:
    - id: Unique identifier
    - text: Original text (for returning to user)
    - embedding: The vector representation
    - metadata: Additional info (source, page number, etc.)
    """
    id: str
    text: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """
    A single search result.
    
    WHY SEPARATE FROM VectorDocument:
    - Includes similarity score (only relevant during search)
    - Can include additional search-specific metadata
    """
    document: VectorDocument
    score: float
    
    def __repr__(self):
        preview = self.document.text[:50] + "..." if len(self.document.text) > 50 else self.document.text
        return f"SearchResult(score={self.score:.4f}, text='{preview}')"


class InMemoryVectorStore:
    """
    Simple in-memory vector store for learning.
    
    WHY START HERE:
    - No external dependencies
    - Easy to understand
    - Good for small datasets (<10K documents)
    - Perfect for learning RAG concepts
    
    LIMITATIONS:
    - Data lost when program ends (unless saved to file)
    - Linear search - slow for large datasets
    - Single machine only
    
    FOR PRODUCTION:
    - Use Azure AI Search
    - Supports hybrid search (vector + keyword)
    - Managed, scalable, persistent
    """
    
    def __init__(self):
        """Initialize empty vector store."""
        self.documents: List[VectorDocument] = []
        self._id_counter = 0
    
    def add(
        self,
        text: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a document to the store.
        
        Args:
            text: The original text
            embedding: The vector representation
            metadata: Additional info (source, page, etc.)
            
        Returns:
            The document ID
        """
        doc_id = f"doc_{self._id_counter}"
        self._id_counter += 1
        
        self.documents.append(VectorDocument(
            id=doc_id,
            text=text,
            embedding=embedding,
            metadata=metadata or {}
        ))
        
        return doc_id
    
    def add_chunks(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]]
    ) -> List[str]:
        """
        Add multiple chunks with their embeddings.
        
        Args:
            chunks: List of Chunk objects (from chunking module)
            embeddings: Corresponding embeddings
            
        Returns:
            List of document IDs
            
        WHY THIS METHOD:
        - Common pattern: chunk document, embed chunks, store
        - Keeps chunk metadata (source, index, etc.)
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks and embeddings must have same length. "
                f"Got {len(chunks)} chunks and {len(embeddings)} embeddings."
            )
        
        ids = []
        for chunk, embedding in zip(chunks, embeddings):
            doc_id = self.add(
                text=chunk.text,
                embedding=embedding,
                metadata={
                    "source": chunk.source,
                    "chunk_index": chunk.chunk_index,
                    "page_number": chunk.page_number,
                    "total_chunks": chunk.total_chunks
                }
            )
            ids.append(doc_id)
        
        return ids
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> List[SearchResult]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: The query vector
            top_k: Number of results to return
            score_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of SearchResult objects, sorted by similarity
            
        HOW IT WORKS:
        1. Calculate cosine similarity with every document
        2. Filter by threshold
        3. Sort by similarity (descending)
        4. Return top K
        
        TIME COMPLEXITY: O(n * d) where n=docs, d=dimensions
        """
        results = []
        
        for doc in self.documents:
            score = cosine_similarity(query_embedding, doc.embedding)
            
            if score >= score_threshold:
                results.append(SearchResult(
                    document=doc,
                    score=score
                ))
        
        # Sort by score (highest first)
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results[:top_k]
    
    def get(self, doc_id: str) -> Optional[VectorDocument]:
        """Get a document by ID."""
        for doc in self.documents:
            if doc.id == doc_id:
                return doc
        return None
    
    def delete(self, doc_id: str) -> bool:
        """Delete a document by ID."""
        for i, doc in enumerate(self.documents):
            if doc.id == doc_id:
                del self.documents[i]
                return True
        return False
    
    def clear(self):
        """Remove all documents."""
        self.documents = []
        self._id_counter = 0
    
    def __len__(self):
        """Number of documents in store."""
        return len(self.documents)
    
    # Persistence methods
    
    def save(self, file_path: str):
        """
        Save the vector store to a file.
        
        WHY PICKLE:
        - Preserves Python objects exactly
        - Fast for numpy arrays
        - Note: Not secure for untrusted data
        """
        with open(file_path, "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "id_counter": self._id_counter
            }, f)
    
    @classmethod
    def load(cls, file_path: str) -> "InMemoryVectorStore":
        """Load a vector store from a file."""
        store = cls()
        
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            store.documents = data["documents"]
            store._id_counter = data["id_counter"]
        
        return store
    
    def export_json(self, file_path: str):
        """
        Export to JSON (human-readable, but large).
        
        WHY JSON:
        - Human readable
        - Can be loaded by other tools
        - Good for debugging
        """
        data = {
            "document_count": len(self.documents),
            "documents": [
                {
                    "id": doc.id,
                    "text": doc.text,
                    "embedding": doc.embedding[:10] + ["..."],  # Truncate for readability
                    "metadata": doc.metadata
                }
                for doc in self.documents
            ]
        }
        
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)


# Example usage
if __name__ == "__main__":
    print("Vector Store Demo\n")
    
    # Create store
    store = InMemoryVectorStore()
    
    # Add some fake documents with fake embeddings (in reality, use EmbeddingClient)
    fake_docs = [
        ("The return policy allows returns within 30 days.", [0.1, 0.2, 0.3]),
        ("Refunds are processed within 5 business days.", [0.15, 0.22, 0.28]),
        ("Contact customer support for help.", [0.5, 0.1, 0.4]),
        ("Our headquarters is in Seattle.", [0.8, 0.9, 0.1]),
    ]
    
    for text, embedding in fake_docs:
        store.add(text, embedding, metadata={"source": "demo.txt"})
    
    print(f"Added {len(store)} documents\n")
    
    # Search for similar documents
    query_embedding = [0.12, 0.21, 0.29]  # Similar to first two docs
    results = store.search(query_embedding, top_k=3)
    
    print("Search results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result.score:.4f}")
        print(f"   Text: {result.document.text}")
        print()
