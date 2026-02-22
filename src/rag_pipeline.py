"""
RAG Pipeline - The Complete System

This module orchestrates all the components into a working RAG system:
1. Document Ingestion → Chunking → Embedding → Storage
2. Query → Embedding → Retrieval → Generation → Answer

THE RAG FLOW VISUALIZED:

INDEXING PHASE (run once per document):
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ Document │───▶│ Chunking │───▶│Embedding │───▶│  Vector  │
│  (PDF)   │    │ (split)  │    │(ada-002) │    │  Store   │
└──────────┘    └──────────┘    └──────────┘    └──────────┘

QUERY PHASE (run for each question):
┌──────────┐    ┌──────────┐    ┌──────────┐
│ Question │───▶│Embedding │───▶│  Vector  │
│          │    │(ada-002) │    │  Search  │
└──────────┘    └──────────┘    └────┬─────┘
                                     │
                                     ▼
                               ┌──────────┐
                               │ Top K    │
                               │ Chunks   │
                               └────┬─────┘
                                    │
                                    ▼
┌──────────┐    ┌──────────────────────────┐
│  Answer  │◀───│ GPT-4o                   │
│          │    │ "Given context: [chunks] │
└──────────┘    │  Answer: [question]"     │
                └──────────────────────────┘

WHY THIS ARCHITECTURE:

1. SEPARATION OF CONCERNS:
   - Each component has one job
   - Easy to test individually
   - Easy to swap implementations

2. FLEXIBILITY:
   - Can use different chunking strategies
   - Can swap vector stores (in-memory → Azure AI Search)
   - Can change LLM (GPT-4o → Claude)

3. OBSERVABILITY:
   - Each step returns metadata
   - Can track tokens, latency, sources
   - Easy to debug issues
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import time

from src.chunking import chunk_document, Chunk, RecursiveChunker, DocumentLoader
from src.embeddings import EmbeddingClient, EmbeddingResult
from src.vector_store import InMemoryVectorStore, SearchResult
from src.generator import Generator, GenerationResult
from config.settings import get_settings


@dataclass
class IndexingResult:
    """
    Result of indexing a document.
    
    WHY TRACK ALL THIS:
    - Debugging: Know what happened during indexing
    - Monitoring: Track performance and costs
    - Verification: Confirm document was processed correctly
    """
    source: str
    chunks_created: int
    embeddings_generated: int
    tokens_used: int
    time_seconds: float
    chunk_ids: List[str] = field(default_factory=list)


@dataclass
class QueryResult:
    """
    Result of a RAG query.
    
    THE COMPLETE PICTURE:
    - answer: What we tell the user
    - sources: Where the answer came from (for trust)
    - retrieved_chunks: The actual context used
    - usage: Token counts for cost tracking
    - timing: Performance metrics
    """
    question: str
    answer: str
    sources: List[str]
    retrieved_chunks: List[SearchResult]
    generation_result: GenerationResult
    timing: Dict[str, float]


class RAGPipeline:
    """
    Complete RAG system that handles both indexing and querying.
    
    USAGE:
        # Initialize
        rag = RAGPipeline()
        
        # Index documents (do this once)
        rag.index_document("company_policy.pdf")
        rag.index_document("faq.txt")
        
        # Query (do this for each question)
        result = rag.query("What is the return policy?")
        print(result.answer)
    
    COMPONENTS:
    - EmbeddingClient: Converts text to vectors
    - InMemoryVectorStore: Stores and searches vectors
    - Generator: Creates answers from context
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 5,
        score_threshold: float = 0.5
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            chunk_size: Target size for document chunks
            chunk_overlap: Overlap between chunks
            top_k: Number of chunks to retrieve
            score_threshold: Minimum similarity score
        """
        settings = get_settings()
        
        # Initialize components
        self.embedding_client = EmbeddingClient()
        self.vector_store = InMemoryVectorStore()
        self.generator = Generator()
        
        # Chunking configuration
        self.chunker = RecursiveChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Retrieval configuration
        self.top_k = top_k
        self.score_threshold = score_threshold
        
        # Track indexed documents
        self.indexed_documents: List[str] = []
    
    def index_document(self, file_path: str) -> IndexingResult:
        """
        Index a document into the RAG system.
        
        WHAT HAPPENS:
        1. Load the document (PDF, TXT, etc.)
        2. Split into chunks
        3. Generate embedding for each chunk
        4. Store in vector store
        
        Args:
            file_path: Path to the document
            
        Returns:
            IndexingResult with stats about the indexing
        """
        start_time = time.time()
        
        # Step 1: Load and chunk the document
        print(f"Loading document: {file_path}")
        text, metadata = DocumentLoader.load(file_path)
        
        print(f"Chunking document...")
        chunks = self.chunker.chunk_text(text, source=file_path)
        print(f"Created {len(chunks)} chunks")
        
        # Step 2: Generate embeddings
        print(f"Generating embeddings...")
        chunk_texts = [chunk.text for chunk in chunks]
        embedding_results = self.embedding_client.embed_batch(chunk_texts)
        embeddings = [result.embedding for result in embedding_results]
        
        total_tokens = sum(result.token_count for result in embedding_results)
        print(f"Used {total_tokens} tokens for embeddings")
        
        # Step 3: Store in vector store
        print(f"Storing in vector store...")
        chunk_ids = self.vector_store.add_chunks(chunks, embeddings)
        
        # Track this document
        self.indexed_documents.append(file_path)
        
        elapsed_time = time.time() - start_time
        print(f"Indexed in {elapsed_time:.2f} seconds")
        
        return IndexingResult(
            source=file_path,
            chunks_created=len(chunks),
            embeddings_generated=len(embeddings),
            tokens_used=total_tokens,
            time_seconds=elapsed_time,
            chunk_ids=chunk_ids
        )
    
    def index_text(self, text: str, source: str = "manual_input") -> IndexingResult:
        """
        Index raw text directly.
        
        WHY THIS METHOD:
        - Index content that's not in a file
        - Good for testing
        - Index from APIs, databases, etc.
        """
        start_time = time.time()
        
        # Chunk the text
        chunks = self.chunker.chunk_text(text, source=source)
        
        # Generate embeddings
        chunk_texts = [chunk.text for chunk in chunks]
        embedding_results = self.embedding_client.embed_batch(chunk_texts)
        embeddings = [result.embedding for result in embedding_results]
        
        total_tokens = sum(result.token_count for result in embedding_results)
        
        # Store
        chunk_ids = self.vector_store.add_chunks(chunks, embeddings)
        self.indexed_documents.append(source)
        
        return IndexingResult(
            source=source,
            chunks_created=len(chunks),
            embeddings_generated=len(embeddings),
            tokens_used=total_tokens,
            time_seconds=time.time() - start_time,
            chunk_ids=chunk_ids
        )
    
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> QueryResult:
        """
        Ask a question and get an answer based on indexed documents.
        
        WHAT HAPPENS:
        1. Convert question to embedding
        2. Search for similar chunks
        3. Generate answer from retrieved context
        
        Args:
            question: The user's question
            top_k: Override default number of chunks to retrieve
            score_threshold: Override default similarity threshold
            
        Returns:
            QueryResult with answer and metadata
        """
        timing = {}
        
        # Step 1: Embed the question
        start = time.time()
        question_embedding = self.embedding_client.embed(question)
        timing["embedding_ms"] = (time.time() - start) * 1000
        
        # Step 2: Search for relevant chunks
        start = time.time()
        results = self.vector_store.search(
            query_embedding=question_embedding.embedding,
            top_k=top_k or self.top_k,
            score_threshold=score_threshold or self.score_threshold
        )
        timing["search_ms"] = (time.time() - start) * 1000
        
        # Step 3: Generate answer
        start = time.time()
        if results:
            generation_result = self.generator.generate(
                question=question,
                context_chunks=results
            )
        else:
            # No relevant documents found
            generation_result = GenerationResult(
                answer="I couldn't find any relevant information to answer your question.",
                sources=[],
                model=self.generator.deployment,
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            )
        timing["generation_ms"] = (time.time() - start) * 1000
        
        timing["total_ms"] = sum(timing.values())
        
        return QueryResult(
            question=question,
            answer=generation_result.answer,
            sources=generation_result.sources,
            retrieved_chunks=results,
            generation_result=generation_result,
            timing=timing
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system."""
        return {
            "indexed_documents": len(self.indexed_documents),
            "total_chunks": len(self.vector_store),
            "documents": self.indexed_documents
        }
    
    def save(self, directory: str):
        """
        Save the RAG system state.
        
        WHY:
        - Don't re-index documents every time
        - Persist between runs
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save vector store
        self.vector_store.save(str(path / "vector_store.pkl"))
        
        # Save metadata
        import json
        with open(path / "metadata.json", "w") as f:
            json.dump({
                "indexed_documents": self.indexed_documents,
                "chunk_size": self.chunker.chunk_size,
                "chunk_overlap": self.chunker.chunk_overlap,
                "top_k": self.top_k,
                "score_threshold": self.score_threshold
            }, f, indent=2)
        
        print(f"Saved RAG system to {directory}")
    
    @classmethod
    def load(cls, directory: str) -> "RAGPipeline":
        """Load a saved RAG system."""
        path = Path(directory)
        
        # Load metadata
        import json
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        # Create pipeline with saved settings
        pipeline = cls(
            chunk_size=metadata["chunk_size"],
            chunk_overlap=metadata["chunk_overlap"],
            top_k=metadata["top_k"],
            score_threshold=metadata["score_threshold"]
        )
        
        # Load vector store
        pipeline.vector_store = InMemoryVectorStore.load(str(path / "vector_store.pkl"))
        pipeline.indexed_documents = metadata["indexed_documents"]
        
        print(f"Loaded RAG system from {directory}")
        return pipeline


# Convenience function for quick start
def create_rag_system() -> RAGPipeline:
    """Create a RAG system with default settings."""
    return RAGPipeline()


# Example usage
if __name__ == "__main__":
    print("""
RAG Pipeline Module
===================

This is the main orchestrator for the RAG system.

Quick Start:
    from src.rag_pipeline import RAGPipeline
    
    # Create the pipeline
    rag = RAGPipeline()
    
    # Index a document
    rag.index_document("my_document.pdf")
    
    # Ask questions
    result = rag.query("What is the main topic?")
    print(result.answer)
    
    # Save for later
    rag.save("my_rag_system")
    
    # Load later
    rag = RAGPipeline.load("my_rag_system")
""")
