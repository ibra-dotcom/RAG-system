"""
Embeddings Module

WHAT ARE EMBEDDINGS:
Embeddings convert text into vectors (lists of numbers) that capture meaning.
Similar texts have similar vectors, allowing us to find related content
using mathematical distance calculations instead of keyword matching.

HOW EMBEDDINGS WORK (Conceptual):
1. Text goes into a neural network (the embedding model)
2. The model outputs a fixed-size vector (text-embedding-ada-002 outputs 1536 numbers)
3. Each dimension captures some aspect of meaning
4. We can't interpret individual dimensions, but similar texts cluster together

EXAMPLE:
"How do I return a product?"  →  [0.023, -0.041, 0.089, ..., 0.012]
"What's your return policy?"  →  [0.025, -0.038, 0.091, ..., 0.010]
                                  ↑ Very similar vectors (close in 1536-dimensional space)

"What's the weather today?"   →  [0.512, 0.103, -0.234, ..., 0.891]
                                  ↑ Very different vector

DISTANCE METRICS:
- Cosine Similarity: Measures angle between vectors (most common)
  - 1.0 = identical direction
  - 0.0 = perpendicular (unrelated)
  - -1.0 = opposite (rare in practice)
  
- Euclidean Distance: Straight-line distance in vector space
  - Smaller = more similar

WHY text-embedding-ada-002:
- 1536 dimensions: Good balance of expressiveness and efficiency
- Fast and cheap: ~$0.0001 per 1000 tokens
- High quality: Captures semantic meaning well
- Industry standard: Well-documented, proven in production
"""

import os
from typing import List, Optional
from dataclasses import dataclass
import numpy as np
from openai import AzureOpenAI

from config.settings import get_settings


@dataclass
class EmbeddingResult:
    """
    Result of embedding a piece of text.
    
    WHY DATACLASS:
    - Clean container for embedding + metadata
    - Type hints help catch errors
    - Easy to serialize/deserialize
    """
    text: str
    embedding: List[float]
    model: str
    token_count: int
    
    @property
    def dimension(self) -> int:
        """Get the embedding dimension (1536 for ada-002)."""
        return len(self.embedding)
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array for math operations."""
        return np.array(self.embedding)


class EmbeddingClient:
    """
    Client for generating embeddings using Azure OpenAI.
    
    WHY A CLASS:
    - Manages the Azure client connection
    - Handles batching (more efficient than one-at-a-time)
    - Provides consistent interface
    - Easy to mock for testing
    """
    
    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        deployment: Optional[str] = None,
        api_version: Optional[str] = None
    ):
        """
        Initialize the embedding client.
        
        Args:
            endpoint: Azure OpenAI endpoint (defaults to settings)
            api_key: API key (defaults to settings)
            deployment: Deployment name (defaults to settings)
            api_version: API version (defaults to settings)
            
        WHY OPTIONAL ARGS:
        - Allows override for testing
        - Falls back to centralized settings
        """
        settings = get_settings()
        
        self.endpoint = endpoint or settings.azure.endpoint
        self.api_key = api_key or settings.azure.api_key
        self.deployment = deployment or settings.azure.embedding_deployment
        self.api_version = api_version or settings.azure.api_version
        
        # Initialize the Azure OpenAI client
        # WHY AzureOpenAI vs OpenAI: Different authentication, endpoints
        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version
        )
    
    def embed(self, text: str) -> EmbeddingResult:
        """
        Generate embedding for a single text.
        
        Args:
            text: The text to embed
            
        Returns:
            EmbeddingResult with the embedding vector
            
        WHAT HAPPENS:
        1. Text is sent to Azure OpenAI
        2. Model tokenizes text
        3. Tokens pass through transformer layers
        4. Output is a 1536-dimensional vector
        """
        response = self.client.embeddings.create(
            input=text,
            model=self.deployment  # In Azure, this is the deployment name
        )
        
        return EmbeddingResult(
            text=text,
            embedding=response.data[0].embedding,
            model=self.deployment,
            token_count=response.usage.total_tokens
        )
    
    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 16
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            batch_size: How many to process at once (API limit is usually 16)
            
        Returns:
            List of EmbeddingResult objects
            
        WHY BATCHING:
        - API supports multiple texts per call
        - Reduces network overhead
        - Faster than one-at-a-time
        - API has limits, so we chunk into batches
        """
        results = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            response = self.client.embeddings.create(
                input=batch,
                model=self.deployment
            )
            
            # Match embeddings to original texts
            for j, embedding_data in enumerate(response.data):
                results.append(EmbeddingResult(
                    text=batch[j],
                    embedding=embedding_data.embedding,
                    model=self.deployment,
                    token_count=response.usage.total_tokens // len(batch)  # Approximate
                ))
        
        return results


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    FORMULA:
    cosine_similarity = (A · B) / (||A|| * ||B||)
    
    WHERE:
    - A · B is the dot product (sum of element-wise multiplication)
    - ||A|| is the magnitude (length) of vector A
    
    WHY COSINE:
    - Measures angle, not magnitude
    - Normalized embeddings have magnitude ~1, so this simplifies
    - Range: -1 to 1 (in practice, 0 to 1 for most embeddings)
    - 1.0 = identical, 0.0 = unrelated
    
    EXAMPLE:
    A = [1, 0, 0]
    B = [1, 0, 0]
    similarity = 1.0 (identical)
    
    A = [1, 0, 0]
    B = [0, 1, 0]
    similarity = 0.0 (perpendicular/unrelated)
    """
    a = np.array(vec1)
    b = np.array(vec2)
    
    # Dot product divided by magnitudes
    dot_product = np.dot(a, b)
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)
    
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    
    return dot_product / (magnitude_a * magnitude_b)


def find_most_similar(
    query_embedding: List[float],
    document_embeddings: List[List[float]],
    top_k: int = 5
) -> List[tuple[int, float]]:
    """
    Find the most similar documents to a query.
    
    Args:
        query_embedding: The embedding of the user's question
        document_embeddings: List of document chunk embeddings
        top_k: Number of results to return
        
    Returns:
        List of (index, similarity_score) tuples, sorted by similarity
        
    HOW IT WORKS:
    1. Calculate similarity between query and each document
    2. Sort by similarity (highest first)
    3. Return top K
    
    TIME COMPLEXITY:
    - O(n * d) where n = number of documents, d = embedding dimension
    - For large collections, use vector databases with indexing (ANN search)
    """
    similarities = []
    
    for i, doc_embedding in enumerate(document_embeddings):
        similarity = cosine_similarity(query_embedding, doc_embedding)
        similarities.append((i, similarity))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]


# Example usage
if __name__ == "__main__":
    # This will only work with proper Azure credentials
    print("Embedding module loaded successfully")
    print(f"Expected embedding dimension: 1536 (text-embedding-ada-002)")
    
    # Demo cosine similarity
    print("\nCosine similarity examples:")
    
    # Identical vectors
    v1 = [1, 0, 0]
    v2 = [1, 0, 0]
    print(f"Identical vectors: {cosine_similarity(v1, v2):.4f}")
    
    # Similar vectors
    v1 = [1, 0.1, 0]
    v2 = [1, 0.2, 0]
    print(f"Similar vectors: {cosine_similarity(v1, v2):.4f}")
    
    # Perpendicular vectors
    v1 = [1, 0, 0]
    v2 = [0, 1, 0]
    print(f"Perpendicular vectors: {cosine_similarity(v1, v2):.4f}")
