"""
Configuration settings for the RAG Document Q&A system.

WHY THIS FILE EXISTS:
- Centralizes all configuration in one place
- Makes it easy to switch between environments (dev/prod)
- Keeps secrets separate from code (loaded from .env)

AZURE AI FOUNDRY CONCEPTS:
- Endpoint: The URL where your AI services are hosted
- API Key: Authentication to access your deployed models
- Deployment Name: The name you gave when deploying a model
  (This is different from the model name itself)
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
# WHY: Keeps secrets out of code, different values for different environments
load_dotenv()


@dataclass
class AzureOpenAIConfig:
    """
    Configuration for Azure OpenAI services.
    
    WHY DATACLASS:
    - Clean way to group related settings
    - Provides type hints for IDE support
    - Easy to pass around as a single object
    """
    endpoint: str
    api_key: str
    api_version: str
    chat_deployment: str      # For generating answers (gpt-4o)
    embedding_deployment: str  # For creating vectors (text-embedding-ada-002)


@dataclass
class ChunkingConfig:
    """
    Configuration for document chunking.
    
    WHY THESE DEFAULTS:
    - chunk_size=1000: ~250 words, fits well in context window
    - chunk_overlap=200: 20% overlap preserves context at boundaries
    - Too small: Loses context, too many chunks to search
    - Too large: May exceed token limits, less precise retrieval
    """
    chunk_size: int = 1000       # Characters per chunk
    chunk_overlap: int = 200     # Overlap between chunks
    min_chunk_size: int = 100    # Minimum chunk size to keep


@dataclass
class RetrievalConfig:
    """
    Configuration for document retrieval.
    
    WHY THESE DEFAULTS:
    - top_k=5: Return 5 most relevant chunks
      - Too few: May miss relevant info
      - Too many: Adds noise, uses more tokens
    - score_threshold=0.7: Minimum similarity score
      - Filters out weakly related chunks
    """
    top_k: int = 5                    # Number of chunks to retrieve
    score_threshold: float = 0.7      # Minimum similarity score (0-1)


@dataclass
class Settings:
    """
    Main settings container.
    
    WHY NESTED CONFIGS:
    - Organized by domain (Azure, chunking, retrieval)
    - Easy to modify one area without affecting others
    - Clear what settings belong together
    """
    azure: AzureOpenAIConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig


def load_settings() -> Settings:
    """
    Load settings from environment variables.
    
    WHY ENVIRONMENT VARIABLES:
    - Security: Secrets not in code
    - Flexibility: Different values per environment
    - Standard practice: Works with CI/CD, Docker, cloud platforms
    
    REQUIRED ENVIRONMENT VARIABLES:
    - AZURE_OPENAI_ENDPOINT: Your Foundry endpoint URL
    - AZURE_OPENAI_API_KEY: Your API key
    """
    
    # Validate required environment variables exist
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    
    if not endpoint:
        raise ValueError(
            "AZURE_OPENAI_ENDPOINT not set. "
            "Add it to your .env file or set it as an environment variable."
        )
    
    if not api_key:
        raise ValueError(
            "AZURE_OPENAI_API_KEY not set. "
            "Add it to your .env file or set it as an environment variable."
        )
    
    return Settings(
        azure=AzureOpenAIConfig(
            endpoint=endpoint,
            api_key=api_key,
            api_version="2024-02-15-preview",  # Azure OpenAI API version
            chat_deployment="gpt-4o",           # Your deployment name
            embedding_deployment="text-embedding",  # Your deployment name
        ),
        chunking=ChunkingConfig(),
        retrieval=RetrievalConfig(),
    )


# Singleton pattern - load settings once and reuse
# WHY: Avoid repeated file I/O and validation
_settings = None

def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings
