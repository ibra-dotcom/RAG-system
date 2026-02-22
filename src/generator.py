"""
Generator Module

WHAT THIS DOES:
Takes a user question + retrieved context and generates an answer using GPT-4o.
This is the "G" in RAG - the Generation part.

THE PROMPT ENGINEERING:
How we construct the prompt dramatically affects answer quality.
Key principles:
1. Clear role definition ("You are a helpful assistant...")
2. Explicit instructions ("Answer ONLY based on context")
3. Context before question (LLM sees info before being asked)
4. Fallback instructions ("If you don't know, say so")

PROMPT TEMPLATE ANATOMY:

┌─────────────────────────────────────────────────┐
│ SYSTEM MESSAGE                                  │
│ - Defines the assistant's role and behavior     │
│ - Sets constraints (only use provided context)  │
│ - Establishes tone and format                   │
└─────────────────────────────────────────────────┘
                    +
┌─────────────────────────────────────────────────┐
│ USER MESSAGE                                    │
│ - Contains the retrieved context                │
│ - Contains the user's question                  │
│ - May include format instructions               │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ ASSISTANT RESPONSE                              │
│ - Generated answer based on context             │
│ - Should cite sources when possible             │
│ - Should admit uncertainty when appropriate     │
└─────────────────────────────────────────────────┘

WHY GPT-4o:
- High quality reasoning
- Good at following instructions
- Handles long contexts well
- Fast response times
- Available on Azure
"""

from typing import List, Optional
from dataclasses import dataclass
from openai import AzureOpenAI

from config.settings import get_settings
from src.vector_store import SearchResult


# Default system prompt for RAG
DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on provided documents.

INSTRUCTIONS:
1. Answer the question using ONLY the information in the provided context.
2. If the context doesn't contain enough information to answer, say "I don't have enough information to answer that question."
3. When possible, mention which part of the context supports your answer.
4. Be concise but complete.
5. If the question is unclear, ask for clarification.

IMPORTANT: Do not make up information. Only use what's in the context."""


@dataclass 
class GenerationResult:
    """
    Result of generating an answer.
    
    WHY TRACK ALL THIS:
    - answer: What we show the user
    - sources: Which chunks were used (for citations)
    - model: Debugging and cost tracking
    - usage: Token consumption for cost monitoring
    - prompt: Debugging - see what was sent to the model
    """
    answer: str
    sources: List[str]
    model: str
    usage: dict
    prompt: Optional[str] = None  # For debugging


class Generator:
    """
    Generate answers using Azure OpenAI GPT-4o.
    
    RESPONSIBILITIES:
    1. Construct effective prompts
    2. Call the Azure OpenAI API
    3. Handle errors gracefully
    4. Track usage for cost monitoring
    """
    
    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        deployment: Optional[str] = None,
        api_version: Optional[str] = None,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize the generator.
        
        Args:
            endpoint: Azure endpoint (defaults to settings)
            api_key: API key (defaults to settings)
            deployment: Model deployment name (defaults to settings)
            system_prompt: Custom system prompt (defaults to DEFAULT_SYSTEM_PROMPT)
        """
        settings = get_settings()
        
        self.endpoint = endpoint or settings.azure.endpoint
        self.api_key = api_key or settings.azure.api_key
        self.deployment = deployment or settings.azure.chat_deployment
        self.api_version = api_version or settings.azure.api_version
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version
        )
    
    def generate(
        self,
        question: str,
        context_chunks: List[SearchResult],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        include_prompt_in_result: bool = False
    ) -> GenerationResult:
        """
        Generate an answer based on retrieved context.
        
        Args:
            question: The user's question
            context_chunks: Retrieved chunks from vector search
            max_tokens: Maximum tokens in the response
            temperature: Creativity (0=deterministic, 1=creative)
            include_prompt_in_result: Whether to include the prompt for debugging
            
        Returns:
            GenerationResult with the answer and metadata
            
        TEMPERATURE EXPLAINED:
        - 0.0: Always pick the most likely token (deterministic)
        - 0.7: Good balance for Q&A (some variety, still focused)
        - 1.0: More creative/random (good for brainstorming)
        For factual Q&A, keep temperature low (0.3-0.7)
        """
        # Build the context string from retrieved chunks
        context = self._build_context(context_chunks)
        
        # Build the user message
        user_message = self._build_user_message(question, context)
        
        # Call the API
        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Extract the answer
        answer = response.choices[0].message.content
        
        # Collect source information
        sources = [
            chunk.document.metadata.get("source", "unknown")
            for chunk in context_chunks
        ]
        # Remove duplicates while preserving order
        sources = list(dict.fromkeys(sources))
        
        return GenerationResult(
            answer=answer,
            sources=sources,
            model=self.deployment,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            prompt=user_message if include_prompt_in_result else None
        )
    
    def _build_context(self, chunks: List[SearchResult]) -> str:
        """
        Build a context string from retrieved chunks.
        
        FORMAT:
        --- Document 1 (source: file.pdf, relevance: 0.92) ---
        [chunk text]
        
        --- Document 2 (source: file.pdf, relevance: 0.87) ---
        [chunk text]
        
        WHY THIS FORMAT:
        - Clear separation between chunks
        - Shows source for potential citations
        - Relevance score helps LLM prioritize
        """
        if not chunks:
            return "No relevant documents found."
        
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            source = chunk.document.metadata.get("source", "unknown")
            score = chunk.score
            
            context_parts.append(
                f"--- Document {i} (source: {source}, relevance: {score:.2f}) ---\n"
                f"{chunk.document.text}"
            )
        
        return "\n\n".join(context_parts)
    
    def _build_user_message(self, question: str, context: str) -> str:
        """
        Build the user message with context and question.
        
        WHY CONTEXT FIRST:
        - LLM reads sequentially
        - Seeing context before question helps comprehension
        - Similar to how humans read a passage before answering questions about it
        """
        return f"""Based on the following context, answer the question.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
    
    def generate_simple(
        self,
        question: str,
        context_text: str,
        max_tokens: int = 1000
    ) -> str:
        """
        Simple generation from raw text context.
        
        WHY THIS METHOD:
        - Sometimes you have context as a string, not SearchResults
        - Useful for testing without full RAG pipeline
        """
        user_message = self._build_user_message(question, context_text)
        
        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=max_tokens,
            temperature=0.7
        )
        
        return response.choices[0].message.content


# Example usage
if __name__ == "__main__":
    print("Generator Module")
    print("=" * 50)
    print("\nDefault System Prompt:")
    print(DEFAULT_SYSTEM_PROMPT)
    print("\n" + "=" * 50)
    print("\nTo use the generator, initialize with Azure credentials")
    print("and call generate() with a question and context chunks.")
