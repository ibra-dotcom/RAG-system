"""
RAG Document Q&A - Demo Script

This script demonstrates the complete RAG workflow:
1. Index a document
2. Ask questions
3. Get answers based on the document content

BEFORE RUNNING:
1. Copy .env.example to .env
2. Fill in your Azure credentials:
   - AZURE_OPENAI_ENDPOINT=https://ibra-ai-foundry.services.ai.azure.com/...
   - AZURE_OPENAI_API_KEY=your-api-key

RUN:
    python demo.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.rag_pipeline import RAGPipeline


def main():
    print("=" * 60)
    print("RAG Document Q&A Demo")
    print("=" * 60)
    print()
    
    # Step 1: Create the RAG pipeline
    print("Step 1: Initializing RAG pipeline...")
    rag = RAGPipeline(
        chunk_size=500,      # Smaller chunks for this demo
        chunk_overlap=100,
        top_k=3,             # Retrieve top 3 chunks
        score_threshold=0.5
    )
    print("✓ Pipeline initialized\n")
    
    # Step 2: Index the sample document
    print("Step 2: Indexing document...")
    result = rag.index_document("data/documents/employee_handbook.txt")
    print(f"✓ Created {result.chunks_created} chunks")
    print(f"✓ Used {result.tokens_used} tokens for embeddings")
    print(f"✓ Indexed in {result.time_seconds:.2f} seconds\n")
    
    # Step 3: Ask questions
    print("Step 3: Asking questions...\n")
    print("-" * 60)
    
    questions = [
        "How many PTO days do new employees get?",
        "What is the remote work policy?",
        "How much does the company match for 401k?",
        "What is the dress code?",
        "How do I submit an expense report?",
    ]
    
    for question in questions:
        print(f"Q: {question}")
        
        result = rag.query(question)
        
        print(f"A: {result.answer}")
        print(f"\n📚 Sources: {', '.join(result.sources)}")
        print(f"⏱️ Time: {result.timing['total_ms']:.0f}ms")
        print(f"📊 Tokens used: {result.generation_result.usage['total_tokens']}")
        print("-" * 60)
        print()
    
    # Step 4: Show system stats
    print("Step 4: System Statistics")
    stats = rag.get_stats()
    print(f"✓ Documents indexed: {stats['indexed_documents']}")
    print(f"✓ Total chunks: {stats['total_chunks']}")
    print()
    
    # Step 5: Interactive mode
    print("=" * 60)
    print("Interactive Mode - Ask your own questions!")
    print("Type 'quit' to exit")
    print("=" * 60)
    print()
    
    while True:
        question = input("Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not question:
            continue
        
        result = rag.query(question)
        print(f"\nAnswer: {result.answer}")
        print(f"(Sources: {', '.join(result.sources)}, Time: {result.timing['total_ms']:.0f}ms)\n")


if __name__ == "__main__":
    main()
