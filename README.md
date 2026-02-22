# RAG Document Q&A System

A Retrieval-Augmented Generation (RAG) system for answering questions based on your documents.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Azure OpenAI](https://img.shields.io/badge/Azure-OpenAI-0078D4.svg)

## What is RAG?

**Retrieval-Augmented Generation (RAG)** is a technique that enhances LLM responses by:
1. **Retrieving** relevant documents based on the user's question
2. **Augmenting** the LLM prompt with this retrieved context
3. **Generating** an answer grounded in your actual documents

This prevents hallucination and allows the LLM to answer questions about your specific data.

## Architecture

```
INDEXING (run once per document):
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ Document │───▶│ Chunking │───▶│Embedding │───▶│  Vector  │
│  (PDF)   │    │ (split)  │    │(ada-002) │    │  Store   │
└──────────┘    └──────────┘    └──────────┘    └──────────┘

QUERYING (run for each question):
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ Question │───▶│Embedding │───▶│  Vector  │───▶│  GPT-4o  │───▶ Answer
│          │    │(ada-002) │    │  Search  │    │          │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
```

## Components

| Component | File | Purpose |
|-----------|------|---------|
| **Chunking** | `src/chunking.py` | Split documents into smaller pieces |
| **Embeddings** | `src/embeddings.py` | Convert text to vectors using ada-002 |
| **Vector Store** | `src/vector_store.py` | Store and search vectors |
| **Generator** | `src/generator.py` | Generate answers using GPT-4o |
| **Pipeline** | `src/rag_pipeline.py` | Orchestrate the complete flow |

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/ibra-dotcom/rag-document-qa.git
cd rag-document-qa
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\Activate
pip install -r requirements.txt
```

### 2. Configure Azure Credentials

```bash
cp .env.example .env
# Edit .env with your Azure AI Foundry credentials
```

Your `.env` file should contain:
```
AZURE_OPENAI_ENDPOINT=https://your-resource.services.ai.azure.com/api/projects/your-project
AZURE_OPENAI_API_KEY=your-api-key
```

### 3. Run the Demo

```bash
python demo.py
```

## Usage

### Basic Usage

```python
from src.rag_pipeline import RAGPipeline

# Create the pipeline
rag = RAGPipeline()

# Index your documents
rag.index_document("company_policy.pdf")
rag.index_document("faq.txt")

# Ask questions
result = rag.query("What is the vacation policy?")
print(result.answer)
print(f"Sources: {result.sources}")
```

### Index Raw Text

```python
rag.index_text("""
Our return policy allows returns within 30 days.
Items must be in original packaging.
""", source="return_policy")
```

### Save and Load

```python
# Save for later
rag.save("my_rag_system")

# Load later
rag = RAGPipeline.load("my_rag_system")
```

## Key Concepts

### Why Chunking?

Documents are split into smaller pieces because:
- LLMs have token limits
- Embeddings work better on focused text
- Retrieval is more precise with smaller chunks

### Why Embeddings?

Embeddings convert text to vectors (lists of numbers) that capture meaning:
- Similar texts have similar vectors
- Enables semantic search (find by meaning, not keywords)
- "How do I return a product?" matches "What's the return policy?"

### Why Vector Search?

Instead of keyword matching:
- Find documents by meaning similarity
- Works across different phrasings
- Handles synonyms automatically

## Configuration

```python
rag = RAGPipeline(
    chunk_size=1000,      # Characters per chunk
    chunk_overlap=200,    # Overlap to preserve context
    top_k=5,              # Number of chunks to retrieve
    score_threshold=0.5   # Minimum similarity score
)
```

## Project Structure

```
rag-document-qa/
├── src/
│   ├── chunking.py        # Document chunking
│   ├── embeddings.py      # Azure OpenAI embeddings
│   ├── vector_store.py    # Vector storage and search
│   ├── generator.py       # Answer generation
│   └── rag_pipeline.py    # Main orchestrator
├── config/
│   └── settings.py        # Configuration management
├── data/
│   └── documents/         # Your documents go here
├── demo.py                # Demo script
├── requirements.txt
└── .env.example
```

## Azure AI Foundry Setup

1. Go to [Azure AI Foundry](https://ai.azure.com)
2. Create a new project
3. Deploy these models:
   - `gpt-4o` (for generation)
   - `text-embedding-ada-002` (for embeddings)
4. Copy your endpoint and API key to `.env`

## Relevant for AI-102 Certification

This project covers key AI-102 topics:
- Azure OpenAI Service
- Embeddings and vector search
- Prompt engineering
- RAG pattern implementation

## License

MIT

## Author

Ibrahima BA - [GitHub](https://github.com/ibra-dotcom)
