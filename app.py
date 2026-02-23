"""
RAG Document Q&A - Streamlit Web Interface

A clean, professional UI for demonstrating the RAG system.

RUN:
    streamlit run app.py

FEATURES:
- Upload documents (PDF, TXT)
- Ask questions in natural language
- See answers with source citations
- View retrieved chunks for transparency
"""

import streamlit as st
import sys
from pathlib import Path
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.rag_pipeline import RAGPipeline
from src.chunking import DocumentLoader


# Page configuration
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #F0F9FF;
        border-left: 4px solid #0EA5E9;
        padding: 1.5rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .source-tag {
        background-color: #E0F2FE;
        color: #0369A1;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.875rem;
        margin-right: 0.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .chunk-box {
        background-color: #FFFBEB;
        border: 1px solid #FCD34D;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = None
    if 'indexed_docs' not in st.session_state:
        st.session_state.indexed_docs = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []


def create_pipeline():
    """Create or get the RAG pipeline."""
    if st.session_state.rag_pipeline is None:
        with st.spinner("Initializing RAG pipeline..."):
            st.session_state.rag_pipeline = RAGPipeline(
                chunk_size=500,
                chunk_overlap=100,
                top_k=3,
                score_threshold=0.5
            )
    return st.session_state.rag_pipeline


def main():
    init_session_state()
    
    # Header
    st.markdown('<p class="main-header">📚 RAG Document Q&A</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions about your documents using AI-powered retrieval</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Document Management")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=['txt', 'pdf'],
            accept_multiple_files=True,
            help="Upload PDF or TXT files to query"
        )
        
        # Index uploaded files
        if uploaded_files:
            if st.button("📥 Index Documents", type="primary", use_container_width=True):
                rag = create_pipeline()
                
                for uploaded_file in uploaded_files:
                    if uploaded_file.name not in st.session_state.indexed_docs:
                        with st.spinner(f"Indexing {uploaded_file.name}..."):
                            # Save temporarily
                            temp_path = Path(f"data/documents/{uploaded_file.name}")
                            temp_path.parent.mkdir(parents=True, exist_ok=True)
                            temp_path.write_bytes(uploaded_file.getvalue())
                            
                            # Index
                            try:
                                result = rag.index_document(str(temp_path))
                                st.session_state.indexed_docs.append(uploaded_file.name)
                                st.success(f"✅ {uploaded_file.name}: {result.chunks_created} chunks")
                            except Exception as e:
                                st.error(f"❌ Error indexing {uploaded_file.name}: {str(e)}")
        
        # Or use sample document
        st.divider()
        st.subheader("📄 Or use sample document")
        
        if st.button("Load Employee Handbook", use_container_width=True):
            rag = create_pipeline()
            sample_path = "data/documents/employee_handbook.txt"
            
            if Path(sample_path).exists():
                if "employee_handbook.txt" not in st.session_state.indexed_docs:
                    with st.spinner("Indexing sample document..."):
                        result = rag.index_document(sample_path)
                        st.session_state.indexed_docs.append("employee_handbook.txt")
                        st.success(f"✅ Indexed {result.chunks_created} chunks")
                else:
                    st.info("Sample document already indexed")
            else:
                st.error("Sample document not found")
        
        # Show indexed documents
        st.divider()
        st.subheader("📋 Indexed Documents")
        
        if st.session_state.indexed_docs:
            for doc in st.session_state.indexed_docs:
                st.markdown(f"• {doc}")
            
            if st.session_state.rag_pipeline:
                stats = st.session_state.rag_pipeline.get_stats()
                st.caption(f"Total chunks: {stats['total_chunks']}")
        else:
            st.caption("No documents indexed yet")
        
        # Clear button
        if st.session_state.indexed_docs:
            st.divider()
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.rag_pipeline = None
                st.session_state.indexed_docs = []
                st.session_state.chat_history = []
                st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask a Question")
        
        # Check if documents are indexed
        if not st.session_state.indexed_docs:
            st.info("👈 Please upload and index documents first, or load the sample document.")
        else:
            # Question input
            question = st.text_input(
                "Your question:",
                placeholder="e.g., What is the remote work policy?",
                label_visibility="collapsed"
            )
            
            col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
            with col_btn1:
                ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)
            with col_btn2:
                show_chunks = st.checkbox("Show sources", value=True)
            
            # Process question
            if ask_button and question:
                rag = st.session_state.rag_pipeline
                
                with st.spinner("Searching and generating answer..."):
                    start_time = time.time()
                    result = rag.query(question)
                    total_time = time.time() - start_time
                
                # Add to history
                st.session_state.chat_history.append({
                    'question': question,
                    'answer': result.answer,
                    'sources': result.sources,
                    'chunks': result.retrieved_chunks,
                    'timing': result.timing,
                    'tokens': result.generation_result.usage['total_tokens']
                })
            
            # Display chat history (most recent first)
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                st.markdown(f"**Q: {chat['question']}**")
                
                # Answer box
                st.markdown(f"""
                <div class="answer-box">
                    {chat['answer']}
                </div>
                """, unsafe_allow_html=True)
                
                # Metrics row
                metric_cols = st.columns(4)
                with metric_cols[0]:
                    st.metric("⏱️ Time", f"{chat['timing']['total_ms']:.0f}ms")
                with metric_cols[1]:
                    st.metric("📊 Tokens", chat['tokens'])
                with metric_cols[2]:
                    st.metric("📚 Sources", len(chat['chunks']))
                with metric_cols[3]:
                    if chat['chunks']:
                        avg_score = sum(c.score for c in chat['chunks']) / len(chat['chunks'])
                        st.metric("🎯 Relevance", f"{avg_score:.0%}")
                
                # Show retrieved chunks
                if show_chunks and chat['chunks']:
                    with st.expander(f"📄 View retrieved chunks ({len(chat['chunks'])})"):
                        for j, chunk in enumerate(chat['chunks']):
                            st.markdown(f"""
                            <div class="chunk-box">
                                <strong>Chunk {j+1}</strong> (Score: {chunk.score:.2%})<br>
                                <em>Source: {chunk.document.metadata.get('source', 'unknown')}</em>
                                <hr style="margin: 0.5rem 0;">
                                {chunk.document.text[:500]}{'...' if len(chunk.document.text) > 500 else ''}
                            </div>
                            """, unsafe_allow_html=True)
                
                st.divider()
    
    with col2:
        st.header("ℹ️ How It Works")
        
        st.markdown("""
        **RAG (Retrieval-Augmented Generation)**
        
        1. **📄 Index**: Documents are split into chunks and converted to vectors
        
        2. **🔍 Retrieve**: Your question is matched against document chunks
        
        3. **✨ Generate**: GPT-4o creates an answer using relevant chunks
        
        ---
        
        **Why RAG?**
        - ✅ Answers grounded in your documents
        - ✅ No hallucination
        - ✅ Source citations
        - ✅ Works with any document
        
        ---
        
        **Tech Stack**
        - 🔷 Azure OpenAI (GPT-4o)
        - 🔷 text-embedding-ada-002
        - 🔷 Vector similarity search
        - 🔷 Streamlit UI
        """)
        
        # Sample questions
        st.header("💡 Try These Questions")
        
        sample_questions = [
            "How many PTO days do employees get?",
            "What is the remote work policy?",
            "How much does the company match for 401k?",
            "What is the dress code?",
            "How do I request time off?"
        ]
        
        for q in sample_questions:
            if st.button(q, key=f"sample_{q}", use_container_width=True):
                # This will set the question and trigger a rerun
                st.session_state.sample_question = q
                st.rerun()


if __name__ == "__main__":
    main()
