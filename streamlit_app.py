"""
Streamlit Frontend for BBC News Articles RAG System
A beautiful web interface for querying BBC news articles using RAG.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
try:
    load_dotenv()
    print("✅ .env file loaded successfully")
except Exception as e:
    print(f"⚠️  Could not load .env file: {e}")
    # Try to load with explicit path
    try:
        load_dotenv('.env')
        print("✅ .env file loaded with explicit path")
    except Exception as e2:
        print(f"⚠️  Could not load .env file with explicit path: {e2}")

# Add the current directory to Python path
sys.path.append('.')

from rag_system import RAGSystem

# Page configuration
st.set_page_config(
    page_title="BBC News RAG System",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .query-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .result-box {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .source-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #28a745;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #0d5a8a;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system with caching."""
    try:
        # Check for API keys
        gemini_key = os.getenv('GOOGLE_API_KEY')
        openai_key = os.getenv('OPENAI_API_KEY')
        
        if gemini_key:
            llm_provider = "gemini"
            api_key = gemini_key
        elif openai_key:
            llm_provider = "openai"
            api_key = openai_key
        else:
            llm_provider = "gemini"
            api_key = None
        
        # Initialize RAG system
        rag = RAGSystem(use_llm=bool(api_key), llm_provider=llm_provider)
        
        # Initialize database if needed
        if not rag.is_initialized:
            with st.spinner("Initializing database... This may take a few minutes for the first time."):
                success = rag.initialize_database()
                if not success:
                    st.error("Failed to initialize database. Please check the error logs.")
                    return None
        
        return rag
    except Exception as e:
        st.error(f"Error initializing RAG system: {e}")
        return None

def display_database_stats(rag_system):
    """Display database statistics in the sidebar."""
    try:
        stats = rag_system.get_database_stats()
        
        st.sidebar.markdown("### 📊 Database Statistics")
        
        # Main metrics
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric(
                label="📰 Articles",
                value=stats['vector_database'].get('document_count', 'N/A')
            )
        with col2:
            st.metric(
                label="🤖 Model",
                value="all-MiniLM-L6-v2"
            )
        
        # Label distribution
        label_dist = stats['preprocessor_stats'].get('label_distribution', {})
        if label_dist:
            st.sidebar.markdown("### 🏷️ Article Categories")
            
            # Create a simple bar chart
            labels = list(label_dist.keys())[:5]  # Top 5 categories
            counts = [label_dist[label] for label in labels]
            
            fig = px.bar(
                x=counts,
                y=labels,
                orientation='h',
                title="Top Categories",
                color=counts,
                color_continuous_scale="Blues"
            )
            fig.update_layout(
                height=300,
                showlegend=False,
                xaxis_title="Number of Articles",
                yaxis_title="Category"
            )
            st.sidebar.plotly_chart(fig, use_container_width=True)
        
        # Text length statistics
        text_stats = stats['preprocessor_stats'].get('text_length_stats', {})
        if text_stats:
            st.sidebar.markdown("### 📝 Text Statistics")
            st.sidebar.metric("Avg Length", f"{text_stats.get('mean', 0):.0f} chars")
            st.sidebar.metric("Max Length", f"{text_stats.get('max', 0):.0f} chars")
        
    except Exception as e:
        st.sidebar.error(f"Error loading statistics: {e}")

def display_query_interface():
    """Display the main query interface."""
    st.markdown('<div class="query-box">', unsafe_allow_html=True)
    
    # Query type selection
    query_type = st.radio(
        "Choose query type:",
        ["Ask a Question", "Search by Topic"],
        horizontal=True
    )
    
    # Query input
    if query_type == "Ask a Question":
        query = st.text_area(
            "Ask a question about BBC news:",
            placeholder="e.g., What are the latest business news?",
            height=100
        )
        n_results = st.slider("Number of sources to retrieve:", 3, 10, 5)
    else:
        query = st.text_area(
            "Search for a topic:",
            placeholder="e.g., technology, politics, sports",
            height=100
        )
        n_results = st.slider("Number of articles to show:", 5, 20, 10)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return query, query_type, n_results

def display_results(query, query_type, n_results, rag_system):
    """Display query results."""
    if not query.strip():
        return
    
    with st.spinner("Processing your query..."):
        if query_type == "Ask a Question":
            result = rag_system.query(query, n_results=n_results)
            
            if result.get('error'):
                st.error(f"Error: {result['error']}")
                return
            
            # Display answer
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.markdown("### 📝 Answer")
            st.write(result['answer'])
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display sources
            if result['retrieved_documents']:
                st.markdown("### 📚 Sources")
                for i, doc in enumerate(result['retrieved_documents'], 1):
                    metadata = doc.get('metadata', {})
                    article_id = metadata.get('article_id', 'Unknown')
                    labels = metadata.get('labels', 'Unknown category')
                    
                    with st.expander(f"Source {i}: Article {article_id} ({labels})"):
                        st.markdown(f"**Preview:** {doc['text'][:300]}...")
                        st.markdown(f"**Category:** {labels}")
                        if 'flesch_score' in metadata:
                            st.markdown(f"**Readability Score:** {metadata['flesch_score']:.1f}")
        
        else:  # Search by topic
            results = rag_system.search_by_topic(query, n_results=n_results)
            
            if not results:
                st.warning("No articles found for this topic.")
                return
            
            st.markdown(f"### 📰 Found {len(results)} articles about '{query}':")
            
            for i, doc in enumerate(results, 1):
                metadata = doc.get('metadata', {})
                article_id = metadata.get('article_id', 'Unknown')
                labels = metadata.get('labels', 'Unknown category')
                distance = doc.get('distance', 0)
                relevance = (1 - distance) * 100
                
                with st.expander(f"Article {i}: {article_id} ({labels}) - {relevance:.1f}% relevant"):
                    st.markdown(f"**Content:** {doc['text'][:500]}...")
                    st.markdown(f"**Category:** {labels}")
                    st.markdown(f"**Relevance:** {relevance:.1f}%")
                    if 'flesch_score' in metadata:
                        st.markdown(f"**Readability Score:** {metadata['flesch_score']:.1f}")

def main():
    """Main Streamlit application."""
    # Header
    st.markdown('<h1 class="main-header">📰 BBC News RAG System</h1>', unsafe_allow_html=True)
    st.markdown("Ask questions about BBC news articles using AI-powered search and retrieval.")
    
    # Initialize RAG system
    rag_system = initialize_rag_system()
    
    if rag_system is None:
        st.error("Failed to initialize the RAG system. Please check your setup.")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎯 Welcome!")
        st.markdown("This system can answer questions about BBC news articles using AI.")
        
        # Display database stats
        display_database_stats(rag_system)
        
        # API Key status
        st.markdown("### 🔑 API Status")
        gemini_key = os.getenv('GOOGLE_API_KEY')
        openai_key = os.getenv('OPENAI_API_KEY')
        
        # Debug information
        st.markdown("**Debug Info:**")
        st.text(f"GOOGLE_API_KEY: {'Set' if gemini_key else 'Not set'}")
        st.text(f"OPENAI_API_KEY: {'Set' if openai_key else 'Not set'}")
        
        if gemini_key:
            st.success("✅ Google Gemini API (FREE)")
            st.text(f"Key: {gemini_key[:8]}...")
        elif openai_key:
            st.success("✅ OpenAI API")
            st.text(f"Key: {openai_key[:8]}...")
        else:
            st.warning("⚠️ No API key found")
            
            # Manual API key input
            st.markdown("**Set API Key Manually:**")
            manual_key = st.text_input("Enter Google Gemini API Key:", type="password", help="Get your FREE key from https://makersuite.google.com/app/apikey")
            
            if manual_key and st.button("Set API Key"):
                os.environ['GOOGLE_API_KEY'] = manual_key
                st.success("✅ API key set! Please refresh the page.")
                st.rerun()
            
            st.markdown("Or run `python setup_gemini_key.py` for FREE AI features")
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        if st.button("🔄 Rebuild Database"):
            with st.spinner("Rebuilding database..."):
                success = rag_system.initialize_database(force_rebuild=True)
                if success:
                    st.success("Database rebuilt successfully!")
                    st.rerun()
                else:
                    st.error("Failed to rebuild database.")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Query interface
        query, query_type, n_results = display_query_interface()
        
        # Process query button
        if st.button("🚀 Process Query", type="primary"):
            if query.strip():
                display_results(query, query_type, n_results, rag_system)
            else:
                st.warning("Please enter a query first.")
    
    with col2:
        # Example queries
        st.markdown("### 💡 Example Queries")
        
        example_queries = [
            "What are the latest business news?",
            "Tell me about technology developments",
            "What happened in politics recently?",
            "Are there any sports updates?",
            "Search for climate change",
            "Find articles about economy"
        ]
        
        for example in example_queries:
            if st.button(f"💬 {example}", key=f"example_{example}"):
                st.session_state.example_query = example
                st.rerun()
        
        # Handle example query selection
        if 'example_query' in st.session_state:
            query = st.session_state.example_query
            del st.session_state.example_query
            display_results(query, "Ask a Question", 5, rag_system)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using Streamlit, ChromaDB, and SentenceTransformers. "
        "Powered by Google Gemini (FREE) or OpenAI for intelligent answers."
    )

if __name__ == "__main__":
    main()
