"""
RAG (Retrieval-Augmented Generation) System for BBC News Articles
This module provides the main RAG functionality for querying news articles.
"""

import os
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

# Import our custom modules
from data_preprocessor import BBCNewsPreprocessor
from vector_database import VectorDatabase

# Import LLM components
try:
    from langchain.llms import OpenAI
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("⚠️  LLM components not available. Install langchain and google-generativeai for better answers.")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    """Main RAG system for BBC News Articles."""
    
    def __init__(self, 
                 csv_path: str = "bbc_news_articles/bbc_news_text_complexity_summarization.csv",
                 collection_name: str = "bbc_news_articles",
                 persist_directory: str = "./chroma_db",
                 use_llm: bool = True,
                 llm_provider: str = "gemini",  # "openai" or "gemini"
                 api_key: Optional[str] = None):
        """
        Initialize the RAG system.
        
        Args:
            csv_path: Path to the BBC news CSV file
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the vector database
            use_llm: Whether to use LLM for answer generation
            llm_provider: LLM provider ("openai" or "gemini")
            api_key: API key (if not provided, will use environment variable)
        """
        self.csv_path = csv_path
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.use_llm = use_llm and LLM_AVAILABLE
        self.llm_provider = llm_provider
        
        # Initialize components
        self.preprocessor = BBCNewsPreprocessor(csv_path)
        self.vector_db = VectorDatabase(collection_name, persist_directory)
        
        # Initialize LLM if available
        self.llm = None
        self.llm_chain = None
        if self.use_llm:
            self._initialize_llm(api_key)
        
        # Check if database is already populated
        self.is_initialized = self._check_initialization()
    
    def _initialize_llm(self, api_key: Optional[str] = None):
        """Initialize the LLM for answer generation."""
        try:
            # Get API key from parameter or environment
            if not api_key:
                if self.llm_provider == "gemini":
                    api_key = os.getenv('GOOGLE_API_KEY')
                else:
                    api_key = os.getenv('OPENAI_API_KEY')
            
            if not api_key:
                provider_name = "Google" if self.llm_provider == "gemini" else "OpenAI"
                env_var = "GOOGLE_API_KEY" if self.llm_provider == "gemini" else "OPENAI_API_KEY"
                logger.warning(f"No {provider_name} API key found. Set {env_var} environment variable or pass api_key parameter.")
                self.use_llm = False
                return
            
            # Initialize LLM based on provider
            if self.llm_provider == "gemini":
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-pro",
                    google_api_key=api_key,
                    temperature=0.7,
                    max_output_tokens=500
                )
                logger.info("✅ Gemini LLM initialized successfully")
            else:
                self.llm = OpenAI(
                    openai_api_key=api_key,
                    temperature=0.7,
                    max_tokens=500
                )
                logger.info("✅ OpenAI LLM initialized successfully")
            
            # Create prompt template
            prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template="""You are a helpful assistant that answers questions based on BBC news articles.

Context from BBC news articles:
{context}

Question: {question}

Please provide a comprehensive answer based on the context above. If the context doesn't contain enough information to answer the question, say so. Keep your answer informative and well-structured.

Answer:"""
            )
            
            # Create LLM chain
            self.llm_chain = LLMChain(llm=self.llm, prompt=prompt_template)
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            self.use_llm = False
        
    def _check_initialization(self) -> bool:
        """Check if the vector database is already populated."""
        try:
            info = self.vector_db.get_collection_info()
            return info.get('document_count', 0) > 0
        except Exception:
            return False
    
    def initialize_database(self, force_rebuild: bool = False) -> bool:
        """
        Initialize the vector database with news articles.
        
        Args:
            force_rebuild: Whether to rebuild the database even if it exists
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.is_initialized and not force_rebuild:
                logger.info("Database already initialized. Use force_rebuild=True to rebuild.")
                return True
            
            if force_rebuild:
                logger.info("Force rebuilding database...")
                self.vector_db.clear_collection()
            
            # Load and preprocess articles
            logger.info("Loading and preprocessing articles...")
            articles = self.preprocessor.preprocess_articles()
            
            if not articles:
                logger.error("No articles to process")
                return False
            
            # Chunk articles
            logger.info("Chunking articles...")
            chunked_docs = self.vector_db.chunk_articles(articles)
            
            if not chunked_docs:
                logger.error("No document chunks created")
                return False
            
            # Add to vector database
            logger.info("Adding documents to vector database...")
            success = self.vector_db.add_documents(chunked_docs)
            
            if success:
                self.is_initialized = True
                logger.info("Database initialization completed successfully!")
                
                # Save initialization info
                self._save_initialization_info(len(articles), len(chunked_docs))
                
            return success
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            return False
    
    def _save_initialization_info(self, article_count: int, chunk_count: int):
        """Save initialization information to a file."""
        try:
            info = {
                'initialization_date': datetime.now().isoformat(),
                'article_count': article_count,
                'chunk_count': chunk_count,
                'csv_path': self.csv_path,
                'collection_name': self.collection_name
            }
            
            with open('rag_initialization_info.json', 'w') as f:
                import json
                json.dump(info, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Could not save initialization info: {e}")
    
    def query(self, 
              question: str, 
              n_results: int = 5,
              filter_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Query the RAG system with a question.
        
        Args:
            question: The question to ask
            n_results: Number of relevant documents to retrieve
            filter_metadata: Optional metadata filters
            
        Returns:
            Dictionary containing the question, retrieved documents, and answer
        """
        if not self.is_initialized:
            return {
                'error': 'Database not initialized. Please run initialize_database() first.',
                'question': question,
                'retrieved_documents': [],
                'answer': None
            }
        
        try:
            logger.info(f"Processing query: {question}")
            
            # Retrieve relevant documents
            retrieved_docs = self.vector_db.search(
                query=question,
                n_results=n_results,
                filter_metadata=filter_metadata
            )
            
            if not retrieved_docs:
                return {
                    'question': question,
                    'retrieved_documents': [],
                    'answer': "No relevant documents found for your question.",
                    'error': None
                }
            
            # Generate answer based on retrieved documents
            answer = self._generate_answer(question, retrieved_docs)
            
            return {
                'question': question,
                'retrieved_documents': retrieved_docs,
                'answer': answer,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'question': question,
                'retrieved_documents': [],
                'answer': None,
                'error': str(e)
            }
    
    def _generate_answer(self, question: str, documents: List[Dict[str, Any]]) -> str:
        """
        Generate an answer based on the question and retrieved documents.
        
        Args:
            question: The original question
            documents: List of retrieved documents
            
        Returns:
            Generated answer
        """
        try:
            # Combine relevant document texts
            context_texts = []
            for doc in documents:
                context_texts.append(doc['text'])
            
            # Create context from retrieved documents
            context = "\n\n".join(context_texts)
            
            # Use LLM if available, otherwise fallback to simple generation
            if self.use_llm and self.llm_chain:
                try:
                    # Generate answer using LLM
                    answer = self.llm_chain.run(context=context, question=question)
                    return answer
                except Exception as e:
                    logger.warning(f"LLM generation failed, using fallback: {e}")
            
            # Fallback: Simple answer generation
            answer = f"""Based on the BBC news articles, here's what I found:

{context[:2000]}{'...' if len(context) > 2000 else ''}

This information is derived from {len(documents)} relevant news articles. The articles cover various topics and provide different perspectives on the subject matter."""
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I found relevant articles but couldn't generate a proper answer. Please try rephrasing your question."
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the database."""
        try:
            vector_info = self.vector_db.get_collection_info()
            preprocessor_stats = self.preprocessor.get_article_stats()
            
            return {
                'vector_database': vector_info,
                'preprocessor_stats': preprocessor_stats,
                'is_initialized': self.is_initialized
            }
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {'error': str(e)}
    
    def search_by_topic(self, topic: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search for articles by topic.
        
        Args:
            topic: Topic to search for
            n_results: Number of results to return
            
        Returns:
            List of relevant documents
        """
        return self.vector_db.search(topic, n_results)
    
    def get_article_by_id(self, article_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific article by its ID.
        
        Args:
            article_id: ID of the article to retrieve
            
        Returns:
            Article document if found, None otherwise
        """
        try:
            results = self.vector_db.search(article_id, n_results=1)
            if results and results[0]['metadata'].get('article_id') == article_id:
                return results[0]
            return None
        except Exception as e:
            logger.error(f"Error getting article by ID: {e}")
            return None

def main():
    """Main function to demonstrate the RAG system."""
    # Initialize RAG system
    rag = RAGSystem()
    
    # Check if database needs initialization
    if not rag.is_initialized:
        print("Initializing database... This may take a few minutes.")
        success = rag.initialize_database()
        if not success:
            print("Failed to initialize database.")
            return
    else:
        print("Database already initialized.")
    
    # Get database stats
    stats = rag.get_database_stats()
    print("\n=== Database Statistics ===")
    print(f"Articles in database: {stats['vector_database'].get('document_count', 'N/A')}")
    print(f"Total articles processed: {stats['preprocessor_stats'].get('total_articles', 'N/A')}")
    
    # Example queries
    example_queries = [
        "What are the latest business news?",
        "Tell me about technology developments",
        "What happened in politics recently?",
        "Are there any sports updates?"
    ]
    
    print("\n=== Example Queries ===")
    for query in example_queries:
        print(f"\nQuery: {query}")
        result = rag.query(query, n_results=3)
        if result['answer']:
            print(f"Answer: {result['answer'][:200]}...")
        else:
            print("No answer generated.")

if __name__ == "__main__":
    main()
