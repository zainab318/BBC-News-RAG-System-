"""
Main Application for BBC News Articles RAG System
This is the main entry point for the RAG application with a user-friendly interface.
"""

import sys
import logging
# Import our RAG system
from rag_system import RAGSystem

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGApplication:
    """Main application class for the RAG system."""
    
    def __init__(self):
        """Initialize the RAG application."""
        self.rag_system = None
        self.running = True
        
    def initialize_system(self) -> bool:
        """Initialize the RAG system."""
        try:
            print("🚀 Initializing BBC News RAG System...")
            print("=" * 50)
            
            # Check for API key
            import os
            gemini_key = os.getenv('GOOGLE_API_KEY')
            openai_key = os.getenv('OPENAI_API_KEY')
            
            if gemini_key:
                print("🤖 LLM features enabled (Google Gemini API key found)")
                llm_provider = "gemini"
                api_key = gemini_key
            elif openai_key:
                print("🤖 LLM features enabled (OpenAI API key found)")
                llm_provider = "openai"
                api_key = openai_key
            else:
                print("⚠️  LLM features disabled (No API key found)")
                print("   Run 'python setup_gemini_key.py' for FREE Gemini API")
                print("   Or run 'python setup_api_key.py' for OpenAI API")
                llm_provider = "gemini"
                api_key = None
            
            # Initialize RAG system
            self.rag_system = RAGSystem(use_llm=bool(api_key), llm_provider=llm_provider)
            
            # Check if database needs initialization
            if not self.rag_system.is_initialized:
                print("📊 Database not found. Initializing with BBC news articles...")
                print("⏳ This may take a few minutes for the first time...")
                
                success = self.rag_system.initialize_database()
                if not success:
                    print("❌ Failed to initialize database. Please check the error logs.")
                    return False
                
                print("✅ Database initialized successfully!")
            else:
                print("✅ Database already initialized and ready to use!")
            
            # Display database statistics
            self.show_database_stats()
            
            return True
            
        except Exception as e:
            print(f"❌ Error initializing system: {e}")
            logger.error(f"System initialization error: {e}")
            return False
    
    def show_database_stats(self):
        """Display database statistics."""
        try:
            stats = self.rag_system.get_database_stats()
            
            print("\n📈 Database Statistics:")
            print("-" * 30)
            print(f"📰 Articles in database: {stats['vector_database'].get('document_count', 'N/A')}")
            print(f"📊 Total articles processed: {stats['preprocessor_stats'].get('total_articles', 'N/A')}")
            print(f"🤖 Embedding model: {stats['vector_database'].get('embedding_model', 'N/A')}")
            
            # Show label distribution if available
            label_dist = stats['preprocessor_stats'].get('label_distribution', {})
            if label_dist:
                print(f"🏷️  Top categories:")
                for i, (label, count) in enumerate(list(label_dist.items())[:3]):
                    print(f"   {i+1}. {label}: {count} articles")
            
        except Exception as e:
            print(f"⚠️  Could not retrieve database statistics: {e}")
    
    def show_help(self):
        """Display help information."""
        print("\n📖 Available Commands:")
        print("-" * 30)
        print("🔍 ask <question>     - Ask a question about the news")
        print("🔎 search <topic>     - Search for articles by topic")
        print("📊 stats             - Show database statistics")
        print("🔄 rebuild           - Rebuild the database")
        print("❓ help              - Show this help message")
        print("🚪 quit/exit         - Exit the application")
        print("\n💡 Example queries:")
        print("   ask What are the latest business news?")
        print("   ask Tell me about technology developments")
        print("   search politics")
        print("   search sports")
    
    def process_query(self, query: str, n_results: int = 5):
        """Process a user query."""
        try:
            print(f"\n🔍 Processing query: '{query}'")
            print("⏳ Searching through articles...")
            
            # Query the RAG system
            result = self.rag_system.query(query, n_results=n_results)
            
            if result.get('error'):
                print(f"❌ Error: {result['error']}")
                return
            
            # Display results
            print(f"\n📝 Answer:")
            print("-" * 40)
            print(result['answer'])
            
            # Show source information
            if result['retrieved_documents']:
                print(f"\n📚 Sources ({len(result['retrieved_documents'])} articles):")
                print("-" * 40)
                for i, doc in enumerate(result['retrieved_documents'][:3], 1):
                    metadata = doc.get('metadata', {})
                    article_id = metadata.get('article_id', 'Unknown')
                    labels = metadata.get('labels', 'Unknown category')
                    print(f"{i}. Article {article_id} ({labels})")
                    print(f"   Preview: {doc['text'][:100]}...")
                    print()
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            logger.error(f"Query processing error: {e}")
    
    def process_search(self, topic: str, n_results: int = 10):
        """Process a topic search."""
        try:
            print(f"\n🔎 Searching for: '{topic}'")
            print("⏳ Finding relevant articles...")
            
            # Search for articles
            results = self.rag_system.search_by_topic(topic, n_results=n_results)
            
            if not results:
                print("❌ No articles found for this topic.")
                return
            
            print(f"\n📰 Found {len(results)} articles about '{topic}':")
            print("-" * 50)
            
            for i, doc in enumerate(results, 1):
                metadata = doc.get('metadata', {})
                article_id = metadata.get('article_id', 'Unknown')
                labels = metadata.get('labels', 'Unknown category')
                distance = doc.get('distance', 0)
                
                print(f"{i}. Article {article_id} ({labels}) [Relevance: {1-distance:.2f}]")
                print(f"   {doc['text'][:150]}...")
                print()
            
        except Exception as e:
            print(f"❌ Error searching: {e}")
            logger.error(f"Search error: {e}")
    
    def rebuild_database(self):
        """Rebuild the database."""
        try:
            print("\n🔄 Rebuilding database...")
            print("⏳ This will take several minutes...")
            
            success = self.rag_system.initialize_database(force_rebuild=True)
            
            if success:
                print("✅ Database rebuilt successfully!")
                self.show_database_stats()
            else:
                print("❌ Failed to rebuild database.")
                
        except Exception as e:
            print(f"❌ Error rebuilding database: {e}")
            logger.error(f"Database rebuild error: {e}")
    
    def run(self):
        """Run the main application loop."""
        print("🎯 BBC News Articles RAG System")
        print("=" * 50)
        print("Welcome! This system can answer questions about BBC news articles.")
        print("Type 'help' for available commands or 'quit' to exit.")
        
        # Initialize the system
        if not self.initialize_system():
            print("❌ Failed to initialize system. Exiting.")
            return
        
        # Main application loop
        while self.running:
            try:
                # Get user input
                user_input = input("\n💬 Enter command: ").strip()
                
                if not user_input:
                    continue
                
                # Parse command
                parts = user_input.split(' ', 1)
                command = parts[0].lower()
                argument = parts[1] if len(parts) > 1 else ""
                
                # Process commands
                if command in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    self.running = False
                
                elif command == 'help' or command == 'h':
                    self.show_help()
                
                elif command == 'ask':
                    if not argument:
                        print("❌ Please provide a question. Example: ask What is the latest news?")
                    else:
                        self.process_query(argument)
                
                elif command == 'search':
                    if not argument:
                        print("❌ Please provide a topic. Example: search technology")
                    else:
                        self.process_search(argument)
                
                elif command == 'stats':
                    self.show_database_stats()
                
                elif command == 'rebuild':
                    confirm = input("⚠️  This will rebuild the entire database. Continue? (y/N): ")
                    if confirm.lower() in ['y', 'yes']:
                        self.rebuild_database()
                    else:
                        print("❌ Rebuild cancelled.")
                
                else:
                    print(f"❌ Unknown command: {command}")
                    print("Type 'help' for available commands.")
            
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                self.running = False
            
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                logger.error(f"Unexpected error in main loop: {e}")

def main():
    """Main entry point."""
    try:
        app = RAGApplication()
        app.run()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
