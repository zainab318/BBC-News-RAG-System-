"""
Vector Database Setup for BBC News Articles RAG System
This module handles document chunking, embedding generation, and vector storage.
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any, Optional
import logging
import os
import json
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDatabase:
    """Manages vector database operations for the RAG system."""
    
    def __init__(self, 
                 collection_name: str = "bbc_news_articles",
                 persist_directory: str = "./chroma_db",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the vector database.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
            embedding_model: Sentence transformer model for embeddings
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
        
    def _get_or_create_collection(self):
        """Get existing collection or create a new one."""
        try:
            # Try to get existing collection
            collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Found existing collection: {self.collection_name}")
            return collection
        except Exception:
            # Create new collection if it doesn't exist
            logger.info(f"Creating new collection: {self.collection_name}")
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "BBC News Articles for RAG System"}
            )
            return collection
    
    def chunk_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split articles into smaller chunks for better retrieval.
        
        Args:
            articles: List of processed articles
            
        Returns:
            List of chunked documents
        """
        chunked_docs = []
        
        logger.info("Starting article chunking...")
        
        for article in articles:
            try:
                # Split the article text into chunks
                chunks = self.text_splitter.split_text(article['text'])
                
                for i, chunk in enumerate(chunks):
                    if len(chunk.strip()) < 50:  # Skip very short chunks
                        continue
                    
                    chunk_doc = {
                        'id': f"{article['id']}_chunk_{i}",
                        'text': chunk,
                        'metadata': {
                            **article['metadata'],
                            'article_id': article['id'],
                            'chunk_index': i,
                            'total_chunks': len(chunks),
                            'original_index': article['original_index']
                        }
                    }
                    chunked_docs.append(chunk_doc)
                
                if len(chunked_docs) % 100 == 0:
                    logger.info(f"Created {len(chunked_docs)} chunks...")
                    
            except Exception as e:
                logger.warning(f"Error chunking article {article['id']}: {e}")
                continue
        
        logger.info(f"Successfully created {len(chunked_docs)} chunks from {len(articles)} articles")
        return chunked_docs
    
    def add_documents(self, documents: List[Dict[str, Any]], batch_size: int = 1000) -> bool:
        """
        Add documents to the vector database in batches.
        
        Args:
            documents: List of document dictionaries
            batch_size: Number of documents to process in each batch
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Adding {len(documents)} documents to vector database in batches of {batch_size}...")
            
            total_docs = len(documents)
            processed = 0
            
            # Process documents in batches
            for i in range(0, total_docs, batch_size):
                batch_end = min(i + batch_size, total_docs)
                batch_docs = documents[i:batch_end]
                
                logger.info(f"Processing batch {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size} ({len(batch_docs)} documents)...")
                
                # Prepare data for ChromaDB
                ids = [doc['id'] for doc in batch_docs]
                texts = [doc['text'] for doc in batch_docs]
                metadatas = [doc['metadata'] for doc in batch_docs]
                
                # Generate embeddings for this batch
                logger.info("Generating embeddings for batch...")
                embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
                
                # Add batch to collection
                self.collection.add(
                    ids=ids,
                    documents=texts,
                    metadatas=metadatas,
                    embeddings=embeddings.tolist()
                )
                
                processed += len(batch_docs)
                logger.info(f"Added {processed}/{total_docs} documents to vector database")
            
            logger.info("Successfully added all documents to vector database")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to vector database: {e}")
            return False
    
    def search(self, 
               query: str, 
               n_results: int = 5,
               filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents in the vector database.
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of search results
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results,
                where=filter_metadata
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i]
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching vector database: {e}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            count = self.collection.count()
            return {
                'collection_name': self.collection_name,
                'document_count': count,
                'embedding_model': self.embedding_model_name,
                'persist_directory': self.persist_directory
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection."""
        try:
            # Delete the collection
            self.client.delete_collection(name=self.collection_name)
            
            # Recreate it
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "BBC News Articles for RAG System"}
            )
            
            logger.info("Collection cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False

def main():
    """Main function to demonstrate the vector database."""
    # Initialize vector database
    vector_db = VectorDatabase()
    
    # Get collection info
    info = vector_db.get_collection_info()
    print("=== Vector Database Info ===")
    print(f"Collection: {info.get('collection_name', 'N/A')}")
    print(f"Document count: {info.get('document_count', 'N/A')}")
    print(f"Embedding model: {info.get('embedding_model', 'N/A')}")

if __name__ == "__main__":
    main()
