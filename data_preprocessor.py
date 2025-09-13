"""
Data Preprocessing Script for BBC News Articles RAG System
This script loads, cleans, and prepares the BBC news articles for vector database storage.
"""

import pandas as pd
import re
from typing import List, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BBCNewsPreprocessor:
    """Preprocesses BBC news articles for RAG system."""
    
    def __init__(self, csv_path: str):
        """
        Initialize the preprocessor.
        
        Args:
            csv_path: Path to the BBC news CSV file
        """
        self.csv_path = csv_path
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """Load the CSV data."""
        try:
            logger.info(f"Loading data from {self.csv_path}")
            self.df = pd.read_csv(self.csv_path)
            logger.info(f"Loaded {len(self.df)} articles")
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text data.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if pd.isna(text) or text == "":
            return ""
        
        # Convert to string if not already
        text = str(text)
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()-]', '', text)
        
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def extract_metadata(self, row: pd.Series) -> Dict[str, Any]:
        """
        Extract metadata from a news article row.
        
        Args:
            row: Pandas Series representing one article
            
        Returns:
            Dictionary containing metadata
        """
        metadata = {
            'labels': str(row.get('labels', '')),
            'no_sentences': int(row.get('no_sentences', 0)),
            'flesch_score': float(row.get('Flesch Reading Ease Score', 0)),
            'dale_chall_score': float(row.get('Dale-Chall Readability Score', 0)),
            'text_rank_summary': str(row.get('text_rank_summary', '')),
            'lsa_summary': str(row.get('lsa_summary', ''))
        }
        return metadata
    
    def preprocess_articles(self) -> List[Dict[str, Any]]:
        """
        Preprocess all articles in the dataset.
        
        Returns:
            List of dictionaries containing processed articles
        """
        if self.df is None:
            self.load_data()
        
        processed_articles = []
        
        logger.info("Starting article preprocessing...")
        
        for idx, row in self.df.iterrows():
            try:
                # Clean the main text
                cleaned_text = self.clean_text(row['text'])
                
                if len(cleaned_text) < 50:  # Skip very short articles
                    continue
                
                # Extract metadata
                metadata = self.extract_metadata(row)
                
                # Create article document
                article = {
                    'id': f"article_{idx}",
                    'text': cleaned_text,
                    'metadata': metadata,
                    'original_index': idx
                }
                
                processed_articles.append(article)
                
                if (idx + 1) % 100 == 0:
                    logger.info(f"Processed {idx + 1} articles...")
                    
            except Exception as e:
                logger.warning(f"Error processing article {idx}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(processed_articles)} articles")
        return processed_articles
    
    def get_article_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the processed articles.
        
        Returns:
            Dictionary containing statistics
        """
        if self.df is None:
            self.load_data()
        
        stats = {
            'total_articles': len(self.df),
            'columns': list(self.df.columns),
            'text_length_stats': {
                'mean': self.df['text'].str.len().mean(),
                'median': self.df['text'].str.len().median(),
                'min': self.df['text'].str.len().min(),
                'max': self.df['text'].str.len().max()
            },
            'label_distribution': self.df['labels'].value_counts().to_dict() if 'labels' in self.df.columns else {}
        }
        
        return stats

def main():
    """Main function to demonstrate the preprocessor."""
    # Initialize preprocessor
    preprocessor = BBCNewsPreprocessor('bbc_news_articles/bbc_news_text_complexity_summarization.csv')
    
    # Load and preprocess data
    articles = preprocessor.preprocess_articles()
    
    # Get statistics
    stats = preprocessor.get_article_stats()
    
    print("=== BBC News Articles Preprocessing Complete ===")
    print(f"Total articles processed: {len(articles)}")
    print(f"Average text length: {stats['text_length_stats']['mean']:.0f} characters")
    print(f"Text length range: {stats['text_length_stats']['min']} - {stats['text_length_stats']['max']} characters")
    
    if stats['label_distribution']:
        print("\nLabel distribution:")
        for label, count in list(stats['label_distribution'].items())[:5]:
            print(f"  {label}: {count}")
    
    # Show sample article
    if articles:
        print(f"\nSample article (ID: {articles[0]['id']}):")
        print(f"Text preview: {articles[0]['text'][:200]}...")
        print(f"Metadata: {articles[0]['metadata']}")

if __name__ == "__main__":
    main()
