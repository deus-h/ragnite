#!/usr/bin/env python3
"""
setup_test_data.py - Populates test data into vector databases for testing
"""
import argparse
import os
import sys
import logging
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Sample text data for testing
SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast auburn fox leaps above the sleepy canine.",
    "The rapid red fox hops over the inactive hound.",
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning is a type of machine learning.",
    "Neural networks are used in deep learning algorithms.",
    "Python is a popular programming language for data science.",
    "R is another language commonly used for statistical analysis.",
    "Julia is gaining popularity for numerical computing.",
    "Vector databases are specialized for similarity search."
]

def generate_test_embeddings(count=10, dim=384):
    """
    Generate random embeddings for testing.
    
    Args:
        count: Number of embeddings to generate
        dim: Dimensionality of embeddings
        
    Returns:
        List of embeddings
    """
    np.random.seed(42)  # For reproducibility
    
    # Generate random embeddings
    embeddings = []
    for i in range(count):
        # Create a random embedding
        embedding = np.random.randn(dim)
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        embeddings.append(embedding.tolist())
    
    return embeddings

def setup_chromadb():
    """Set up test data in ChromaDB"""
    try:
        import chromadb
        
        logger.info("Setting up ChromaDB test data...")
        
        # Connect to ChromaDB
        client = chromadb.HttpClient(host="localhost", port=8000)
        
        # Create a test collection
        try:
            client.delete_collection("test_collection")
            logger.info("Deleted existing test_collection")
        except Exception:
            pass
        
        collection = client.create_collection(
            name="test_collection",
            metadata={"description": "Test collection for RAG"}
        )
        
        # Generate test embeddings
        embeddings = generate_test_embeddings(len(SAMPLE_TEXTS))
        
        # Add documents to collection
        collection.add(
            embeddings=embeddings,
            documents=SAMPLE_TEXTS,
            metadatas=[{"index": i, "category": "test"} for i in range(len(SAMPLE_TEXTS))],
            ids=[f"doc_{i}" for i in range(len(SAMPLE_TEXTS))]
        )
        
        logger.info(f"Added {len(SAMPLE_TEXTS)} documents to ChromaDB test_collection")
        
    except ImportError:
        logger.error("chromadb package not installed. Install with: pip install chromadb")
        
    except Exception as e:
        logger.error(f"Error setting up ChromaDB: {str(e)}")

def setup_postgres():
    """Set up test data in PostgreSQL with pgvector"""
    try:
        import psycopg2
        import psycopg2.extras
        
        logger.info("Setting up PostgreSQL/pgvector test data...")
        
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            dbname="vectordb",
            user="postgres",
            password="postgres"
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Generate test embeddings
        embeddings = generate_test_embeddings(len(SAMPLE_TEXTS))
        
        # Add vectors to test_collection
        for i, (text, embedding) in enumerate(zip(SAMPLE_TEXTS, embeddings)):
            cursor.execute(
                """
                INSERT INTO vector_store.test_collection (id, vector, metadata)
                VALUES (%s, %s, %s)
                ON CONFLICT (id) DO UPDATE
                SET vector = EXCLUDED.vector, metadata = EXCLUDED.metadata
                """,
                (
                    f"doc_{i}",
                    embedding,
                    {"index": i, "category": "test", "text": text}
                )
            )
        
        # Close connection
        cursor.close()
        conn.close()
        
        logger.info(f"Added {len(SAMPLE_TEXTS)} vectors to PostgreSQL test_collection")
        
    except ImportError:
        logger.error("psycopg2 package not installed. Install with: pip install psycopg2-binary")
        
    except Exception as e:
        logger.error(f"Error setting up PostgreSQL: {str(e)}")

def setup_qdrant():
    """Set up test data in Qdrant"""
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models
        
        logger.info("Setting up Qdrant test data...")
        
        # Connect to Qdrant
        client = QdrantClient(host="localhost", port=6333)
        
        # Create a test collection
        try:
            client.delete_collection("test_collection")
            logger.info("Deleted existing test_collection")
        except Exception:
            pass
        
        client.create_collection(
            collection_name="test_collection",
            vectors_config=models.VectorParams(
                size=384,
                distance=models.Distance.COSINE
            )
        )
        
        # Generate test embeddings
        embeddings = generate_test_embeddings(len(SAMPLE_TEXTS))
        
        # Prepare points for insertion
        points = []
        for i, (text, embedding) in enumerate(zip(SAMPLE_TEXTS, embeddings)):
            points.append(
                models.PointStruct(
                    id=i,
                    vector=embedding,
                    payload={"text": text, "index": i, "category": "test"}
                )
            )
        
        # Add points to collection
        client.upsert(
            collection_name="test_collection",
            points=points
        )
        
        logger.info(f"Added {len(SAMPLE_TEXTS)} points to Qdrant test_collection")
        
    except ImportError:
        logger.error("qdrant_client package not installed. Install with: pip install qdrant-client")
        
    except Exception as e:
        logger.error(f"Error setting up Qdrant: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Set up test data for vector databases")
    parser.add_argument("--db", choices=["chromadb", "postgres", "qdrant", "all"], 
                       default="all", help="Which database to set up")
    
    args = parser.parse_args()
    
    if args.db in ["chromadb", "all"]:
        setup_chromadb()
    
    if args.db in ["postgres", "all"]:
        setup_postgres()
    
    if args.db in ["qdrant", "all"]:
        setup_qdrant()

if __name__ == "__main__":
    main() 