#!/usr/bin/env python3
"""
Embedding Tools Example

This script demonstrates how to use the embedding tools to generate, visualize,
analyze, and transform embeddings.
"""

import os
import sys
import logging
import numpy as np
from typing import List

# Add the parent directory to the path so we can import the tools package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embeddings import (
    get_embedding_generator,
    get_embedding_visualizer,
    get_embedding_analyzer,
    get_dimensionality_reducer
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """
    Main function to demonstrate embedding tools.
    """
    # Sample texts
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast auburn fox leaps above the sleepy canine.",
        "The rapid red fox hops over the inactive hound.",
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning is a type of machine learning.",
        "Neural networks are used in deep learning algorithms.",
        "Python is a popular programming language for data science.",
        "R is another language commonly used for statistical analysis.",
        "Julia is gaining popularity for numerical computing.",
    ]
    
    # Create labels for visualization
    labels = [
        "fox1", "fox2", "fox3",
        "ai1", "ai2", "ai3",
        "lang1", "lang2", "lang3"
    ]
    
    # Step 1: Generate embeddings
    logger.info("Generating embeddings...")
    try:
        # Try to use sentence-transformers if available
        generator = get_embedding_generator(
            generator_type="sentence-transformers",
            model_name="all-MiniLM-L6-v2",
            normalize_embeddings=True
        )
    except ImportError:
        # Fall back to a simple custom generator for demonstration
        logger.info("Sentence Transformers not available, using a simple custom generator.")
        generator = get_embedding_generator(
            generator_type="custom",
            embedding_function=simple_embedding_function,
            normalize_embeddings=True
        )
    
    # Generate embeddings
    embeddings = generator.batch_generate(
        texts=texts,
        batch_size=3,
        show_progress=True
    )
    
    logger.info(f"Generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}")
    
    # Step 2: Analyze embeddings
    logger.info("Analyzing embeddings...")
    
    # Similarity analysis
    similarity_analyzer = get_embedding_analyzer(
        analyzer_type="similarity",
        metric="cosine"
    )
    
    similarity_results = similarity_analyzer.analyze(
        embeddings=embeddings,
        texts=texts
    )
    
    logger.info("Similarity Analysis Results:")
    for i, (text1, similarities) in enumerate(zip(texts, similarity_results["similarity_matrix"])):
        top_similar = sorted(
            [(j, sim) for j, sim in enumerate(similarities) if j != i],
            key=lambda x: x[1],
            reverse=True
        )[:2]
        
        logger.info(f"Text: {text1[:30]}...")
        for j, sim in top_similar:
            logger.info(f"  Similar to: {texts[j][:30]}... (similarity: {sim:.4f})")
    
    # Clustering analysis
    clustering_analyzer = get_embedding_analyzer(
        analyzer_type="clustering",
        n_clusters=3
    )
    
    clustering_results = clustering_analyzer.analyze(
        embeddings=embeddings,
        texts=texts
    )
    
    logger.info("Clustering Analysis Results:")
    clusters = {}
    for i, cluster_id in enumerate(clustering_results["cluster_labels"]):
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(texts[i])
    
    for cluster_id, cluster_texts in clusters.items():
        logger.info(f"Cluster {cluster_id}:")
        for text in cluster_texts:
            logger.info(f"  - {text[:50]}...")
    
    # Step 3: Reduce dimensionality for visualization
    logger.info("Reducing dimensionality...")
    reducer = get_dimensionality_reducer(
        reducer_type="pca",
        n_components=2
    )
    
    reduced_embeddings = reducer.fit_transform(embeddings)
    
    # Step 4: Visualize embeddings
    logger.info("Visualizing embeddings...")
    try:
        # Try to use matplotlib if available
        visualizer = get_embedding_visualizer(
            visualizer_type="matplotlib",
            n_components=2
        )
        
        # Visualize
        fig = visualizer.visualize(
            embeddings=reduced_embeddings,
            labels=labels,
            title="Text Embeddings Visualization"
        )
        
        # Save visualization
        os.makedirs("outputs", exist_ok=True)
        visualizer.save("outputs/embedding_visualization.png")
        logger.info("Visualization saved to outputs/embedding_visualization.png")
        
    except ImportError:
        logger.info("Matplotlib not available, skipping visualization.")
    
    # Step 5: Save embeddings
    logger.info("Saving embeddings...")
    os.makedirs("outputs", exist_ok=True)
    generator.save_embeddings(
        embeddings=embeddings,
        path="outputs/embeddings.json",
        metadata=[{"text": text, "label": label} for text, label in zip(texts, labels)]
    )
    
    logger.info("Embeddings saved to outputs/embeddings.json")
    
    logger.info("Example completed successfully!")


def simple_embedding_function(texts: List[str]) -> List[List[float]]:
    """
    A simple embedding function that creates random embeddings.
    This is used as a fallback when sentence-transformers is not available.
    
    Args:
        texts: List of texts to generate embeddings for
        
    Returns:
        List of embeddings
    """
    # Create random embeddings with consistent dimensions for the same text
    embeddings = []
    for text in texts:
        # Use the hash of the text as a seed for reproducibility
        seed = hash(text) % 10000
        np.random.seed(seed)
        
        # Create a random embedding
        embedding = np.random.randn(384)  # 384 dimensions like MiniLM
        
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        
        embeddings.append(embedding.tolist())
    
    return embeddings


if __name__ == "__main__":
    main() 