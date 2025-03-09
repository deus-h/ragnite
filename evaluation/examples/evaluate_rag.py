#!/usr/bin/env python3
"""
RAG Evaluation Example

This script demonstrates how to use the RAG evaluation framework to evaluate
a RAG system's performance.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))
from evaluation.src.evaluator import RAGEvaluator


def setup_argparse():
    """Set up command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Evaluate a Retrieval-Augmented Generation (RAG) system"
    )
    
    # Required arguments
    parser.add_argument(
        "--rag_system", 
        help="Path to the RAG system module to evaluate",
    )
    
    # Evaluation options
    parser.add_argument(
        "--benchmark", 
        help="Path to benchmark dataset JSON file",
    )
    parser.add_argument(
        "--output_dir", 
        default="./evaluation_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=100,
        help="Number of samples to evaluate",
    )
    
    # Evaluation components
    parser.add_argument(
        "--skip_retrieval", 
        action="store_true",
        help="Skip retrieval evaluation",
    )
    parser.add_argument(
        "--skip_generation", 
        action="store_true",
        help="Skip generation evaluation",
    )
    parser.add_argument(
        "--skip_end_to_end", 
        action="store_true",
        help="Skip end-to-end evaluation",
    )
    parser.add_argument(
        "--skip_efficiency", 
        action="store_true",
        help="Skip efficiency measurement",
    )
    
    # Visualization options
    parser.add_argument(
        "--skip_visualization", 
        action="store_true",
        help="Skip result visualization",
    )
    parser.add_argument(
        "--show_plots", 
        action="store_true",
        help="Show plots during evaluation",
    )
    
    # Advanced options
    parser.add_argument(
        "--semantic_model", 
        default="all-MiniLM-L6-v2",
        help="Semantic similarity model to use",
    )
    parser.add_argument(
        "--disable_semantic", 
        action="store_true",
        help="Disable semantic similarity for evaluation",
    )
    
    return parser


def load_rag_system(path):
    """
    Load a RAG system from a Python module path.
    
    Args:
        path: Path to the RAG system module
        
    Returns:
        Loaded RAG system instance
    """
    import importlib.util
    
    # Check if the path exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"RAG system file not found: {path}")
    
    # Load the module
    module_name = os.path.basename(path).replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, path)
    
    if spec is None:
        raise ImportError(f"Could not load module from {path}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Look for a RAG class or function in the module
    rag_system = None
    
    # Try to find a class or function named RAG or containing RAG
    for attr_name in dir(module):
        if "rag" in attr_name.lower():
            rag_attr = getattr(module, attr_name)
            if callable(rag_attr):
                rag_system = rag_attr
                break
    
    if rag_system is None:
        raise ValueError(f"Could not find a RAG system in {path}")
    
    # Initialize the RAG system if it's a class
    if isinstance(rag_system, type):
        try:
            return rag_system()
        except Exception as e:
            print(f"Warning: Could not initialize RAG system: {e}")
            print("Falling back to class reference.")
            return rag_system
    
    return rag_system


def create_dummy_rag_system():
    """
    Create a dummy RAG system for demonstration purposes.
    
    Returns:
        A simple dummy RAG system
    """
    class DummyRAG:
        def query(self, query_text):
            """Simple dummy query function that returns a fixed response."""
            # Simulate retrieval and generation
            retrieved_docs = [
                f"This is a dummy document about {query_text}",
                f"Another dummy document mentioning {query_text}",
            ]
            
            # Simulate a generated answer
            answer = f"This is a dummy answer to: {query_text}. "
            answer += "The dummy RAG system retrieved some documents and generated this response."
            
            # Return in a format expected by the evaluator
            return {
                "answer": answer,
                "retrieved_docs": retrieved_docs,
                "context": " ".join(retrieved_docs),
            }
    
    return DummyRAG()


def main():
    """Main function."""
    # Parse command-line arguments
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load RAG system
    rag_system = None
    if args.rag_system:
        print(f"Loading RAG system from {args.rag_system}...")
        try:
            rag_system = load_rag_system(args.rag_system)
            print("RAG system loaded successfully.")
        except Exception as e:
            print(f"Error loading RAG system: {e}")
            print("Falling back to dummy RAG system.")
            rag_system = create_dummy_rag_system()
    else:
        print("No RAG system specified. Using dummy RAG system.")
        rag_system = create_dummy_rag_system()
    
    # Initialize the evaluator
    evaluator = RAGEvaluator(
        use_semantic_similarity=not args.disable_semantic,
        semantic_model_name=args.semantic_model,
        output_dir=args.output_dir,
    )
    
    # Run the evaluation
    results = evaluator.evaluate(
        rag_system=rag_system,
        benchmark_dataset=args.benchmark,
        num_samples=args.num_samples,
        evaluate_retrieval=not args.skip_retrieval,
        evaluate_generation=not args.skip_generation,
        evaluate_end_to_end=not args.skip_end_to_end,
        measure_system_efficiency=not args.skip_efficiency,
        visualize=not args.skip_visualization,
    )
    
    # Print summary
    summary = evaluator.summary(results)
    print("\n" + summary)
    
    # Save summary
    summary_file = os.path.join(args.output_dir, "evaluation_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary)
    
    print(f"\nEvaluation complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main() 