"""
Code RAG Example

This script demonstrates how to use the Code RAG implementation.
It shows how to:
1. Ingest code files
2. Query the Code RAG system with and without code context
3. Analyze and improve code snippets
"""

import os
import sys

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from code_rag_pipeline import CodeRAG


def create_sample_code_file(filename, code):
    """
    Create a sample code file for demonstration.
    
    Args:
        filename: The name of the file to create.
        code: The code content to write to the file.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        f.write(code)
    print(f"Created sample file: {filename}")


def run_code_rag_example():
    """
    Run an example of the Code RAG pipeline.
    """
    # Create a directory for sample code files
    sample_dir = "../data/raw/samples"
    os.makedirs(sample_dir, exist_ok=True)
    
    # Create sample Python code files
    create_sample_code_file(f"{sample_dir}/binary_search.py", """
def binary_search(arr, target):
    \"\"\"
    Perform binary search on a sorted array.
    
    Args:
        arr: A sorted array
        target: The target element to find
        
    Returns:
        The index of the target element, or -1 if not found
    \"\"\"
    left = 0
    right = len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1
""")

    create_sample_code_file(f"{sample_dir}/bubble_sort.py", """
def bubble_sort(arr):
    \"\"\"
    Sort an array using bubble sort algorithm.
    
    Args:
        arr: An array to sort
        
    Returns:
        The sorted array
    \"\"\"
    n = len(arr)
    
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    
    return arr
""")

    create_sample_code_file(f"{sample_dir}/calculator.py", """
class Calculator:
    \"\"\"A simple calculator class.\"\"\"
    
    def __init__(self):
        self.result = 0
    
    def add(self, x, y):
        \"\"\"Add two numbers.\"\"\"
        return x + y
    
    def subtract(self, x, y):
        \"\"\"Subtract y from x.\"\"\"
        return x - y
    
    def multiply(self, x, y):
        \"\"\"Multiply two numbers.\"\"\"
        return x * y
    
    def divide(self, x, y):
        \"\"\"Divide x by y.\"\"\"
        if y == 0:
            raise ValueError("Cannot divide by zero")
        return x / y
""")

    # Initialize the Code RAG pipeline
    vector_store_path = "../data/processed/python_vector_store"
    print("Initializing Code RAG pipeline...")
    code_rag = CodeRAG(
        vector_store_path=vector_store_path if os.path.exists(vector_store_path) else None,
        language="python"
    )
    
    # Ingest the sample code files
    print("\nIngesting sample code files...")
    code_rag.ingest_code_directory(sample_dir)
    
    # Save the vector store
    print("\nSaving the vector store...")
    code_rag.save_vector_store(vector_store_path)
    
    # Example queries
    example_queries = [
        {
            "title": "Basic Code Information",
            "query": "How does the binary search algorithm work?",
            "code_context": ""
        },
        {
            "title": "Code Improvement Suggestion",
            "query": "How can I optimize the bubble sort algorithm?",
            "code_context": """def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr"""
        },
        {
            "title": "Debugging Help",
            "query": "What is wrong with this code and how can I fix it?",
            "code_context": """def binary_search(arr, target):
    left = 0
    right = len(arr)
    
    while left < right:
        mid = (left + right) / 2  # Error: should be integer division
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid
        else:
            right = mid
    
    return -1"""
        },
        {
            "title": "Code Extension",
            "query": "How can I add a power function to the Calculator class?",
            "code_context": """class Calculator:
    def __init__(self):
        self.result = 0
    
    def add(self, x, y):
        return x + y
    
    def subtract(self, x, y):
        return x - y
    
    def multiply(self, x, y):
        return x * y
    
    def divide(self, x, y):
        if y == 0:
            raise ValueError("Cannot divide by zero")
        return x / y"""
        }
    ]
    
    # Run queries
    print("\n=== Running Example Queries with Code RAG ===\n")
    for example in example_queries:
        print(f"\n{'='*80}")
        print(f"QUERY: {example['title']}")
        print(f"{'='*80}")
        
        print(f"Question: {example['query']}")
        
        if example['code_context']:
            print("\nCode Context:")
            print(example['code_context'])
        
        # Run the query
        result = code_rag.query(example['query'], example['code_context'])
        
        print("\nRetrieved Documents:")
        for i, doc in enumerate(result["retrieved_documents"]):
            print(f"\nDocument {i+1} (from {doc.metadata.get('source', 'Unknown')}):")
            chunk_type = doc.metadata.get('chunk_type', 'Unknown')
            print(f"Type: {chunk_type}")
            
            if chunk_type == 'class':
                print(f"Class: {doc.metadata.get('class_name', 'Unknown')}")
            elif chunk_type == 'function':
                print(f"Function: {doc.metadata.get('function_name', 'Unknown')}")
            elif chunk_type == 'method':
                print(f"Method: {doc.metadata.get('method_name', 'Unknown')}")
            
            # Print a short preview
            lines = doc.page_content.split('\n')
            preview = '\n'.join(lines[:3]) + ('\n...' if len(lines) > 3 else '')
            print(f"\n{preview}")
        
        print("\nResponse:")
        print(result["response"])
        print("\n" + "="*80)
    
    print("\nCode RAG example completed successfully!")


if __name__ == "__main__":
    run_code_rag_example() 