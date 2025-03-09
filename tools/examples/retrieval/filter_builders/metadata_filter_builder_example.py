#!/usr/bin/env python3
"""
Metadata Filter Builder Example

This script demonstrates how to use the MetadataFilterBuilder to create various types of metadata filters
for vector databases. It shows how to:

1. Create basic equality filters
2. Create text matching filters
3. Create list-based filters
4. Create existence filters
5. Combine filters with logical operators
6. Generate filters for different vector database formats
"""

import sys
import os
import json
import pprint
from typing import Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from tools.src.retrieval import get_filter_builder

def main():
    """
    Main function demonstrating the MetadataFilterBuilder with multiple examples.
    """
    print("\n" + "="*80)
    print("Metadata Filter Builder Example")
    print("="*80)
    
    # Create a MetadataFilterBuilder
    metadata_filter = get_filter_builder(builder_type="metadata")
    
    # Example 1: Basic Equality Filters
    print("\n1. Basic Equality Filters")
    print("-" * 60)
    
    # Create a filter for a specific category
    metadata_filter.field("category").equals("technology")
    
    # Get the filter dictionary
    filter_dict = metadata_filter.build()
    print("Filter for documents with category = 'technology':")
    pprint.pprint(filter_dict)
    
    # Reset the filter builder
    metadata_filter.reset()
    
    # Create a filter for a specific author
    metadata_filter.field("author").equals("John Doe")
    
    # Get the filter dictionary
    filter_dict = metadata_filter.build()
    print("\nFilter for documents with author = 'John Doe':")
    pprint.pprint(filter_dict)
    
    # Example 2: Text Matching Filters
    print("\n2. Text Matching Filters")
    print("-" * 60)
    
    # Reset the filter builder
    metadata_filter.reset()
    
    # Create a filter for text containing a substring
    metadata_filter.field("title").contains("python")
    
    # Get the filter dictionary
    filter_dict = metadata_filter.build()
    print("Filter for documents with title containing 'python':")
    pprint.pprint(filter_dict)
    
    # Reset the filter builder
    metadata_filter.reset()
    
    # Create a filter for text starting with a prefix
    metadata_filter.field("title").starts_with("How to")
    
    # Get the filter dictionary
    filter_dict = metadata_filter.build()
    print("\nFilter for documents with title starting with 'How to':")
    pprint.pprint(filter_dict)
    
    # Reset the filter builder
    metadata_filter.reset()
    
    # Create a filter for text ending with a suffix
    metadata_filter.field("title").ends_with("Guide")
    
    # Get the filter dictionary
    filter_dict = metadata_filter.build()
    print("\nFilter for documents with title ending with 'Guide':")
    pprint.pprint(filter_dict)
    
    # Reset the filter builder
    metadata_filter.reset()
    
    # Create a filter for text matching a regex pattern
    metadata_filter.field("email").matches_regex(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
    
    # Get the filter dictionary
    filter_dict = metadata_filter.build()
    print("\nFilter for documents with email matching a valid email pattern:")
    pprint.pprint(filter_dict)
    
    # Example 3: List-based Filters
    print("\n3. List-based Filters")
    print("-" * 60)
    
    # Reset the filter builder
    metadata_filter.reset()
    
    # Create a filter for values in a list
    metadata_filter.field("category").in_list(["technology", "programming", "data science"])
    
    # Get the filter dictionary
    filter_dict = metadata_filter.build()
    print("Filter for documents with category in ['technology', 'programming', 'data science']:")
    pprint.pprint(filter_dict)
    
    # Reset the filter builder
    metadata_filter.reset()
    
    # Create a filter for values not in a list
    metadata_filter.field("category").not_in_list(["sports", "entertainment", "politics"])
    
    # Get the filter dictionary
    filter_dict = metadata_filter.build()
    print("\nFilter for documents with category not in ['sports', 'entertainment', 'politics']:")
    pprint.pprint(filter_dict)
    
    # Reset the filter builder
    metadata_filter.reset()
    
    # Create a filter for array contains
    metadata_filter.field("tags").array_contains("python")
    
    # Get the filter dictionary
    filter_dict = metadata_filter.build()
    print("\nFilter for documents with tags array containing 'python':")
    pprint.pprint(filter_dict)
    
    # Reset the filter builder
    metadata_filter.reset()
    
    # Create a filter for array contains any
    metadata_filter.field("tags").array_contains_any(["python", "javascript", "rust"])
    
    # Get the filter dictionary
    filter_dict = metadata_filter.build()
    print("\nFilter for documents with tags array containing any of ['python', 'javascript', 'rust']:")
    pprint.pprint(filter_dict)
    
    # Reset the filter builder
    metadata_filter.reset()
    
    # Create a filter for array contains all
    metadata_filter.field("tags").array_contains_all(["programming", "tutorial"])
    
    # Get the filter dictionary
    filter_dict = metadata_filter.build()
    print("\nFilter for documents with tags array containing all of ['programming', 'tutorial']:")
    pprint.pprint(filter_dict)
    
    # Example 4: Existence Filters
    print("\n4. Existence Filters")
    print("-" * 60)
    
    # Reset the filter builder
    metadata_filter.reset()
    
    # Create a filter for field exists
    metadata_filter.field("summary").exists()
    
    # Get the filter dictionary
    filter_dict = metadata_filter.build()
    print("Filter for documents where 'summary' field exists:")
    pprint.pprint(filter_dict)
    
    # Reset the filter builder
    metadata_filter.reset()
    
    # Create a filter for field does not exist
    metadata_filter.field("summary").not_exists()
    
    # Get the filter dictionary
    filter_dict = metadata_filter.build()
    print("\nFilter for documents where 'summary' field does not exist:")
    pprint.pprint(filter_dict)
    
    # Example 5: Complex Filters with Logical Operators
    print("\n5. Complex Filters with Logical Operators")
    print("-" * 60)
    
    # Reset the filter builder
    metadata_filter.reset()
    
    # Create a complex filter with AND operator
    metadata_filter.field("category").equals("technology").and_().field("author").equals("John Doe")
    
    # Get the filter dictionary
    filter_dict = metadata_filter.build()
    print("Complex filter with AND operator (category = 'technology' AND author = 'John Doe'):")
    pprint.pprint(filter_dict)
    
    # Reset the filter builder
    metadata_filter.reset()
    
    # Create a complex filter with OR operator
    metadata_filter.field("category").equals("technology").or_().field("category").equals("programming")
    
    # Get the filter dictionary
    filter_dict = metadata_filter.build()
    print("\nComplex filter with OR operator (category = 'technology' OR category = 'programming'):")
    pprint.pprint(filter_dict)
    
    # Reset the filter builder
    metadata_filter.reset()
    
    # Create a complex filter with NOT operator
    metadata_filter.not_().field("category").equals("sports")
    
    # Get the filter dictionary
    filter_dict = metadata_filter.build()
    print("\nComplex filter with NOT operator (NOT category = 'sports'):")
    pprint.pprint(filter_dict)
    
    # Reset the filter builder
    metadata_filter.reset()
    
    # Create a complex filter with nested operators
    metadata_filter.field("category").equals("technology").and_().open_group().field("author").equals("John Doe").or_().field("author").equals("Jane Smith").close_group()
    
    # Get the filter dictionary
    filter_dict = metadata_filter.build()
    print("\nComplex filter with nested operators (category = 'technology' AND (author = 'John Doe' OR author = 'Jane Smith')):")
    pprint.pprint(filter_dict)
    
    # Example 6: Targeting Different Vector Databases
    print("\n6. Targeting Different Vector Databases")
    print("-" * 60)
    
    # Create filters for different vector database formats
    db_formats = ["generic", "chroma", "qdrant", "pinecone"]
    
    for db_format in db_formats:
        # Reset the filter builder
        metadata_filter.reset()
        
        # Create a metadata filter
        metadata_filter.field("category").equals("technology").and_().field("tags").array_contains("python")
        
        # Get the filter dictionary for the specific database format
        filter_dict = metadata_filter.build(format=db_format)
        print(f"\nFilter for {db_format.capitalize()} format:")
        pprint.pprint(filter_dict)
    
    print("\n" + "="*80)
    print("End of Metadata Filter Builder Example")
    print("="*80 + "\n")


if __name__ == "__main__":
    main() 