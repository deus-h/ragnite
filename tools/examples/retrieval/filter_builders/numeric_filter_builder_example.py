#!/usr/bin/env python3
"""
Numeric Filter Builder Example

This script demonstrates how to use the NumericFilterBuilder to create various types of numeric filters
for vector databases. It shows how to:

1. Create basic numeric equality filters
2. Create numeric range filters
3. Create greater than and less than filters
4. Create filters for multiple numeric values
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
    Main function demonstrating the NumericFilterBuilder with multiple examples.
    """
    print("\n" + "="*80)
    print("Numeric Filter Builder Example")
    print("="*80)
    
    # Create a NumericFilterBuilder
    numeric_filter = get_filter_builder(builder_type="numeric")
    
    # Example 1: Basic Numeric Equality Filter
    print("\n1. Basic Numeric Equality Filter")
    print("-" * 60)
    
    # Create a filter for a specific price
    numeric_filter.field("price").equals(99.99)
    
    # Get the filter dictionary
    filter_dict = numeric_filter.build()
    print("Filter for documents with price = 99.99:")
    pprint.pprint(filter_dict)
    
    # Example 2: Numeric Range Filter
    print("\n2. Numeric Range Filter")
    print("-" * 60)
    
    # Reset the filter builder
    numeric_filter.reset()
    
    # Create a filter for a price range
    numeric_filter.field("price").between(10.0, 50.0)
    
    # Get the filter dictionary
    filter_dict = numeric_filter.build()
    print("Filter for documents with price between 10.0 and 50.0:")
    pprint.pprint(filter_dict)
    
    # Example 3: Greater Than and Less Than Filters
    print("\n3. Greater Than and Less Than Filters")
    print("-" * 60)
    
    # Reset the filter builder
    numeric_filter.reset()
    
    # Create a filter for prices greater than a specific value
    numeric_filter.field("price").greater_than(100.0)
    
    # Get the filter dictionary
    filter_dict = numeric_filter.build()
    print("Filter for documents with price > 100.0:")
    pprint.pprint(filter_dict)
    
    # Reset the filter builder
    numeric_filter.reset()
    
    # Create a filter for prices less than a specific value
    numeric_filter.field("price").less_than(50.0)
    
    # Get the filter dictionary
    filter_dict = numeric_filter.build()
    print("Filter for documents with price < 50.0:")
    pprint.pprint(filter_dict)
    
    # Reset the filter builder
    numeric_filter.reset()
    
    # Create a filter for prices greater than or equal to a specific value
    numeric_filter.field("price").greater_than_or_equal(100.0)
    
    # Get the filter dictionary
    filter_dict = numeric_filter.build()
    print("Filter for documents with price >= 100.0:")
    pprint.pprint(filter_dict)
    
    # Reset the filter builder
    numeric_filter.reset()
    
    # Create a filter for prices less than or equal to a specific value
    numeric_filter.field("price").less_than_or_equal(50.0)
    
    # Get the filter dictionary
    filter_dict = numeric_filter.build()
    print("Filter for documents with price <= 50.0:")
    pprint.pprint(filter_dict)
    
    # Example 4: Multiple Numeric Values
    print("\n4. Multiple Numeric Values")
    print("-" * 60)
    
    # Reset the filter builder
    numeric_filter.reset()
    
    # Create a filter for a list of specific prices
    numeric_filter.field("price").in_list([9.99, 19.99, 29.99, 39.99])
    
    # Get the filter dictionary
    filter_dict = numeric_filter.build()
    print("Filter for documents with price in [9.99, 19.99, 29.99, 39.99]:")
    pprint.pprint(filter_dict)
    
    # Reset the filter builder
    numeric_filter.reset()
    
    # Create a filter for prices not in a list
    numeric_filter.field("price").not_in_list([9.99, 19.99, 29.99, 39.99])
    
    # Get the filter dictionary
    filter_dict = numeric_filter.build()
    print("Filter for documents with price not in [9.99, 19.99, 29.99, 39.99]:")
    pprint.pprint(filter_dict)
    
    # Example 5: Complex Numeric Filters with Logical Operators
    print("\n5. Complex Numeric Filters with Logical Operators")
    print("-" * 60)
    
    # Reset the filter builder
    numeric_filter.reset()
    
    # Create a complex filter with AND operator
    numeric_filter.field("price").greater_than(10.0).and_().field("quantity").greater_than(5)
    
    # Get the filter dictionary
    filter_dict = numeric_filter.build()
    print("Complex filter with AND operator (price > 10.0 AND quantity > 5):")
    pprint.pprint(filter_dict)
    
    # Reset the filter builder
    numeric_filter.reset()
    
    # Create a complex filter with OR operator
    numeric_filter.field("price").less_than(10.0).or_().field("price").greater_than(100.0)
    
    # Get the filter dictionary
    filter_dict = numeric_filter.build()
    print("Complex filter with OR operator (price < 10.0 OR price > 100.0):")
    pprint.pprint(filter_dict)
    
    # Reset the filter builder
    numeric_filter.reset()
    
    # Create a complex filter with NOT operator
    numeric_filter.not_().field("price").between(50.0, 100.0)
    
    # Get the filter dictionary
    filter_dict = numeric_filter.build()
    print("Complex filter with NOT operator (NOT (price between 50.0 and 100.0)):")
    pprint.pprint(filter_dict)
    
    # Reset the filter builder
    numeric_filter.reset()
    
    # Create a complex filter with nested operators
    numeric_filter.field("price").less_than(20.0).and_().open_group().field("quantity").greater_than(10).or_().field("is_featured").equals(1).close_group()
    
    # Get the filter dictionary
    filter_dict = numeric_filter.build()
    print("Complex filter with nested operators (price < 20.0 AND (quantity > 10 OR is_featured = 1)):")
    pprint.pprint(filter_dict)
    
    # Example 6: Targeting Different Vector Databases
    print("\n6. Targeting Different Vector Databases")
    print("-" * 60)
    
    # Create filters for different vector database formats
    db_formats = ["generic", "chroma", "qdrant", "pinecone"]
    
    for db_format in db_formats:
        # Reset the filter builder
        numeric_filter.reset()
        
        # Create a numeric range filter
        numeric_filter.field("price").between(10.0, 50.0)
        
        # Get the filter dictionary for the specific database format
        filter_dict = numeric_filter.build(format=db_format)
        print(f"\nFilter for {db_format.capitalize()} format:")
        pprint.pprint(filter_dict)
    
    print("\n" + "="*80)
    print("End of Numeric Filter Builder Example")
    print("="*80 + "\n")


if __name__ == "__main__":
    main() 