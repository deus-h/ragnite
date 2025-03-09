#!/usr/bin/env python3
"""
Composite Filter Builder Example

This script demonstrates how to use the CompositeFilterBuilder to combine different types of filters
(metadata, date, numeric) for vector databases. It shows how to:

1. Create basic composite filters
2. Combine different filter types with logical operators
3. Create nested composite filters
4. Generate filters for different vector database formats
"""

import sys
import os
import json
import datetime
import pprint
from typing import Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from tools.src.retrieval import get_filter_builder

def main():
    """
    Main function demonstrating the CompositeFilterBuilder with multiple examples.
    """
    print("\n" + "="*80)
    print("Composite Filter Builder Example")
    print("="*80)
    
    # Example 1: Basic Composite Filter
    print("\n1. Basic Composite Filter")
    print("-" * 60)
    
    # Create a composite filter builder
    composite_filter = get_filter_builder(builder_type="composite")
    
    # Create individual filter builders
    metadata_filter = get_filter_builder(builder_type="metadata")
    date_filter = get_filter_builder(builder_type="date")
    numeric_filter = get_filter_builder(builder_type="numeric")
    
    # Configure the metadata filter
    metadata_filter.field("category").equals("technology")
    
    # Configure the date filter
    date_filter.field("publication_date").after("2023-01-01")
    
    # Configure the numeric filter
    numeric_filter.field("rating").greater_than(4.0)
    
    # Add all filters to the composite filter with AND logic
    composite_filter.and_filters([
        metadata_filter.build(),
        date_filter.build(),
        numeric_filter.build()
    ])
    
    # Get the composite filter dictionary
    filter_dict = composite_filter.build()
    print("Composite filter combining metadata, date, and numeric filters with AND logic:")
    pprint.pprint(filter_dict)
    
    # Example 2: Combining Different Filter Types with OR Logic
    print("\n2. Combining Different Filter Types with OR Logic")
    print("-" * 60)
    
    # Reset the composite filter
    composite_filter.reset()
    
    # Create new individual filter builders
    metadata_filter1 = get_filter_builder(builder_type="metadata")
    metadata_filter2 = get_filter_builder(builder_type="metadata")
    date_filter = get_filter_builder(builder_type="date")
    
    # Configure the first metadata filter
    metadata_filter1.field("category").equals("technology")
    
    # Configure the second metadata filter
    metadata_filter2.field("category").equals("programming")
    
    # Configure the date filter
    date_filter.field("publication_date").in_last_days(30)
    
    # Add filters to the composite filter with OR logic for categories and AND logic for date
    composite_filter.and_filters([
        get_filter_builder(builder_type="composite").or_filters([
            metadata_filter1.build(),
            metadata_filter2.build()
        ]).build(),
        date_filter.build()
    ])
    
    # Get the composite filter dictionary
    filter_dict = composite_filter.build()
    print("Composite filter with OR logic for categories and AND logic for date:")
    print("(category = 'technology' OR category = 'programming') AND (publication_date in last 30 days)")
    pprint.pprint(filter_dict)
    
    # Example 3: Nested Composite Filters
    print("\n3. Nested Composite Filters")
    print("-" * 60)
    
    # Reset the composite filter
    composite_filter.reset()
    
    # Create individual filter builders
    category_filter = get_filter_builder(builder_type="metadata")
    author_filter1 = get_filter_builder(builder_type="metadata")
    author_filter2 = get_filter_builder(builder_type="metadata")
    date_filter = get_filter_builder(builder_type="date")
    rating_filter = get_filter_builder(builder_type="numeric")
    
    # Configure the filters
    category_filter.field("category").equals("technology")
    author_filter1.field("author").equals("John Doe")
    author_filter2.field("author").equals("Jane Smith")
    date_filter.field("publication_date").this_year()
    rating_filter.field("rating").greater_than_or_equal(4.5)
    
    # Create a nested composite filter structure:
    # category = 'technology' AND 
    # (author = 'John Doe' OR author = 'Jane Smith') AND
    # publication_date in this year AND
    # rating >= 4.5
    
    # First, combine the author filters with OR logic
    author_composite = get_filter_builder(builder_type="composite")
    author_composite.or_filters([
        author_filter1.build(),
        author_filter2.build()
    ])
    
    # Then, combine all filters with AND logic
    composite_filter.and_filters([
        category_filter.build(),
        author_composite.build(),
        date_filter.build(),
        rating_filter.build()
    ])
    
    # Get the composite filter dictionary
    filter_dict = composite_filter.build()
    print("Nested composite filter:")
    print("category = 'technology' AND")
    print("(author = 'John Doe' OR author = 'Jane Smith') AND")
    print("publication_date in this year AND")
    print("rating >= 4.5")
    pprint.pprint(filter_dict)
    
    # Example 4: Complex Filter with Multiple Nesting Levels
    print("\n4. Complex Filter with Multiple Nesting Levels")
    print("-" * 60)
    
    # Reset the composite filter
    composite_filter.reset()
    
    # Create a complex filter structure:
    # (category = 'technology' OR category = 'programming') AND
    # (
    #   (publication_date >= '2023-01-01' AND publication_date <= '2023-12-31') OR
    #   (author = 'John Doe' AND rating >= 4.8)
    # )
    
    # Create individual filter builders
    tech_category = get_filter_builder(builder_type="metadata")
    tech_category.field("category").equals("technology")
    
    prog_category = get_filter_builder(builder_type="metadata")
    prog_category.field("category").equals("programming")
    
    date_range = get_filter_builder(builder_type="date")
    date_range.field("publication_date").between("2023-01-01", "2023-12-31")
    
    author_filter = get_filter_builder(builder_type="metadata")
    author_filter.field("author").equals("John Doe")
    
    rating_filter = get_filter_builder(builder_type="numeric")
    rating_filter.field("rating").greater_than_or_equal(4.8)
    
    # Combine category filters with OR
    category_composite = get_filter_builder(builder_type="composite")
    category_composite.or_filters([
        tech_category.build(),
        prog_category.build()
    ])
    
    # Combine author and rating filters with AND
    author_rating_composite = get_filter_builder(builder_type="composite")
    author_rating_composite.and_filters([
        author_filter.build(),
        rating_filter.build()
    ])
    
    # Combine date range and author-rating composite with OR
    date_author_composite = get_filter_builder(builder_type="composite")
    date_author_composite.or_filters([
        date_range.build(),
        author_rating_composite.build()
    ])
    
    # Finally, combine category composite and date-author composite with AND
    composite_filter.and_filters([
        category_composite.build(),
        date_author_composite.build()
    ])
    
    # Get the composite filter dictionary
    filter_dict = composite_filter.build()
    print("Complex filter with multiple nesting levels:")
    print("(category = 'technology' OR category = 'programming') AND")
    print("(")
    print("  (publication_date >= '2023-01-01' AND publication_date <= '2023-12-31') OR")
    print("  (author = 'John Doe' AND rating >= 4.8)")
    print(")")
    pprint.pprint(filter_dict)
    
    # Example 5: Targeting Different Vector Databases
    print("\n5. Targeting Different Vector Databases")
    print("-" * 60)
    
    # Create a simple composite filter for demonstration
    metadata_filter = get_filter_builder(builder_type="metadata")
    metadata_filter.field("category").equals("technology")
    
    date_filter = get_filter_builder(builder_type="date")
    date_filter.field("publication_date").this_year()
    
    # Create filters for different vector database formats
    db_formats = ["generic", "chroma", "qdrant", "pinecone"]
    
    for db_format in db_formats:
        # Create a composite filter for the specific database format
        composite_filter = get_filter_builder(builder_type="composite")
        
        # Add the filters with AND logic
        composite_filter.and_filters([
            metadata_filter.build(format=db_format),
            date_filter.build(format=db_format)
        ])
        
        # Get the filter dictionary for the specific database format
        filter_dict = composite_filter.build(format=db_format)
        print(f"\nFilter for {db_format.capitalize()} format:")
        pprint.pprint(filter_dict)
    
    print("\n" + "="*80)
    print("End of Composite Filter Builder Example")
    print("="*80 + "\n")


if __name__ == "__main__":
    main() 