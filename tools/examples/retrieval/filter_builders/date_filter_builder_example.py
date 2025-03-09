#!/usr/bin/env python3
"""
Date Filter Builder Example

This script demonstrates how to use the DateFilterBuilder to create various types of date filters
for vector databases. It shows how to:

1. Create basic date equality filters
2. Create date range filters
3. Create before and after date filters
4. Use exact datetime objects
5. Create relative date range filters
6. Create filters for specific time periods
7. Create filters with multiple dates
8. Combine filters with logical operators
9. Generate filters for different vector database formats
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
    Main function demonstrating the DateFilterBuilder with multiple examples.
    """
    print("\n" + "="*80)
    print("Date Filter Builder Example")
    print("="*80)
    
    # Create a DateFilterBuilder
    date_filter = get_filter_builder(builder_type="date")
    
    # Example 1: Basic Date Equality Filter
    print("\n1. Basic Date Equality Filter")
    print("-" * 60)
    
    # Create a filter for a specific publication date
    date_filter.field("publication_date").equals("2023-06-15")
    
    # Get the filter dictionary
    filter_dict = date_filter.build()
    print("Filter for documents published on 2023-06-15:")
    pprint.pprint(filter_dict)
    
    # Example 2: Date Range Filter
    print("\n2. Date Range Filter")
    print("-" * 60)
    
    # Reset the filter builder
    date_filter.reset()
    
    # Create a filter for a date range
    date_filter.field("publication_date").between("2023-01-01", "2023-12-31")
    
    # Get the filter dictionary
    filter_dict = date_filter.build()
    print("Filter for documents published in 2023:")
    pprint.pprint(filter_dict)
    
    # Example 3: Before and After Date Filters
    print("\n3. Before and After Date Filters")
    print("-" * 60)
    
    # Reset the filter builder
    date_filter.reset()
    
    # Create a filter for dates before a specific date
    date_filter.field("publication_date").before("2023-06-15")
    
    # Get the filter dictionary
    filter_dict = date_filter.build()
    print("Filter for documents published before 2023-06-15:")
    pprint.pprint(filter_dict)
    
    # Reset the filter builder
    date_filter.reset()
    
    # Create a filter for dates after a specific date
    date_filter.field("publication_date").after("2023-06-15")
    
    # Get the filter dictionary
    filter_dict = date_filter.build()
    print("Filter for documents published after 2023-06-15:")
    pprint.pprint(filter_dict)
    
    # Example 4: Using Exact Date/Time Objects
    print("\n4. Using Exact Date/Time Objects")
    print("-" * 60)
    
    # Reset the filter builder
    date_filter.reset()
    
    # Create a filter using datetime objects
    start_date = datetime.datetime(2023, 1, 1)
    end_date = datetime.datetime(2023, 12, 31, 23, 59, 59)
    
    date_filter.field("publication_date").between(start_date, end_date)
    
    # Get the filter dictionary
    filter_dict = date_filter.build()
    print("Filter using datetime objects:")
    pprint.pprint(filter_dict)
    
    # Example 5: Relative Date Ranges
    print("\n5. Relative Date Ranges")
    print("-" * 60)
    
    # Reset the filter builder
    date_filter.reset()
    
    # Create a filter for the last 30 days
    date_filter.field("publication_date").in_last_days(30)
    
    # Get the filter dictionary
    filter_dict = date_filter.build()
    print("Filter for documents published in the last 30 days:")
    pprint.pprint(filter_dict)
    
    # Reset the filter builder
    date_filter.reset()
    
    # Create a filter for the next 2 weeks
    date_filter.field("publication_date").in_next_days(14)
    
    # Get the filter dictionary
    filter_dict = date_filter.build()
    print("Filter for documents with publication dates in the next 2 weeks:")
    pprint.pprint(filter_dict)
    
    # Example 6: Specific Time Periods
    print("\n6. Specific Time Periods")
    print("-" * 60)
    
    # Reset the filter builder
    date_filter.reset()
    
    # Create a filter for today
    date_filter.field("publication_date").today()
    
    # Get the filter dictionary
    filter_dict = date_filter.build()
    print("Filter for documents published today:")
    pprint.pprint(filter_dict)
    
    # Reset the filter builder
    date_filter.reset()
    
    # Create a filter for this week
    date_filter.field("publication_date").this_week()
    
    # Get the filter dictionary
    filter_dict = date_filter.build()
    print("Filter for documents published this week:")
    pprint.pprint(filter_dict)
    
    # Reset the filter builder
    date_filter.reset()
    
    # Create a filter for this month
    date_filter.field("publication_date").this_month()
    
    # Get the filter dictionary
    filter_dict = date_filter.build()
    print("Filter for documents published this month:")
    pprint.pprint(filter_dict)
    
    # Reset the filter builder
    date_filter.reset()
    
    # Create a filter for this year
    date_filter.field("publication_date").this_year()
    
    # Get the filter dictionary
    filter_dict = date_filter.build()
    print("Filter for documents published this year:")
    pprint.pprint(filter_dict)
    
    # Example 7: Working with Specific Dates
    print("\n7. Working with Specific Dates")
    print("-" * 60)
    
    # Reset the filter builder
    date_filter.reset()
    
    # Create a filter for a specific date
    date_filter.field("publication_date").on_date("2023-06-15")
    
    # Get the filter dictionary
    filter_dict = date_filter.build()
    print("Filter for documents published on 2023-06-15:")
    pprint.pprint(filter_dict)
    
    # Reset the filter builder
    date_filter.reset()
    
    # Create a filter for a list of dates
    date_filter.field("publication_date").in_dates(["2023-06-15", "2023-07-15", "2023-08-15"])
    
    # Get the filter dictionary
    filter_dict = date_filter.build()
    print("Filter for documents published on specific dates:")
    pprint.pprint(filter_dict)
    
    # Example 8: Complex Date Filters with Logical Operators
    print("\n8. Complex Date Filters with Logical Operators")
    print("-" * 60)
    
    # Reset the filter builder
    date_filter.reset()
    
    # Create a complex filter with AND, OR, and NOT operators
    date_filter.field("created_date").after("2023-01-01").and_().field("updated_date").before("2023-12-31")
    
    # Get the filter dictionary
    filter_dict = date_filter.build()
    print("Complex filter with AND operator:")
    pprint.pprint(filter_dict)
    
    # Reset the filter builder
    date_filter.reset()
    
    # Create a filter with OR operator
    date_filter.field("publication_date").equals("2023-06-15").or_().field("publication_date").equals("2023-07-15")
    
    # Get the filter dictionary
    filter_dict = date_filter.build()
    print("Complex filter with OR operator:")
    pprint.pprint(filter_dict)
    
    # Reset the filter builder
    date_filter.reset()
    
    # Create a filter with NOT operator
    date_filter.not_().field("publication_date").equals("2023-06-15")
    
    # Get the filter dictionary
    filter_dict = date_filter.build()
    print("Complex filter with NOT operator:")
    pprint.pprint(filter_dict)
    
    # Example 9: Targeting Different Vector Databases
    print("\n9. Targeting Different Vector Databases")
    print("-" * 60)
    
    # Create filters for different vector database formats
    db_formats = ["generic", "chroma", "qdrant", "pinecone"]
    
    for db_format in db_formats:
        # Reset the filter builder
        date_filter.reset()
        
        # Create a date range filter
        date_filter.field("publication_date").between("2023-01-01", "2023-12-31")
        
        # Get the filter dictionary for the specific database format
        filter_dict = date_filter.build(format=db_format)
        print(f"\nFilter for {db_format.capitalize()} format:")
        pprint.pprint(filter_dict)
    
    print("\n" + "="*80)
    print("End of Date Filter Builder Example")
    print("="*80 + "\n")


if __name__ == "__main__":
    main() 