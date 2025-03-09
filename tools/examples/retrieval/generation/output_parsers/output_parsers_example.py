#!/usr/bin/env python3
"""
Output Parsers Example

This script demonstrates the usage of various output parsers for extracting
structured data from language model outputs.
"""

import json
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Callable

# Import the output parsers
from tools.src.retrieval import (
    JSONOutputParser,
    XMLOutputParser,
    MarkdownOutputParser,
    StructuredOutputParser,
    get_output_parser,
    ParsingError
)

# Sample LLM outputs for demonstration
SAMPLE_OUTPUTS = {
    "json": """
I'll provide the information about the book in JSON format:

```json
{
  "title": "The Great Gatsby",
  "author": "F. Scott Fitzgerald",
  "publication_year": 1925,
  "genres": ["Novel", "Fiction", "Tragedy"],
  "summary": "A tale of wealth, love, and the American Dream in the Roaring Twenties."
}
```

The book is considered one of the great American novels.
    """,
    
    "xml": """
Here's the product information in XML format:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<product>
  <name>Wireless Bluetooth Headphones</name>
  <price>79.99</price>
  <category>Electronics</category>
  <features>
    <feature>Noise Cancellation</feature>
    <feature>40h Battery Life</feature>
    <feature>Fast Charging</feature>
  </features>
  <rating>4.5</rating>
</product>
```

These headphones are currently in stock and ship within 2 business days.
    """,
    
    "markdown": """
# Customer Feedback Report

## Positive Comments

- User interface is intuitive and easy to navigate
- Fast response times for customer service inquiries
- Mobile app works flawlessly across different devices

## Areas for Improvement

1. Expand payment options to include PayPal
2. Add dark mode to reduce eye strain
3. Include more detailed product descriptions

## Recommendations

> Consider implementing a loyalty program to reward repeat customers and encourage engagement.

```python
def calculate_loyalty_points(purchase_amount):
    return int(purchase_amount * 0.1)  # 10% of purchase as points
```

Thank you for reviewing this feedback report!
    """,
    
    "structured": """
Here's the analysis of the quarterly sales data:

Sales Summary:
- Total Revenue: $1,245,678
- Units Sold: 45,672
- Average Order Value: $273

Top Performing Products:
1. Smartphone X Pro - $325,000
2. Wireless Earbuds - $189,000
3. Smart Watch Series 5 - $156,000

Regional Performance:
North America: 42%
Europe: 31%
Asia: 18%
Other: 9%
    """
}

def run_json_parser_example():
    """Demonstrate using the JSONOutputParser."""
    print("\n" + "="*50)
    print("JSON Output Parser Example")
    print("="*50)
    
    # Create a JSONOutputParser with a schema
    json_parser = JSONOutputParser({
        "schema": {
            "title": str,
            "author": str,
            "publication_year": int,
            "genres": list,
            "summary": str
        }
    })
    
    # Get the format instructions
    print("Format Instructions:")
    print(json_parser.get_format_instructions())
    print()
    
    # Parse the sample JSON output
    try:
        start_time = time.time()
        result = json_parser.parse(SAMPLE_OUTPUTS["json"])
        parse_time = time.time() - start_time
        
        print(f"Parsed JSON (in {parse_time:.4f} seconds):")
        print(json.dumps(result, indent=2))
        print()
        
        # Access specific fields
        print(f"Book Title: {result['title']}")
        print(f"Author: {result['author']}")
        print(f"Genres: {', '.join(result['genres'])}")
    except ParsingError as e:
        print(f"Error parsing JSON: {e}")

def run_xml_parser_example():
    """Demonstrate using the XMLOutputParser."""
    print("\n" + "="*50)
    print("XML Output Parser Example")
    print("="*50)
    
    # Create an XMLOutputParser with a root tag and required elements
    xml_parser = XMLOutputParser({
        "root_tag": "product",
        "required_elements": ["name", "price", "category", "features"]
    })
    
    # Get the format instructions
    print("Format Instructions:")
    print(xml_parser.get_format_instructions())
    print()
    
    # Parse the sample XML output
    try:
        start_time = time.time()
        result = xml_parser.parse(SAMPLE_OUTPUTS["xml"])
        parse_time = time.time() - start_time
        
        print(f"Parsed XML (in {parse_time:.4f} seconds):")
        print(f"Root tag: {result.tag}")
        
        # Print the XML structure
        for child in result:
            if child.tag == "features":
                print(f"{child.tag}:")
                for feature in child:
                    print(f"  - {feature.text}")
            else:
                print(f"{child.tag}: {child.text}")
        
        # Use ElementTree's functionality
        print("\nProduct Details:")
        print(f"Name: {result.find('name').text}")
        print(f"Price: ${result.find('price').text}")
        print(f"Rating: {result.find('rating').text} stars")
    except ParsingError as e:
        print(f"Error parsing XML: {e}")

def run_markdown_parser_example():
    """Demonstrate using the MarkdownOutputParser."""
    print("\n" + "="*50)
    print("Markdown Output Parser Example")
    print("="*50)
    
    # Create a MarkdownOutputParser
    markdown_parser = MarkdownOutputParser({
        "required_sections": ["Positive Comments", "Areas for Improvement"]
    })
    
    # Get the format instructions
    print("Format Instructions:")
    print(markdown_parser.get_format_instructions())
    print()
    
    # Parse the sample Markdown output
    try:
        start_time = time.time()
        result = markdown_parser.parse(SAMPLE_OUTPUTS["markdown"])
        parse_time = time.time() - start_time
        
        print(f"Parsed Markdown (in {parse_time:.4f} seconds):")
        
        # Print headers
        print("\nHeaders:")
        for header in result["headers"]:
            print(f"Level {header['level']} - {header['title']}")
        
        # Print lists
        print("\nLists:")
        for list_item in result["lists"]:
            print(f"Type: {list_item['type']}")
            for item in list_item["items"]:
                print(f"  - {item}")
        
        # Print code blocks
        print("\nCode Blocks:")
        for code_block in result["code_blocks"]:
            print(f"Language: {code_block['language']}")
            print(f"Code:\n{code_block['code']}")
        
        # Print blockquotes
        print("\nBlockquotes:")
        for quote in result["blockquotes"]:
            print(f"  \"{quote}\"")
    except ParsingError as e:
        print(f"Error parsing Markdown: {e}")

def run_structured_parser_example():
    """Demonstrate using the StructuredOutputParser."""
    print("\n" + "="*50)
    print("Structured Output Parser Example")
    print("="*50)
    
    # Define a schema and extraction patterns
    schema = {
        "total_revenue": float,
        "units_sold": int,
        "average_order_value": float,
        "top_products": list,
        "regional_performance": dict
    }
    
    # Define extraction patterns
    extraction_patterns = {
        "total_revenue": r"Total Revenue: \$([0-9,]+(?:\.[0-9]+)?)",
        "units_sold": r"Units Sold: ([0-9,]+)",
        "average_order_value": r"Average Order Value: \$([0-9,]+(?:\.[0-9]+)?)",
        "top_products": r"Top Performing Products:(.*?)(?:\n\n|\Z)",
        "regional_performance": r"Regional Performance:(.*?)(?:\n\n|\Z)"
    }
    
    # Define transformers to clean up the extracted data
    def clean_number(value: str) -> float:
        """Remove commas from number strings."""
        if isinstance(value, str):
            return float(value.replace(",", ""))
        return float(value)
    
    def parse_products(text: str) -> List[Dict[str, Any]]:
        """Parse product list from text."""
        products = []
        lines = text.strip().split("\n")
        for line in lines:
            match = re.search(r'(\d+)\.\s+(.*?)\s+-\s+\$([0-9,]+)', line)
            if match:
                products.append({
                    "rank": int(match.group(1)),
                    "name": match.group(2).strip(),
                    "revenue": clean_number(match.group(3))
                })
        return products
    
    def parse_regions(text: str) -> Dict[str, float]:
        """Parse regional performance from text."""
        regions = {}
        lines = text.strip().split("\n")
        for line in lines:
            parts = line.split(":")
            if len(parts) == 2:
                region = parts[0].strip()
                percentage = float(parts[1].strip().replace("%", ""))
                regions[region] = percentage
        return regions
    
    # Create transformers dictionary
    transformers = {
        "total_revenue": clean_number,
        "units_sold": lambda x: int(x.replace(",", "")),
        "average_order_value": clean_number,
        "top_products": parse_products,
        "regional_performance": parse_regions
    }
    
    # Create a StructuredOutputParser
    structured_parser = StructuredOutputParser({
        "schema": schema,
        "extraction_patterns": extraction_patterns,
        "transformers": transformers
    })
    
    # Get the format instructions
    print("Format Instructions:")
    print(structured_parser.get_format_instructions())
    print()
    
    # Parse the sample structured output
    try:
        start_time = time.time()
        result = structured_parser.parse(SAMPLE_OUTPUTS["structured"])
        parse_time = time.time() - start_time
        
        print(f"Parsed Structured Data (in {parse_time:.4f} seconds):")
        print(json.dumps(result, indent=2, default=str))
        
        # Use the parsed data
        print("\nSales Summary:")
        print(f"Revenue: ${result['total_revenue']:,.2f}")
        print(f"Units: {result['units_sold']:,}")
        print(f"AOV: ${result['average_order_value']:.2f}")
        
        print("\nTop 3 Products:")
        for product in result["top_products"]:
            print(f"{product['rank']}. {product['name']} - ${product['revenue']:,.2f}")
        
        print("\nRegional Breakdown:")
        for region, percentage in result["regional_performance"].items():
            print(f"{region}: {percentage}%")
    except ParsingError as e:
        print(f"Error parsing structured data: {e}")

def run_factory_function_example():
    """Demonstrate using the get_output_parser factory function."""
    print("\n" + "="*50)
    print("Factory Function Example")
    print("="*50)
    
    parsers = []
    
    # Create a JSON parser using the factory
    json_parser = get_output_parser(
        parser_type="json",
        schema={
            "title": str,
            "author": str,
            "publication_year": int
        }
    )
    parsers.append(("JSON", json_parser, SAMPLE_OUTPUTS["json"]))
    
    # Create an XML parser using the factory
    xml_parser = get_output_parser(
        parser_type="xml",
        root_tag="product",
        required_elements=["name", "price"]
    )
    parsers.append(("XML", xml_parser, SAMPLE_OUTPUTS["xml"]))
    
    # Create a Markdown parser using the factory
    markdown_parser = get_output_parser(
        parser_type="markdown",
        required_sections=["Positive Comments"]
    )
    parsers.append(("Markdown", markdown_parser, SAMPLE_OUTPUTS["markdown"]))
    
    # Test each parser
    for name, parser, sample in parsers:
        print(f"\n{name} Parser (via factory):")
        try:
            result = parser.parse(sample)
            print(f"  Success! Parsed {type(result).__name__}")
        except ParsingError as e:
            print(f"  Error: {e}")

def custom_validation_example():
    """Demonstrate custom validation with output parsers."""
    print("\n" + "="*50)
    print("Custom Validation Example")
    print("="*50)
    
    # Define a schema with validators
    schema = {
        "user_id": str,
        "age": int,
        "email": str,
        "interests": list
    }
    
    # Define custom validators
    validators = {
        "user_id": lambda x: len(x) >= 6,  # User ID must be at least 6 chars
        "age": lambda x: 18 <= x <= 120,   # Age must be between 18 and 120
        "email": lambda x: "@" in x,       # Simple email validation
        "interests": lambda x: len(x) > 0  # Must have at least one interest
    }
    
    # Create the parser with validators
    parser = JSONOutputParser({
        "schema": schema,
        "validators": validators,
        "strict": True
    })
    
    # Test with valid data
    valid_json = """
    {
        "user_id": "user123",
        "age": 35,
        "email": "user@example.com",
        "interests": ["music", "technology", "books"]
    }
    """
    
    # Test with invalid data
    invalid_json = """
    {
        "user_id": "u12",
        "age": 12,
        "email": "notanemail",
        "interests": []
    }
    """
    
    # Parse valid data
    print("Parsing valid data:")
    try:
        result = parser.parse(valid_json)
        print("  Success! Validation passed.")
        print("  " + json.dumps(result))
    except ParsingError as e:
        print(f"  Error: {e}")
    
    # Parse invalid data
    print("\nParsing invalid data:")
    try:
        result = parser.parse(invalid_json)
        print("  Success (unexpected)! Validation passed.")
    except ParsingError as e:
        print(f"  Error (expected): {str(e)}")
        if hasattr(e, "underlying_error") and e.underlying_error:
            print(f"  Underlying error: {e.underlying_error}")

def main():
    """Run all output parser examples."""
    print("Output Parsers - Examples")
    print("\nThis script demonstrates the various output parsers for extracting structured data.")
    
    # Run individual examples
    run_json_parser_example()
    run_xml_parser_example()
    run_markdown_parser_example()
    run_structured_parser_example()
    run_factory_function_example()
    custom_validation_example()
    
    print("\n" + "="*50)
    print("All examples completed!")
    print("="*50)

if __name__ == "__main__":
    import re  # For the structured parser example
    main() 