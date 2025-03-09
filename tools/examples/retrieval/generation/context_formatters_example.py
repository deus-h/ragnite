#!/usr/bin/env python3
"""
Context Formatters Example

This script demonstrates the usage of various context formatters for transforming
retrieved documents into formatted contexts for language models.
"""

import sys
import os
import json
from typing import Dict, Any, List

# Add the parent directory to sys.path to import the tools
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from tools.src.retrieval import (
    BasicContextFormatter,
    MetadataEnrichedFormatter,
    SourceAttributionFormatter,
    HierarchicalContextFormatter,
    get_context_formatter
)


# Sample documents for demonstration
SAMPLE_DOCUMENTS = [
    {
        "content": "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically typed and garbage-collected.",
        "title": "Python Programming Language",
        "source": "Wikipedia",
        "url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
        "date": "2023-01-15",
        "category": "Programming",
        "subcategory": "Languages"
    },
    {
        "content": "JavaScript, often abbreviated as JS, is a programming language that is one of the core technologies of the World Wide Web, alongside HTML and CSS. Over 97% of websites use JavaScript on the client side for web page behavior.",
        "title": "JavaScript",
        "source": "Mozilla Developer Network",
        "url": "https://developer.mozilla.org/en-US/docs/Web/JavaScript",
        "date": "2023-02-20",
        "category": "Programming",
        "subcategory": "Web Development"
    },
    {
        "content": "Machine learning is a field of inquiry devoted to understanding and building methods that 'learn', that is, methods that leverage data to improve performance on some set of tasks.",
        "title": "Introduction to Machine Learning",
        "source": "AI Textbook",
        "url": "https://example.com/ai-textbook",
        "date": "2023-03-10",
        "category": "Artificial Intelligence",
        "subcategory": "Machine Learning"
    },
    {
        "content": "Deep learning is a subset of machine learning where artificial neural networks, algorithms inspired by the human brain, learn from large amounts of data.",
        "title": "Deep Learning",
        "source": "AI Research Journal",
        "url": "https://example.com/ai-journal",
        "date": "2023-04-05",
        "category": "Artificial Intelligence",
        "subcategory": "Deep Learning"
    },
    {
        "content": "SQL (Structured Query Language) is a domain-specific language used in programming and designed for managing data held in a relational database management system.",
        "title": "SQL Fundamentals",
        "source": "Database Handbook",
        "url": "https://example.com/db-handbook",
        "date": "2023-05-12",
        "category": "Programming",
        "subcategory": "Databases"
    }
]


def basic_formatter_example():
    """
    Example using the BasicContextFormatter.
    """
    print("\n" + "="*50)
    print("BASIC CONTEXT FORMATTER EXAMPLE")
    print("="*50)
    
    # Create a basic context formatter
    formatter = BasicContextFormatter(config={
        "separator": "\n\n---\n\n",
        "numbered": True,
        "max_length": 100,
        "header": "Here are some relevant documents:",
        "footer": "Please use the information above to answer the question."
    })
    
    # Format the documents
    formatted_context = formatter.format_context(SAMPLE_DOCUMENTS)
    
    print(formatted_context)


def metadata_enriched_formatter_example():
    """
    Example using the MetadataEnrichedFormatter.
    """
    print("\n" + "="*50)
    print("METADATA ENRICHED FORMATTER EXAMPLE")
    print("="*50)
    
    # Create a metadata enriched formatter
    formatter = MetadataEnrichedFormatter(config={
        "metadata_fields": ["title", "source", "date", "url"],
        "metadata_format": "**{field}**: {value}",
        "metadata_separator": " | ",
        "max_length": 100,
        "header": "I've found the following relevant information:",
        "footer": "Based on these sources, I can provide you with accurate information."
    })
    
    # Format the documents
    formatted_context = formatter.format_context(SAMPLE_DOCUMENTS[:3])
    
    print(formatted_context)


def source_attribution_formatter_example():
    """
    Example using the SourceAttributionFormatter.
    """
    print("\n" + "="*50)
    print("SOURCE ATTRIBUTION FORMATTER EXAMPLE")
    print("="*50)
    
    # Create a source attribution formatter
    formatter = SourceAttributionFormatter(config={
        "citation_format": "[{index}]",
        "reference_format": "[{index}] {source}",
        "reference_header": "\n\nReferences:",
        "source_fields": ["title", "source", "date"],
        "source_separator": ", ",
        "citation_placement": "suffix",
        "max_length": 100
    })
    
    # Format the documents
    formatted_context = formatter.format_context(SAMPLE_DOCUMENTS)
    
    print(formatted_context)
    
    # Example with different citation placement
    print("\n" + "-"*40)
    print("Different Citation Placement (prefix)")
    print("-"*40)
    
    formatter.set_citation_placement("prefix")
    formatted_context = formatter.format_context(SAMPLE_DOCUMENTS[:2])
    
    print(formatted_context)


def hierarchical_formatter_example():
    """
    Example using the HierarchicalContextFormatter.
    """
    print("\n" + "="*50)
    print("HIERARCHICAL CONTEXT FORMATTER EXAMPLE")
    print("="*50)
    
    # Create a hierarchical context formatter
    formatter = HierarchicalContextFormatter(config={
        "hierarchy_field": "category",
        "secondary_field": "subcategory",
        "group_format": "# {group}",
        "subgroup_format": "## {subgroup}",
        "include_item_headers": True,
        "item_header_format": "### {index}. {title}",
        "max_length": 100
    })
    
    # Format the documents
    formatted_context = formatter.format_context(SAMPLE_DOCUMENTS)
    
    print(formatted_context)
    
    # Single level hierarchy
    print("\n" + "-"*40)
    print("Single Level Hierarchy")
    print("-"*40)
    
    formatter.set_hierarchy_fields("category")
    formatted_context = formatter.format_context(SAMPLE_DOCUMENTS)
    
    print(formatted_context)


def factory_function_example():
    """
    Example using the get_context_formatter factory function.
    """
    print("\n" + "="*50)
    print("FACTORY FUNCTION EXAMPLE")
    print("="*50)
    
    # Create formatters using the factory function
    basic = get_context_formatter(
        formatter_type="basic",
        numbered=True,
        separator="\n\n",
        header="# Retrieved Documents"
    )
    
    metadata = get_context_formatter(
        formatter_type="metadata",
        metadata_fields=["title", "source"],
        separator="\n\n---\n\n"
    )
    
    attribution = get_context_formatter(
        formatter_type="source_attribution",
        citation_placement="both"
    )
    
    hierarchical = get_context_formatter(
        formatter_type="hierarchical",
        hierarchy_field="category"
    )
    
    # Format with each formatter
    print("\nBASIC FORMATTER:")
    print(basic.format_context(SAMPLE_DOCUMENTS[:2]))
    
    print("\nMETADATA FORMATTER:")
    print(metadata.format_context(SAMPLE_DOCUMENTS[:2]))
    
    print("\nSOURCE ATTRIBUTION FORMATTER:")
    print(attribution.format_context(SAMPLE_DOCUMENTS[:2]))
    
    print("\nHIERARCHICAL FORMATTER:")
    print(hierarchical.format_context(SAMPLE_DOCUMENTS[:2]))


def custom_configuration_example():
    """
    Example showing more custom configurations.
    """
    print("\n" + "="*50)
    print("CUSTOM CONFIGURATION EXAMPLE")
    print("="*50)
    
    # Custom configuration for MetadataEnrichedFormatter
    metadata_formatter = MetadataEnrichedFormatter(config={
        "metadata_fields": ["title", "source", "date"],
        "metadata_format": "• {field}: {value}",
        "metadata_separator": "\n",
        "separator": "\n\n====================\n\n",
        "max_length": 150
    })
    
    print("\nCUSTOM METADATA FORMATTER:")
    print(metadata_formatter.format_context(SAMPLE_DOCUMENTS[:2]))
    
    # Custom configuration for SourceAttributionFormatter
    attribution_formatter = SourceAttributionFormatter(config={
        "citation_format": "(ref: {index})",
        "reference_format": "Reference {index}: {source}",
        "reference_header": "\n\nSource Information:",
        "citation_placement": "suffix"
    })
    
    print("\nCUSTOM ATTRIBUTION FORMATTER:")
    print(attribution_formatter.format_context(SAMPLE_DOCUMENTS[:2]))
    
    # Custom configuration for HierarchicalContextFormatter
    hierarchical_formatter = HierarchicalContextFormatter(config={
        "hierarchy_field": "category",
        "group_format": "📚 {group} 📚",
        "include_item_headers": True,
        "item_header_format": "📄 {title}",
        "numbered": False
    })
    
    print("\nCUSTOM HIERARCHICAL FORMATTER:")
    print(hierarchical_formatter.format_context(SAMPLE_DOCUMENTS[:3]))


def override_parameters_example():
    """
    Example showing parameter overrides in format_context.
    """
    print("\n" + "="*50)
    print("PARAMETER OVERRIDE EXAMPLE")
    print("="*50)
    
    # Create a basic formatter
    formatter = BasicContextFormatter()
    
    # Format with default configuration
    print("\nDEFAULT CONFIGURATION:")
    print(formatter.format_context(SAMPLE_DOCUMENTS[:2]))
    
    # Format with overridden parameters
    print("\nOVERRIDDEN PARAMETERS:")
    print(formatter.format_context(
        SAMPLE_DOCUMENTS[:2],
        numbered=True,
        separator="\n\n===\n\n",
        header="CUSTOM HEADER",
        footer="CUSTOM FOOTER",
        max_length=50
    ))


def main():
    """
    Main function to run all examples.
    """
    basic_formatter_example()
    metadata_enriched_formatter_example()
    source_attribution_formatter_example()
    hierarchical_formatter_example()
    factory_function_example()
    custom_configuration_example()
    override_parameters_example()
    
    print("\nAll examples completed successfully!")


if __name__ == "__main__":
    main() 