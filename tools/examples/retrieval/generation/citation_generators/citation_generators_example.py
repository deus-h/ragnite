#!/usr/bin/env python3
"""
Citation Generators Example

This script demonstrates how to use the various citation generators
to create properly formatted citations for different types of sources.
"""

import sys
import time
from typing import Dict, List, Any

sys.path.append("../../../..")  # Add the repository root to the path

from tools.src.retrieval.generation.citation_generators import (
    BaseCitationGenerator,
    AcademicCitationGenerator,
    LegalCitationGenerator,
    WebCitationGenerator,
    CustomCitationGenerator,
    get_citation_generator
)

# Sample sources for citation
ACADEMIC_SOURCES = [
    {
        "type": "article",
        "title": "Large Language Models as Retrieval Augmentation Generators",
        "authors": [
            {"first_name": "Jane", "last_name": "Smith"},
            {"first_name": "John", "last_name": "Doe"},
            {"first_name": "Alice", "middle_name": "B", "last_name": "Johnson"}
        ],
        "journal": "Journal of Artificial Intelligence Research",
        "journal_abbrev": "JAIR",
        "volume": "45",
        "issue": "2",
        "pages": "123-145",
        "year": "2023",
        "doi": "10.1234/jair.2023.45.2.123"
    },
    {
        "type": "book",
        "title": "Advanced Natural Language Processing: Theory and Practice",
        "authors": [
            {"first_name": "Robert", "last_name": "Williams"},
            {"first_name": "Sarah", "last_name": "Chen"}
        ],
        "publisher": "Academic Press",
        "location": "New York",
        "edition": "2nd",
        "year": "2022"
    },
    {
        "type": "conference",
        "title": "A Novel Approach to Knowledge Retrieval in Large Language Models",
        "authors": [
            {"first_name": "Michael", "last_name": "Brown"}
        ],
        "conference": "Proceedings of the 35th Conference on Neural Information Processing Systems",
        "location": "Virtual",
        "year": "2021"
    }
]

LEGAL_SOURCES = [
    {
        "type": "case",
        "name": "Smith v. Jones",
        "volume": "567",
        "reporter": "U.S.",
        "page": "123",
        "court": "Supreme Court of the United States",
        "year": "2020",
        "pincite": "126"
    },
    {
        "type": "statute",
        "title": "17",
        "code": "U.S.C.",
        "section": "107",
        "year": "2021"
    },
    {
        "type": "article",
        "title": "The Evolution of Fair Use in Copyright Law",
        "authors": [
            {"first_name": "Elizabeth", "last_name": "Rodriguez"}
        ],
        "journal": "Harvard Law Review",
        "volume": "134",
        "page": "1723",
        "year": "2021"
    }
]

WEB_SOURCES = [
    {
        "type": "webpage",
        "title": "Retrieval Augmented Generation: A Primer",
        "authors": [
            {"first_name": "David", "last_name": "Lee"}
        ],
        "site_name": "AI Research Blog",
        "date": "2023-05-15",
        "url": "https://example.com/rag-primer",
        "accessed": "2023-10-01"
    },
    {
        "type": "article",
        "title": "The Future of Search Engines with RAG Technology",
        "site_name": "Tech Insights",
        "date": "2023-08-22",
        "url": "https://example.com/search-engines-rag",
        "accessed": "2023-10-05"
    },
    {
        "type": "blog",
        "title": "Implementing RAG Systems: Lessons Learned",
        "authors": [
            {"first_name": "Maria", "last_name": "Garcia"}
        ],
        "site_name": "Engineering at Scale",
        "date": "2023-09-10",
        "url": "https://example.com/engineering/rag-lessons",
        "accessed": "2023-10-10"
    },
    {
        "type": "social",
        "title": "Just published our new paper on RAG systems. Check it out!",
        "authors": [
            {"first_name": "Andrew", "last_name": "Taylor"}
        ],
        "platform": "Twitter",
        "username": "andrewtaylor",
        "date": "2023-10-01",
        "url": "https://twitter.com/andrewtaylor/status/1234567890",
        "accessed": "2023-10-02"
    }
]

CUSTOM_TEMPLATES = {
    "simple": "{authors} ({year}). {title}.",
    "detailed": "{authors} ({year}). {title}. {source}. Retrieved from {url}.",
    "inline": "({authors}, {year}, \"{title}\")",
    "numbered": "[{index}] {authors} ({year}). {title}. {source}.",
    "annotated": "{authors} ({year}). {title}. {source}. {url:?Available at: {url}.} {notes:?Notes: {notes}}"
}

def print_separator():
    """Print a separator line for better readability."""
    print("\n" + "=" * 80 + "\n")

def print_citations(title: str, citations: List[str]):
    """Print a list of citations with a title."""
    print(f"\n{title}:")
    print("-" * len(title))
    for i, citation in enumerate(citations, 1):
        print(f"{i}. {citation}")

def run_academic_citation_example():
    """Run examples using the AcademicCitationGenerator."""
    print_separator()
    print("ACADEMIC CITATION GENERATOR EXAMPLES")
    
    # Create an academic citation generator
    academic_generator = AcademicCitationGenerator({
        'include_doi': True,
        'include_url': True,
        'abbreviate_journal': True
    })
    
    # Get supported styles
    styles = academic_generator.get_supported_styles()
    print(f"Supported styles: {', '.join(styles)}")
    
    # Generate citations for academic sources in different styles
    for style in styles:
        start_time = time.time()
        citations = academic_generator.generate_citations(ACADEMIC_SOURCES, style)
        end_time = time.time()
        
        print_citations(f"{style} Style Citations", citations)
        print(f"Generation time: {(end_time - start_time) * 1000:.2f} ms")
    
    # Example of missing fields
    incomplete_source = {
        "type": "article",
        "title": "Incomplete Source Example",
        # Missing authors and journal
        "year": "2023"
    }
    
    print("\nHandling Incomplete Source:")
    incomplete_citation = academic_generator.generate_citation(incomplete_source, "APA")
    print(incomplete_citation)
    
    # Get required fields
    required_fields = academic_generator.get_required_fields("APA")
    print(f"Required fields for APA: {', '.join(required_fields)}")

def run_legal_citation_example():
    """Run examples using the LegalCitationGenerator."""
    print_separator()
    print("LEGAL CITATION GENERATOR EXAMPLES")
    
    # Create a legal citation generator
    legal_generator = LegalCitationGenerator({
        'include_url': True,
        'jurisdiction': 'US',
        'court_abbreviations': {
            'Supreme Court of the United States': 'U.S.',
            'United States Court of Appeals': 'F.',
            'District Court': 'F. Supp.',
            'California Supreme Court': 'Cal.',
            'New York Court of Appeals': 'N.Y.'
        }
    })
    
    # Get supported styles
    styles = legal_generator.get_supported_styles()
    print(f"Supported styles: {', '.join(styles)}")
    
    # Generate citations for legal sources in different styles
    for style in styles:
        start_time = time.time()
        citations = legal_generator.generate_citations(LEGAL_SOURCES, style)
        end_time = time.time()
        
        print_citations(f"{style} Style Citations", citations)
        print(f"Generation time: {(end_time - start_time) * 1000:.2f} ms")
    
    # Example with parties instead of a case name
    case_with_parties = {
        "type": "case",
        "parties": {
            "plaintiff": "Brown",
            "defendant": "Board of Education"
        },
        "volume": "347",
        "reporter": "U.S.",
        "page": "483",
        "court": "Supreme Court of the United States",
        "year": "1954"
    }
    
    print("\nCase with Parties:")
    party_citation = legal_generator.generate_citation(case_with_parties, "BLUEBOOK")
    print(party_citation)

def run_web_citation_example():
    """Run examples using the WebCitationGenerator."""
    print_separator()
    print("WEB CITATION GENERATOR EXAMPLES")
    
    # Create a web citation generator
    web_generator = WebCitationGenerator({
        'include_access_date': True,
        'access_date_format': '%B %d, %Y',
        'include_url': True
    })
    
    # Get supported styles
    styles = web_generator.get_supported_styles()
    print(f"Supported styles: {', '.join(styles)}")
    
    # Generate citations for web sources in different styles
    for style in styles:
        start_time = time.time()
        citations = web_generator.generate_citations(WEB_SOURCES, style)
        end_time = time.time()
        
        print_citations(f"{style} Style Citations", citations)
        print(f"Generation time: {(end_time - start_time) * 1000:.2f} ms")
    
    # Example with group author
    group_author_source = {
        "type": "webpage",
        "title": "Guidelines for Artificial Intelligence Ethics",
        "group_author": "Institute for Ethical AI",
        "site_name": "Ethical AI Organization",
        "date": "2023-01-15",
        "url": "https://example.com/ai-ethics-guidelines",
        "accessed": "2023-10-15"
    }
    
    print("\nSource with Group Author:")
    group_citation = web_generator.generate_citation(group_author_source, "APA")
    print(group_citation)

def run_custom_citation_example():
    """Run examples using the CustomCitationGenerator."""
    print_separator()
    print("CUSTOM CITATION GENERATOR EXAMPLES")
    
    # Create a custom citation generator with templates
    custom_generator = CustomCitationGenerator({
        'templates': CUSTOM_TEMPLATES,
        'date_format': '%B %d, %Y'
    })
    
    # Get supported styles (template names)
    styles = custom_generator.get_supported_styles()
    print(f"Available templates: {', '.join(styles)}")
    
    # Prepare sources for custom citation
    custom_sources = [
        {
            "authors": "Smith, J.",
            "year": "2023",
            "title": "The Future of RAG Systems",
            "source": "AI Journal",
            "url": "https://example.com/future-rag",
            "notes": "Seminal paper on RAG technology"
        },
        {
            "authors": "Johnson, A. & Williams, B.",
            "year": "2022",
            "title": "Knowledge Retrieval Mechanisms in LLMs",
            "source": "Proceedings of AI Conference",
            "url": "https://example.com/knowledge-retrieval",
            "index": "42"
        }
    ]
    
    # Generate citations using different templates
    for template_name in styles:
        start_time = time.time()
        citations = []
        for source in custom_sources:
            citation = custom_generator.generate_citation(source, template_name)
            citations.append(citation)
        end_time = time.time()
        
        print_citations(f"Template: {template_name}", citations)
        print(f"Generation time: {(end_time - start_time) * 1000:.2f} ms")
    
    # Demonstrate adding a new template
    print("\nAdding a new template:")
    custom_generator.add_template("footnote", "{index}. {authors}, \"{title},\" {source} ({year}).")
    
    footnote_citations = []
    for i, source in enumerate(custom_sources, 1):
        source_with_index = {**source, "index": i}
        citation = custom_generator.generate_citation(source_with_index, "footnote")
        footnote_citations.append(citation)
    
    print_citations("Footnote Template", footnote_citations)
    
    # Demonstrate template export/import
    print("\nExporting templates to JSON:")
    json_str = custom_generator.to_json()
    print(json_str[:100] + "..." if len(json_str) > 100 else json_str)
    
    # Create a new generator and import the templates
    new_generator = CustomCitationGenerator()
    new_generator.from_json(json_str)
    print(f"Imported templates: {', '.join(new_generator.get_supported_styles())}")

def run_factory_function_example():
    """Run examples using the factory function."""
    print_separator()
    print("FACTORY FUNCTION EXAMPLES")
    
    try:
        # Create different types of citation generators using the factory function
        generators = {
            "academic": get_citation_generator("academic", {"default_style": "APA"}),
            "legal": get_citation_generator("legal", {"default_style": "BLUEBOOK"}),
            "web": get_citation_generator("web", {"default_style": "MLA"}),
            "custom": get_citation_generator("custom", {"templates": CUSTOM_TEMPLATES})
        }
        
        # Generate a citation with each generator
        examples = {
            "academic": ACADEMIC_SOURCES[0],
            "legal": LEGAL_SOURCES[0],
            "web": WEB_SOURCES[0],
            "custom": {"authors": "Smith, J.", "year": "2023", "title": "Example Paper", "source": "Journal"}
        }
        
        for name, generator in generators.items():
            citation = generator.generate_citation(examples[name])
            print(f"{name.capitalize()} generator: {citation}")
        
        # Demonstrate error handling with invalid type
        try:
            invalid_generator = get_citation_generator("invalid_type")
            print("This should not be printed")
        except ValueError as e:
            print(f"Error handling: {e}")
            
    except Exception as e:
        print(f"Error: {e}")

def main():
    """Run all citation generator examples."""
    print("CITATION GENERATORS EXAMPLE")
    print("This script demonstrates how to use the citation generators to create properly formatted citations.")
    
    # Run individual examples
    run_academic_citation_example()
    run_legal_citation_example()
    run_web_citation_example()
    run_custom_citation_example()
    run_factory_function_example()
    
    print_separator()
    print("Examples completed successfully.")

if __name__ == "__main__":
    main() 