# Citation Generators

Citation generators create properly formatted citations for various types of sources used in retrieval-augmented generation, such as academic papers, legal documents, web resources, and custom sources.

## Overview

Proper citation is a crucial component of responsible AI-generated content. When an LLM generates content based on retrieved information, providing citations helps establish credibility, acknowledge original sources, and enable verification. The citation generators in this module help create properly formatted citations for various source types and in different citation styles.

## Available Citation Generators

### AcademicCitationGenerator

Generate citations for academic sources like journal articles, books, conference papers, dissertations/theses, and book chapters.

**Features:**
- Support for multiple academic citation styles (APA, MLA, Chicago, Harvard)
- Specialized formatting for different source types (articles, books, conferences, etc.)
- Options for DOI inclusion, URL inclusion, and journal name abbreviation
- Proper formatting of author names, dates, and page numbers according to style guidelines

**Supported Source Types:**
- Journal articles
- Books
- Conference papers
- Theses/dissertations
- Book chapters

### LegalCitationGenerator

Generate citations for legal sources like cases, statutes, regulations, law review articles, and legal treatises.

**Features:**
- Support for legal citation styles (Bluebook, ALWD)
- Jurisdiction-specific formatting
- Court name abbreviation
- Proper handling of parallel citations, pinpoint citations, and party names

**Supported Source Types:**
- Cases
- Statutes
- Regulations
- Law review articles
- Legal treatises

### WebCitationGenerator

Generate citations for web resources such as websites, online articles, blog posts, and social media content.

**Features:**
- Support for multiple citation styles (APA, MLA, Chicago, Harvard)
- Specialized formatting for different web source types
- Automatic date formatting and access date inclusion
- Proper handling of URLs, group authors, and platform-specific details

**Supported Source Types:**
- Web pages
- Online articles
- Blog posts
- Social media posts

### CustomCitationGenerator

Generate citations using user-defined templates for any type of source.

**Features:**
- Flexible template system with field placeholders
- Conditional text based on field presence
- Custom field formatters
- Template export/import via JSON
- Auto-cleaning of citations to fix punctuation and spacing

## Usage

### Basic Usage

You can use each citation generator directly by creating an instance and calling its `generate_citation` or `generate_citations` methods:

```python
from tools.src.retrieval.generation.citation_generators import AcademicCitationGenerator

# Create the generator
academic_generator = AcademicCitationGenerator({
    'include_doi': True,
    'include_url': True
})

# Source to cite
source = {
    "type": "article",
    "title": "Large Language Models as Retrieval Augmentation Generators",
    "authors": [
        {"first_name": "Jane", "last_name": "Smith"},
        {"first_name": "John", "last_name": "Doe"}
    ],
    "journal": "Journal of AI Research",
    "volume": "45",
    "issue": "2",
    "pages": "123-145",
    "year": "2023",
    "doi": "10.1234/jair.2023.45.2.123"
}

# Generate the citation in APA style
citation = academic_generator.generate_citation(source, "APA")
print(citation)
```

### Factory Function

The module provides a factory function to create citation generators dynamically:

```python
from tools.src.retrieval.generation.citation_generators import get_citation_generator

# Create a citation generator using the factory function
generator = get_citation_generator(
    generator_type="academic",  # "academic", "legal", "web", or "custom"
    config={"default_style": "APA"}
)

# Use the generator
citation = generator.generate_citation(source)
```

### Using CustomCitationGenerator with Templates

```python
from tools.src.retrieval.generation.citation_generators import CustomCitationGenerator

# Define custom templates
templates = {
    "simple": "{authors}. ({year}). {title}.",
    "detailed": "{authors} ({year}). {title}. {source}. Retrieved from {url}.",
    "inline": "({authors}, {year}, \"{title}\")"
}

# Create the generator with templates
custom_generator = CustomCitationGenerator({
    'templates': templates,
    'default_style': 'simple'
})

# Source to cite
source = {
    "authors": "Smith, J.",
    "year": "2023",
    "title": "The Future of RAG Systems",
    "source": "AI Journal",
    "url": "https://example.com/future-rag"
}

# Generate the citation using a template
citation = custom_generator.generate_citation(source, "detailed")
print(citation)
```

## Configuration Options

### AcademicCitationGenerator Options

```python
{
    'default_style': 'APA',  # Default citation style
    'include_doi': True,     # Whether to include DOIs in citations
    'include_url': True,     # Whether to include URLs
    'abbreviate_journal': False  # Whether to abbreviate journal names
}
```

### LegalCitationGenerator Options

```python
{
    'default_style': 'BLUEBOOK',  # Default citation style
    'include_url': True,          # Whether to include URLs
    'jurisdiction': 'US',         # Default jurisdiction
    'court_abbreviations': {      # Court name abbreviations
        'Supreme Court of the United States': 'U.S.',
        'United States Court of Appeals': 'F.',
        'United States District Court': 'F. Supp.'
    }
}
```

### WebCitationGenerator Options

```python
{
    'default_style': 'APA',         # Default citation style
    'include_access_date': True,    # Whether to include access date
    'access_date_format': '%B %d, %Y', # Format for access dates
    'include_url': True,            # Whether to include URLs
    'default_publisher': 'Web'      # Default publisher if none provided
}
```

### CustomCitationGenerator Options

```python
{
    'default_style': 'default',  # Default template to use
    'templates': {              # Citation templates
        'default': '{authors}. ({year}). {title}. {source}.'
    },
    'date_format': '%B %d, %Y'  # Format for dates
}
```

## Creating Custom Field Formatters

With the `CustomCitationGenerator`, you can create custom field formatters for specialized formatting needs:

```python
def custom_title_formatter(title, source):
    """Format title with specific capitalization rules."""
    # Custom formatting logic
    return title.title()  # Title case

# Add the formatter to the generator
custom_generator.add_field_formatter('title', custom_title_formatter)
```

## Validation and Error Handling

Each citation generator validates that sources contain all required fields. If required fields are missing, the generator returns an error message instead of an improperly formatted citation:

```python
# Missing required fields
incomplete_source = {
    "type": "article",
    "title": "Incomplete Source Example",
    # Missing authors and year
}

# Will return something like: "[Incomplete citation - missing: authors, year]"
citation = generator.generate_citation(incomplete_source)
```

## Examples

For complete examples of how to use each citation generator, see the [citation_generators_example.py](../../../../../examples/retrieval/generation/citation_generators/citation_generators_example.py) script. 