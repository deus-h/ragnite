# Output Parsers

This directory contains documentation for the output parsers in the RAG Research project. Output parsers transform raw language model outputs into structured, validated formats for downstream use.

## Overview

Output parsers solve a critical challenge in RAG systems: extracting structured data from the free-text responses generated by LLMs. They allow you to parse JSON, XML, Markdown, or custom structured outputs, making it possible to:

1. Convert LLM text responses into structured data (objects, arrays, types)
2. Validate that the output meets your expectations and schema requirements
3. Handle edge cases and inconsistencies in LLM output formatting
4. Extract specific components from multi-part responses

## Available Output Parsers

### JSONOutputParser

The `JSONOutputParser` extracts JSON objects from language model outputs, even when embedded in other text.

Features:
- Extracts JSON from markdown code blocks or explanatory text
- Validates JSON against a schema with type checking
- Handles malformed JSON with helpful error messages
- Provides format instructions for prompting LLMs
- See [json_output_parser.md](./json_output_parser.md) for detailed documentation

### XMLOutputParser

The `XMLOutputParser` extracts XML elements from language model outputs.

Features:
- Extracts XML from markdown code blocks or explanatory text
- Validates against required elements and root tag constraints
- Parses into ElementTree objects for easy traversal
- Handles malformed XML with helpful error messages
- See [xml_output_parser.md](./xml_output_parser.md) for detailed documentation

### MarkdownOutputParser

The `MarkdownOutputParser` extracts structured information from Markdown-formatted text.

Features:
- Identifies and extracts headers, lists, code blocks, and blockquotes
- Organizes content hierarchically based on header levels
- Validates against required section headers
- Preserves content relationships and nesting
- See [markdown_output_parser.md](./markdown_output_parser.md) for detailed documentation

### StructuredOutputParser

The `StructuredOutputParser` extracts custom structured data according to predefined schemas and patterns.

Features:
- Combines JSON parsing with custom extraction patterns
- Applies transformations to parsed values
- Validates fields with custom validation functions
- Handles both well-formed JSON and informal structured text
- See [structured_output_parser.md](./structured_output_parser.md) for detailed documentation

## Usage

To use an output parser, create an instance with your configuration and then call the `parse` method:

```python
from tools.src.retrieval import JSONOutputParser, ParsingError

# Create a parser with a schema
parser = JSONOutputParser({
    "schema": {
        "title": str,
        "author": str,
        "publication_year": int,
        "genres": list,
        "summary": str
    }
})

# Get format instructions to include in your prompt
instructions = parser.get_format_instructions()
print(instructions)

# Parse LLM output
llm_output = """
Here's the book information:

```json
{
  "title": "Dune",
  "author": "Frank Herbert",
  "publication_year": 1965,
  "genres": ["Science Fiction", "Adventure"],
  "summary": "A complex tale of politics, ecology, and destiny on the desert planet Arrakis."
}
```
"""

try:
    result = parser.parse(llm_output)
    print(f"Title: {result['title']}")
    print(f"Author: {result['author']}")
    print(f"Year: {result['publication_year']}")
except ParsingError as e:
    print(f"Error parsing output: {e}")
```

## Factory Function

The `get_output_parser` factory function provides a simplified interface for creating parser instances:

```python
from tools.src.retrieval import get_output_parser

# Create a JSON parser
json_parser = get_output_parser(
    parser_type="json",
    schema={"name": str, "age": int, "skills": list}
)

# Create an XML parser
xml_parser = get_output_parser(
    parser_type="xml",
    root_tag="person",
    required_elements=["name", "age", "skills"]
)

# Create a Markdown parser
markdown_parser = get_output_parser(
    parser_type="markdown",
    required_sections=["Introduction", "Methodology", "Results"]
)

# Create a structured parser
structured_parser = get_output_parser(
    parser_type="structured",
    schema={"revenue": float, "customers": int},
    extraction_patterns={
        "revenue": r"Revenue: \$([0-9,]+(?:\.[0-9]+)?)",
        "customers": r"Customers: ([0-9,]+)"
    }
)
```

## Choosing an Output Parser

Different output parsers are suitable for different use cases:

- **JSONOutputParser**: Use when you need structured data with strict typing and validation.
- **XMLOutputParser**: Use when you need hierarchical data with elements and attributes.
- **MarkdownOutputParser**: Use when you want to extract components from a formatted document.
- **StructuredOutputParser**: Use when you need to extract specific patterns from semi-structured text.

## Error Handling

All output parsers raise a `ParsingError` when parsing fails. This exception provides details about what went wrong:

```python
from tools.src.retrieval import JSONOutputParser, ParsingError

parser = JSONOutputParser({"schema": {"name": str, "score": int}})

try:
    result = parser.parse('{"name": "Alice", "score": "invalid"}')
except ParsingError as e:
    print(f"Error: {e}")
    if e.underlying_error:
        print(f"Underlying error: {e.underlying_error}")
    print(f"Original text: {e.original_text}")
```

## Examples

See the [examples directory](../../../../examples/retrieval/generation/output_parsers/) for example scripts demonstrating the use of output parsers.

## Combining with Prompt Templates

Output parsers work best when combined with prompt templates that include the parser's formatting instructions:

```python
from tools.src.retrieval import (
    BasicPromptTemplate,
    JSONOutputParser
)

# Create an output parser
json_parser = JSONOutputParser({
    "schema": {
        "answer": str,
        "confidence": int,
        "sources": list
    }
})

# Create a prompt template
template = """Answer the following question:
{question}

{format_instructions}
"""

prompt = BasicPromptTemplate(template)

# Get the format instructions from the parser
format_instructions = json_parser.get_format_instructions()

# Generate the prompt
formatted_prompt = prompt.format(
    question="What is the capital of France?",
    format_instructions=format_instructions
)

# Send formatted_prompt to your LLM, then parse the response
# response = llm.generate(formatted_prompt)
# result = json_parser.parse(response)
```

## Extending the Parsers

You can create custom output parsers by extending the `BaseOutputParser` class:

```python
from typing import Dict, Any, Optional, List
from tools.src.retrieval import BaseOutputParser, ParsingError

class CustomTSVParser(BaseOutputParser[List[Dict[str, str]]]):
    """Parser for tab-separated values."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.headers = self.config.get("headers", [])
        self.required_headers = self.config.get("required_headers", [])
    
    def parse(self, text: str) -> List[Dict[str, str]]:
        """Parse TSV text into a list of dictionaries."""
        lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
        if not lines:
            raise ParsingError("No content to parse", text)
            
        # Extract headers from first line if not provided
        headers = self.headers
        if not headers:
            headers = [h.strip() for h in lines[0].split('\t')]
            lines = lines[1:]
            
        # Check required headers
        for req in self.required_headers:
            if req not in headers:
                raise ParsingError(f"Required header '{req}' not found", text)
                
        # Parse rows
        result = []
        for line in lines:
            values = [v.strip() for v in line.split('\t')]
            if len(values) != len(headers):
                continue  # Skip malformed rows
            result.append(dict(zip(headers, values)))
            
        return result
    
    def get_format_instructions(self) -> str:
        """Get instructions for formatting output as TSV."""
        headers = self.headers or ["column1", "column2", "column3"]
        return f"Respond with data in tab-separated format with these columns: {', '.join(headers)}"
``` 