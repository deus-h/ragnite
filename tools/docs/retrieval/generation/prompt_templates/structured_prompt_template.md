# Structured Prompt Template

The `StructuredPromptTemplate` helps language models generate responses in specific structured formats like JSON, XML, or CSV. By providing explicit format instructions and schemas, this template significantly improves the consistency and usability of structured outputs from language models.

## Features

- Support for multiple output formats (JSON, XML, YAML, CSV, markdown tables)
- Schema definition to guide the model's output structure
- Multiple schema formats (JSON Schema, examples, descriptions)
- Customizable format instructions
- Built-in schema simplification for readability
- Inherits all functionality from BasicPromptTemplate

## When to Use

The `StructuredPromptTemplate` is particularly valuable when:

- You need the LLM's output in a machine-readable format
- Your application requires parsing or processing the LLM's response
- You need consistent, structured data from LLM responses
- You're building applications that integrate LLMs with other systems
- You want to reduce parsing errors and post-processing complexity
- You need outputs in specific formats like JSON, XML, or YAML

## Supported Formats

- `json`: JavaScript Object Notation
- `xml`: Extensible Markup Language
- `yaml`: YAML Ain't Markup Language
- `csv`: Comma Separated Values
- `markdown_table`: Markdown formatted tables
- `custom`: Custom formats defined by the user

## Usage

### Basic JSON Format

```python
from tools.src.retrieval import get_prompt_template

# Create a structured prompt template for JSON output
json_template = get_prompt_template(
    template_type="structured",
    template="Extract the person's information from this text: {text}",
    output_format="json",
    schema={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "occupation": {"type": "string"},
            "email": {"type": "string", "format": "email"}
        },
        "required": ["name", "age"]
    }
)

# Format the template
text = "John Doe is a 35-year-old software engineer. You can contact him at john.doe@example.com."
formatted_prompt = json_template.format(text=text)

print(formatted_prompt)
# Output will include:
# - The main instruction
# - Format instructions with JSON schema
# - Directives to follow the format
```

### Using Direct Instantiation

```python
from tools.src.retrieval import StructuredPromptTemplate

# Create a structured prompt template
template = StructuredPromptTemplate(
    template="Analyze the sentiment of this product review: {review}",
    output_format="json",
    schema={
        "type": "object",
        "properties": {
            "sentiment": {
                "type": "string", 
                "enum": ["positive", "negative", "neutral"]
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1
            },
            "key_points": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["sentiment", "confidence"]
    }
)

# Format the template
review = "This product exceeded my expectations. The quality is great and it works perfectly. Highly recommended!"
formatted_prompt = template.format(review=review)
print(formatted_prompt)
```

### XML Format with Example Schema

```python
from tools.src.retrieval import StructuredPromptTemplate

# Create a template for XML output using an example
xml_template = StructuredPromptTemplate(
    template="Extract the book information from this text: {text}",
    output_format="xml",
    schema="""
<book>
    <title>The Great Gatsby</title>
    <author>F. Scott Fitzgerald</author>
    <year>1925</year>
    <genre>Novel</genre>
    <summary>A story of wealth, love, and the American Dream.</summary>
</book>
    """,
    schema_format="example"
)

# Format the template
text = "To Kill a Mockingbird was written by Harper Lee and published in 1960. It's a classic American novel that deals with serious issues of race and injustice."
formatted_prompt = xml_template.format(text=text)
print(formatted_prompt)
```

### YAML Format

```python
from tools.src.retrieval import StructuredPromptTemplate

# Create a template for YAML output
yaml_template = StructuredPromptTemplate(
    template="Create a recipe based on these ingredients: {ingredients}",
    output_format="yaml",
    schema="""
name: Chocolate Chip Cookies
ingredients:
  - item: Flour
    amount: 2 cups
  - item: Sugar
    amount: 1 cup
  - item: Chocolate Chips
    amount: 1 cup
instructions:
  - Preheat oven to 350°F
  - Mix ingredients
  - Bake for 10-12 minutes
preparation_time: 30 minutes
serves: 24 cookies
    """,
    schema_format="example"
)

# Format the template
ingredients = "chicken, rice, garlic, onions, bell peppers, tomatoes"
formatted_prompt = yaml_template.format(ingredients=ingredients)
print(formatted_prompt)
```

### CSV Format

```python
from tools.src.retrieval import StructuredPromptTemplate

# Create a template for CSV output
csv_template = StructuredPromptTemplate(
    template="Generate a list of top 5 {category} with their key features:",
    output_format="csv",
    schema="""
name,price_range,features,rating
Product A,$100-$200,"Feature 1, Feature 2",4.5
Product B,$150-$250,"Feature 1, Feature 3",4.2
Product C,$300-$400,"Feature 2, Feature 4",4.8
    """,
    schema_format="example"
)

# Format the template
formatted_prompt = csv_template.format(category="smartphones")
print(formatted_prompt)
```

### Markdown Table Format

```python
from tools.src.retrieval import StructuredPromptTemplate

# Create a template for markdown table output
md_table_template = StructuredPromptTemplate(
    template="Compare the following programming languages: {languages}",
    output_format="markdown_table",
    schema="""
| Language | Paradigm | Typing | Use Cases | Popularity |
|----------|----------|--------|-----------|------------|
| Python   | Multi    | Dynamic| Data Science, Web | Very High |
| Java     | OOP      | Static | Enterprise, Android | High |
| JavaScript | Multi  | Dynamic | Web, Frontend | Very High |
    """,
    schema_format="example"
)

# Format the template
formatted_prompt = md_table_template.format(languages="Go, Rust, TypeScript")
print(formatted_prompt)
```

### Custom Format with Description

```python
from tools.src.retrieval import StructuredPromptTemplate

# Create a template with a custom format
custom_template = StructuredPromptTemplate(
    template="Generate a SQL query to {task}:",
    output_format="custom",
    schema="""
Your response should be a valid SQL query with the following components:
1. SELECT statement listing specific columns (not SELECT *)
2. FROM clause with table name(s)
3. WHERE clause with appropriate conditions
4. Optional: GROUP BY, ORDER BY, LIMIT clauses as needed
5. Comments explaining your query logic

Example:
-- Query to find high-value customers
SELECT 
    customer_id,
    first_name,
    last_name,
    SUM(order_total) as total_spent
FROM 
    customers
JOIN
    orders ON customers.customer_id = orders.customer_id
WHERE 
    order_date > '2023-01-01'
GROUP BY 
    customer_id, first_name, last_name
HAVING 
    total_spent > 1000
ORDER BY 
    total_spent DESC
LIMIT 10;
    """,
    schema_format="description"
)

# Format the template
formatted_prompt = custom_template.format(task="find all products with inventory below 10 units")
print(formatted_prompt)
```

## Configuration Options

The `StructuredPromptTemplate` accepts the following configuration options:

| Option | Description | Default |
|--------|-------------|---------|
| `include_format_instructions` | Whether to include format instructions | `True` |
| `format_strict` | Whether to enforce strict format compliance | `True` |
| `format_intro` | Custom introduction for format instructions | "Your response must be in the following format:" |
| `format_outro` | Custom conclusion for format instructions | "Ensure your response exactly follows this format." |

Plus all the configuration options from `BasicPromptTemplate`.

## Methods

| Method | Description |
|--------|-------------|
| `format(**kwargs)` | Format the template with provided variables |
| `set_output_format(output_format)` | Set the output format |
| `set_schema(schema, schema_format)` | Set the schema definition and optionally the schema format |
| `set_template(template)` | Update the main template string |
| `get_template()` | Get the current template string |
| `set_config(config)` | Update configuration options |
| `get_config()` | Get current configuration options |

## Example: Entity Extraction

```python
from tools.src.retrieval import StructuredPromptTemplate

# Create a template for entity extraction
entity_template = StructuredPromptTemplate(
    template="Extract the entities from this text: {text}",
    output_format="json",
    schema={
        "type": "object",
        "properties": {
            "people": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "role": {"type": "string"}
                    }
                }
            },
            "organizations": {
                "type": "array",
                "items": {"type": "string"}
            },
            "locations": {
                "type": "array",
                "items": {"type": "string"}
            },
            "dates": {
                "type": "array",
                "items": {"type": "string"}
            }
        }
    },
    config={
        "format_intro": "Extract entities into this JSON structure:",
        "format_outro": "Include only entities explicitly mentioned in the text."
    }
)

# Example usage
text = """
On May 15, 2023, CEO Jane Smith announced that Acme Corporation will open a new headquarters 
in Austin, Texas. The project will be overseen by CTO Mark Johnson and is expected to be 
completed by Q3 2024. The company has partnered with BuildCo Construction for this project.
"""
formatted_prompt = entity_template.format(text=text)
print(formatted_prompt)
```

## Best Practices

1. **Clear Schema**: Provide a clear, detailed schema that precisely defines the structure you want.

2. **Examples**: When using example schemas, make them representative of the expected output.

3. **Required Fields**: Specify which fields are required in your JSON Schema to ensure they're included.

4. **Simple Structure**: Keep the output structure as simple as possible while meeting your needs.

5. **Format Strictness**: Use `format_strict=True` (default) to emphasize the importance of format compliance.

6. **Error Handling**: Implement robust error handling when parsing the structured outputs, as LLMs may occasionally deviate from the specified format.

7. **Field Descriptions**: Include descriptions for fields in your JSON Schema to guide the model.

## Schema Format Options

The `StructuredPromptTemplate` supports four schema format options:

1. **json_schema**: A JSON Schema specification defining the structure and validations.
2. **example**: An example of the desired output structure.
3. **description**: A textual description of the expected format.
4. **custom**: A custom format specification defined by the user.

## Known Limitations

1. Very complex schemas may not be followed precisely by all language models.
2. Output format guarantees depend on the capabilities of the underlying language model.
3. For extremely structured data needs, consider post-processing or using specialized output parsers.

## See Also

- [BasicPromptTemplate](./basic_prompt_template.md) - For simple variable substitution
- [FewShotPromptTemplate](./few_shot_prompt_template.md) - For example-based prompting
- [ChainOfThoughtPromptTemplate](./chain_of_thought_prompt_template.md) - For reasoning tasks 