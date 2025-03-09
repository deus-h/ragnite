# Basic Prompt Template

The `BasicPromptTemplate` provides a simple way to create prompts with variable substitution. It uses Python's `string.format()` method to replace placeholders in the template with provided values.

## Features

- Simple variable substitution using Python's string formatting
- Automatic variable detection and validation
- Configurable whitespace and newline handling
- Error handling for missing variables

## When to Use

The `BasicPromptTemplate` is ideal for:

- Simple prompts that don't require examples or complex structure
- Cases where you need to insert variables into a predefined template
- General purpose text generation without special formatting needs
- Quick prototyping of prompts

## Usage

### Basic Usage

```python
from tools.src.retrieval import get_prompt_template

# Create a basic prompt template
template = get_prompt_template(
    template_type="basic",
    template="You are a helpful assistant. Please answer the following question: {question}"
)

# Format the template with a question
formatted_prompt = template.format(question="What is machine learning?")

print(formatted_prompt)
# Output:
# You are a helpful assistant. Please answer the following question: What is machine learning?
```

### Using Direct Instantiation

```python
from tools.src.retrieval import BasicPromptTemplate

# Create a basic prompt template
template = BasicPromptTemplate(
    template="Summarize the following text in {format_style} style:\n\n{text}"
)

# Format the template with variables
formatted_prompt = template.format(
    format_style="academic",
    text="Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed."
)

print(formatted_prompt)
```

### Working with Multiple Variables

```python
from tools.src.retrieval import BasicPromptTemplate

# Template with multiple variables
template = BasicPromptTemplate(
    template="""
    System: {system_message}
    
    User: {user_message}
    
    Assistant:
    """
)

# Format with multiple variables
formatted_prompt = template.format(
    system_message="You are a helpful programming assistant.",
    user_message="How do I read a CSV file in Python?"
)

print(formatted_prompt)
```

### Inspecting Template Variables

```python
from tools.src.retrieval import BasicPromptTemplate

template = BasicPromptTemplate(
    template="Generate a {length} {content_type} about {topic}."
)

# List all variables in the template
variables = template.list_variables()
print(variables)  # ['length', 'content_type', 'topic']

# Validate variables
has_all_vars = template.validate_variables({
    "length": "short",
    "content_type": "essay",
    "topic": "climate change"
})
print(has_all_vars)  # True

missing_vars = template.validate_variables({
    "length": "short",
    "topic": "climate change"
})
print(missing_vars)  # False - missing 'content_type'
```

## Configuration Options

The `BasicPromptTemplate` accepts the following configuration options:

| Option | Description | Default |
|--------|-------------|---------|
| `strip_whitespace` | Whether to strip leading/trailing whitespace | `True` |
| `strip_newlines` | Whether to replace excessive newlines with a single newline | `False` |

Example with configuration:

```python
from tools.src.retrieval import BasicPromptTemplate

template = BasicPromptTemplate(
    template="""
    
    Answer the question: {question}
    
    
    """,
    config={
        "strip_whitespace": True,  # Remove leading/trailing whitespace
        "strip_newlines": True     # Remove excessive newlines
    }
)

formatted_prompt = template.format(question="What is AI?")
print(formatted_prompt)
# Output: "Answer the question: What is AI?"
```

## Error Handling

The template will raise a `ValueError` if a required variable is missing:

```python
from tools.src.retrieval import BasicPromptTemplate

template = BasicPromptTemplate(
    template="Answer the question: {question}"
)

try:
    # Missing 'question' variable
    formatted_prompt = template.format(wrong_variable="What is AI?")
except ValueError as e:
    print(e)  # Missing required variable in prompt template: question
```

## Methods

| Method | Description |
|--------|-------------|
| `format(**kwargs)` | Format the template with provided variables |
| `list_variables()` | Get a list of all variables in the template |
| `validate_variables(variables)` | Check if all required variables are provided |
| `set_template(template)` | Update the template string |
| `get_template()` | Get the current template string |
| `set_config(config)` | Update configuration options |
| `get_config()` | Get current configuration options |

## Example

```python
from tools.src.retrieval import BasicPromptTemplate

# Create a template for generating questions
question_template = BasicPromptTemplate(
    template="Generate {question_count} questions about {topic} at the {difficulty} level."
)

# Format with specific values
easy_math_prompt = question_template.format(
    question_count=5,
    topic="algebra",
    difficulty="beginner"
)
print(easy_math_prompt)
# Output: "Generate 5 questions about algebra at the beginner level."

# Format with different values
hard_history_prompt = question_template.format(
    question_count=10,
    topic="World War II",
    difficulty="advanced"
)
print(hard_history_prompt)
# Output: "Generate 10 questions about World War II at the advanced level."
```

## See Also

- [FewShotPromptTemplate](./few_shot_prompt_template.md) - For templates with examples
- [ChainOfThoughtPromptTemplate](./chain_of_thought_prompt_template.md) - For reasoning-oriented prompts
- [StructuredPromptTemplate](./structured_prompt_template.md) - For generating structured outputs 