# Prompt Templates

This directory contains documentation for the prompt templates in the RAG Research project. Prompt templates help create well-structured prompts for large language models, making it easier to get consistent and high-quality responses.

## Overview

Prompt templates provide a way to structure and format prompts for language models. They allow you to define a template with placeholders and then fill those placeholders with specific values at runtime, creating a complete prompt.

## Available Prompt Templates

### BasicPromptTemplate

The `BasicPromptTemplate` provides simple string-based templating with variable substitution.

Features:
- Simple variable substitution using Python's string formatting
- Automatic variable detection and validation
- Configurable whitespace and newline handling
- See [basic_prompt_template.md](./basic_prompt_template.md) for detailed documentation

### FewShotPromptTemplate

The `FewShotPromptTemplate` lets you create prompts that include examples to demonstrate the desired behavior.

Features:
- Example-based prompting for better performance
- Configurable example selection and formatting
- Support for example randomization and limiting
- See [few_shot_prompt_template.md](./few_shot_prompt_template.md) for detailed documentation

### ChainOfThoughtPromptTemplate

The `ChainOfThoughtPromptTemplate` helps create prompts that encourage step-by-step reasoning in language models.

Features:
- Structured reasoning with consistent prompting
- Example-based demonstration of reasoning steps
- Configurable reasoning and answer prompts
- See [chain_of_thought_prompt_template.md](./chain_of_thought_prompt_template.md) for detailed documentation

### StructuredPromptTemplate

The `StructuredPromptTemplate` creates prompts that guide language models to output structured data formats like JSON or XML.

Features:
- Support for multiple output formats (JSON, XML, YAML, CSV, markdown tables)
- Schema definition and validation
- Configurable format instructions
- See [structured_prompt_template.md](./structured_prompt_template.md) for detailed documentation

## Usage

To use a prompt template, you can create an instance directly or use the factory function:

```python
from tools.src.retrieval import get_prompt_template, BasicPromptTemplate

# Using the factory function
basic_template = get_prompt_template(
    template_type="basic",
    template="Answer the following question: {question}"
)

# Or creating an instance directly
from tools.src.retrieval import BasicPromptTemplate
basic_template = BasicPromptTemplate(template="Answer the following question: {question}")

# Format the template
formatted_prompt = basic_template.format(question="What is machine learning?")
```

## Factory Function

The `get_prompt_template` factory function provides a unified interface for creating prompt templates:

```python
from tools.src.retrieval import get_prompt_template

# Create a basic template
basic = get_prompt_template(
    template_type="basic",
    template="Answer the following question: {question}"
)

# Create a few-shot template
few_shot = get_prompt_template(
    template_type="few_shot",
    template="Please answer the question: {question}",
    examples=[
        {"input": "What is the capital of France?", "output": "The capital of France is Paris."}
    ],
    example_template="Question: {input}\nAnswer: {output}"
)

# Create a chain-of-thought template
cot = get_prompt_template(
    template_type="chain_of_thought",
    template="Solve the problem: {problem}",
    examples=[
        {
            "question": "What is 25 + 13?",
            "reasoning": "To add 25 and 13, I add the digits: 5 + 3 = 8, and 20 + 10 = 30. So 25 + 13 = 30 + 8 = 38.",
            "answer": "38"
        }
    ]
)

# Create a structured template
structured = get_prompt_template(
    template_type="structured",
    template="Analyze the sentiment of this text: {text}",
    output_format="json",
    schema={
        "type": "object",
        "properties": {
            "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
            "confidence": {"type": "number"}
        }
    }
)
```

## Choosing a Prompt Template

Different prompt templates have different strengths and use cases:

- **BasicPromptTemplate**: Use for simple, straightforward prompts where you just need variable substitution.
- **FewShotPromptTemplate**: Use when you want to guide the model by providing examples of the expected behavior.
- **ChainOfThoughtPromptTemplate**: Use for complex reasoning tasks where you want the model to show its work and reason step by step.
- **StructuredPromptTemplate**: Use when you need the model to output structured data in a specific format (JSON, XML, etc.).

## Examples

See the [examples directory](../../../../examples/retrieval/generation/) for example scripts demonstrating the use of prompt templates. 