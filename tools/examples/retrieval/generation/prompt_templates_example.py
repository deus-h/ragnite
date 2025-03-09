#!/usr/bin/env python3
"""
Prompt Templates Example

This script demonstrates the usage of various prompt templates for generating prompts
for language models.
"""

import sys
import os
import json
from typing import Dict, Any, List

# Add the parent directory to sys.path to import the tools
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from tools.src.retrieval import (
    BasicPromptTemplate,
    FewShotPromptTemplate,
    ChainOfThoughtPromptTemplate,
    StructuredPromptTemplate,
    get_prompt_template
)


def basic_template_example():
    """
    Example using the BasicPromptTemplate.
    """
    print("\n" + "="*50)
    print("BASIC PROMPT TEMPLATE EXAMPLE")
    print("="*50)
    
    # Create a basic prompt template
    template = """
    You are a helpful assistant.
    
    User query: {query}
    
    Please provide information about {topic} that addresses the user's query.
    """
    
    basic_template = BasicPromptTemplate(template=template)
    
    # Format the prompt
    formatted_prompt = basic_template.format(
        query="How does photosynthesis work?",
        topic="photosynthesis"
    )
    
    print(formatted_prompt)
    
    # List template variables
    print("\nTemplate Variables:", basic_template.list_variables())


def few_shot_template_example():
    """
    Example using the FewShotPromptTemplate.
    """
    print("\n" + "="*50)
    print("FEW-SHOT PROMPT TEMPLATE EXAMPLE")
    print("="*50)
    
    # Create examples
    examples = [
        {
            "input": "What is the capital of France?",
            "output": "The capital of France is Paris."
        },
        {
            "input": "Who wrote the play 'Romeo and Juliet'?",
            "output": "William Shakespeare wrote the play 'Romeo and Juliet'."
        },
        {
            "input": "What is the boiling point of water?",
            "output": "The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure."
        }
    ]
    
    # Create a few-shot prompt template
    template = "Please answer the following question: {question}"
    example_template = "Question: {input}\nAnswer: {output}"
    
    few_shot_template = FewShotPromptTemplate(
        template=template,
        example_template=example_template,
        examples=examples,
        prefix="Here are some examples of question-answer pairs:\n\n",
        suffix="\nNow, I'll answer your question:\n",
        config={"max_examples": 2}
    )
    
    # Format the prompt
    formatted_prompt = few_shot_template.format(question="What is photosynthesis?")
    
    print(formatted_prompt)


def chain_of_thought_template_example():
    """
    Example using the ChainOfThoughtPromptTemplate.
    """
    print("\n" + "="*50)
    print("CHAIN OF THOUGHT PROMPT TEMPLATE EXAMPLE")
    print("="*50)
    
    # Create examples
    examples = [
        {
            "question": "If John has 5 apples and gives 2 to Mary, how many apples does John have left?",
            "reasoning": "John starts with 5 apples. He gives 2 apples to Mary. So, 5 - 2 = 3 apples.",
            "answer": "3 apples"
        },
        {
            "question": "A train travels at 60 mph. How far will it travel in 2.5 hours?",
            "reasoning": "The train travels at 60 miles per hour. In 2.5 hours, it will travel 60 × 2.5 = 150 miles.",
            "answer": "150 miles"
        }
    ]
    
    # Create a chain of thought prompt template
    template = "Solve the following problem: {problem}"
    
    cot_template = ChainOfThoughtPromptTemplate(
        template=template,
        examples=examples,
        reasoning_prefix="Reasoning: ",
        answer_prefix="Answer: "
    )
    
    # Format the prompt
    formatted_prompt = cot_template.format(problem="If a rectangle has a length of 8 cm and a width of 5 cm, what is its area?")
    
    print(formatted_prompt)


def structured_template_example():
    """
    Example using the StructuredPromptTemplate.
    """
    print("\n" + "="*50)
    print("STRUCTURED PROMPT TEMPLATE EXAMPLE")
    print("="*50)
    
    # Create a JSON schema
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Name of the person"},
            "age": {"type": "integer", "description": "Age of the person"},
            "address": {
                "type": "object",
                "properties": {
                    "street": {"type": "string"},
                    "city": {"type": "string"},
                    "zipCode": {"type": "string"}
                }
            },
            "interests": {
                "type": "array",
                "items": {
                    "type": "string"
                }
            }
        },
        "required": ["name", "age"]
    }
    
    # Create a structured prompt template
    template = """
    Create a user profile for a person with the following information:
    
    Name: {name}
    Age: {age}
    Location: {location}
    Occupation: {occupation}
    Hobbies: {hobbies}
    """
    
    structured_template = StructuredPromptTemplate(
        template=template,
        output_format="json",
        schema=schema,
        schema_format="json_schema"
    )
    
    # Format the prompt
    formatted_prompt = structured_template.format(
        name="Jane Smith",
        age=32,
        location="New York City",
        occupation="Software Engineer",
        hobbies="Reading, hiking, photography"
    )
    
    print(formatted_prompt)
    
    # Show XML format example
    print("\n" + "-"*50)
    print("XML FORMAT EXAMPLE")
    print("-"*50)
    
    xml_schema = """
<user>
  <name>John Doe</name>
  <age>30</age>
  <address>
    <street>123 Main St</street>
    <city>San Francisco</city>
    <zipCode>94102</zipCode>
  </address>
  <interests>
    <interest>Reading</interest>
    <interest>Hiking</interest>
  </interests>
</user>
"""
    
    structured_template_xml = StructuredPromptTemplate(
        template=template,
        output_format="xml",
        schema=xml_schema,
        schema_format="example"
    )
    
    formatted_prompt_xml = structured_template_xml.format(
        name="Jane Smith",
        age=32,
        location="New York City",
        occupation="Software Engineer",
        hobbies="Reading, hiking, photography"
    )
    
    print(formatted_prompt_xml)


def factory_function_example():
    """
    Example using the get_prompt_template factory function.
    """
    print("\n" + "="*50)
    print("FACTORY FUNCTION EXAMPLE")
    print("="*50)
    
    # Create a basic template using factory
    basic = get_prompt_template(
        template_type="basic",
        template="Answer the following question: {question}"
    )
    
    print("Basic Template:", basic.format(question="What is AI?"))
    
    # Create a few-shot template using factory
    few_shot = get_prompt_template(
        template_type="few_shot",
        template="Translate the following text to {language}: {text}",
        examples=[
            {"input": "Hello, how are you?", "output": "Hola, ¿cómo estás?", "language": "Spanish"},
            {"input": "What's your name?", "output": "¿Cómo te llamas?", "language": "Spanish"}
        ],
        example_template="English: {input}\n{language}: {output}"
    )
    
    print("\nFew-Shot Template:", few_shot.format(text="I love programming", language="Spanish"))
    
    # Create a structured template using factory
    structured = get_prompt_template(
        template_type="structured",
        template="Analyze the sentiment of the following text: {text}",
        output_format="json",
        schema={
            "type": "object",
            "properties": {
                "sentiment": {
                    "type": "string", 
                    "enum": ["positive", "negative", "neutral"]
                },
                "confidence": {"type": "number"},
                "reasoning": {"type": "string"}
            }
        }
    )
    
    print("\nStructured Template:", structured.format(text="I really enjoyed this movie. It was fantastic!"))


def main():
    """
    Main function to run all examples.
    """
    # Run examples
    basic_template_example()
    few_shot_template_example()
    chain_of_thought_template_example()
    structured_template_example()
    factory_function_example()
    
    print("\nAll examples completed successfully!")


if __name__ == "__main__":
    main() 