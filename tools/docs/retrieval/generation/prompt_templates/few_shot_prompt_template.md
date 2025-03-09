# Few-Shot Prompt Template

The `FewShotPromptTemplate` enhances prompt effectiveness by including examples that demonstrate the desired behavior before presenting the actual task. This technique, known as "few-shot learning," helps guide the language model to produce responses in the style and format you want.

## Features

- Example-based prompting for better model performance
- Configurable example template for consistent formatting
- Support for example selection based on relevance to the query
- Control over example ordering and quantity
- Customizable prefix and suffix text
- Built on top of BasicPromptTemplate for variable substitution

## When to Use

The `FewShotPromptTemplate` is particularly effective when:

- You want to guide the model's responses by showing examples
- The task might be ambiguous without examples
- You need consistent output formatting
- You're using smaller models that benefit from examples
- You want to demonstrate a specific style, tone, or approach

## Usage

### Basic Usage

```python
from tools.src.retrieval import get_prompt_template

# Create a few-shot template with examples
few_shot_template = get_prompt_template(
    template_type="few_shot",
    template="Translate the following English text to French: {text}",
    examples=[
        {"input": "Hello, how are you?", "output": "Bonjour, comment allez-vous?"},
        {"input": "I love machine learning.", "output": "J'adore l'apprentissage automatique."}
    ],
    example_template="English: {input}\nFrench: {output}",
    prefix="Here are some examples of English to French translations:\n\n",
    suffix="\nNow, translate this new text:\n\n"
)

# Format the template
formatted_prompt = few_shot_template.format(text="The weather is nice today.")

print(formatted_prompt)
# Output:
# Here are some examples of English to French translations:
#
# English: Hello, how are you?
# French: Bonjour, comment allez-vous?
#
# English: I love machine learning.
# French: J'adore l'apprentissage automatique.
#
# Now, translate this new text:
#
# Translate the following English text to French: The weather is nice today.
```

### Using Direct Instantiation

```python
from tools.src.retrieval import FewShotPromptTemplate

# Create a few-shot prompt template
template = FewShotPromptTemplate(
    template="Write a {adj} poem about {topic}",
    example_template="Topic: {topic}\nStyle: {style}\n\nPoem:\n{poem}",
    examples=[
        {
            "topic": "Mountains",
            "style": "Haiku",
            "poem": "Silent mountains rise\nWhispers of ancient stories\nStone giants sleeping"
        },
        {
            "topic": "Ocean",
            "style": "Free verse",
            "poem": "Endless blue expanse\nRolling waves that crash and foam\nSalt spray in the air\nThe ocean breathes with the tide\nA rhythm eternal"
        }
    ],
    prefix="Here are some examples of poems in different styles:\n\n",
    suffix="\nNow, write a new poem based on these instructions:\n"
)

# Format the template
formatted_prompt = template.format(adj="lyrical", topic="autumn leaves")
print(formatted_prompt)
```

### Limiting the Number of Examples

```python
from tools.src.retrieval import FewShotPromptTemplate

# Create a template with many examples
template = FewShotPromptTemplate(
    template="Classify the sentiment of the following text as positive, negative, or neutral: {text}",
    example_template="Text: {text}\nSentiment: {sentiment}",
    examples=[
        {"text": "I love this product!", "sentiment": "positive"},
        {"text": "This is terrible service.", "sentiment": "negative"},
        {"text": "The package arrived on time.", "sentiment": "neutral"},
        {"text": "Amazing experience, highly recommend!", "sentiment": "positive"},
        {"text": "I'm disappointed with the quality.", "sentiment": "negative"}
    ],
    config={"max_examples": 3}  # Only use the first 3 examples
)

formatted_prompt = template.format(text="The food was okay, nothing special.")
print(formatted_prompt)
```

### Randomizing Examples

```python
from tools.src.retrieval import FewShotPromptTemplate

# Create a template with randomized examples
template = FewShotPromptTemplate(
    template="Generate a creative name for a {business_type} business:",
    example_template="Business Type: {type}\nName: {name}",
    examples=[
        {"type": "Coffee shop", "name": "Brew Haven"},
        {"type": "Bookstore", "name": "Chapter & Verse"},
        {"type": "Pet grooming", "name": "Paws & Relax"},
        {"type": "Bakery", "name": "Flour Power"},
        {"type": "Tech startup", "name": "Quantum Leap Solutions"}
    ],
    config={
        "max_examples": 3,        # Use 3 examples
        "randomize_examples": True  # Randomly select examples
    }
)

formatted_prompt = template.format(business_type="yoga studio")
print(formatted_prompt)
```

### Using an Example Selector

```python
from tools.src.retrieval import FewShotPromptTemplate

# Define a custom example selector function
def select_examples_by_language(examples, query_vars):
    """Select examples matching the target language."""
    target_language = query_vars.get("target_language", "").lower()
    return [ex for ex in examples if ex.get("language", "").lower() == target_language]

# Create examples for different languages
examples = [
    {"input": "Hello", "output": "Hola", "language": "spanish"},
    {"input": "Goodbye", "output": "Adiós", "language": "spanish"},
    {"input": "Hello", "output": "Bonjour", "language": "french"},
    {"input": "Goodbye", "output": "Au revoir", "language": "french"},
    {"input": "Hello", "output": "Guten Tag", "language": "german"},
    {"input": "Goodbye", "output": "Auf Wiedersehen", "language": "german"}
]

# Create a template with the custom selector
template = FewShotPromptTemplate(
    template="Translate the following English word to {target_language}: {word}",
    example_template="English: {input}\n{language}: {output}",
    examples=examples,
    example_selector=select_examples_by_language
)

# Will only include Spanish examples
spanish_prompt = template.format(target_language="Spanish", word="Thank you")
print(spanish_prompt)

# Will only include French examples
french_prompt = template.format(target_language="French", word="Thank you")
print(french_prompt)
```

## Configuration Options

The `FewShotPromptTemplate` accepts the following configuration options:

| Option | Description | Default |
|--------|-------------|---------|
| `max_examples` | Maximum number of examples to include | All examples |
| `randomize_examples` | Whether to randomize the order of examples | `False` |

Plus all the configuration options from `BasicPromptTemplate`.

## Methods

| Method | Description |
|--------|-------------|
| `format(**kwargs)` | Format the template with provided variables |
| `add_example(example)` | Add a new example to the examples list |
| `set_examples(examples)` | Replace the entire examples list |
| `set_example_template(example_template)` | Set a new example template |
| `set_example_selector(example_selector)` | Set a new example selector function |
| `set_template(template)` | Update the main template string |
| `get_template()` | Get the current template string |
| `set_config(config)` | Update configuration options |
| `get_config()` | Get current configuration options |

## Example: Question Answering

```python
from tools.src.retrieval import FewShotPromptTemplate

# Create a template for question answering
qa_template = FewShotPromptTemplate(
    template="Please answer the following question: {question}",
    example_template="Q: {question}\nA: {answer}",
    examples=[
        {
            "question": "What is the capital of France?",
            "answer": "The capital of France is Paris."
        },
        {
            "question": "How many planets are in our solar system?",
            "answer": "There are eight planets in our solar system: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune."
        },
        {
            "question": "Who wrote Romeo and Juliet?",
            "answer": "Romeo and Juliet was written by William Shakespeare."
        }
    ],
    prefix="I'll answer your question based on the following examples:\n\n",
    suffix="\nNow for your question:\n\n"
)

# Format with a new question
formatted_prompt = qa_template.format(question="What is the largest ocean on Earth?")
print(formatted_prompt)
```

## Best Practices

1. **Quality Examples**: Choose high-quality, diverse examples that demonstrate the expected output format.

2. **Example Relevance**: Use examples that are relevant to the types of queries your system will handle.

3. **Example Order**: Consider ordering examples from simple to complex, as the model may be influenced by the order.

4. **Example Count**: Start with 3-5 examples for most tasks. More examples can help for complex tasks but may hit token limits.

5. **Consistency**: Ensure the format of your examples is consistent with what you want the model to produce.

6. **Clear Instructions**: Include clear instructions in your prefix, suffix, and template to guide the model's behavior.

## See Also

- [BasicPromptTemplate](./basic_prompt_template.md) - For simple variable substitution
- [ChainOfThoughtPromptTemplate](./chain_of_thought_prompt_template.md) - For reasoning tasks
- [StructuredPromptTemplate](./structured_prompt_template.md) - For structured outputs 