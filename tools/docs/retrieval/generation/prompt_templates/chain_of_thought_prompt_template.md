# Chain of Thought Prompt Template

The `ChainOfThoughtPromptTemplate` is designed to encourage step-by-step reasoning in language models. This prompting technique significantly improves performance on complex reasoning tasks by guiding the model to break down problems into intermediate steps before arriving at a final answer.

## Features

- Encourages step-by-step reasoning for better problem-solving
- Builds on the few-shot learning approach with reasoning-specific structures
- Customizable reasoning and answer prefixes
- Support for adding additional reasoning prompts
- Configurable example formatting
- Inherits all functionality from FewShotPromptTemplate

## When to Use

The `ChainOfThoughtPromptTemplate` is particularly effective for:

- Complex reasoning tasks (math problems, logical puzzles)
- Multi-step problem solving
- Tasks requiring explanation or justification
- Reducing errors in language model outputs
- Generating more reliable and verifiable answers
- Improving transparency in the model's thinking process

## Usage

### Basic Usage

```python
from tools.src.retrieval import get_prompt_template

# Create a chain of thought template
cot_template = get_prompt_template(
    template_type="chain_of_thought",
    template="Solve the following problem: {problem}",
    examples=[
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
    ],
    reasoning_prefix="Let's think step by step:",
    answer_prefix="Therefore, the answer is:"
)

# Format the template with a problem
formatted_prompt = cot_template.format(problem="If a rectangle has a length of 8 cm and a width of 5 cm, what is its area?")

print(formatted_prompt)
# Output will include:
# - Two example problems with reasoning and answers
# - The new problem
# - "Let's think step by step:" prefix to encourage reasoning
# - "Therefore, the answer is:" prompt for the final answer
```

### Using Direct Instantiation

```python
from tools.src.retrieval import ChainOfThoughtPromptTemplate

# Create a chain of thought template
cot_template = ChainOfThoughtPromptTemplate(
    template="Answer the following question: {question}",
    examples=[
        {
            "question": "What is the capital of France?",
            "reasoning": "France is a country in Europe. The capital city of France is Paris.",
            "answer": "Paris"
        },
        {
            "question": "Who wrote 'Romeo and Juliet'?",
            "reasoning": "Romeo and Juliet is a famous play. It was written by William Shakespeare, who was an English playwright in the 16th century.",
            "answer": "William Shakespeare"
        }
    ],
    reasoning_prefix="I'll think through this step by step:",
    answer_prefix="So the answer is:"
)

# Format the template
formatted_prompt = cot_template.format(question="What is the largest planet in our solar system?")
print(formatted_prompt)
```

### Customizing the Example Template

```python
from tools.src.retrieval import ChainOfThoughtPromptTemplate

# Create a template with custom example format
cot_template = ChainOfThoughtPromptTemplate(
    template="Translate the following English sentence to French: {text}",
    example_template=(
        "English: {question}\n\n"
        "Thinking: {reasoning}\n\n"
        "French Translation: {answer}"
    ),
    examples=[
        {
            "question": "The book is on the table.",
            "reasoning": "Let me break this down:\n- 'The book' = 'Le livre'\n- 'is on' = 'est sur'\n- 'the table' = 'la table'\nPutting it together: 'Le livre est sur la table.'",
            "answer": "Le livre est sur la table."
        },
        {
            "question": "I like to read in the evening.",
            "reasoning": "Let me translate each part:\n- 'I like' = 'J'aime'\n- 'to read' = 'lire'\n- 'in the evening' = 'le soir'\nPutting it together: 'J'aime lire le soir.'",
            "answer": "J'aime lire le soir."
        }
    ],
    reasoning_prefix="Let me translate this step by step:",
    answer_prefix="The French translation is:"
)

formatted_prompt = cot_template.format(text="The weather is beautiful today.")
print(formatted_prompt)
```

### Adding Additional Reasoning Prompts

```python
from tools.src.retrieval import ChainOfThoughtPromptTemplate

# Create a template
cot_template = ChainOfThoughtPromptTemplate(
    template="Solve this math problem: {problem}",
    examples=[
        {
            "question": "What is 25 × 12?",
            "reasoning": "I can break this down: 25 × 12 = 25 × 10 + 25 × 2 = 250 + 50 = 300",
            "answer": "300"
        }
    ]
)

# Add additional reasoning prompts to guide the thinking process
cot_template.add_reasoning_prompt("First, I'll understand what the question is asking.")
cot_template.add_reasoning_prompt("Next, I'll identify the relevant information and variables.")
cot_template.add_reasoning_prompt("Then, I'll choose the appropriate formula or method to solve this.")
cot_template.add_reasoning_prompt("Finally, I'll calculate the result carefully.")

formatted_prompt = cot_template.format(problem="If a car travels at 60 km/h, how far will it travel in 2.5 hours?")
print(formatted_prompt)
```

### Configuring Reasoning and Answer Inclusion

```python
from tools.src.retrieval import ChainOfThoughtPromptTemplate

# Create a template with custom configuration
cot_template = ChainOfThoughtPromptTemplate(
    template="Answer this question: {question}",
    examples=[
        {
            "question": "What is the sum of the angles in a triangle?",
            "reasoning": "In any triangle, the sum of the interior angles is 180 degrees. This is a fundamental property of Euclidean geometry.",
            "answer": "180 degrees"
        }
    ],
    config={
        "include_reasoning_prefix": True,   # Include the reasoning prefix in the prompt
        "include_answer_in_query": False,   # Don't include the answer prefix
        "additional_reasoning_prompts": ["Break this down into smaller steps."]
    }
)

formatted_prompt = cot_template.format(question="What is the sum of the angles in a quadrilateral?")
print(formatted_prompt)
```

## Configuration Options

The `ChainOfThoughtPromptTemplate` accepts the following configuration options:

| Option | Description | Default |
|--------|-------------|---------|
| `include_reasoning_prefix` | Whether to include the reasoning prefix | `True` |
| `include_answer_in_query` | Whether to include the answer prefix | `True` |
| `additional_reasoning_prompts` | List of additional prompts to encourage reasoning | `[]` |

Plus all the configuration options from `FewShotPromptTemplate`.

## Methods

| Method | Description |
|--------|-------------|
| `format(**kwargs)` | Format the template with provided variables |
| `set_reasoning_prefix(reasoning_prefix)` | Set the reasoning prefix |
| `set_answer_prefix(answer_prefix)` | Set the answer prefix |
| `add_reasoning_prompt(prompt)` | Add an additional reasoning prompt |
| `add_example(example)` | Add a new example to the examples list |
| `set_examples(examples)` | Replace the entire examples list |
| `set_example_template(example_template)` | Set a new example template |
| `set_example_selector(example_selector)` | Set a new example selector function |
| `set_template(template)` | Update the main template string |
| `get_template()` | Get the current template string |
| `set_config(config)` | Update configuration options |
| `get_config()` | Get current configuration options |

## Example: Math Problem Solving

```python
from tools.src.retrieval import ChainOfThoughtPromptTemplate

# Create a template for math problem solving
math_cot_template = ChainOfThoughtPromptTemplate(
    template="Solve the following math problem: {problem}",
    examples=[
        {
            "question": "If x + y = 10 and x - y = 4, what are the values of x and y?",
            "reasoning": "We have two equations:\n1. x + y = 10\n2. x - y = 4\n\nFrom equation 2, we get x = 4 + y\nSubstituting this into equation 1:\n(4 + y) + y = 10\n4 + 2y = 10\n2y = 6\ny = 3\n\nNow, we can find x by substituting y = 3 into equation 1:\nx + 3 = 10\nx = 7",
            "answer": "x = 7 and y = 3"
        },
        {
            "question": "A rectangle has a perimeter of 30 cm and an area of 56 cm². What are its dimensions?",
            "reasoning": "Let's call the length l and width w.\n\nFor a rectangle, we know that:\n- Perimeter = 2l + 2w = 30 cm\n- Area = l × w = 56 cm²\n\nFrom the perimeter equation:\n2l + 2w = 30\nl + w = 15\nl = 15 - w\n\nSubstituting into the area equation:\n(15 - w) × w = 56\n15w - w² = 56\n-w² + 15w - 56 = 0\nw² - 15w + 56 = 0\n\nUsing the quadratic formula:\nw = (-b ± √(b² - 4ac)) / 2a\nw = (15 ± √(225 - 224)) / 2\nw = (15 ± √1) / 2\nw = (15 ± 1) / 2\nw = 8 or w = 7\n\nIf w = 8, then l = 15 - 8 = 7\nIf w = 7, then l = 15 - 7 = 8\n\nSo the dimensions are 7 cm and 8 cm.",
            "answer": "The dimensions are 7 cm and 8 cm."
        }
    ],
    reasoning_prefix="I'll solve this step by step:",
    answer_prefix="Therefore,"
)

# Format with a new problem
problem = "A train travels at 60 km/h. How far will it travel in 3.5 hours?"
formatted_prompt = math_cot_template.format(problem=problem)
print(formatted_prompt)
```

## Best Practices

1. **Clear Examples**: Provide clear, detailed reasoning steps in your examples.

2. **Explicit Steps**: Break down the reasoning into explicit, discrete steps.

3. **Varied Examples**: Include examples that demonstrate different types of reasoning.

4. **Appropriate Complexity**: Match the complexity of your examples to your target problems.

5. **Consistent Format**: Use a consistent format for all examples to help the model understand the pattern.

6. **Encourage Verbosity**: Chain-of-thought works better when the model explains its reasoning verbosely.

7. **Guide, Don't Solve**: In the reasoning prompts, guide the process but don't solve the specific problem.

## See Also

- [BasicPromptTemplate](./basic_prompt_template.md) - For simple variable substitution
- [FewShotPromptTemplate](./few_shot_prompt_template.md) - For example-based prompting
- [StructuredPromptTemplate](./structured_prompt_template.md) - For structured outputs 