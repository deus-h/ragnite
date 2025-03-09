# Self-RAG

## Overview
This folder contains an implementation of Self-RAG (Retrieval-Augmented Generation with Self-Reflection), an advanced RAG technique that incorporates self-reflection and critique into the generation process.

## What is Self-RAG?
Self-RAG enhances traditional RAG by adding a self-reflection mechanism that allows the model to:

1. Decide when to retrieve information vs. when to rely on its parametric knowledge
2. Critique its own generated content for factuality, relevance, and coherence
3. Iteratively refine its responses based on self-evaluation
4. Provide confidence scores and citations for generated content

The key innovation is that the model itself learns to determine when and what to retrieve, and how to evaluate the quality of its generations.

## Implementation Details
Our implementation uses:
- **Retrieval Decision**: LLM-based decision making (GPT-3.5-turbo)
- **Embedding Model**: OpenAI's text-embedding-ada-002
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Self-Reflection**: Specialized prompting for self-critique
- **Final Generation**: GPT-3.5-turbo with reflection-based refinement

## Project Structure
```
self-rag/
├── src/
│   ├── retrieval_critic.py    # Decide when to retrieve
│   ├── generation_critic.py   # Self-critique generation
│   ├── retriever.py           # Embedding-based retrieval
│   └── self_rag_pipeline.py   # End-to-end Self-RAG pipeline
├── examples/                  # Example usage scripts
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Getting Started
1. Install dependencies:
   ```
   pnpm install
   ```

2. Set up your OpenAI API key:
   ```
   export OPENAI_API_KEY=your_api_key_here
   ```

3. Run a query through the Self-RAG pipeline:
   ```
   python src/self_rag_pipeline.py --query "Your question here"
   ```

## Example
```python
from self_rag_pipeline import SelfRAG

# Initialize the Self-RAG pipeline
self_rag = SelfRAG(
    vector_store_path="../basic-rag/data/processed/vector_store",
    model_name="gpt-3.5-turbo"
)

# Query the Self-RAG system
response = self_rag.query("What are the effects of climate change?")
print(response)

# You can also see the reflection process
print("Retrieval decision:", self_rag.last_retrieval_decision)
print("Self-critique:", self_rag.last_critique)
print("Confidence score:", self_rag.last_confidence_score)
```

## Benefits of Self-RAG
- More judicious use of retrieval (only when needed)
- Higher factual accuracy through self-critique
- Transparent confidence scoring
- Reduced hallucinations
- Better source attribution and citation

## Limitations
- Increased computational cost due to multiple LLM calls
- Complex prompt engineering required
- Potential for self-reinforcing errors if initial critique is flawed

## References
- [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511)
- [RARR: Researching and Revising What Language Models Say](https://arxiv.org/abs/2210.08726)
- [Improving Factuality and Reasoning in Language Models through Multiagent Debate](https://arxiv.org/abs/2305.14325) 