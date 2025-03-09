# Multi-Query RAG

## Overview
This folder contains an implementation of Multi-Query Retrieval-Augmented Generation, an advanced RAG technique that improves retrieval quality by generating multiple query variations from a single user query.

## What is Multi-Query RAG?
Multi-Query RAG addresses a key limitation of basic RAG: the dependency on the exact wording of a user's query for retrieval. Instead of using only the original query, Multi-Query RAG:

1. Takes the user's original query
2. Generates multiple semantically similar but lexically diverse query variations
3. Performs retrieval for each query variation
4. Aggregates and deduplicates the retrieved documents
5. Uses the combined context for generation

This approach increases the likelihood of retrieving relevant information that might be missed by a single query formulation.

## Implementation Details
Our implementation uses:
- **Query Generation**: LLM-based query reformulation (GPT-3.5-turbo)
- **Embedding Model**: OpenAI's text-embedding-ada-002
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Aggregation Strategy**: Fusion methods (reciprocal rank fusion)
- **LLM**: OpenAI's GPT-3.5-turbo

## Project Structure
```
multi-query/
├── src/
│   ├── query_generator.py     # Generate query variations
│   ├── retriever.py           # Multi-query retrieval
│   ├── fusion.py              # Result fusion strategies
│   └── mq_rag_pipeline.py     # End-to-end Multi-Query RAG pipeline
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

3. Run a query through the Multi-Query RAG pipeline:
   ```
   python src/mq_rag_pipeline.py --query "Your question here" --num_queries 3
   ```

## Example
```python
from mq_rag_pipeline import MultiQueryRAG

# Initialize the Multi-Query RAG pipeline
mq_rag = MultiQueryRAG(
    vector_store_path="../basic-rag/data/processed/vector_store",
    model_name="gpt-3.5-turbo",
    num_queries=3
)

# Query the Multi-Query RAG system
response = mq_rag.query("What are the effects of climate change?")
print(response)

# You can also see the generated query variations
print("Generated queries:", mq_rag.last_generated_queries)
```

## Benefits of Multi-Query RAG
- Improved recall of relevant information
- Reduced dependency on perfect query formulation
- Better handling of ambiguous queries
- More comprehensive context for generation

## Limitations
- Increased computational cost (multiple retrievals)
- Potential for irrelevant information if query variations drift too far
- Requires careful prompt engineering for query generation

## References
- [Query Generation with Large Language Models](https://arxiv.org/abs/2305.15294)
- [Multi-Query Retrieval](https://github.com/langchain-ai/langchain/blob/master/docs/extras/use_cases/question_answering/multi-query-retriever.ipynb)
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) 