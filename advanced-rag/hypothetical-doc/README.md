# Hypothetical Document Embeddings (HyDE)

## Overview
This folder contains an implementation of Hypothetical Document Embeddings (HyDE), an advanced RAG technique that improves retrieval by generating a hypothetical answer document before performing retrieval.

## What is HyDE?
HyDE addresses the "lexical gap" problem in information retrieval - the mismatch between query terms and relevant document terms. The approach works as follows:

1. Take a user query
2. Use an LLM to generate a hypothetical document that would answer the query
3. Embed this hypothetical document (not the original query)
4. Use this embedding to retrieve similar real documents
5. Feed the retrieved real documents back to the LLM for final answer generation

This technique leverages the LLM's knowledge to bridge the gap between query and document vocabulary, improving retrieval performance.

## Implementation Details
Our implementation uses:
- **Hypothetical Document Generation**: GPT-3.5-turbo
- **Embedding Model**: OpenAI's text-embedding-ada-002
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Final Generation**: GPT-3.5-turbo

## Project Structure
```
hypothetical-doc/
├── src/
│   ├── hyde_generator.py     # Generate hypothetical documents
│   ├── retriever.py          # Embedding-based retrieval
│   └── hyde_rag_pipeline.py  # End-to-end HyDE RAG pipeline
├── examples/                 # Example usage scripts
├── requirements.txt          # Dependencies
└── README.md                 # This file
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

3. Run a query through the HyDE RAG pipeline:
   ```
   python src/hyde_rag_pipeline.py --query "Your question here"
   ```

## Example
```python
from hyde_rag_pipeline import HyDERAG

# Initialize the HyDE RAG pipeline
hyde_rag = HyDERAG(
    vector_store_path="../basic-rag/data/processed/vector_store",
    model_name="gpt-3.5-turbo"
)

# Query the HyDE RAG system
response = hyde_rag.query("What are the effects of climate change?")
print(response)

# You can also see the generated hypothetical document
print("Hypothetical document:", hyde_rag.last_hypothetical_doc)
```

## Benefits of HyDE
- Better handling of semantic search by bridging the lexical gap
- Improved retrieval for complex or abstract queries
- Leverages LLM knowledge to guide retrieval
- Can work with existing embedding models and vector stores

## Limitations
- Computational overhead of generating hypothetical documents
- Quality depends on the LLM's ability to generate relevant hypothetical content
- May introduce biases from the LLM's training data

## References
- [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496)
- [HyDE: Hypothetical Document Embeddings](https://github.com/langchain-ai/langchain/blob/master/docs/extras/use_cases/question_answering/hyde.ipynb)
- [Improving Passage Retrieval with Zero-Shot Question Generation](https://arxiv.org/abs/2204.07496) 