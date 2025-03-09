# Basic RAG Implementation

## Overview
This folder contains a simple implementation of the Retrieval-Augmented Generation (RAG) pattern. This serves as a foundation for understanding more complex RAG systems.

## What is Basic RAG?
Basic RAG follows a straightforward pipeline:
1. **Document Ingestion**: Process and chunk documents into manageable pieces
2. **Embedding Generation**: Convert text chunks into vector embeddings
3. **Vector Storage**: Store embeddings in a vector database
4. **Query Processing**: Convert user queries into the same embedding space
5. **Retrieval**: Find the most relevant document chunks based on embedding similarity
6. **Augmented Generation**: Feed retrieved context along with the query to an LLM for response generation

## Implementation Details
Our implementation uses:
- **Embedding Model**: OpenAI's text-embedding-ada-002
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **LLM**: OpenAI's GPT-3.5-turbo
- **Document Processing**: Langchain's text splitters

## Project Structure
```
basic-rag/
├── src/
│   ├── ingest.py          # Document ingestion and embedding
│   ├── retrieve.py        # Vector similarity search
│   ├── generate.py        # LLM response generation
│   └── rag_pipeline.py    # End-to-end RAG pipeline
├── data/
│   ├── raw/               # Raw documents
│   └── processed/         # Processed chunks and metadata
├── examples/              # Example usage scripts
├── requirements.txt       # Dependencies
└── README.md              # This file
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

3. Run the document ingestion process:
   ```
   python src/ingest.py --data_dir data/raw
   ```

4. Run a query through the RAG pipeline:
   ```
   python src/rag_pipeline.py --query "Your question here"
   ```

## Example
```python
from rag_pipeline import RAGPipeline

# Initialize the RAG pipeline
rag = RAGPipeline(
    vector_store_path="data/processed/vector_store",
    model_name="gpt-3.5-turbo"
)

# Query the RAG system
response = rag.query("What is the capital of France?")
print(response)
```

## Limitations of Basic RAG
- No query reformulation or expansion
- Limited to single-hop retrieval
- No re-ranking of retrieved documents
- Potential for hallucinations when retrieved context is insufficient

## Next Steps
After understanding this basic implementation, explore the advanced RAG techniques in the other folders of this repository. 