"""
Basic RAG Pipeline Implementation

This module implements a simple Retrieval-Augmented Generation (RAG) pipeline
that demonstrates the core components of a RAG system.
"""

import os
import argparse
from typing import List, Dict, Any, Optional

import faiss
import numpy as np
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


class RAGPipeline:
    """
    A basic RAG pipeline that implements document retrieval and generation.
    """

    def __init__(
        self,
        vector_store_path: Optional[str] = None,
        embedding_model: str = "text-embedding-ada-002",
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        top_k: int = 4,
    ):
        """
        Initialize the RAG pipeline.

        Args:
            vector_store_path: Path to the FAISS vector store. If None, a new one will be created.
            embedding_model: Name of the embedding model to use.
            model_name: Name of the LLM to use.
            temperature: Temperature for the LLM.
            top_k: Number of documents to retrieve.
        """
        self.embedding_model = OpenAIEmbeddings(model=embedding_model)
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.top_k = top_k
        
        # Load vector store if path is provided
        if vector_store_path and os.path.exists(vector_store_path):
            self.vector_store = FAISS.load_local(vector_store_path, self.embedding_model)
        else:
            self.vector_store = None
            
        # Define the prompt template for generation
        self.prompt_template = PromptTemplate(
            input_variables=["question", "context"],
            template="""
            You are a helpful assistant that answers questions based on the provided context.
            
            Context:
            {context}
            
            Question:
            {question}
            
            Answer the question based on the context provided. If the answer cannot be found in the context, 
            say "I don't have enough information to answer this question." and suggest what additional 
            information would be needed to answer it accurately.
            """,
        )
        
        self.generation_chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def ingest_documents(
        self, documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> None:
        """
        Ingest documents into the vector store.

        Args:
            documents: List of documents to ingest.
            chunk_size: Size of each document chunk.
            chunk_overlap: Overlap between chunks.
        """
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create or update vector store
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(chunks, self.embedding_model)
        else:
            self.vector_store.add_documents(chunks)
            
        print(f"Ingested {len(chunks)} document chunks.")

    def save_vector_store(self, path: str) -> None:
        """
        Save the vector store to disk.

        Args:
            path: Path to save the vector store.
        """
        if self.vector_store is None:
            raise ValueError("No vector store to save. Ingest documents first.")
            
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.vector_store.save_local(path)
        print(f"Vector store saved to {path}")

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: The query to retrieve documents for.

        Returns:
            List of retrieved documents.
        """
        if self.vector_store is None:
            raise ValueError("No vector store available. Ingest documents first.")
            
        retrieved_docs = self.vector_store.similarity_search(query, k=self.top_k)
        return retrieved_docs

    def generate(self, query: str, context_docs: List[Document]) -> str:
        """
        Generate a response based on the query and retrieved documents.

        Args:
            query: The user query.
            context_docs: The retrieved context documents.

        Returns:
            Generated response.
        """
        # Combine document texts into a single context string
        context = "\n\n".join([doc.page_content for doc in context_docs])
        
        # Generate response
        response = self.generation_chain.run(question=query, context=context)
        return response

    def query(self, query: str) -> Dict[str, Any]:
        """
        End-to-end RAG pipeline: retrieve and generate.

        Args:
            query: The user query.

        Returns:
            Dictionary containing the query, retrieved documents, and generated response.
        """
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(query)
        
        # Generate response
        response = self.generate(query, retrieved_docs)
        
        return {
            "query": query,
            "retrieved_documents": retrieved_docs,
            "response": response,
        }


def main():
    """
    Main function to run the RAG pipeline from the command line.
    """
    parser = argparse.ArgumentParser(description="Run the RAG pipeline")
    parser.add_argument("--query", type=str, required=True, help="The query to process")
    parser.add_argument(
        "--vector_store_path",
        type=str,
        default="data/processed/vector_store",
        help="Path to the vector store",
    )
    args = parser.parse_args()

    # Initialize the RAG pipeline
    rag = RAGPipeline(vector_store_path=args.vector_store_path)
    
    # Process the query
    result = rag.query(args.query)
    
    # Print the response
    print("\nQuery:", result["query"])
    print("\nRetrieved Documents:")
    for i, doc in enumerate(result["retrieved_documents"]):
        print(f"\nDocument {i+1}:")
        print(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
    
    print("\nResponse:")
    print(result["response"])


if __name__ == "__main__":
    main() 