"""
Hypothetical Document Embeddings (HyDE) RAG Pipeline Implementation

This module implements a HyDE Retrieval-Augmented Generation (RAG) pipeline
that generates hypothetical documents to improve retrieval quality.
"""

import os
import argparse
from typing import List, Dict, Any, Optional

import numpy as np
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


class HyDERAG:
    """
    A HyDE RAG pipeline that generates hypothetical documents
    to improve retrieval quality.
    """

    def __init__(
        self,
        vector_store_path: Optional[str] = None,
        embedding_model: str = "text-embedding-ada-002",
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,  # Higher temperature for creative hypothetical docs
        top_k: int = 4,
    ):
        """
        Initialize the HyDE RAG pipeline.

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
        self.last_hypothetical_doc = ""
        
        # Load vector store if path is provided
        if vector_store_path and os.path.exists(vector_store_path):
            self.vector_store = FAISS.load_local(vector_store_path, self.embedding_model)
        else:
            self.vector_store = None
            
        # Define the prompt template for hypothetical document generation
        self.hyde_gen_template = PromptTemplate(
            input_variables=["question"],
            template="""
            You are an AI assistant that generates a hypothetical document or passage that would contain information to answer a given question.
            
            Question: {question}
            
            Generate a detailed, informative passage that would directly answer this question. 
            This passage should be written as if it were extracted from an authoritative document on the topic.
            Use a factual, neutral tone and include relevant details that would help answer the question comprehensively.
            The passage should be around 3-4 paragraphs in length.
            
            Hypothetical Document:
            """,
        )
        
        self.hyde_gen_chain = LLMChain(llm=self.llm, prompt=self.hyde_gen_template)
        
        # Define the prompt template for final generation
        self.generation_template = PromptTemplate(
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
        
        self.generation_chain = LLMChain(llm=self.llm, prompt=self.generation_template)

    def generate_hypothetical_document(self, query: str) -> str:
        """
        Generate a hypothetical document that would answer the query.

        Args:
            query: The user query.

        Returns:
            Hypothetical document text.
        """
        # Generate a hypothetical document using the LLM
        hypothetical_doc = self.hyde_gen_chain.run(question=query).strip()
        
        # Store for later reference
        self.last_hypothetical_doc = hypothetical_doc
        
        return hypothetical_doc

    def retrieve_with_hyde(self, hypothetical_doc: str) -> List[Document]:
        """
        Retrieve documents using the hypothetical document as the query.

        Args:
            hypothetical_doc: The hypothetical document.

        Returns:
            List of retrieved documents.
        """
        if self.vector_store is None:
            raise ValueError("No vector store available. Ingest documents first.")
            
        # Use the hypothetical document as the query for retrieval
        retrieved_docs = self.vector_store.similarity_search(hypothetical_doc, k=self.top_k)
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
        End-to-end HyDE RAG pipeline: generate hypothetical document, retrieve, and generate.

        Args:
            query: The user query.

        Returns:
            Dictionary containing the query, hypothetical document, retrieved documents, and generated response.
        """
        # Generate hypothetical document
        hypothetical_doc = self.generate_hypothetical_document(query)
        
        # Retrieve documents using the hypothetical document
        retrieved_docs = self.retrieve_with_hyde(hypothetical_doc)
        
        # Generate response
        response = self.generate(query, retrieved_docs)
        
        return {
            "query": query,
            "hypothetical_document": hypothetical_doc,
            "retrieved_documents": retrieved_docs,
            "response": response,
        }


def main():
    """
    Main function to run the HyDE RAG pipeline from the command line.
    """
    parser = argparse.ArgumentParser(description="Run the HyDE RAG pipeline")
    parser.add_argument("--query", type=str, required=True, help="The query to process")
    parser.add_argument(
        "--vector_store_path",
        type=str,
        default="../../../basic-rag/data/processed/vector_store",
        help="Path to the vector store",
    )
    args = parser.parse_args()

    # Initialize the HyDE RAG pipeline
    hyde_rag = HyDERAG(vector_store_path=args.vector_store_path)
    
    # Process the query
    result = hyde_rag.query(args.query)
    
    # Print the response
    print("\nQuery:", result["query"])
    
    print("\nHypothetical Document:")
    print(result["hypothetical_document"])
    
    print("\nRetrieved Documents:")
    for i, doc in enumerate(result["retrieved_documents"]):
        print(f"\nDocument {i+1}:")
        print(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
    
    print("\nResponse:")
    print(result["response"])


if __name__ == "__main__":
    main() 