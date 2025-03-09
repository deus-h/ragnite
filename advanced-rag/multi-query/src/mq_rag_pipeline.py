"""
Multi-Query RAG Pipeline Implementation

This module implements a Multi-Query Retrieval-Augmented Generation (RAG) pipeline
that generates multiple query variations to improve retrieval quality.
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


class MultiQueryRAG:
    """
    A Multi-Query RAG pipeline that generates multiple query variations
    to improve retrieval quality.
    """

    def __init__(
        self,
        vector_store_path: Optional[str] = None,
        embedding_model: str = "text-embedding-ada-002",
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        top_k: int = 3,
        num_queries: int = 3,
    ):
        """
        Initialize the Multi-Query RAG pipeline.

        Args:
            vector_store_path: Path to the FAISS vector store. If None, a new one will be created.
            embedding_model: Name of the embedding model to use.
            model_name: Name of the LLM to use.
            temperature: Temperature for the LLM.
            top_k: Number of documents to retrieve per query.
            num_queries: Number of query variations to generate.
        """
        self.embedding_model = OpenAIEmbeddings(model=embedding_model)
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.top_k = top_k
        self.num_queries = num_queries
        self.last_generated_queries = []
        
        # Load vector store if path is provided
        if vector_store_path and os.path.exists(vector_store_path):
            self.vector_store = FAISS.load_local(vector_store_path, self.embedding_model)
        else:
            self.vector_store = None
            
        # Define the prompt template for query generation
        self.query_gen_template = PromptTemplate(
            input_variables=["question"],
            template="""
            You are an AI assistant helping to generate different versions of a search query to improve retrieval.
            
            Original question: {question}
            
            Please generate {num_queries} different versions of this question. The versions should:
            1. Rephrase the question using different words and sentence structures
            2. Include different but related terms that might appear in relevant documents
            3. Focus on different aspects of the question
            4. Vary in specificity (some more general, some more specific)
            
            Return ONLY the list of questions, one per line, with no additional text or explanations.
            """.replace("{num_queries}", str(num_queries)),
        )
        
        self.query_gen_chain = LLMChain(llm=self.llm, prompt=self.query_gen_template)
        
        # Define the prompt template for generation
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

    def generate_query_variations(self, query: str) -> List[str]:
        """
        Generate variations of the original query.

        Args:
            query: The original query.

        Returns:
            List of query variations.
        """
        # Generate query variations using the LLM
        response = self.query_gen_chain.run(question=query)
        
        # Parse the response into a list of queries
        query_variations = [q.strip() for q in response.strip().split("\n") if q.strip()]
        
        # Always include the original query
        if query not in query_variations:
            query_variations.insert(0, query)
            
        # Limit to the specified number of queries
        query_variations = query_variations[:self.num_queries]
        
        # Store for later reference
        self.last_generated_queries = query_variations
        
        return query_variations

    def retrieve_multi_query(self, query_variations: List[str]) -> List[Document]:
        """
        Retrieve documents using multiple query variations.

        Args:
            query_variations: List of query variations.

        Returns:
            List of unique retrieved documents.
        """
        if self.vector_store is None:
            raise ValueError("No vector store available. Ingest documents first.")
        
        all_docs = []
        doc_ids = set()  # To track unique documents
        
        # Retrieve documents for each query variation
        for query in query_variations:
            docs = self.vector_store.similarity_search(query, k=self.top_k)
            
            # Add only unique documents
            for doc in docs:
                # Create a unique identifier for the document
                doc_id = hash(doc.page_content)
                
                if doc_id not in doc_ids:
                    all_docs.append(doc)
                    doc_ids.add(doc_id)
        
        return all_docs

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
        End-to-end Multi-Query RAG pipeline: generate query variations, retrieve, and generate.

        Args:
            query: The user query.

        Returns:
            Dictionary containing the query, query variations, retrieved documents, and generated response.
        """
        # Generate query variations
        query_variations = self.generate_query_variations(query)
        
        # Retrieve documents using multiple queries
        retrieved_docs = self.retrieve_multi_query(query_variations)
        
        # Generate response
        response = self.generate(query, retrieved_docs)
        
        return {
            "query": query,
            "query_variations": query_variations,
            "retrieved_documents": retrieved_docs,
            "response": response,
        }


def main():
    """
    Main function to run the Multi-Query RAG pipeline from the command line.
    """
    parser = argparse.ArgumentParser(description="Run the Multi-Query RAG pipeline")
    parser.add_argument("--query", type=str, required=True, help="The query to process")
    parser.add_argument(
        "--vector_store_path",
        type=str,
        default="../../basic-rag/data/processed/vector_store",
        help="Path to the vector store",
    )
    parser.add_argument(
        "--num_queries",
        type=int,
        default=3,
        help="Number of query variations to generate",
    )
    args = parser.parse_args()

    # Initialize the Multi-Query RAG pipeline
    mq_rag = MultiQueryRAG(
        vector_store_path=args.vector_store_path,
        num_queries=args.num_queries
    )
    
    # Process the query
    result = mq_rag.query(args.query)
    
    # Print the response
    print("\nOriginal Query:", result["query"])
    
    print("\nQuery Variations:")
    for i, query in enumerate(result["query_variations"]):
        print(f"{i+1}. {query}")
    
    print("\nRetrieved Documents:")
    for i, doc in enumerate(result["retrieved_documents"]):
        print(f"\nDocument {i+1}:")
        print(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
    
    print("\nResponse:")
    print(result["response"])


if __name__ == "__main__":
    main() 