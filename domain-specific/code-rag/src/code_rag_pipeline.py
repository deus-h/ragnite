"""
Code RAG Pipeline

This module implements a specialized RAG pipeline for code-related queries.
It handles code-aware chunking, code-specific embeddings, and code-tailored generation.
"""

import os
import argparse
from typing import List, Dict, Any, Optional, Union

import numpy as np
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from code_chunker import split_code_file, create_code_chunker


class CodeRAG:
    """
    A RAG pipeline specialized for code-related queries.
    """

    def __init__(
        self,
        vector_store_path: Optional[str] = None,
        language: str = "python",
        embedding_model: str = "text-embedding-ada-002",
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        top_k: int = 4,
        min_chunk_size: int = 50,
        max_chunk_size: int = 1500,
    ):
        """
        Initialize the Code RAG pipeline.

        Args:
            vector_store_path: Path to the FAISS vector store. If None, a new one will be created.
            language: The programming language to focus on.
            embedding_model: Name of the embedding model to use.
            model_name: Name of the LLM to use.
            temperature: Temperature for the LLM.
            top_k: Number of documents to retrieve.
            min_chunk_size: Minimum chunk size for code chunking.
            max_chunk_size: Maximum chunk size for code chunking.
        """
        self.language = language.lower()
        self.embedding_model = OpenAIEmbeddings(model=embedding_model)
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.top_k = top_k
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        # Load vector store if path is provided
        if vector_store_path and os.path.exists(vector_store_path):
            self.vector_store = FAISS.load_local(vector_store_path, self.embedding_model)
        else:
            self.vector_store = None
            
        # Define the prompt template for code generation
        self.generation_template = PromptTemplate(
            input_variables=["question", "context", "language", "code_context"],
            template="""
            You are an expert software developer specializing in {language} programming.
            
            Relevant code context:
            {context}
            
            User's code (if provided):
            {code_context}
            
            Question:
            {question}
            
            Please provide a detailed, accurate, and helpful response. If the question asks for code, make sure the code is:
            1. Correct and follows best practices for {language}
            2. Well-commented and explained
            3. Efficient and maintainable
            4. Secure and handles edge cases
            
            If you're suggesting improvements to existing code, explain the rationale behind each change.
            If you're explaining concepts, provide concrete examples in {language}.
            
            Cite specific parts of the context that informed your answer when relevant.
            """,
        )
        
        self.generation_chain = LLMChain(llm=self.llm, prompt=self.generation_template)

    def ingest_code_file(self, file_path: str) -> None:
        """
        Ingest a code file into the vector store.

        Args:
            file_path: Path to the code file to ingest.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Split the code file into chunks
        chunks = split_code_file(
            file_path, 
            min_chunk_size=self.min_chunk_size, 
            max_chunk_size=self.max_chunk_size
        )
        
        print(f"Split {file_path} into {len(chunks)} chunks.")
        
        # Create or update vector store
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(chunks, self.embedding_model)
        else:
            self.vector_store.add_documents(chunks)

    def ingest_code_directory(self, directory_path: str, extensions: Optional[List[str]] = None) -> None:
        """
        Ingest all code files in a directory into the vector store.

        Args:
            directory_path: Path to the directory containing code files.
            extensions: List of file extensions to include (e.g., ['.py', '.js']).
                        If None, all files will be processed.
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Set default extensions based on language if not provided
        if extensions is None:
            if self.language == "python":
                extensions = ['.py']
            elif self.language == "javascript":
                extensions = ['.js', '.jsx']
            elif self.language == "typescript":
                extensions = ['.ts', '.tsx']
            else:
                extensions = []
        
        # Convert extensions to lowercase
        extensions = [ext.lower() for ext in extensions]
        
        # Walk through the directory and process matching files
        processed_files = 0
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_ext = os.path.splitext(file)[1].lower()
                if not extensions or file_ext in extensions:
                    file_path = os.path.join(root, file)
                    try:
                        self.ingest_code_file(file_path)
                        processed_files += 1
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
        
        print(f"Processed {processed_files} files from {directory_path}.")

    def save_vector_store(self, path: str) -> None:
        """
        Save the vector store to disk.

        Args:
            path: Path to save the vector store.
        """
        if self.vector_store is None:
            raise ValueError("No vector store to save. Ingest code files first.")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.vector_store.save_local(path)
        print(f"Vector store saved to {path}")

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve relevant code chunks for a query.

        Args:
            query: The query to retrieve code chunks for.

        Returns:
            List of retrieved code chunks as Document objects.
        """
        if self.vector_store is None:
            raise ValueError("No vector store available. Ingest code files first.")
        
        retrieved_docs = self.vector_store.similarity_search(query, k=self.top_k)
        return retrieved_docs

    def generate(
        self, 
        query: str, 
        context_docs: List[Document], 
        code_context: str = ""
    ) -> str:
        """
        Generate a response based on the query and retrieved code chunks.

        Args:
            query: The user query.
            context_docs: The retrieved context documents.
            code_context: Optional code snippet provided by the user for context.

        Returns:
            Generated response.
        """
        # Combine document texts into a single context string
        context = "\n\n".join([
            f"--- {doc.metadata.get('chunk_type', 'code')} from {doc.metadata.get('source', 'unknown')} ---\n"
            + (f"Class: {doc.metadata.get('class_name')}\n" if 'class_name' in doc.metadata else "")
            + (f"Function: {doc.metadata.get('function_name')}\n" if 'function_name' in doc.metadata else "")
            + (f"Method: {doc.metadata.get('method_name')}\n" if 'method_name' in doc.metadata else "")
            + "\n" + doc.page_content
            for doc in context_docs
        ])
        
        # Generate response
        response = self.generation_chain.run(
            question=query,
            context=context,
            language=self.language,
            code_context=code_context
        )
        
        return response

    def query(
        self, 
        query: str, 
        code_context: str = ""
    ) -> Dict[str, Any]:
        """
        End-to-end Code RAG pipeline: retrieve and generate.

        Args:
            query: The user query.
            code_context: Optional code snippet provided by the user for context.

        Returns:
            Dictionary containing the query, retrieved documents, and generated response.
        """
        # Retrieve relevant code chunks
        retrieved_docs = self.retrieve(query)
        
        # Generate response
        response = self.generate(query, retrieved_docs, code_context)
        
        return {
            "query": query,
            "code_context": code_context,
            "retrieved_documents": retrieved_docs,
            "response": response,
        }


def main():
    """
    Main function to run the Code RAG pipeline from the command line.
    """
    parser = argparse.ArgumentParser(description="Run the Code RAG pipeline")
    parser.add_argument("--query", type=str, required=True, help="The query to process")
    parser.add_argument("--language", type=str, default="python", help="The programming language")
    parser.add_argument(
        "--vector_store_path",
        type=str,
        default="data/processed/vector_store",
        help="Path to the vector store",
    )
    parser.add_argument(
        "--code_context",
        type=str,
        default="",
        help="Optional code context provided by the user",
    )
    parser.add_argument(
        "--ingest_file",
        type=str,
        help="Path to a code file to ingest before querying",
    )
    parser.add_argument(
        "--ingest_dir",
        type=str,
        help="Path to a directory containing code files to ingest before querying",
    )
    args = parser.parse_args()

    # Initialize the Code RAG pipeline
    code_rag = CodeRAG(
        vector_store_path=args.vector_store_path if os.path.exists(args.vector_store_path) else None,
        language=args.language
    )
    
    # Ingest code if specified
    if args.ingest_file:
        code_rag.ingest_code_file(args.ingest_file)
    
    if args.ingest_dir:
        code_rag.ingest_code_directory(args.ingest_dir)
    
    # Save the vector store if ingestion was performed
    if args.ingest_file or args.ingest_dir:
        code_rag.save_vector_store(args.vector_store_path)
    
    # Process the query
    result = code_rag.query(args.query, args.code_context)
    
    # Print the response
    print("\nQuery:", result["query"])
    
    if result["code_context"]:
        print("\nCode Context:")
        print(result["code_context"])
    
    print("\nRetrieved Documents:")
    for i, doc in enumerate(result["retrieved_documents"]):
        print(f"\nDocument {i+1}:")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Type: {doc.metadata.get('chunk_type', 'Unknown')}")
        if 'class_name' in doc.metadata:
            print(f"Class: {doc.metadata['class_name']}")
        if 'function_name' in doc.metadata:
            print(f"Function: {doc.metadata['function_name']}")
        if 'method_name' in doc.metadata:
            print(f"Method: {doc.metadata['method_name']}")
        
        # Print a preview of the content
        content_preview = "\n".join(doc.page_content.split("\n")[:5])
        if len(doc.page_content.split("\n")) > 5:
            content_preview += "\n..."
        print(f"\n{content_preview}")
    
    print("\nResponse:")
    print(result["response"])


if __name__ == "__main__":
    main() 