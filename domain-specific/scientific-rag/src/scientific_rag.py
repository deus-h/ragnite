"""
Scientific RAG: Main implementation

This module provides the main ScientificRAG class that integrates all components
of the scientific domain-specific RAG system.
"""

import os
import json
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from .chunking import ScientificPaperChunker, SectionSplitter, MathFormulaHandler


class ScientificRAG:
    """
    Scientific RAG: A domain-specific RAG system for scientific research.
    
    This class provides the main interface for using the Scientific RAG system,
    integrating document processing, embedding, retrieval, and generation components.
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        use_openai: bool = False,
        openai_api_key: Optional[str] = None,
        llm_model: str = "gpt-3.5-turbo",
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
        handle_math: bool = True,
        db_directory: str = "./scientific_rag_db",
    ):
        """
        Initialize the Scientific RAG system.
        
        Args:
            embedding_model: Embedding model to use (local HF model or "openai")
            use_openai: Whether to use OpenAI for embeddings
            openai_api_key: OpenAI API key (required if use_openai is True)
            llm_model: LLM model to use for generation
            chunk_size: Maximum chunk size for document splitting
            chunk_overlap: Overlap between chunks
            handle_math: Whether to handle mathematical content specially
            db_directory: Directory to store the vector database
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.handle_math = handle_math
        self.db_directory = db_directory
        
        # Initialize embeddings
        if use_openai:
            if not openai_api_key:
                openai_api_key = os.environ.get("OPENAI_API_KEY")
                if not openai_api_key:
                    raise ValueError("OpenAI API key is required when use_openai is True")
            
            self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        else:
            self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Initialize the chunker for scientific papers
        self.chunker = ScientificPaperChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            handle_math=handle_math,
        )
        
        # Initialize math formula handler if needed
        if handle_math:
            self.math_handler = MathFormulaHandler(
                preserve_original=True,
                convert_to_unicode=True,
            )
        
        # Initialize LLM for generation
        if use_openai:
            self.llm = ChatOpenAI(model_name=llm_model, temperature=0.1, openai_api_key=openai_api_key)
        else:
            # Placeholder for local LLM support
            # In a full implementation, we would support local models
            self.llm = None
            print("WARNING: Local LLM not implemented yet, using OpenAI is recommended")
        
        # Initialize vector store if already exists
        self.vectorstore = None
        if os.path.exists(db_directory):
            try:
                self.vectorstore = Chroma(
                    persist_directory=db_directory,
                    embedding_function=self.embeddings,
                )
                print(f"Loaded existing vector database from {db_directory}")
            except Exception as e:
                print(f"Failed to load vector database: {str(e)}")
    
    def ingest_document(self, file_path: str) -> int:
        """
        Ingest a single scientific document into the RAG system.
        
        Args:
            file_path: Path to the document (PDF)
            
        Returns:
            Number of chunks ingested
        """
        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Process the document
        chunks = self.chunker.chunk_document(file_path)
        
        # Handle math formulas if needed
        if self.handle_math:
            for chunk in chunks:
                chunk["text"] = self.math_handler.process_text(chunk["text"])
        
        # Extract text and metadata
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        # Initialize vector store if needed
        if self.vectorstore is None:
            os.makedirs(self.db_directory, exist_ok=True)
            self.vectorstore = Chroma.from_texts(
                texts=texts,
                metadatas=metadatas,
                embedding=self.embeddings,
                persist_directory=self.db_directory,
            )
            self.vectorstore.persist()
        else:
            # Add documents to existing vector store
            self.vectorstore.add_texts(texts=texts, metadatas=metadatas)
            self.vectorstore.persist()
        
        return len(chunks)
    
    def ingest_documents(self, directory_path: str, file_extension: str = ".pdf") -> Dict[str, int]:
        """
        Ingest all documents in a directory into the RAG system.
        
        Args:
            directory_path: Path to the directory containing documents
            file_extension: File extension to filter by (default: .pdf)
            
        Returns:
            Dictionary mapping file paths to number of chunks ingested
        """
        results = {}
        
        # Check if the directory exists
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Get all files with the specified extension
        file_paths = [
            os.path.join(directory_path, filename)
            for filename in os.listdir(directory_path)
            if filename.endswith(file_extension)
        ]
        
        # Ingest each document
        for file_path in file_paths:
            try:
                num_chunks = self.ingest_document(file_path)
                results[file_path] = num_chunks
                print(f"Ingested {file_path}: {num_chunks} chunks")
            except Exception as e:
                print(f"Error ingesting {file_path}: {str(e)}")
                results[file_path] = 0
        
        return results
    
    def _create_scientific_prompt(self) -> PromptTemplate:
        """
        Create a specialized prompt template for scientific queries.
        
        Returns:
            PromptTemplate for scientific queries
        """
        template = """
        You are a scientific research assistant with expertise across multiple scientific fields.
        Use the following pieces of scientific literature to answer the question. 
        
        If you don't know the answer, just say you don't know. Don't try to make up an answer.
        Always indicate the source of your information and citation when possible.
        If the retrieved scientific information contains mathematical formulas, explain them clearly.
        
        Be precise and scientific in your language. Define specialized terms if they're likely unfamiliar.
        Indicate levels of scientific consensus and evidence quality (e.g., "well-established", 
        "emerging research suggests", "limited evidence indicates").
        
        Scientific Information:
        {context}
        
        Question: {question}
        
        Scientific Answer:
        """
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"],
        )
    
    def query(
        self,
        query: str,
        num_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Query the Scientific RAG system.
        
        Args:
            query: Scientific question or query
            num_results: Number of relevant document chunks to retrieve
            filter_metadata: Optional metadata filter for retrieval
            
        Returns:
            Dictionary containing the answer and sources
        """
        # Check if vector store exists
        if self.vectorstore is None:
            return {
                "answer": "No documents have been ingested yet. Please ingest scientific documents first.",
                "sources": [],
            }
        
        # Create the prompt
        prompt = self._create_scientific_prompt()
        
        # Create a retriever
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": num_results, "filter": filter_metadata}
        )
        
        # Create a QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
        )
        
        # Run the query
        response = qa_chain({"query": query})
        
        # Extract the answer and sources
        answer = response.get("result", "")
        source_documents = response.get("source_documents", [])
        
        # Format the sources
        sources = []
        for doc in source_documents:
            metadata = doc.metadata.copy()
            content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            sources.append({
                "content": content,
                "metadata": metadata,
            })
        
        return {
            "answer": answer,
            "sources": sources,
        }
    
    def query_by_section(
        self,
        query: str,
        section_name: str,
        num_results: int = 5,
    ) -> Dict[str, Any]:
        """
        Query the Scientific RAG system filtering by section.
        
        Args:
            query: Scientific question or query
            section_name: Section name to filter by (e.g., "methods", "results")
            num_results: Number of relevant document chunks to retrieve
            
        Returns:
            Dictionary containing the answer and sources
        """
        # Create metadata filter for the specified section
        filter_metadata = {"section": section_name}
        
        # Use the regular query method with the filter
        return self.query(query, num_results, filter_metadata)
    
    def save_ingestion_stats(self, output_file: str = "ingestion_stats.json") -> None:
        """
        Save statistics about the ingested documents.
        
        Args:
            output_file: Path to the output JSON file
            
        Returns:
            None
        """
        if self.vectorstore is None:
            print("No documents have been ingested yet.")
            return
        
        # Get all documents in the vector store
        if hasattr(self.vectorstore, "_collection"):
            collection = self.vectorstore._collection
            count = collection.count()
            metadatas = collection.get(include=["metadatas"])["metadatas"]
            
            # Analyze the metadatas
            stats = {
                "total_chunks": count,
                "documents": {},
                "sections": {},
            }
            
            for metadata in metadatas:
                source = metadata.get("source", "unknown")
                section = metadata.get("section", "unknown")
                
                # Count by document
                if source not in stats["documents"]:
                    stats["documents"][source] = 0
                stats["documents"][source] += 1
                
                # Count by section
                if section not in stats["sections"]:
                    stats["sections"][section] = 0
                stats["sections"][section] += 1
            
            # Save to file
            with open(output_file, "w") as f:
                json.dump(stats, f, indent=2)
                
            print(f"Ingestion statistics saved to {output_file}")
        else:
            print("Unable to extract statistics from the current vector store implementation.") 