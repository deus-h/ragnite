"""
Medical RAG Pipeline

This module implements a specialized RAG pipeline for medical and healthcare applications.
It combines medical document chunking, biomedical embeddings, and medical fact verification
to provide accurate, evidence-based, and ethically sound responses.
"""

import os
import argparse
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from medical_chunker import MedicalChunker, process_medical_document
from fact_verifier import MedicalFactVerifier


class MedicalEntityRecognizer:
    """
    A component that recognizes medical entities in text and links them to standard ontologies.
    
    This is a simplified version for demonstration purposes. In a real implementation,
    this would use specialized biomedical NER models and ontology linking tools.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.0
    ):
        """
        Initialize the medical entity recognizer.
        
        Args:
            model_name: Name of the LLM to use.
            temperature: Temperature for the LLM.
        """
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        
        # Define the prompt template for entity recognition
        self.entity_recognition_template = PromptTemplate(
            input_variables=["text"],
            template="""
            You are a medical entity recognition system. Extract all medical entities from the text
            and link them to standard medical ontologies where possible.
            
            Text: {text}
            
            For each entity, provide:
            1. The entity text
            2. The entity type (e.g., disease, medication, procedure, etc.)
            3. The ontology ID if available (e.g., UMLS:C0123456, SNOMED:123456, RxNorm:123456)
            
            Format your response as a JSON-like structure:
            [
              {{
                "text": "entity text",
                "type": "entity type",
                "ontology": "ontology ID if available, otherwise null"
              }},
              ...
            ]
            
            Only provide this structured output, no additional text.
            """,
        )
        
        self.entity_recognition_chain = LLMChain(llm=self.llm, prompt=self.entity_recognition_template)
    
    def recognize_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Recognize medical entities in text.
        
        Args:
            text: The text to analyze.
            
        Returns:
            List of dictionaries containing entity information.
        """
        # This is a simplified implementation using an LLM
        # In a real system, this would use specialized biomedical NER models
        result = self.entity_recognition_chain.run(text=text)
        
        # Parse the result - in a real system, this would be more robust
        try:
            # Clean up the result to make it valid JSON
            result = result.strip()
            if result.startswith("```json"):
                result = result[7:]
            if result.endswith("```"):
                result = result[:-3]
            
            import json
            entities = json.loads(result)
            return entities
        except Exception as e:
            print(f"Error parsing entity recognition result: {e}")
            return []


class MedicalRAG:
    """
    A RAG pipeline specialized for medical and healthcare applications.
    """

    def __init__(
        self,
        vector_store_path: Optional[str] = None,
        embedding_model: str = "text-embedding-ada-002",
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        top_k: int = 5,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000,
        verify_facts: bool = True,
        entity_recognition: bool = True,
        confidence_threshold: float = 0.7,
        strict_mode: bool = False
    ):
        """
        Initialize the Medical RAG pipeline.

        Args:
            vector_store_path: Path to the FAISS vector store. If None, a new one will be created.
            embedding_model: Name of the embedding model to use.
            model_name: Name of the LLM to use.
            temperature: Temperature for the LLM.
            top_k: Number of documents to retrieve.
            min_chunk_size: Minimum chunk size for medical chunking.
            max_chunk_size: Maximum chunk size for medical chunking.
            verify_facts: Whether to verify medical facts in responses.
            entity_recognition: Whether to recognize medical entities in queries.
            confidence_threshold: Threshold for confidence score to determine if a claim is verifiable.
            strict_mode: If True, only include claims that can be verified with high confidence.
        """
        self.embedding_model = OpenAIEmbeddings(model=embedding_model)
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.top_k = top_k
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.verify_facts = verify_facts
        self.entity_recognition = entity_recognition
        
        # Initialize medical-specific components
        self.medical_chunker = MedicalChunker(min_chunk_size=min_chunk_size, max_chunk_size=max_chunk_size)
        self.fact_verifier = MedicalFactVerifier(
            model_name=model_name,
            temperature=temperature,
            confidence_threshold=confidence_threshold,
            strict_mode=strict_mode
        )
        self.entity_recognizer = MedicalEntityRecognizer(
            model_name=model_name,
            temperature=temperature
        ) if entity_recognition else None
        
        # Load vector store if path is provided
        if vector_store_path and os.path.exists(vector_store_path):
            self.vector_store = FAISS.load_local(vector_store_path, self.embedding_model)
        else:
            self.vector_store = None
            
        # Define the prompt template for medical generation
        self.generation_template = PromptTemplate(
            input_variables=["question", "context", "entities"],
            template="""
            You are a medical AI assistant providing evidence-based information. Your responses should be:
            1. Accurate and grounded in the medical literature
            2. Clear and understandable for the target audience
            3. Ethically sound, avoiding harmful or misleading information
            4. Properly cited when making specific claims
            5. Transparent about limitations and uncertainties
            
            Medical entities identified in the query:
            {entities}
            
            Relevant medical context:
            {context}
            
            Question:
            {question}
            
            Please provide a comprehensive, accurate response based on the provided context.
            
            If the context doesn't contain enough information to fully answer the question:
            1. Clearly state what information is missing
            2. Provide what information you can based on the available context
            3. Avoid making unsubstantiated claims beyond what's in the context
            
            Always distinguish clearly between:
            - Well-established medical facts with strong evidence
            - Emerging research with limited evidence
            - Areas of clinical uncertainty or ongoing debate
            
            When appropriate, indicate the level of evidence supporting key claims (e.g., from systematic reviews, RCTs, observational studies, expert opinion, etc.).
            """,
        )
        
        self.generation_chain = LLMChain(llm=self.llm, prompt=self.generation_template)
        
        # Define the prompt template for general medical knowledge fallback
        self.fallback_template = PromptTemplate(
            input_variables=["question", "entities"],
            template="""
            You are a medical AI assistant providing evidence-based information. Your responses should be:
            1. Accurate and grounded in the medical literature
            2. Clear and understandable for the target audience
            3. Ethically sound, avoiding harmful or misleading information
            4. Transparent about limitations and uncertainties
            
            Medical entities identified in the query:
            {entities}
            
            Question:
            {question}
            
            Please provide a response based on general medical knowledge, but note that:
            
            1. You should only provide information that is well-established in medical literature
            2. You should acknowledge the limitations of not having specific reference materials for this query
            3. You should clearly indicate when you're providing general information versus specific recommendations
            4. You should encourage consultation with healthcare providers for clinical questions
            
            For any specific medical claims, indicate what level of evidence typically supports such information.
            """,
        )
        
        self.fallback_chain = LLMChain(llm=self.llm, prompt=self.fallback_template)

    def ingest_medical_document(self, file_path: str) -> None:
        """
        Ingest a medical document into the vector store.

        Args:
            file_path: Path to the medical document to ingest.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Process the medical document using the medical chunker
        chunks = process_medical_document(
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

    def ingest_medical_directory(self, directory_path: str, extensions: Optional[List[str]] = None) -> None:
        """
        Ingest all medical documents in a directory into the vector store.

        Args:
            directory_path: Path to the directory containing medical documents.
            extensions: List of file extensions to include (e.g., ['.pdf', '.txt']).
                        If None, common medical document extensions will be used.
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Set default extensions if not provided
        if extensions is None:
            extensions = ['.pdf', '.txt', '.md', '.html', '.xml']
        
        # Convert extensions to lowercase
        extensions = [ext.lower() if not ext.startswith('.') else ext.lower() for ext in extensions]
        extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in extensions]
        
        # Process all matching files
        processed_files = 0
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in extensions:
                    file_path = os.path.join(root, file)
                    try:
                        self.ingest_medical_document(file_path)
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
            raise ValueError("No vector store to save. Ingest medical documents first.")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.vector_store.save_local(path)
        print(f"Vector store saved to {path}")

    def retrieve(self, query: str, identified_entities: List[Dict[str, Any]] = None) -> List[Document]:
        """
        Retrieve relevant medical documents for a query.

        Args:
            query: The query to retrieve documents for.
            identified_entities: Optional list of medical entities identified in the query.

        Returns:
            List of retrieved documents.
        """
        if self.vector_store is None:
            raise ValueError("No vector store available. Ingest medical documents first.")
        
        # Enhance retrieval with entity information if available
        if identified_entities and len(identified_entities) > 0:
            # Extract entity texts
            entity_texts = [entity["text"] for entity in identified_entities]
            
            # Create an enhanced query that emphasizes the medical entities
            enhanced_query = f"{query} {' '.join(entity_texts)}"
            
            # Retrieve documents using the enhanced query
            retrieved_docs = self.vector_store.similarity_search(enhanced_query, k=self.top_k)
        else:
            # Standard retrieval without entity enhancement
            retrieved_docs = self.vector_store.similarity_search(query, k=self.top_k)
            
        return retrieved_docs

    def generate(
        self, 
        query: str, 
        context_docs: List[Document],
        identified_entities: List[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a response based on the query and retrieved medical documents.

        Args:
            query: The user query.
            context_docs: The retrieved context documents.
            identified_entities: Optional list of medical entities identified in the query.

        Returns:
            Generated response.
        """
        # Format entity information
        entity_text = "No specific medical entities identified." if not identified_entities else "\n".join([
            f"- {entity['text']} (Type: {entity['type']}" + 
            (f", Ontology ID: {entity['ontology']})" if entity.get('ontology') else ")")
            for entity in identified_entities
        ])
        
        # Combine document texts into a single context string
        context = "\n\n".join([
            f"--- {doc.metadata.get('chunk_type', 'section')} from {doc.metadata.get('source', 'unknown')} ---\n" +
            (f"Section: {doc.metadata.get('section_name', 'Unknown')}\n" if 'section_name' in doc.metadata else "") +
            doc.page_content 
            for doc in context_docs
        ])
        
        # Generate response
        if context_docs:
            # Use the context for generation
            response = self.generation_chain.run(
                question=query,
                context=context,
                entities=entity_text
            )
        else:
            # Fallback to general medical knowledge
            response = self.fallback_chain.run(
                question=query,
                entities=entity_text
            )
        
        # Verify medical facts if enabled
        if self.verify_facts and context_docs:
            verification_result = self.fact_verifier.verify_text(response, context_docs)
            response = self.fact_verifier.generate_cited_text(verification_result)
            
            # Add a disclaimer when facts couldn't be fully verified
            if verification_result["unverified_claims"] > 0 or verification_result["modified_claims"] > 0:
                disclaimer = "\n\n[Note: Some medical claims in this response could not be fully verified with the available context. Please consult healthcare professionals for medical advice.]"
                response += disclaimer
        
        return response

    def query(self, query: str) -> Dict[str, Any]:
        """
        End-to-end Medical RAG pipeline: entity recognition, retrieval, generation, and fact verification.

        Args:
            query: The user query.

        Returns:
            Dictionary containing the query, identified entities, retrieved documents,
            generated response, and fact verification results.
        """
        # Initialize results
        result = {
            "query": query,
            "identified_entities": [],
            "retrieved_documents": [],
            "response": "",
            "citations": [],
            "confidence_score": 0.0
        }
        
        # Entity recognition
        if self.entity_recognition and self.entity_recognizer:
            identified_entities = self.entity_recognizer.recognize_entities(query)
            result["identified_entities"] = identified_entities
        else:
            identified_entities = []
        
        try:
            # Retrieve relevant documents
            retrieved_docs = self.retrieve(query, identified_entities)
            result["retrieved_documents"] = retrieved_docs
            
            # Generate response
            response = self.generate(query, retrieved_docs, identified_entities)
            result["response"] = response
            
            # Extract citations
            if self.verify_facts:
                citations = []
                citation_pattern = r'\[(.*?)\]'
                import re
                for match in re.finditer(citation_pattern, response):
                    citation = match.group(1)
                    if citation and not citation.startswith("Note:") and not citation.startswith("Corrected"):
                        citations.append(citation)
                
                result["citations"] = list(set(citations))  # Deduplicate
                
                # Use the fact verifier's overall confidence score if available
                verification_metrics = getattr(self.fact_verifier, "verified_claims", 0) + getattr(self.fact_verifier, "unverified_claims", 0)
                if verification_metrics > 0:
                    confidence = getattr(self.fact_verifier, "verified_claims", 0) / verification_metrics
                    result["confidence_score"] = confidence
        except Exception as e:
            # Fallback to general medical knowledge if retrieval/generation fails
            print(f"Error in retrieval/generation: {e}. Falling back to general knowledge.")
            
            response = self.fallback_chain.run(
                question=query,
                entities="\n".join([f"- {entity['text']} (Type: {entity['type']})" for entity in identified_entities]) 
                if identified_entities else "No specific medical entities identified."
            )
            
            result["response"] = response + "\n\n[Note: This response is based on general medical knowledge and not on specific medical literature retrieval.]"
        
        return result


def main():
    """
    Main function to run the Medical RAG pipeline from the command line.
    """
    parser = argparse.ArgumentParser(description="Run the Medical RAG pipeline")
    parser.add_argument("--query", type=str, required=True, help="The medical query to process")
    parser.add_argument(
        "--vector_store_path",
        type=str,
        default="../data/processed/medical_vector_store",
        help="Path to the vector store",
    )
    parser.add_argument(
        "--verify_facts",
        action="store_true",
        help="Whether to verify medical facts in the response",
    )
    parser.add_argument(
        "--no_entity_recognition",
        action="store_true",
        help="Disable medical entity recognition",
    )
    parser.add_argument(
        "--ingest_file",
        type=str,
        help="Path to a medical document to ingest before querying",
    )
    parser.add_argument(
        "--ingest_dir",
        type=str,
        help="Path to a directory containing medical documents to ingest before querying",
    )
    args = parser.parse_args()

    # Initialize the Medical RAG pipeline
    medical_rag = MedicalRAG(
        vector_store_path=args.vector_store_path if os.path.exists(args.vector_store_path) else None,
        verify_facts=args.verify_facts,
        entity_recognition=not args.no_entity_recognition
    )
    
    # Ingest medical documents if specified
    if args.ingest_file:
        medical_rag.ingest_medical_document(args.ingest_file)
    
    if args.ingest_dir:
        medical_rag.ingest_medical_directory(args.ingest_dir)
    
    # Save the vector store if ingestion was performed
    if args.ingest_file or args.ingest_dir:
        medical_rag.save_vector_store(args.vector_store_path)
    
    # Process the query
    result = medical_rag.query(args.query)
    
    # Print the response
    print("\nQuery:", result["query"])
    
    if result["identified_entities"]:
        print("\nIdentified Medical Entities:")
        for entity in result["identified_entities"]:
            print(f"- {entity['text']} (Type: {entity['type']}" + 
                  (f", Ontology ID: {entity['ontology']})" if entity.get('ontology') else ")"))
    
    if result["retrieved_documents"]:
        print("\nRetrieved Medical Documents:")
        for i, doc in enumerate(result["retrieved_documents"]):
            print(f"\nDocument {i+1}:")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"Type: {doc.metadata.get('chunk_type', 'Unknown')}")
            if 'section_name' in doc.metadata:
                print(f"Section: {doc.metadata['section_name']}")
            
            # Print a preview of the content
            content_preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
            print(f"\n{content_preview}")
    
    print("\nResponse:")
    print(result["response"])
    
    if result["citations"]:
        print("\nCitations:")
        for citation in result["citations"]:
            print(f"- {citation}")
    
    print(f"\nConfidence Score: {result['confidence_score']:.2f}")


if __name__ == "__main__":
    main() 