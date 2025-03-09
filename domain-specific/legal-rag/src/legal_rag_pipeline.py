"""
Legal RAG Pipeline

This module provides the main pipeline for Legal RAG, integrating legal document chunking,
citation extraction, retrieval, and authority verification.
"""

import os
import json
from typing import List, Dict, Any, Optional, Union, Tuple
import logging

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.docstore.document import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

from legal_chunker import LegalChunker, process_legal_document
from citation_extractor import CitationExtractor
from authority_verifier import LegalAuthorityVerifier

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LegalRAG:
    """
    Legal RAG pipeline that integrates specialized components for legal research
    and document analysis.
    """

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        embedding_model: str = "text-embedding-ada-002",
        temperature: float = 0.0,
        vector_store_path: Optional[str] = None,
        verify_authority: bool = True,
        strict_mode: bool = False,
        jurisdiction_filter: Optional[str] = None
    ):
        """
        Initialize the Legal RAG pipeline.

        Args:
            model_name (str): The name of the LLM to use
            embedding_model (str): The name of the embedding model
            temperature (float): Temperature setting for the LLM
            vector_store_path (str, optional): Path to save/load the vector store
            verify_authority (bool): Whether to verify legal authority
            strict_mode (bool): If True, applies stricter verification standards
            jurisdiction_filter (str, optional): Filter results by jurisdiction
        """
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.vector_store_path = vector_store_path
        self.verify_authority = verify_authority
        self.strict_mode = strict_mode
        self.jurisdiction_filter = jurisdiction_filter
        
        # Initialize components
        self.chunker = LegalChunker()
        self.citation_extractor = CitationExtractor()
        self.authority_verifier = LegalAuthorityVerifier(
            model_name=model_name,
            temperature=temperature,
            strict_mode=strict_mode
        )
        
        # Initialize LLM
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        
        # Initialize vector store if path is provided
        self.vector_store = None
        if vector_store_path and os.path.exists(vector_store_path):
            try:
                self.vector_store = FAISS.load_local(vector_store_path, self.embeddings)
                logger.info(f"Loaded vector store from {vector_store_path}")
            except Exception as e:
                logger.error(f"Failed to load vector store: {str(e)}")
                
        # Legal-specific prompts
        self.legal_qa_prompt = ChatPromptTemplate.from_template(
            """You are a legal research assistant with expertise in legal analysis and research.
            
            QUESTION:
            {question}
            
            CONTEXT:
            {context}
            
            JURISDICTION:
            {jurisdiction}
            
            Please provide a comprehensive legal analysis based on the provided context.
            Your response should:
            1. Directly address the question
            2. Cite relevant legal authorities from the context
            3. Distinguish between binding and persuasive authority
            4. Note any jurisdictional limitations
            5. Identify any areas where the law is unsettled or where authorities conflict
            6. Include proper legal citations in your response
            
            If the context doesn't provide sufficient information to fully answer the question,
            acknowledge the limitations of your response.
            
            RESPONSE:"""
        )
        
        self.legal_summarization_prompt = ChatPromptTemplate.from_template(
            """You are a legal expert tasked with summarizing legal information.
            
            LEGAL CONTENT:
            {content}
            
            Please provide a concise yet comprehensive summary of this legal content.
            Your summary should:
            1. Identify the key legal principles or holdings
            2. Preserve important citations and authorities
            3. Maintain legal precision and accuracy
            4. Organize information logically (e.g., facts, issues, holdings)
            5. Retain the legal significance of the original content
            
            SUMMARY:"""
        )

    def ingest_document(self, file_path: str) -> List[str]:
        """
        Process and ingest a legal document into the vector store.

        Args:
            file_path (str): Path to the legal document

        Returns:
            List[str]: IDs of the ingested document chunks
        """
        # Process the document
        try:
            chunks = process_legal_document(file_path)
            logger.info(f"Processed {file_path} into {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Failed to process document {file_path}: {str(e)}")
            return []
            
        # Extract and standardize citations in each chunk
        for chunk in chunks:
            text = chunk.page_content
            
            # Extract citations
            citations = self.citation_extractor.extract_citations(text)
            
            # Add citation metadata
            if citations:
                chunk.metadata["citations"] = [
                    {
                        "text": citation["citation"],
                        "standardized": self.citation_extractor.standardize_citation(citation),
                        "type": citation["type"]
                    }
                    for citation in citations
                ]
                
                # Add citation metadata for improved retrieval
                citation_metadata = {}
                for citation in citations:
                    metadata = self.citation_extractor.get_citation_metadata(citation)
                    for key, value in metadata.items():
                        if key not in citation_metadata:
                            citation_metadata[key] = []
                        if value not in citation_metadata[key]:
                            citation_metadata[key].append(value)
                
                # Add to chunk metadata
                for key, values in citation_metadata.items():
                    chunk.metadata[f"citation_{key}"] = values
            
        # Create or update vector store
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            logger.info(f"Created new vector store with {len(chunks)} chunks")
        else:
            self.vector_store.add_documents(chunks)
            logger.info(f"Added {len(chunks)} chunks to existing vector store")
            
        # Save vector store if path is provided
        if self.vector_store_path:
            os.makedirs(os.path.dirname(self.vector_store_path), exist_ok=True)
            self.vector_store.save_local(self.vector_store_path)
            logger.info(f"Saved vector store to {self.vector_store_path}")
            
        # Return document IDs
        return [str(i) for i in range(len(chunks))]

    def ingest_directory(self, directory_path: str) -> Dict[str, List[str]]:
        """
        Process and ingest all legal documents in a directory.

        Args:
            directory_path (str): Path to the directory containing legal documents

        Returns:
            Dict[str, List[str]]: Mapping of file paths to their chunk IDs
        """
        results = {}
        
        for root, _, files in os.walk(directory_path):
            for file in files:
                # Skip hidden files and non-document files
                if file.startswith('.'):
                    continue
                    
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                
                # Process only document files
                if file_ext in ['.pdf', '.docx', '.txt', '.md', '.rtf']:
                    chunk_ids = self.ingest_document(file_path)
                    results[file_path] = chunk_ids
                    
        return results

    def retrieve(
        self, 
        query: str, 
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        use_citation_retrieval: bool = True
    ) -> List[Document]:
        """
        Retrieve relevant legal documents for a query.

        Args:
            query (str): The query string
            k (int): Number of documents to retrieve
            filter_metadata (Dict[str, Any], optional): Metadata filters
            use_citation_retrieval (bool): Whether to use citation-aware retrieval

        Returns:
            List[Document]: Retrieved documents
        """
        if self.vector_store is None:
            logger.error("No vector store available for retrieval")
            return []
            
        # Apply jurisdiction filter if specified
        if self.jurisdiction_filter and not filter_metadata:
            filter_metadata = {"citation_jurisdiction": self.jurisdiction_filter}
        elif self.jurisdiction_filter and filter_metadata:
            filter_metadata["citation_jurisdiction"] = self.jurisdiction_filter
            
        # Extract any citations from the query
        query_citations = self.citation_extractor.extract_citations(query)
        
        # If citations found and citation retrieval enabled, use them to enhance retrieval
        if query_citations and use_citation_retrieval:
            # Get standardized citations
            std_citations = [self.citation_extractor.standardize_citation(c) for c in query_citations]
            
            # Try to find exact citation matches first
            citation_filter = {"citations.standardized": {"$in": std_citations}}
            if filter_metadata:
                citation_filter.update(filter_metadata)
                
            # Retrieve documents with matching citations
            citation_docs = self.vector_store.similarity_search(
                query, k=k, filter=citation_filter
            )
            
            # If we found enough citation matches, return them
            if len(citation_docs) >= k // 2:
                return citation_docs
                
        # Standard semantic search
        retrieved_docs = self.vector_store.similarity_search(
            query, k=k, filter=filter_metadata
        )
        
        # Use contextual compression to focus on relevant parts
        compressor = LLMChainExtractor.from_llm(self.llm)
        compression_retriever = ContextualCompressionRetriever(
            base_retriever=self.vector_store.as_retriever(search_kwargs={"k": k}),
            doc_compressor=compressor
        )
        
        try:
            compressed_docs = compression_retriever.get_relevant_documents(query)
            if compressed_docs:
                return compressed_docs
        except Exception as e:
            logger.warning(f"Contextual compression failed: {str(e)}")
            
        return retrieved_docs

    def query(
        self, 
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        verify_response: bool = True
    ) -> Dict[str, Any]:
        """
        Query the Legal RAG system.

        Args:
            query (str): The query string
            k (int): Number of documents to retrieve
            filter_metadata (Dict[str, Any], optional): Metadata filters
            verify_response (bool): Whether to verify the response

        Returns:
            Dict[str, Any]: Response with legal analysis and citations
        """
        # Retrieve relevant documents
        docs = self.retrieve(query, k, filter_metadata)
        
        if not docs:
            return {
                "response": "I couldn't find any relevant legal information to answer your question.",
                "citations": [],
                "confidence_score": 0.0
            }
            
        # Determine jurisdiction from query or default
        jurisdiction = self.jurisdiction_filter or "general"
        if "jurisdiction" in query.lower():
            # Try to extract jurisdiction from query
            jurisdiction_match = re.search(r"(?:in|under|for)\s+([a-zA-Z\s]+)\s+(?:law|jurisdiction)", query.lower())
            if jurisdiction_match:
                jurisdiction = jurisdiction_match.group(1).strip()
                
        # Format context from retrieved documents
        context = "\n\n".join([f"SOURCE {i+1}:\n{doc.page_content}\n\nMETADATA: {json.dumps(doc.metadata)}" 
                              for i, doc in enumerate(docs)])
        
        # Generate response using legal QA prompt
        qa_chain = LLMChain(llm=self.llm, prompt=self.legal_qa_prompt)
        response_text = qa_chain.run(
            question=query,
            context=context,
            jurisdiction=jurisdiction
        )
        
        # Extract citations from response
        response_citations = self.citation_extractor.extract_and_process_citations(response_text)
        
        # Format citations for output
        formatted_citations = []
        for citation in response_citations:
            formatted_citation = {
                "text": citation["standardized"],
                "type": citation["type"]
            }
            
            # Add source document if available
            for i, doc in enumerate(docs):
                if citation["citation"] in doc.page_content:
                    formatted_citation["source_document"] = f"SOURCE {i+1}"
                    formatted_citation["source_metadata"] = doc.metadata
                    break
                    
            formatted_citations.append(formatted_citation)
            
        # Verify response if requested
        verification_result = None
        confidence_score = 0.5  # Default confidence
        
        if verify_response and self.verify_authority:
            try:
                verification_result = self.authority_verifier.verify_legal_content(response_text, docs)
                confidence_score = verification_result["overall_confidence"]
            except Exception as e:
                logger.error(f"Verification failed: {str(e)}")
                
        # Prepare final response
        result = {
            "response": response_text,
            "citations": formatted_citations,
            "confidence_score": confidence_score,
            "retrieved_documents": [
                {
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in docs
            ]
        }
        
        # Add verification details if available
        if verification_result:
            result["verification"] = {
                "verified": verification_result["verified"],
                "claims": [
                    {
                        "claim": claim["claim"],
                        "status": claim["status"],
                        "confidence": claim["confidence"],
                        "authority_level": claim.get("authority_level", "unknown")
                    }
                    for claim in verification_result["claims"]
                ]
            }
            
        return result

    def summarize_legal_document(self, document: Union[str, Document]) -> str:
        """
        Generate a legal summary of a document.

        Args:
            document (Union[str, Document]): Document to summarize

        Returns:
            str: Legal summary
        """
        # Get document content
        if isinstance(document, Document):
            content = document.page_content
        else:
            content = document
            
        # Create summarization chain
        summarization_chain = LLMChain(llm=self.llm, prompt=self.legal_summarization_prompt)
        
        # Generate summary
        summary = summarization_chain.run(content=content)
        
        # Standardize citations in the summary
        summary = self.citation_extractor.replace_with_standardized_citations(summary)
        
        return summary

    def analyze_legal_document(self, file_path: str) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a legal document.

        Args:
            file_path (str): Path to the legal document

        Returns:
            Dict[str, Any]: Analysis results
        """
        # Process the document
        try:
            chunks = process_legal_document(file_path)
        except Exception as e:
            logger.error(f"Failed to process document {file_path}: {str(e)}")
            return {"error": f"Failed to process document: {str(e)}"}
            
        if not chunks:
            return {"error": "No content extracted from document"}
            
        # Combine chunks for full document
        full_text = "\n\n".join([chunk.page_content for chunk in chunks])
        
        # Extract document type and metadata
        doc_type = chunks[0].metadata.get("doc_type", "unknown")
        
        # Extract all citations
        all_citations = self.citation_extractor.extract_and_process_citations(full_text)
        
        # Generate summary
        summary = self.summarize_legal_document(full_text)
        
        # Extract key legal concepts
        legal_concepts = self._extract_legal_concepts(full_text)
        
        # Analyze document structure
        structure = self._analyze_document_structure(chunks)
        
        return {
            "document_type": doc_type,
            "summary": summary,
            "citations": all_citations,
            "legal_concepts": legal_concepts,
            "structure": structure,
            "metadata": chunks[0].metadata
        }

    def _extract_legal_concepts(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract key legal concepts from text.

        Args:
            text (str): Legal text

        Returns:
            List[Dict[str, Any]]: Extracted legal concepts
        """
        # Create a prompt for concept extraction
        concept_prompt = ChatPromptTemplate.from_template(
            """You are a legal expert tasked with identifying key legal concepts in the following text.
            
            TEXT:
            {text}
            
            Please identify the most important legal concepts, principles, doctrines, or tests mentioned
            in this text. For each concept, provide:
            1. The name of the concept
            2. A brief definition or explanation
            3. The legal domain it belongs to (e.g., constitutional law, contract law)
            
            Format your response as a list of concepts with these three elements for each.
            
            LEGAL CONCEPTS:"""
        )
        
        # Create extraction chain
        concept_chain = LLMChain(llm=self.llm, prompt=concept_prompt)
        
        # Extract concepts
        result = concept_chain.run(text=text[:10000])  # Limit text length
        
        # Parse results
        concepts = []
        current_concept = {}
        
        for line in result.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith(('1.', '2.', '3.')) and current_concept:
                # Add completed concept
                if len(current_concept) >= 3:
                    concepts.append(current_concept)
                current_concept = {}
                
            if line.startswith('1.') or (not current_concept and line[0].isdigit() and line[1] == '.'):
                # Start new concept
                current_concept = {"name": line.split('.', 1)[1].strip()}
            elif "name:" in line.lower() and not current_concept.get("name"):
                current_concept["name"] = line.split(':', 1)[1].strip()
            elif "definition:" in line.lower() or "explanation:" in line.lower():
                current_concept["definition"] = line.split(':', 1)[1].strip()
            elif "domain:" in line.lower():
                current_concept["domain"] = line.split(':', 1)[1].strip()
                
        # Add the last concept if complete
        if current_concept and len(current_concept) >= 3:
            concepts.append(current_concept)
            
        return concepts

    def _analyze_document_structure(self, chunks: List[Document]) -> Dict[str, Any]:
        """
        Analyze the structure of a legal document.

        Args:
            chunks (List[Document]): Document chunks

        Returns:
            Dict[str, Any]: Document structure analysis
        """
        # Get document type
        doc_type = chunks[0].metadata.get("doc_type", "unknown")
        
        # Analyze based on document type
        if doc_type == "case_law":
            return self._analyze_case_structure(chunks)
        elif doc_type == "statute":
            return self._analyze_statute_structure(chunks)
        elif doc_type == "contract":
            return self._analyze_contract_structure(chunks)
        else:
            return self._analyze_generic_structure(chunks)

    def _analyze_case_structure(self, chunks: List[Document]) -> Dict[str, Any]:
        """Analyze case law structure"""
        sections = {}
        
        for chunk in chunks:
            section = chunk.metadata.get("section", "unknown")
            if section not in sections:
                sections[section] = []
            sections[section].append(chunk.page_content[:100] + "...")
            
        # Extract case name and citation if available
        case_name = None
        citation = None
        
        for chunk in chunks:
            if "case_name" in chunk.metadata:
                case_name = chunk.metadata["case_name"]
            if "citation" in chunk.metadata:
                citation = chunk.metadata["citation"]
                
        return {
            "type": "case_law",
            "case_name": case_name,
            "citation": citation,
            "sections": sections
        }

    def _analyze_statute_structure(self, chunks: List[Document]) -> Dict[str, Any]:
        """Analyze statute structure"""
        sections = {}
        
        for chunk in chunks:
            section = chunk.metadata.get("section", "unknown")
            if section not in sections:
                sections[section] = []
            sections[section].append(chunk.page_content[:100] + "...")
            
        return {
            "type": "statute",
            "sections": sections
        }

    def _analyze_contract_structure(self, chunks: List[Document]) -> Dict[str, Any]:
        """Analyze contract structure"""
        sections = {}
        parties = []
        
        for chunk in chunks:
            section = chunk.metadata.get("section", "unknown")
            if section not in sections:
                sections[section] = []
            sections[section].append(chunk.page_content[:100] + "...")
            
            # Extract parties
            if "party1" in chunk.metadata and chunk.metadata["party1"] not in parties:
                parties.append(chunk.metadata["party1"])
            if "party2" in chunk.metadata and chunk.metadata["party2"] not in parties:
                parties.append(chunk.metadata["party2"])
                
        return {
            "type": "contract",
            "parties": parties,
            "sections": sections
        }

    def _analyze_generic_structure(self, chunks: List[Document]) -> Dict[str, Any]:
        """Analyze generic legal document structure"""
        sections = {}
        
        for chunk in chunks:
            section = chunk.metadata.get("section", "unknown")
            if section not in sections:
                sections[section] = []
            sections[section].append(chunk.page_content[:100] + "...")
            
        return {
            "type": "legal_document",
            "sections": sections
        } 