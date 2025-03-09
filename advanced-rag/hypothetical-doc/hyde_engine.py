"""
Hypothetical Document Embeddings (HyDE) Engine

This module implements HyDE (Hypothetical Document Embeddings) techniques for RAG,
where hypothetical document(s) are generated based on a query and then used for retrieval.
"""

import logging
import re
import time
from typing import List, Dict, Any, Optional, Union, Callable, Tuple
import numpy as np

# Try to import the model provider and related components
try:
    from tools.src.models import (
        LLMProvider, Message, Role, get_model_provider
    )
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


class HyDEEngine:
    """
    Hypothetical Document Embeddings (HyDE) Engine for enhanced retrieval.
    
    HyDE works by first generating a hypothetical document that would answer a query,
    then embedding that document and using it for retrieval. This helps bridge the
    lexical gap between queries and documents.
    
    This implementation adds:
    - Multi-perspective generation (generates multiple hypothetical documents)
    - Incremental refinement (iteratively improves hypothetical documents)
    - Domain-specific templates (uses domain knowledge for better generation)
    """
    
    def __init__(
        self,
        model_provider: Optional[Union[str, LLMProvider]] = None,
        embedding_provider: Optional[Union[str, LLMProvider]] = None,
        retriever: Optional[Callable] = None,
        num_perspectives: int = 3,
        refinement_steps: int = 1,
        domain: str = "general",  # "general", "code", "medical", "legal", "scientific", etc.
        system_prompt: Optional[str] = None,
        generation_prompts: Optional[Dict[str, str]] = None,
        refinement_prompt: Optional[str] = None,
        temperature_decay: float = 0.1,  # Temperature decreases with each refinement
        base_temperature: float = 0.7,
        use_hybrid_retrieval: bool = True,
    ):
        """
        Initialize the HyDE Engine.
        
        Args:
            model_provider: LLM provider for generating hypothetical documents
            embedding_provider: LLM provider for generating embeddings (if different)
            retriever: Function for retrieving passages from a knowledge base
            num_perspectives: Number of different perspectives to generate
            refinement_steps: Number of refinement iterations to apply
            domain: Domain for specialized templates
            system_prompt: Custom system prompt for document generation
            generation_prompts: Custom prompts for different perspectives
            refinement_prompt: Custom prompt for document refinement
            temperature_decay: How much to decrease temperature for each refinement
            base_temperature: Base temperature for document generation
            use_hybrid_retrieval: Whether to combine query and document embeddings
        """
        if not MODELS_AVAILABLE:
            raise ImportError(
                "The models module is required. Make sure the 'tools.src.models' module is available."
            )
        
        # Set up the model provider for generation
        if model_provider is None:
            try:
                # Default to OpenAI GPT-4 if not specified
                self.model_provider = get_model_provider("openai", model="gpt-4o")
                logger.info("Using default OpenAI GPT-4o model provider for HyDE generation")
            except Exception as e:
                logger.warning(f"Error initializing default model provider: {str(e)}")
                raise
        elif isinstance(model_provider, str):
            try:
                self.model_provider = get_model_provider(model_provider)
                logger.info(f"Using {model_provider} model provider for HyDE generation")
            except Exception as e:
                logger.warning(f"Error initializing model provider {model_provider}: {str(e)}")
                raise
        else:
            self.model_provider = model_provider
        
        # Set up the embedding provider (same as model provider if not specified)
        if embedding_provider is None:
            self.embedding_provider = self.model_provider
        elif isinstance(embedding_provider, str):
            try:
                self.embedding_provider = get_model_provider(embedding_provider)
                logger.info(f"Using {embedding_provider} model provider for embeddings")
            except Exception as e:
                logger.warning(f"Error initializing embedding provider {embedding_provider}: {str(e)}")
                self.embedding_provider = self.model_provider
        else:
            self.embedding_provider = embedding_provider
        
        # Set retriever (can be updated later if None)
        self.retriever = retriever
        
        # Set HyDE parameters
        self.num_perspectives = num_perspectives
        self.refinement_steps = refinement_steps
        self.domain = domain
        self.base_temperature = base_temperature
        self.temperature_decay = temperature_decay
        self.use_hybrid_retrieval = use_hybrid_retrieval
        
        # Initialize default prompts
        self._init_default_prompts()
        
        # Override with custom prompts if provided
        if system_prompt:
            self.system_prompt = system_prompt
        if generation_prompts:
            self.generation_prompts.update(generation_prompts)
        if refinement_prompt:
            self.refinement_prompt = refinement_prompt
        
        logger.debug(f"Initialized HyDE Engine with {num_perspectives} perspectives and {refinement_steps} refinement steps")
    
    def _init_default_prompts(self):
        """Initialize default prompts for the HyDE pipeline."""
        
        # Main system prompt for document generation
        self.system_prompt = """
        You are an expert at generating hypothetical documents that contain information
        that would answer a query. Your goal is to create realistic, accurate, and 
        detailed documents that might exist in a knowledge base.
        
        For each query, create a document that:
        1. Directly addresses the query
        2. Contains factual information that would be useful for answering the query
        3. Is written in a style appropriate for the domain and perspective requested
        4. Includes specific details, examples, or explanations
        5. Mimics the structure and format of real documents in this domain
        
        Do not format your response as a direct answer to the query. Instead, create a
        document that contains the information someone would need to answer the query.
        """
        
        # Initialize generation prompts for different perspectives
        self.generation_prompts = {
            "expert": """
            Generate a hypothetical document written by an expert in the field that would contain 
            information to answer this query. The document should be detailed, technically accurate, 
            and demonstrate deep expertise in the subject.
            
            Query: {query}
            
            Domain: {domain}
            
            Expert Hypothetical Document:
            """,
            
            "educational": """
            Generate a hypothetical educational resource or tutorial that would contain 
            information to answer this query. The document should be clear, instructional,
            and designed to teach someone about the topic.
            
            Query: {query}
            
            Domain: {domain}
            
            Educational Hypothetical Document:
            """,
            
            "summary": """
            Generate a hypothetical summary document that would contain key information
            to answer this query. The document should be concise, well-organized, and 
            highlight the most important points related to the query.
            
            Query: {query}
            
            Domain: {domain}
            
            Summary Hypothetical Document:
            """,
            
            "example_based": """
            Generate a hypothetical document that uses concrete examples to address
            this query. The document should contain specific instances, case studies,
            or examples that illustrate the answer to the query.
            
            Query: {query}
            
            Domain: {domain}
            
            Example-Based Hypothetical Document:
            """,
            
            "qa_format": """
            Generate a hypothetical Q&A document that would contain information to answer
            this query. The document should be structured as a series of questions and 
            answers that cover various aspects of the topic, including the specific query.
            
            Query: {query}
            
            Domain: {domain}
            
            Q&A Hypothetical Document:
            """
        }
        
        # Domain-specific prompt additions
        self.domain_additions = {
            "code": """
            Include code examples, technical specifications, API documentation style content, 
            and focus on implementation details. Use appropriate syntax highlighting and 
            code comments. Describe functions, classes, or algorithms relevant to the query.
            """,
            
            "medical": """
            Include medical terminology, reference research studies, follow evidence-based 
            approaches, and maintain clinical accuracy. Consider patient presentations, 
            diagnostic criteria, treatment protocols, and medical guidelines as appropriate.
            """,
            
            "legal": """
            Include references to relevant laws, statutes, cases, or regulations. Use proper 
            legal terminology and citation formats. Consider jurisdictional issues and 
            precedents. Structure content as a legal memorandum or analysis where appropriate.
            """,
            
            "scientific": """
            Include references to scientific literature, methodology descriptions, data 
            analysis approaches, and results interpretation. Use appropriate scientific 
            terminology and follow scientific writing conventions. Include hypotheses, 
            experimental design, and conclusions where relevant.
            """
        }
        
        # Refinement prompt for iterative improvement
        self.refinement_prompt = """
        Your task is to refine and improve the following hypothetical document to better 
        address the original query. Make this document more detailed, accurate, and useful 
        for retrieval purposes.
        
        Original Query: {query}
        
        Current Hypothetical Document:
        {document}
        
        Specific Improvements Needed:
        1. Add more specific details, facts, or examples
        2. Improve technical accuracy where needed
        3. Enhance the structure and formatting
        4. Make the content more directly relevant to the query
        5. Expand on any unclear or underdeveloped points
        
        Refined Hypothetical Document:
        """
    
    def set_retriever(self, retriever: Callable):
        """
        Set or update the retriever function.
        
        Args:
            retriever: Function for retrieving passages from a knowledge base
        """
        self.retriever = retriever
        logger.info("Retriever function updated")
    
    def _get_perspective_prompts(self, query: str) -> List[Tuple[str, str]]:
        """
        Get a list of prompts for different perspectives.
        
        Args:
            query: The user's query
            
        Returns:
            List of (perspective_name, prompt) tuples
        """
        # Select the perspectives to use
        if self.num_perspectives >= len(self.generation_prompts):
            # Use all available perspectives
            perspective_names = list(self.generation_prompts.keys())
        else:
            # Pick a subset of perspectives
            # Always include "expert" if available
            if "expert" in self.generation_prompts and self.num_perspectives > 0:
                perspective_names = ["expert"]
                remaining = list(set(self.generation_prompts.keys()) - {"expert"})
                # Add additional perspectives up to num_perspectives
                perspective_names.extend(remaining[:self.num_perspectives-1])
            else:
                # Just take the first num_perspectives
                perspective_names = list(self.generation_prompts.keys())[:self.num_perspectives]
        
        # Get domain-specific additions if available
        domain_addition = self.domain_additions.get(self.domain, "")
        
        # Create the prompts
        prompts = []
        for name in perspective_names:
            base_prompt = self.generation_prompts[name]
            
            # Add domain-specific content
            if domain_addition and "{domain_specific_instructions}" in base_prompt:
                prompt = base_prompt.replace("{domain_specific_instructions}", domain_addition)
            elif domain_addition:
                prompt = base_prompt + "\n\n" + domain_addition
            else:
                prompt = base_prompt
            
            # Format with query and domain
            formatted_prompt = prompt.format(query=query, domain=self.domain)
            prompts.append((name, formatted_prompt))
        
        return prompts
    
    def _generate_hypothetical_document(self, 
                                      query: str, 
                                      perspective: str, 
                                      prompt: str, 
                                      temperature: float = None) -> str:
        """
        Generate a hypothetical document for a single perspective.
        
        Args:
            query: The user's query
            perspective: The perspective name
            prompt: The generation prompt
            temperature: Generation temperature (default: base_temperature)
            
        Returns:
            Generated hypothetical document
        """
        if temperature is None:
            temperature = self.base_temperature
        
        # Create messages for generation
        messages = [
            Message(role=Role.SYSTEM, content=self.system_prompt),
            Message(role=Role.USER, content=prompt)
        ]
        
        # Generate the document
        try:
            response = self.model_provider.generate(
                messages=messages,
                temperature=temperature,
                max_tokens=2048  # Adjust as needed for document length
            )
            
            # Extract the generated document
            document = response.get("content", "")
            
            logger.info(f"Generated {len(document.split())} word hypothetical document from {perspective} perspective")
            return document
            
        except Exception as e:
            logger.error(f"Error generating hypothetical document: {str(e)}")
            # Return a minimal fallback document
            return f"Hypothetical document addressing {query} from {perspective} perspective."
    
    def _refine_document(self, 
                       query: str, 
                       document: str, 
                       refinement_step: int) -> str:
        """
        Refine a hypothetical document with additional information and improvements.
        
        Args:
            query: The original query
            document: The hypothetical document to refine
            refinement_step: Current refinement iteration (0-based)
            
        Returns:
            Refined hypothetical document
        """
        # Decrease temperature with each refinement step
        temperature = max(0.2, self.base_temperature - (refinement_step * self.temperature_decay))
        
        # Create refinement prompt
        refinement_content = self.refinement_prompt.format(
            query=query,
            document=document
        )
        
        # Create messages for refinement
        messages = [
            Message(role=Role.SYSTEM, content=self.system_prompt),
            Message(role=Role.USER, content=refinement_content)
        ]
        
        # Generate the refined document
        try:
            response = self.model_provider.generate(
                messages=messages,
                temperature=temperature,
                max_tokens=2048
            )
            
            # Extract the refined document
            refined_document = response.get("content", "")
            
            # If something went wrong and we got no content, return the original
            if not refined_document.strip():
                logger.warning(f"Empty refined document returned for refinement step {refinement_step}")
                return document
            
            logger.info(f"Refined document at step {refinement_step+1}/{self.refinement_steps}")
            return refined_document
            
        except Exception as e:
            logger.error(f"Error refining document at step {refinement_step}: {str(e)}")
            # Return the original document as fallback
            return document
    
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            # Use the embedding provider to generate embeddings
            embeddings = self.embedding_provider.embed(texts)
            logger.info(f"Generated embeddings for {len(texts)} texts")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            # Return empty embeddings as fallback
            return [[] for _ in texts]
    
    def _aggregate_embeddings(self, embeddings: List[List[float]]) -> List[float]:
        """
        Aggregate multiple embedding vectors into a single vector.
        
        Args:
            embeddings: List of embedding vectors
            
        Returns:
            Aggregated embedding vector
        """
        if not embeddings or not embeddings[0]:
            logger.warning("No valid embeddings to aggregate")
            return []
        
        # Convert to numpy for easier operations
        try:
            np_embeddings = np.array(embeddings)
            
            # Calculate mean embedding
            mean_embedding = np.mean(np_embeddings, axis=0)
            
            # Normalize to unit length
            norm = np.linalg.norm(mean_embedding)
            if norm > 0:
                normalized = mean_embedding / norm
            else:
                normalized = mean_embedding
            
            return normalized.tolist()
        except Exception as e:
            logger.error(f"Error aggregating embeddings: {str(e)}")
            # Return the first embedding as fallback if available
            return embeddings[0] if embeddings else []
    
    def generate_hypothetical_documents(self, query: str) -> List[Dict[str, Any]]:
        """
        Generate hypothetical documents from multiple perspectives,
        with optional refinement.
        
        Args:
            query: The user's query
            
        Returns:
            List of hypothetical document dictionaries with metadata
        """
        # Get prompts for different perspectives
        perspective_prompts = self._get_perspective_prompts(query)
        logger.info(f"Generating {len(perspective_prompts)} hypothetical documents")
        
        hypothetical_documents = []
        
        # Generate initial documents from different perspectives
        for perspective, prompt in perspective_prompts:
            # Generate initial document
            document = self._generate_hypothetical_document(
                query=query,
                perspective=perspective,
                prompt=prompt
            )
            
            # Apply refinement if specified
            refined_document = document
            for step in range(self.refinement_steps):
                refined_document = self._refine_document(
                    query=query,
                    document=refined_document,
                    refinement_step=step
                )
            
            # Store the document with metadata
            hypothetical_documents.append({
                "query": query,
                "perspective": perspective,
                "document": refined_document,
                "refinement_steps": self.refinement_steps,
                "domain": self.domain
            })
        
        return hypothetical_documents
    
    def retrieve_with_hypothetical_documents(self, 
                                           query: str, 
                                           limit: int = 10,
                                           **kwargs) -> Dict[str, Any]:
        """
        Perform retrieval using hypothetical documents as queries.
        
        Args:
            query: The user's query
            limit: Maximum number of results to return
            **kwargs: Additional parameters
                - hyde_only: Whether to use only HyDE for retrieval (default: False)
                - hypothetical_documents: Pre-generated hypothetical documents
                - aggregation_method: How to aggregate results ("merge" or "ensemble")
                
        Returns:
            Dictionary with retrieval results and HyDE metadata
        """
        if not self.retriever:
            raise ValueError("Retriever function is required for HyDE retrieval")
        
        # Extract parameters
        hyde_only = kwargs.get("hyde_only", False)
        hypothetical_documents = kwargs.get("hypothetical_documents")
        aggregation_method = kwargs.get("aggregation_method", "merge")
        
        # Generate hypothetical documents if not provided
        if not hypothetical_documents:
            hypothetical_documents = self.generate_hypothetical_documents(query)
        
        # Extract just the document texts
        document_texts = [doc["document"] for doc in hypothetical_documents]
        
        # Generate embeddings for the hypothetical documents
        document_embeddings = self._generate_embeddings(document_texts)
        
        # Also get the query embedding if doing hybrid retrieval
        if not hyde_only and self.use_hybrid_retrieval:
            query_embedding = self._generate_embeddings([query])[0]
        else:
            query_embedding = None
        
        # Aggregate the embeddings from multiple perspectives
        aggregated_embedding = self._aggregate_embeddings(document_embeddings)
        
        all_results = []
        
        # Perform retrieval with each document embedding
        if aggregation_method == "ensemble":
            # Retrieve with each document embedding separately
            for i, embedding in enumerate(document_embeddings):
                doc_results = self.retriever(
                    query=None,  # Using embedding instead of text query
                    embedding=embedding,
                    limit=limit,
                    **kwargs
                )
                
                # Add source perspective to results
                for result in doc_results:
                    result["source_perspective"] = hypothetical_documents[i]["perspective"]
                
                all_results.append(doc_results)
            
            # Merge results from different perspectives
            merged_results = self._merge_results(all_results, limit)
            
        else:  # "merge" method (default)
            # Retrieve with the aggregated embedding
            merged_results = self.retriever(
                query=None,  # Using embedding instead of text query
                embedding=aggregated_embedding,
                limit=limit,
                **kwargs
            )
        
        # If hybrid retrieval, also retrieve with the original query
        if not hyde_only and self.use_hybrid_retrieval and query_embedding:
            query_results = self.retriever(
                query=None,  # Using embedding instead of text query
                embedding=query_embedding,
                limit=limit,
                **kwargs
            )
            
            # Add source metadata
            for result in query_results:
                result["source"] = "original_query"
            
            # Combine with HyDE results
            for result in merged_results:
                result["source"] = "hyde"
            
            # Create final merged results
            final_results = self._combine_results(merged_results, query_results, limit)
        else:
            # Just use the HyDE results
            final_results = merged_results
        
        # Create the result object
        result = {
            "results": final_results,
            "hypothetical_documents": hypothetical_documents,
            "hyde_config": {
                "num_perspectives": self.num_perspectives,
                "refinement_steps": self.refinement_steps,
                "domain": self.domain,
                "hybrid_retrieval": self.use_hybrid_retrieval and not hyde_only,
            }
        }
        
        return result
    
    def _merge_results(self, 
                      result_sets: List[List[Dict[str, Any]]], 
                      limit: int) -> List[Dict[str, Any]]:
        """
        Merge results from multiple retrievals.
        
        Args:
            result_sets: List of result sets from different perspectives
            limit: Maximum number of results to return
            
        Returns:
            Merged and deduplicated results
        """
        # Track unique documents by ID
        seen_docs = {}
        
        # Process all result sets
        for perspective_results in result_sets:
            for result in perspective_results:
                doc_id = result["id"]
                
                # If we haven't seen this document or it has a higher score
                if doc_id not in seen_docs or result["score"] > seen_docs[doc_id]["score"]:
                    seen_docs[doc_id] = result
        
        # Convert to list and sort by score
        merged_results = list(seen_docs.values())
        merged_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Limit to requested number
        return merged_results[:limit]
    
    def _combine_results(self, 
                        hyde_results: List[Dict[str, Any]], 
                        query_results: List[Dict[str, Any]], 
                        limit: int) -> List[Dict[str, Any]]:
        """
        Combine results from HyDE and original query retrievals.
        
        Args:
            hyde_results: Results from HyDE retrieval
            query_results: Results from original query retrieval
            limit: Maximum number of results to return
            
        Returns:
            Combined and deduplicated results
        """
        # Use a simple interleaving strategy to combine results
        # Start with top HyDE result, then top query result, etc.
        combined = []
        seen_ids = set()
        
        # Calculate how many to take from each source
        hyde_count = min(len(hyde_results), limit // 2 + (limit % 2))
        query_count = min(len(query_results), limit - hyde_count)
        
        # Get the selected results from each source
        selected_hyde = hyde_results[:hyde_count]
        selected_query = query_results[:query_count]
        
        # Interleave the results
        for i in range(max(hyde_count, query_count)):
            # Add HyDE result if available
            if i < hyde_count:
                doc_id = selected_hyde[i]["id"]
                if doc_id not in seen_ids:
                    combined.append(selected_hyde[i])
                    seen_ids.add(doc_id)
            
            # Add query result if available
            if i < query_count:
                doc_id = selected_query[i]["id"]
                if doc_id not in seen_ids:
                    combined.append(selected_query[i])
                    seen_ids.add(doc_id)
            
            # Stop if we've reached the limit
            if len(combined) >= limit:
                break
        
        # If we still have room, add more results
        if len(combined) < limit:
            # Add remaining HyDE results
            for result in hyde_results[hyde_count:]:
                if result["id"] not in seen_ids and len(combined) < limit:
                    combined.append(result)
                    seen_ids.add(result["id"])
            
            # Add remaining query results
            for result in query_results[query_count:]:
                if result["id"] not in seen_ids and len(combined) < limit:
                    combined.append(result)
                    seen_ids.add(result["id"])
        
        return combined 