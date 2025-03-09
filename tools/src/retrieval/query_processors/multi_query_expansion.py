"""
Multi-Query Expansion

This module provides query expansion strategies to improve retrieval by generating
multiple query variations and combining the results.
"""

import logging
from typing import List, Dict, Any, Optional, Union, Callable
import re

from ..hybrid_searchers.base_hybrid_searcher import BaseHybridSearcher

# Import model provider for LLM-based expansions
try:
    from ...models import LLMProvider, Message, Role, get_model_provider
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


class MultiQueryExpansion:
    """
    Multi-Query Expansion improves retrieval by generating multiple variations of the query
    to increase coverage and precision of retrieval results.
    
    It uses LLMs to generate alternative phrasings of the original query and
    combines results from all query variations.
    """
    
    def __init__(
        self,
        model_provider: Optional[Union[str, LLMProvider]] = None,
        num_variations: int = 3,
        fusion_strategy: str = "reciprocal_rank_fusion",
        similarity_threshold: float = 0.85,
        system_prompt: Optional[str] = None,
        prompt_template: Optional[str] = None,
        diversity_score_threshold: float = 0.7,
        use_semantic_deduplication: bool = True,
    ):
        """
        Initialize the MultiQueryExpansion processor.
        
        Args:
            model_provider: LLM provider to use for query expansion, either a provider instance or a string
                identifying the provider type ("openai", "anthropic", "local", etc.)
            num_variations: Number of query variations to generate
            fusion_strategy: Strategy for combining results ("reciprocal_rank_fusion", "simple_merge", "weighted")
            similarity_threshold: Threshold for semantic similarity in deduplication
            system_prompt: Custom system prompt for query expansion
            prompt_template: Custom prompt template for query expansion
            diversity_score_threshold: Minimum semantic distance between variations
            use_semantic_deduplication: Whether to deduplicate results semantically
        """
        self.num_variations = num_variations
        self.fusion_strategy = fusion_strategy
        self.similarity_threshold = similarity_threshold
        self.diversity_score_threshold = diversity_score_threshold
        self.use_semantic_deduplication = use_semantic_deduplication
        
        # Initialize default prompts
        self.system_prompt = system_prompt or (
            "You are an expert at generating diverse variations of search queries. "
            "Your task is to generate multiple different ways to phrase the same question or query. "
            "The variations should capture different aspects, terminologies, or approaches to the same information need. "
            "Make sure the variations are diverse but maintain the same core information need."
        )
        
        self.prompt_template = prompt_template or (
            "Generate {num_variations} diverse variations of the following query. "
            "The variations should be semantically similar but phrased differently, "
            "potentially using different terms or perspectives. "
            "Provide ONLY the list of variations as plain text, one per line. "
            "Do not include any other explanations or numbering.\n\n"
            "Original query: {query}"
        )
        
        # Set up the model provider
        if MODELS_AVAILABLE:
            if model_provider is None:
                try:
                    # Default to OpenAI GPT models if not specified
                    self.model_provider = get_model_provider("openai")
                    logger.info("Using default OpenAI model provider for query expansion")
                except Exception as e:
                    logger.warning(f"Error initializing default model provider: {str(e)}")
                    self.model_provider = None
            elif isinstance(model_provider, str):
                try:
                    self.model_provider = get_model_provider(model_provider)
                    logger.info(f"Using {model_provider} model provider for query expansion")
                except Exception as e:
                    logger.warning(f"Error initializing model provider {model_provider}: {str(e)}")
                    self.model_provider = None
            else:
                self.model_provider = model_provider
        else:
            logger.warning("Models module not available, query expansion will use simple strategies")
            self.model_provider = None
    
    def expand_query(self, query: str) -> List[str]:
        """
        Generate multiple variations of the query.
        
        Args:
            query: The original query
            
        Returns:
            List of query variations including the original query
        """
        logger.info(f"Expanding query: '{query}'")
        
        variations = [query]  # Always include the original query
        
        if self.model_provider:
            # Use LLM-based query expansion
            try:
                # Create messages for the LLM request
                messages = [
                    Message(role=Role.SYSTEM, content=self.system_prompt),
                    Message(
                        role=Role.USER, 
                        content=self.prompt_template.format(
                            query=query, 
                            num_variations=self.num_variations
                        )
                    )
                ]
                
                # Request query variations from the LLM
                response = self.model_provider.generate(
                    messages=messages,
                    temperature=0.7,  # Higher temperature for diverse variations
                    max_tokens=1024
                )
                
                # Parse the variations from the response
                content = response.get("content", "")
                if content:
                    # Split by newlines and filter empty lines
                    llm_variations = [
                        line.strip() for line in content.split("\n") 
                        if line.strip() and line.strip() != query
                    ]
                    
                    # Add variations from the LLM
                    variations.extend(llm_variations)
                    
                    # Deduplicate (ignoring case differences)
                    unique_variations = []
                    unique_lower = set()
                    
                    for var in variations:
                        if var.lower() not in unique_lower:
                            unique_variations.append(var)
                            unique_lower.add(var.lower())
                    
                    # Ensure we have the required number of variations, but don't exceed
                    variations = unique_variations[:self.num_variations + 1]
                    
                    logger.info(f"Generated {len(variations) - 1} query variations using LLM")
            except Exception as e:
                logger.error(f"Error generating query variations with LLM: {str(e)}")
                # Fall back to simple variations when LLM fails
                variations.extend(self._generate_simple_variations(query))
        else:
            # Use simple rule-based expansion strategies
            variations.extend(self._generate_simple_variations(query))
        
        logger.debug(f"Query variations: {variations}")
        return variations
    
    def _generate_simple_variations(self, query: str) -> List[str]:
        """
        Generate query variations using simple rule-based methods.
        
        Args:
            query: The original query
            
        Returns:
            List of query variations
        """
        variations = []
        
        # Convert question to statement
        if "?" in query:
            # Simple question to statement conversion
            statement = query.replace("?", "").strip()
            statement = re.sub(r'^(what|who|where|when|why|how|is|are|can|do|does)\s+', '', statement, flags=re.IGNORECASE)
            statement = statement.strip()
            if statement and statement != query:
                variations.append(statement)
        
        # Add "about" prefix for context
        about_query = f"about {query}"
        variations.append(about_query)
        
        # Add "information on" or "details on" prefix
        info_query = f"information on {query}"
        variations.append(info_query)
        
        # Swap synonyms for common words (very simple)
        synonyms = {
            'how': 'what is the method',
            'what': 'describe',
            'why': 'reason for',
            'where': 'location of',
            'when': 'time of',
            'who': 'person',
            'create': 'make',
            'build': 'develop',
            'problem': 'issue',
            'good': 'beneficial',
            'bad': 'problematic',
        }
        
        for word, replacement in synonyms.items():
            pattern = r'\b' + re.escape(word) + r'\b'
            if re.search(pattern, query, re.IGNORECASE):
                new_query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
                variations.append(new_query)
                
                # Only add one synonym variation to avoid too many similar variations
                break
        
        # Return unique variations
        unique_variations = list(set(variations))
        return unique_variations[:self.num_variations]
    
    def retrieve_with_variations(
        self, 
        query: str, 
        retriever: Union[Callable, BaseHybridSearcher],
        limit: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents using multiple query variations and combine the results.
        
        Args:
            query: The original query
            retriever: Retrieval function or hybrid searcher
            limit: Maximum number of results to return
            **kwargs: Additional parameters to pass to the retriever
            
        Returns:
            Combined and deduplicated retrieval results
        """
        logger.info(f"Retrieving with variations for query: '{query}'")
        
        # Generate variations
        variations = self.expand_query(query)
        
        # Retrieve results for each variation
        all_results = []
        
        for i, variation in enumerate(variations):
            logger.debug(f"Retrieving with variation [{i}]: '{variation}'")
            
            try:
                # Call the retriever on each variation
                if isinstance(retriever, BaseHybridSearcher):
                    variation_results = retriever.search(variation, limit=limit, **kwargs)
                else:
                    variation_results = retriever(variation, limit=limit, **kwargs)
                
                # Tag results with the query variation that retrieved them
                for result in variation_results:
                    if "source_queries" not in result:
                        result["source_queries"] = []
                    result["source_queries"].append(variation)
                
                all_results.append(variation_results)
                logger.debug(f"Retrieved {len(variation_results)} results for variation {i}")
                
            except Exception as e:
                logger.error(f"Error retrieving with variation '{variation}': {str(e)}")
        
        # Combine results based on the chosen strategy
        if self.fusion_strategy == "reciprocal_rank_fusion":
            combined_results = self._reciprocal_rank_fusion(all_results, limit)
        elif self.fusion_strategy == "weighted":
            combined_results = self._weighted_fusion(all_results, limit)
        else:  # "simple_merge"
            combined_results = self._simple_merge(all_results, limit)
        
        logger.info(f"Combined results from {len(variations)} variations into {len(combined_results)} results")
        return combined_results
    
    def _simple_merge(
        self, 
        result_sets: List[List[Dict[str, Any]]], 
        limit: int
    ) -> List[Dict[str, Any]]:
        """
        Merge results from multiple queries, removing duplicates.
        
        Args:
            result_sets: List of result sets from each query variation
            limit: Maximum number of results to return
            
        Returns:
            Merged and deduplicated results
        """
        # Use document ID to deduplicate
        seen_docs = {}
        
        # Process each result set
        for results in result_sets:
            for result in results:
                doc_id = result["id"]
                
                # If we haven't seen this document or it has a higher score, add/update it
                if doc_id not in seen_docs or result["score"] > seen_docs[doc_id]["score"]:
                    seen_docs[doc_id] = result
        
        # Convert to list and sort by score
        merged_results = list(seen_docs.values())
        merged_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top-k results
        return merged_results[:limit]
    
    def _weighted_fusion(
        self, 
        result_sets: List[List[Dict[str, Any]]], 
        limit: int
    ) -> List[Dict[str, Any]]:
        """
        Combine results with weights favoring the original query.
        
        Args:
            result_sets: List of result sets from each query variation
            limit: Maximum number of results to return
            
        Returns:
            Merged and reranked results
        """
        seen_docs = {}
        
        # Define weights for query variations (original query has highest weight)
        weights = [1.0]  # Original query
        # Assign decreasing weights to variations
        weights.extend([0.9 - (i * 0.1) for i in range(len(result_sets) - 1)])
        weights = [max(0.2, w) for w in weights]  # Ensure minimum weight
        
        # Process each result set with its weight
        for i, (results, weight) in enumerate(zip(result_sets, weights)):
            for result in results:
                doc_id = result["id"]
                weighted_score = result["score"] * weight
                
                if doc_id not in seen_docs:
                    # Add new document with its weighted score
                    new_result = result.copy()
                    new_result["score"] = weighted_score
                    new_result["original_score"] = result["score"]
                    new_result["weight"] = weight
                    seen_docs[doc_id] = new_result
                else:
                    # Update existing document if this score is higher
                    if weighted_score > seen_docs[doc_id]["score"]:
                        seen_docs[doc_id]["score"] = weighted_score
                        seen_docs[doc_id]["original_score"] = result["score"]
                        seen_docs[doc_id]["weight"] = weight
        
        # Convert to list and sort by weighted score
        merged_results = list(seen_docs.values())
        merged_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top-k results
        return merged_results[:limit]
    
    def _reciprocal_rank_fusion(
        self, 
        result_sets: List[List[Dict[str, Any]]], 
        limit: int,
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Combine results using Reciprocal Rank Fusion algorithm.
        
        This is particularly effective for merging results from multiple queries
        while taking into account the rank of each document.
        
        Args:
            result_sets: List of result sets from each query variation
            limit: Maximum number of results to return
            k: Constant in RRF formula (default: 60)
            
        Returns:
            Merged and reranked results
        """
        doc_scores = {}
        
        # Calculate RRF scores
        for results in result_sets:
            # Get ranks for each document in this result set
            for rank, result in enumerate(results):
                doc_id = result["id"]
                
                # RRF formula: 1 / (k + rank)
                rrf_score = 1.0 / (k + rank)
                
                # Accumulate RRF scores
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        "rrf_score": rrf_score,
                        "doc": result,
                        "best_rank": rank
                    }
                else:
                    doc_scores[doc_id]["rrf_score"] += rrf_score
                    if rank < doc_scores[doc_id]["best_rank"]:
                        doc_scores[doc_id]["best_rank"] = rank
                        doc_scores[doc_id]["doc"] = result
        
        # Create final result list
        merged_results = []
        for doc_id, data in doc_scores.items():
            result = data["doc"].copy()
            result["score"] = data["rrf_score"]
            result["original_score"] = data["doc"]["score"]
            result["best_rank"] = data["best_rank"]
            merged_results.append(result)
        
        # Sort by RRF score
        merged_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top-k results
        return merged_results[:limit] 