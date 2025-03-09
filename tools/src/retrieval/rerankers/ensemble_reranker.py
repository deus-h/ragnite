"""
Ensemble Reranker

This module provides the EnsembleReranker class that combines multiple rerankers
to improve reranking performance.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple

from .base_reranker import BaseReranker

# Configure logging
logger = logging.getLogger(__name__)


class EnsembleReranker(BaseReranker):
    """
    Reranker that combines multiple rerankers to improve reranking performance.
    
    This reranker applies multiple reranking strategies and combines their scores
    using various combination methods such as weighted average, max score, or
    reciprocal rank fusion.
    
    Attributes:
        rerankers: List of reranker objects.
        weights: Dictionary mapping reranker names to weights.
        combination_method: Method to combine scores from different rerankers.
        config: Configuration options for the reranker.
    """
    
    def __init__(self, 
                 rerankers: List[BaseReranker],
                 weights: Optional[Dict[str, float]] = None,
                 combination_method: str = "weighted_average",
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the EnsembleReranker.
        
        Args:
            rerankers: List of reranker objects.
            weights: Dictionary mapping reranker names to weights.
            combination_method: Method to combine scores from different rerankers.
                Options: 'weighted_average', 'max_score', 'reciprocal_rank_fusion'.
            config: Additional configuration options.
        
        Raises:
            ValueError: If no rerankers are provided or if the combination method is invalid.
        """
        super().__init__(config or {})
        
        if not rerankers:
            raise ValueError("At least one reranker must be provided")
        
        self.rerankers = rerankers
        self.combination_method = combination_method
        
        # Initialize weights
        if weights is None:
            # Equal weights by default
            self.weights = {f"reranker_{i}": 1.0 / len(rerankers) 
                           for i in range(len(rerankers))}
        else:
            self.weights = weights
            # Normalize weights if needed
            if sum(self.weights.values()) != 1.0:
                total = sum(self.weights.values())
                if total > 0:
                    self.weights = {k: v / total for k, v in self.weights.items()}
        
        # Validate combination method
        valid_methods = ["weighted_average", "max_score", "reciprocal_rank_fusion"]
        if combination_method not in valid_methods:
            raise ValueError(f"Invalid combination method: {combination_method}. "
                            f"Valid options are: {valid_methods}")
        
        # Update configuration
        self.config.update({
            "reranker_count": len(rerankers),
            "weights": self.weights,
            "combination_method": combination_method
        })
        
        logger.info(f"Initialized EnsembleReranker with {len(rerankers)} rerankers "
                   f"using {combination_method} combination method")
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], 
               top_k: Optional[int] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Rerank documents by combining results from multiple rerankers.
        
        Args:
            query: The query string.
            documents: List of document dictionaries, each with at least 'id' and 'content' or 'text' keys.
            top_k: The number of top documents to return. If None, all documents are returned.
            **kwargs: Additional keyword arguments passed to the rerankers.
        
        Returns:
            List of reranked document dictionaries with updated scores.
        """
        if not documents:
            logger.warning("No documents provided for reranking")
            return []
        
        logger.info(f"Reranking {len(documents)} documents with ensemble of {len(self.rerankers)} rerankers")
        
        # Keep a copy of original documents
        original_docs = documents.copy()
        
        # Get results from each reranker
        all_results = []
        for i, reranker in enumerate(self.rerankers):
            reranker_name = f"reranker_{i}"
            
            try:
                # Apply the current reranker
                reranked_docs = reranker.rerank(query, original_docs.copy(), None, **kwargs)
                
                if not reranked_docs:
                    logger.warning(f"Reranker {reranker_name} returned no results")
                    continue
                
                # Map document IDs to their new scores
                id_to_score = {doc.get("id", f"doc_{j}"): doc.get("score", 0.0) 
                              for j, doc in enumerate(reranked_docs)}
                
                all_results.append({
                    "name": reranker_name,
                    "weight": self.weights.get(reranker_name, 1.0 / len(self.rerankers)),
                    "scores": id_to_score
                })
                
                logger.debug(f"Reranker {reranker_name} processed {len(reranked_docs)} documents")
                
            except Exception as e:
                logger.error(f"Error applying reranker {reranker_name}: {e}")
        
        if not all_results:
            logger.warning("No successful rerankers, returning original documents")
            return original_docs
        
        # Combine the scores according to the selected method
        if self.combination_method == "weighted_average":
            combined_docs = self._combine_by_weighted_average(original_docs, all_results)
        elif self.combination_method == "max_score":
            combined_docs = self._combine_by_max_score(original_docs, all_results)
        elif self.combination_method == "reciprocal_rank_fusion":
            combined_docs = self._combine_by_reciprocal_rank_fusion(original_docs, all_results)
        else:
            logger.warning(f"Unknown combination method {self.combination_method}, using weighted_average")
            combined_docs = self._combine_by_weighted_average(original_docs, all_results)
        
        # Sort by combined score
        combined_docs.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top k if specified
        if top_k is not None:
            combined_docs = combined_docs[:top_k]
        
        logger.info(f"Ensemble reranking complete, returning {len(combined_docs)} documents")
        return combined_docs
    
    def _combine_by_weighted_average(self, 
                                    original_docs: List[Dict[str, Any]], 
                                    reranker_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Combine reranker scores using a weighted average.
        
        Args:
            original_docs: List of original document dictionaries.
            reranker_results: List of dictionaries with reranker names, weights, and scores.
        
        Returns:
            List of document dictionaries with combined scores.
        """
        # Initialize combined scores
        doc_id_to_combined_score = {}
        doc_id_to_source_scores = {}
        
        # For each document, compute the weighted average of scores from all rerankers
        for doc in original_docs:
            doc_id = doc.get("id", f"doc_{original_docs.index(doc)}")
            
            # Collect scores from all rerankers
            scores = []
            weights = []
            sources = []
            
            for result in reranker_results:
                if doc_id in result["scores"]:
                    score = result["scores"][doc_id]
                    weight = result["weight"]
                    
                    scores.append(score)
                    weights.append(weight)
                    sources.append({
                        "name": result["name"],
                        "score": score,
                        "weight": weight
                    })
            
            if scores:
                # Compute weighted average
                combined_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
                doc_id_to_combined_score[doc_id] = combined_score
                doc_id_to_source_scores[doc_id] = sources
        
        # Create new document list with combined scores
        combined_docs = []
        for doc in original_docs:
            doc_id = doc.get("id", f"doc_{original_docs.index(doc)}")
            
            # Skip documents with no scores (should be rare)
            if doc_id not in doc_id_to_combined_score:
                continue
            
            doc_copy = doc.copy()
            
            # Store the original score if available
            if "score" in doc_copy:
                doc_copy["original_score"] = doc_copy["score"]
                
            # Set the combined score
            doc_copy["score"] = doc_id_to_combined_score[doc_id]
            
            # Add source information
            doc_copy["sources"] = doc_id_to_source_scores[doc_id]
            
            combined_docs.append(doc_copy)
        
        return combined_docs
    
    def _combine_by_max_score(self, 
                             original_docs: List[Dict[str, Any]], 
                             reranker_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Combine reranker scores by taking the maximum score for each document.
        
        Args:
            original_docs: List of original document dictionaries.
            reranker_results: List of dictionaries with reranker names, weights, and scores.
        
        Returns:
            List of document dictionaries with combined scores.
        """
        # Initialize combined scores
        doc_id_to_max_score = {}
        doc_id_to_max_source = {}
        doc_id_to_all_sources = {}
        
        # For each document, find the maximum score across all rerankers
        for doc in original_docs:
            doc_id = doc.get("id", f"doc_{original_docs.index(doc)}")
            
            # Collect scores from all rerankers
            max_score = -float('inf')
            max_source = None
            all_sources = []
            
            for result in reranker_results:
                if doc_id in result["scores"]:
                    score = result["scores"][doc_id]
                    source = {
                        "name": result["name"],
                        "score": score,
                        "weight": result["weight"]
                    }
                    
                    all_sources.append(source)
                    
                    if score > max_score:
                        max_score = score
                        max_source = source
            
            if max_source is not None:
                doc_id_to_max_score[doc_id] = max_score
                doc_id_to_max_source[doc_id] = max_source
                doc_id_to_all_sources[doc_id] = all_sources
        
        # Create new document list with max scores
        combined_docs = []
        for doc in original_docs:
            doc_id = doc.get("id", f"doc_{original_docs.index(doc)}")
            
            # Skip documents with no scores (should be rare)
            if doc_id not in doc_id_to_max_score:
                continue
            
            doc_copy = doc.copy()
            
            # Store the original score if available
            if "score" in doc_copy:
                doc_copy["original_score"] = doc_copy["score"]
                
            # Set the max score
            doc_copy["score"] = doc_id_to_max_score[doc_id]
            
            # Add source information
            doc_copy["max_source"] = doc_id_to_max_source[doc_id]
            doc_copy["sources"] = doc_id_to_all_sources[doc_id]
            
            combined_docs.append(doc_copy)
        
        return combined_docs
    
    def _combine_by_reciprocal_rank_fusion(self, 
                                          original_docs: List[Dict[str, Any]], 
                                          reranker_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Combine reranker scores using reciprocal rank fusion.
        
        Args:
            original_docs: List of original document dictionaries.
            reranker_results: List of dictionaries with reranker names, weights, and scores.
        
        Returns:
            List of document dictionaries with combined scores.
        """
        # Initialize structures for tracking
        doc_id_to_ranks = {}
        doc_id_to_sources = {}
        all_doc_ids = set()
        
        # For each reranker, get the ranked list of documents
        for result in reranker_results:
            # Sort documents by score for this reranker
            sorted_doc_ids = sorted(result["scores"].keys(), 
                                   key=lambda doc_id: result["scores"][doc_id], 
                                   reverse=True)
            
            # Record the rank of each document
            for rank, doc_id in enumerate(sorted_doc_ids, start=1):
                all_doc_ids.add(doc_id)
                
                if doc_id not in doc_id_to_ranks:
                    doc_id_to_ranks[doc_id] = []
                    doc_id_to_sources[doc_id] = []
                
                doc_id_to_ranks[doc_id].append({
                    "name": result["name"],
                    "rank": rank,
                    "weight": result["weight"]
                })
                
                doc_id_to_sources[doc_id].append({
                    "name": result["name"],
                    "score": result["scores"][doc_id],
                    "rank": rank,
                    "weight": result["weight"]
                })
        
        # Compute RRF scores (k=60 is a common default value)
        k = self.config.get("rrf_k", 60)
        doc_id_to_rrf_score = {}
        
        for doc_id in all_doc_ids:
            # Get all ranks for this document
            ranks = doc_id_to_ranks.get(doc_id, [])
            
            if not ranks:
                continue
                
            # Compute weighted RRF score
            rrf_score = 0.0
            total_weight = sum(r["weight"] for r in ranks)
            
            for rank_info in ranks:
                rank = rank_info["rank"]
                weight = rank_info["weight"]
                rrf_score += weight * (1.0 / (k + rank))
            
            # Normalize by total weight
            if total_weight > 0:
                rrf_score /= total_weight
                
            doc_id_to_rrf_score[doc_id] = rrf_score
        
        # Create new document list with RRF scores
        combined_docs = []
        for doc in original_docs:
            doc_id = doc.get("id", f"doc_{original_docs.index(doc)}")
            
            # Skip documents with no RRF score
            if doc_id not in doc_id_to_rrf_score:
                continue
            
            doc_copy = doc.copy()
            
            # Store the original score if available
            if "score" in doc_copy:
                doc_copy["original_score"] = doc_copy["score"]
                
            # Set the RRF score
            doc_copy["score"] = doc_id_to_rrf_score[doc_id]
            
            # Add source information
            doc_copy["sources"] = doc_id_to_sources[doc_id]
            doc_copy["ranks"] = doc_id_to_ranks[doc_id]
            
            combined_docs.append(doc_copy)
        
        return combined_docs
    
    def add_reranker(self, reranker: BaseReranker, weight: float = None) -> None:
        """
        Add a new reranker to the ensemble.
        
        Args:
            reranker: The reranker object to add.
            weight: The weight for the new reranker. If None, uses the average of existing weights.
        """
        reranker_name = f"reranker_{len(self.rerankers)}"
        self.rerankers.append(reranker)
        
        # Determine weight for the new reranker
        if weight is None:
            weight = 1.0 / len(self.rerankers)
        
        # Add the new reranker to weights
        self.weights[reranker_name] = weight
        
        # Re-normalize weights
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}
        
        # Update configuration
        self.config["reranker_count"] = len(self.rerankers)
        self.config["weights"] = self.weights
        
        logger.info(f"Added new reranker {reranker_name} with weight {weight}")
    
    def remove_reranker(self, index: int) -> None:
        """
        Remove a reranker from the ensemble.
        
        Args:
            index: The index of the reranker to remove.
        
        Raises:
            ValueError: If the index is out of range.
        """
        if index < 0 or index >= len(self.rerankers):
            raise ValueError(f"Invalid reranker index: {index}")
        
        reranker_name = f"reranker_{index}"
        del self.rerankers[index]
        
        # Remove the reranker from weights
        if reranker_name in self.weights:
            del self.weights[reranker_name]
        
        # Rename the remaining rerankers and update weights
        new_weights = {}
        for i, reranker in enumerate(self.rerankers):
            old_name = f"reranker_{i if i < index else i + 1}"
            new_name = f"reranker_{i}"
            
            if old_name in self.weights:
                new_weights[new_name] = self.weights[old_name]
            else:
                new_weights[new_name] = 1.0 / len(self.rerankers)
        
        self.weights = new_weights
        
        # Re-normalize weights
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}
        
        # Update configuration
        self.config["reranker_count"] = len(self.rerankers)
        self.config["weights"] = self.weights
        
        logger.info(f"Removed reranker at index {index}")
    
    def set_weights(self, weights: Dict[str, float]) -> None:
        """
        Set the weights for the rerankers.
        
        Args:
            weights: Dictionary mapping reranker names to weights.
        """
        self.weights = weights
        
        # Re-normalize weights
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}
        
        # Update configuration
        self.config["weights"] = self.weights
        
        logger.info(f"Updated weights: {self.weights}")
    
    def set_combination_method(self, method: str) -> None:
        """
        Set the combination method.
        
        Args:
            method: The combination method to use.
        
        Raises:
            ValueError: If the method is invalid.
        """
        valid_methods = ["weighted_average", "max_score", "reciprocal_rank_fusion"]
        if method not in valid_methods:
            raise ValueError(f"Invalid combination method: {method}. "
                            f"Valid options are: {valid_methods}")
        
        self.combination_method = method
        self.config["combination_method"] = method
        
        logger.info(f"Set combination method to: {method}")
    
    def get_reranker_info(self) -> List[Dict[str, Any]]:
        """
        Get information about the rerankers in the ensemble.
        
        Returns:
            List of dictionaries with reranker information.
        """
        reranker_info = []
        
        for i, reranker in enumerate(self.rerankers):
            reranker_name = f"reranker_{i}"
            
            info = {
                "name": reranker_name,
                "weight": self.weights.get(reranker_name, 1.0 / len(self.rerankers)),
                "type": type(reranker).__name__
            }
            
            # Include additional information if available
            if hasattr(reranker, "get_model_info"):
                info["model_info"] = reranker.get_model_info()
            
            reranker_info.append(info)
        
        return reranker_info
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Update the reranker configuration.
        
        Args:
            config: New configuration parameters.
        """
        super().set_config(config)
        
        # Update ensemble-specific configuration
        if "combination_method" in config:
            try:
                self.set_combination_method(config["combination_method"])
            except ValueError as e:
                logger.warning(str(e))
        
        if "weights" in config:
            self.set_weights(config["weights"])
        
        if "rrf_k" in config and self.combination_method == "reciprocal_rank_fusion":
            self.config["rrf_k"] = config["rrf_k"]
    
    @property
    def supported_backends(self) -> List[str]:
        """
        Get the list of supported backend types.
        
        Returns:
            List of supported backend types.
        """
        # Collect unique backends supported by all rerankers
        all_backends = set()
        for reranker in self.rerankers:
            if hasattr(reranker, "supported_backends"):
                all_backends.update(reranker.supported_backends)
        
        return list(all_backends) 