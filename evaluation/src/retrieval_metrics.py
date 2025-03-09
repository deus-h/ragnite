"""
Retrieval Metrics

This module provides metrics for evaluating the retrieval component of RAG systems.
"""

import numpy as np
from typing import List, Dict, Any, Union, Optional, Set, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score
import spacy
import torch
from sentence_transformers import SentenceTransformer


class RetrievalMetrics:
    """
    A class for evaluating the retrieval performance of RAG systems.
    """
    
    def __init__(
        self,
        use_semantic_similarity: bool = True,
        semantic_model_name: str = "all-MiniLM-L6-v2",
        relevance_threshold: float = 0.7,
        use_spacy: bool = True,
        spacy_model: str = "en_core_web_md",
    ):
        """
        Initialize the retrieval metrics calculator.
        
        Args:
            use_semantic_similarity: Whether to use semantic similarity for relevance
            semantic_model_name: Name of the sentence transformer model to use
            relevance_threshold: Threshold for considering a document relevant
            use_spacy: Whether to use spaCy for text processing
            spacy_model: Name of the spaCy model to use
        """
        self.use_semantic_similarity = use_semantic_similarity
        self.relevance_threshold = relevance_threshold
        
        # Initialize models if needed
        if use_semantic_similarity:
            try:
                self.semantic_model = SentenceTransformer(semantic_model_name)
            except Exception as e:
                print(f"Warning: Could not load sentence transformer model: {e}")
                self.semantic_model = None
                self.use_semantic_similarity = False
        
        if use_spacy:
            try:
                self.nlp = spacy.load(spacy_model)
            except Exception as e:
                print(f"Warning: Could not load spaCy model: {e}")
                self.nlp = None
                self.use_spacy = False
        else:
            self.nlp = None
            self.use_spacy = False
    
    def precision_at_k(
        self,
        retrieved_docs: List[str],
        relevant_docs: List[str],
        k: Optional[int] = None,
    ) -> float:
        """
        Calculate precision at k for retrieval.
        
        Args:
            retrieved_docs: List of retrieved document IDs or contents
            relevant_docs: List of relevant document IDs or contents
            k: Number of top documents to consider (default: all)
            
        Returns:
            Precision at k score (0 to 1)
        """
        if not retrieved_docs:
            return 0.0
        
        if k is None or k > len(retrieved_docs):
            k = len(retrieved_docs)
        
        # Consider only top-k results
        retrieved_k = retrieved_docs[:k]
        
        # Convert to sets for intersection
        if isinstance(retrieved_docs[0], str) and isinstance(relevant_docs[0], str):
            # If we're working with text content, use semantic matching
            relevant_set = set(relevant_docs)
            if self.use_semantic_similarity and self.semantic_model:
                # Compute embeddings for all documents
                retrieved_embeds = self.semantic_model.encode(retrieved_k)
                relevant_embeds = self.semantic_model.encode(list(relevant_set))
                
                # Calculate similarity and count matches above threshold
                matches = 0
                for r_embed in retrieved_embeds:
                    # Calculate cosine similarity with all relevant docs
                    similarities = torch.nn.functional.cosine_similarity(
                        torch.tensor(r_embed).unsqueeze(0),
                        torch.tensor(relevant_embeds),
                        dim=1
                    )
                    # If any similarity is above threshold, count as match
                    if torch.max(similarities) >= self.relevance_threshold:
                        matches += 1
                
                return matches / k
            else:
                # Fall back to exact matching
                matches = sum(1 for doc in retrieved_k if doc in relevant_set)
                return matches / k
        else:
            # If we're working with document IDs, use exact matching
            retrieved_set = set(retrieved_k)
            relevant_set = set(relevant_docs)
            matches = len(retrieved_set.intersection(relevant_set))
            return matches / k
    
    def recall_at_k(
        self,
        retrieved_docs: List[str],
        relevant_docs: List[str],
        k: Optional[int] = None,
    ) -> float:
        """
        Calculate recall at k for retrieval.
        
        Args:
            retrieved_docs: List of retrieved document IDs or contents
            relevant_docs: List of relevant document IDs or contents
            k: Number of top documents to consider (default: all)
            
        Returns:
            Recall at k score (0 to 1)
        """
        if not relevant_docs:
            return 1.0  # Perfect recall if there are no relevant docs
        
        if not retrieved_docs:
            return 0.0
        
        if k is None or k > len(retrieved_docs):
            k = len(retrieved_docs)
        
        # Consider only top-k results
        retrieved_k = retrieved_docs[:k]
        
        # Convert to sets for intersection
        if isinstance(retrieved_docs[0], str) and isinstance(relevant_docs[0], str):
            # If we're working with text content, use semantic matching
            relevant_set = set(relevant_docs)
            if self.use_semantic_similarity and self.semantic_model:
                # Compute embeddings for all documents
                retrieved_embeds = self.semantic_model.encode(retrieved_k)
                relevant_embeds = self.semantic_model.encode(list(relevant_set))
                
                # Calculate similarity and count matches above threshold
                matched_relevant = set()
                for i, rel_embed in enumerate(relevant_embeds):
                    for ret_embed in retrieved_embeds:
                        similarity = torch.nn.functional.cosine_similarity(
                            torch.tensor(rel_embed).unsqueeze(0),
                            torch.tensor(ret_embed).unsqueeze(0),
                            dim=1
                        ).item()
                        if similarity >= self.relevance_threshold:
                            matched_relevant.add(i)
                            break
                
                return len(matched_relevant) / len(relevant_set)
            else:
                # Fall back to exact matching
                matches = sum(1 for doc in relevant_set if doc in retrieved_k)
                return matches / len(relevant_set)
        else:
            # If we're working with document IDs, use exact matching
            retrieved_set = set(retrieved_k)
            relevant_set = set(relevant_docs)
            matches = len(retrieved_set.intersection(relevant_set))
            return matches / len(relevant_set)
    
    def f1_at_k(
        self,
        retrieved_docs: List[str],
        relevant_docs: List[str],
        k: Optional[int] = None,
    ) -> float:
        """
        Calculate F1 score at k for retrieval.
        
        Args:
            retrieved_docs: List of retrieved document IDs or contents
            relevant_docs: List of relevant document IDs or contents
            k: Number of top documents to consider (default: all)
            
        Returns:
            F1 score at k (0 to 1)
        """
        precision = self.precision_at_k(retrieved_docs, relevant_docs, k)
        recall = self.recall_at_k(retrieved_docs, relevant_docs, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def mean_average_precision(
        self,
        retrieved_docs_list: List[List[str]],
        relevant_docs_list: List[List[str]],
    ) -> float:
        """
        Calculate Mean Average Precision (MAP) across multiple queries.
        
        Args:
            retrieved_docs_list: List of retrieved document lists for each query
            relevant_docs_list: List of relevant document lists for each query
            
        Returns:
            MAP score (0 to 1)
        """
        if not retrieved_docs_list:
            return 0.0
        
        avg_precisions = []
        
        for retrieved_docs, relevant_docs in zip(retrieved_docs_list, relevant_docs_list):
            if not relevant_docs:
                continue
                
            # Calculate precision at each relevant document position
            precisions = []
            relevant_so_far = 0
            
            for i, doc in enumerate(retrieved_docs):
                is_relevant = False
                
                # Check if the current document is relevant
                if isinstance(doc, str) and isinstance(relevant_docs[0], str):
                    # Text content comparison
                    if self.use_semantic_similarity and self.semantic_model:
                        doc_embed = self.semantic_model.encode([doc])[0]
                        relevant_embeds = self.semantic_model.encode(relevant_docs)
                        similarities = torch.nn.functional.cosine_similarity(
                            torch.tensor(doc_embed).unsqueeze(0),
                            torch.tensor(relevant_embeds),
                            dim=1
                        )
                        is_relevant = torch.max(similarities) >= self.relevance_threshold
                    else:
                        is_relevant = doc in relevant_docs
                else:
                    # Document ID comparison
                    is_relevant = doc in relevant_docs
                
                if is_relevant:
                    relevant_so_far += 1
                    precisions.append(relevant_so_far / (i + 1))
            
            if precisions:
                avg_precisions.append(sum(precisions) / len(relevant_docs))
        
        if not avg_precisions:
            return 0.0
            
        return sum(avg_precisions) / len(avg_precisions)
    
    def ndcg_at_k(
        self,
        retrieved_docs: List[str],
        relevant_docs: Dict[str, float],
        k: Optional[int] = None,
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG) at k.
        
        Args:
            retrieved_docs: List of retrieved document IDs or contents
            relevant_docs: Dictionary mapping relevant document IDs or contents to relevance scores
            k: Number of top documents to consider (default: all)
            
        Returns:
            NDCG at k score (0 to 1)
        """
        if not retrieved_docs or not relevant_docs:
            return 0.0
        
        if k is None or k > len(retrieved_docs):
            k = len(retrieved_docs)
        
        # Consider only top-k results
        retrieved_k = retrieved_docs[:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, doc in enumerate(retrieved_k):
            relevance = 0.0
            
            # Determine relevance score for the document
            if isinstance(doc, str) and all(isinstance(key, str) for key in relevant_docs.keys()):
                # Text content comparison
                if self.use_semantic_similarity and self.semantic_model:
                    doc_embed = self.semantic_model.encode([doc])[0]
                    for rel_doc, rel_score in relevant_docs.items():
                        rel_embed = self.semantic_model.encode([rel_doc])[0]
                        similarity = torch.nn.functional.cosine_similarity(
                            torch.tensor(doc_embed).unsqueeze(0),
                            torch.tensor(rel_embed).unsqueeze(0),
                            dim=1
                        ).item()
                        if similarity >= self.relevance_threshold:
                            relevance = rel_score
                            break
                else:
                    relevance = relevant_docs.get(doc, 0.0)
            else:
                # Document ID comparison
                relevance = relevant_docs.get(doc, 0.0)
            
            # Calculate DCG at position i+1
            if i == 0:
                dcg += relevance
            else:
                dcg += relevance / np.log2(i + 2)
        
        # Calculate IDCG (ideal DCG)
        ideal_relevance_scores = sorted(relevant_docs.values(), reverse=True)[:k]
        idcg = 0.0
        for i, relevance in enumerate(ideal_relevance_scores):
            if i == 0:
                idcg += relevance
            else:
                idcg += relevance / np.log2(i + 2)
        
        if idcg == 0:
            return 0.0
            
        return dcg / idcg
    
    def context_relevance(
        self,
        query: str,
        retrieved_contexts: List[str],
        method: str = "semantic",
    ) -> List[float]:
        """
        Calculate relevance scores between a query and retrieved contexts.
        
        Args:
            query: The query text
            retrieved_contexts: List of retrieved context texts
            method: Method to use for relevance calculation ('semantic', 'bm25', 'overlap')
            
        Returns:
            List of relevance scores for each context
        """
        if not retrieved_contexts:
            return []
        
        if method == "semantic" and self.use_semantic_similarity and self.semantic_model:
            # Use semantic similarity
            query_embed = self.semantic_model.encode([query])[0]
            context_embeds = self.semantic_model.encode(retrieved_contexts)
            
            # Calculate cosine similarity between query and each context
            similarities = torch.nn.functional.cosine_similarity(
                torch.tensor(query_embed).unsqueeze(0),
                torch.tensor(context_embeds),
                dim=1
            ).tolist()
            
            return similarities
            
        elif method == "overlap" and self.use_spacy and self.nlp:
            # Use token overlap
            query_doc = self.nlp(query)
            relevance_scores = []
            
            query_tokens = set([token.lemma_.lower() for token in query_doc if not token.is_stop and not token.is_punct])
            if not query_tokens:
                return [0.0] * len(retrieved_contexts)
                
            for context in retrieved_contexts:
                context_doc = self.nlp(context)
                context_tokens = set([token.lemma_.lower() for token in context_doc if not token.is_stop and not token.is_punct])
                
                if not context_tokens:
                    relevance_scores.append(0.0)
                    continue
                
                # Calculate Jaccard similarity
                overlap = len(query_tokens.intersection(context_tokens))
                union = len(query_tokens.union(context_tokens))
                
                relevance_scores.append(overlap / union if union > 0 else 0.0)
            
            return relevance_scores
            
        else:
            # Fall back to simple term overlap
            query_terms = set(query.lower().split())
            relevance_scores = []
            
            for context in retrieved_contexts:
                context_terms = set(context.lower().split())
                overlap = len(query_terms.intersection(context_terms))
                union = len(query_terms.union(context_terms))
                
                relevance_scores.append(overlap / union if union > 0 else 0.0)
            
            return relevance_scores
    
    def evaluate_all(
        self,
        queries: List[str],
        retrieved_docs_list: List[List[str]],
        relevant_docs_list: List[List[str]],
        k_values: List[int] = [1, 3, 5, 10],
    ) -> Dict[str, Any]:
        """
        Evaluate retrieval using multiple metrics.
        
        Args:
            queries: List of query texts
            retrieved_docs_list: List of retrieved document lists for each query
            relevant_docs_list: List of relevant document lists for each query
            k_values: List of k values to calculate metrics at
            
        Returns:
            Dictionary of evaluation results
        """
        results = {
            "precision": {},
            "recall": {},
            "f1": {},
            "context_relevance": [],
            "map": 0.0,
        }
        
        # Calculate precision, recall, and F1 at different k values
        for k in k_values:
            precision_values = []
            recall_values = []
            f1_values = []
            
            for retrieved_docs, relevant_docs in zip(retrieved_docs_list, relevant_docs_list):
                precision = self.precision_at_k(retrieved_docs, relevant_docs, k)
                recall = self.recall_at_k(retrieved_docs, relevant_docs, k)
                f1 = self.f1_at_k(retrieved_docs, relevant_docs, k)
                
                precision_values.append(precision)
                recall_values.append(recall)
                f1_values.append(f1)
            
            results["precision"][f"@{k}"] = sum(precision_values) / len(precision_values) if precision_values else 0.0
            results["recall"][f"@{k}"] = sum(recall_values) / len(recall_values) if recall_values else 0.0
            results["f1"][f"@{k}"] = sum(f1_values) / len(f1_values) if f1_values else 0.0
        
        # Calculate MAP
        results["map"] = self.mean_average_precision(retrieved_docs_list, relevant_docs_list)
        
        # Calculate context relevance
        if queries:
            for query, retrieved_docs in zip(queries, retrieved_docs_list):
                relevance_scores = self.context_relevance(query, retrieved_docs)
                results["context_relevance"].append({
                    "query": query,
                    "scores": relevance_scores,
                    "average": sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
                })
        
        return results 