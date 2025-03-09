"""
Retrieval Inspector

This module provides the RetrievalInspector class for inspecting retrieval results.
"""

import logging
import re
import math
import statistics
from collections import Counter
from typing import List, Dict, Any, Optional, Union, Tuple, Set, Callable

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from .base_debugger import BaseRetrievalDebugger

# Configure logging
logger = logging.getLogger(__name__)


class RetrievalInspector(BaseRetrievalDebugger):
    """
    Retrieval inspector for analyzing and diagnosing retrieval results.
    
    This class helps inspect retrieval results, calculate various metrics,
    identify patterns, and provide insights to improve retrieval performance.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.7,
        relevance_threshold: float = 0.5,
        min_token_overlap: int = 3,
        analyze_content: bool = True,
        analyze_metadata: bool = True,
        similarity_method: str = "tfidf",
        custom_similarity_fn: Optional[Callable[[str, str], float]] = None,
        include_charts: bool = False,
        max_insights: int = 5
    ):
        """
        Initialize a RetrievalInspector.
        
        Args:
            similarity_threshold: Threshold for considering documents similar
            relevance_threshold: Threshold for considering a document relevant
            min_token_overlap: Minimum token overlap for keyword analysis
            analyze_content: Whether to analyze document content
            analyze_metadata: Whether to analyze document metadata
            similarity_method: Method to use for text similarity ("tfidf", "jaccard", "custom")
            custom_similarity_fn: Custom function for text similarity
            include_charts: Whether to include chart data in results
            max_insights: Maximum number of insights to generate
        """
        self.similarity_threshold = similarity_threshold
        self.relevance_threshold = relevance_threshold
        self.min_token_overlap = min_token_overlap
        self.analyze_content = analyze_content
        self.analyze_metadata = analyze_metadata
        self.similarity_method = similarity_method
        self.custom_similarity_fn = custom_similarity_fn
        self.include_charts = include_charts
        self.max_insights = max_insights
        
        # Initialize TF-IDF vectorizer for text similarity
        self.vectorizer = None
        
        # Validate similarity method
        valid_methods = ["tfidf", "jaccard", "custom"]
        if self.similarity_method not in valid_methods:
            raise ValueError(f"Invalid similarity_method: {similarity_method}. Must be one of {valid_methods}")
        
        if self.similarity_method == "custom" and not self.custom_similarity_fn:
            raise ValueError("custom_similarity_fn must be provided when using 'custom' similarity method")
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis."""
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _extract_content(self, result: Dict[str, Any]) -> str:
        """Extract content from a result document."""
        # Try common field names for content
        content_fields = ['content', 'text', 'document', 'passage', 'body', 'chunk']
        
        for field in content_fields:
            if field in result and isinstance(result[field], str):
                return result[field]
        
        # If metadata exists and contains content-like fields
        if 'metadata' in result and isinstance(result['metadata'], dict):
            for field in content_fields:
                if field in result['metadata'] and isinstance(result['metadata'][field], str):
                    return result['metadata'][field]
        
        # Fallback: concatenate all string values
        content = ""
        for key, value in result.items():
            if isinstance(value, str) and key != 'id' and key != 'score':
                content += value + " "
        
        return content.strip()
    
    def _extract_metadata(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from a result document."""
        if 'metadata' in result and isinstance(result['metadata'], dict):
            return result['metadata']
        
        # If no explicit metadata field, construct metadata from non-content fields
        metadata = {}
        content_fields = ['content', 'text', 'document', 'passage']
        score_fields = ['score', 'similarity', 'relevance']
        id_fields = ['id', 'document_id', 'doc_id']
        
        for key, value in result.items():
            if (key not in content_fields and 
                key not in score_fields and 
                key not in id_fields and 
                not isinstance(value, (dict, list))):
                metadata[key] = value
        
        return metadata
    
    def _calculate_jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts."""
        if not text1 or not text2:
            return 0.0
        
        # Tokenize
        tokens1 = set(self._preprocess_text(text1).split())
        tokens2 = set(self._preprocess_text(text2).split())
        
        # Calculate Jaccard similarity
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _calculate_tfidf_similarity(self, text1: str, text2: str) -> float:
        """Calculate TF-IDF cosine similarity between two texts."""
        if not text1 or not text2:
            return 0.0
        
        # Preprocess texts
        text1 = self._preprocess_text(text1)
        text2 = self._preprocess_text(text2)
        
        if not text1 or not text2:
            return 0.0
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer()
        
        try:
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating TF-IDF similarity: {str(e)}")
            return 0.0
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity based on the selected method."""
        if self.similarity_method == "tfidf":
            return self._calculate_tfidf_similarity(text1, text2)
        elif self.similarity_method == "jaccard":
            return self._calculate_jaccard_similarity(text1, text2)
        elif self.similarity_method == "custom" and self.custom_similarity_fn:
            try:
                return float(self.custom_similarity_fn(text1, text2))
            except Exception as e:
                logger.error(f"Error in custom similarity function: {str(e)}")
                return 0.0
        
        return 0.0
    
    def _calculate_query_result_similarity(self, query: str, result: Dict[str, Any]) -> float:
        """Calculate similarity between query and result content."""
        content = self._extract_content(result)
        return self._calculate_text_similarity(query, content)
    
    def _extract_common_keywords(self, query: str, results: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
        """Extract common keywords between query and results."""
        # Preprocess query
        query_tokens = set(self._preprocess_text(query).split())
        
        # Extract content from results
        contents = [self._extract_content(result) for result in results]
        
        # Count keyword occurrences
        keyword_counts = Counter()
        
        for content in contents:
            content_tokens = set(self._preprocess_text(content).split())
            overlapping_tokens = query_tokens.intersection(content_tokens)
            
            for token in overlapping_tokens:
                if len(token) >= self.min_token_overlap:
                    keyword_counts[token] += 1
        
        # Return sorted keywords by frequency
        return sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
    
    def _calculate_result_diversity(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate diversity among retrieval results."""
        contents = [self._extract_content(result) for result in results]
        n_results = len(contents)
        
        if n_results <= 1:
            return {
                "diversity_score": 1.0,
                "similar_pairs": [],
                "duplicate_count": 0
            }
        
        # Calculate pairwise similarities
        similarities = []
        similar_pairs = []
        
        for i in range(n_results):
            for j in range(i + 1, n_results):
                sim = self._calculate_text_similarity(contents[i], contents[j])
                similarities.append(sim)
                
                if sim >= self.similarity_threshold:
                    similar_pairs.append({
                        "index1": i,
                        "index2": j,
                        "similarity": sim,
                        "id1": results[i].get("id", str(i)),
                        "id2": results[j].get("id", str(j))
                    })
        
        # Calculate diversity metrics
        if similarities:
            avg_similarity = sum(similarities) / len(similarities)
            diversity_score = 1.0 - avg_similarity
        else:
            diversity_score = 1.0
        
        duplicate_count = sum(1 for sim in similarities if sim >= self.similarity_threshold)
        
        return {
            "diversity_score": diversity_score,
            "similar_pairs": similar_pairs,
            "duplicate_count": duplicate_count,
            "average_similarity": sum(similarities) / len(similarities) if similarities else 0.0
        }
    
    def _analyze_metadata_patterns(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in result metadata."""
        if not results:
            return {"fields": {}, "common_values": {}}
        
        metadata_fields = {}
        field_values = {}
        
        for result in results:
            metadata = self._extract_metadata(result)
            
            for field, value in metadata.items():
                # Count field occurrences
                metadata_fields[field] = metadata_fields.get(field, 0) + 1
                
                # Track field values
                if field not in field_values:
                    field_values[field] = Counter()
                
                # Handle different value types
                if isinstance(value, (str, int, float, bool)):
                    field_values[field][str(value)] += 1
        
        # Calculate common values (present in at least 30% of results)
        threshold = max(1, len(results) * 0.3)
        common_values = {}
        
        for field, counter in field_values.items():
            common = [
                {"value": value, "count": count}
                for value, count in counter.most_common(5)
                if count >= threshold
            ]
            if common:
                common_values[field] = common
        
        return {
            "fields": {field: count for field, count in metadata_fields.items()},
            "common_values": common_values
        }
    
    def _calculate_metrics(self, query: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate retrieval quality metrics."""
        if not results:
            return {"query_similarity": [], "average_similarity": 0.0}
        
        # Calculate similarity between query and each result
        query_similarities = []
        
        for result in results:
            similarity = self._calculate_query_result_similarity(query, result)
            
            # Add original score if available
            score_info = {
                "id": result.get("id", None),
                "query_similarity": similarity,
            }
            
            if "score" in result:
                score_info["original_score"] = result["score"]
                
            query_similarities.append(score_info)
        
        # Calculate average query similarity
        similarities = [item["query_similarity"] for item in query_similarities]
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        
        # Calculate score distribution
        if len(similarities) >= 2:
            std_dev = statistics.stdev(similarities)
            variance = statistics.variance(similarities)
        else:
            std_dev = 0.0
            variance = 0.0
            
        return {
            "query_similarity": query_similarities,
            "average_similarity": avg_similarity,
            "std_deviation": std_dev,
            "variance": variance,
            "max_similarity": max(similarities) if similarities else 0.0,
            "min_similarity": min(similarities) if similarities else 0.0
        }
    
    def _generate_insights(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate actionable insights based on analysis results."""
        insights = []
        
        # Insight 1: Low query similarity
        avg_similarity = analysis_results.get("metrics", {}).get("average_similarity", 0.0)
        if avg_similarity < self.relevance_threshold:
            insights.append(
                f"Low relevance detected (avg similarity: {avg_similarity:.2f}). "
                "Consider reformulating the query or using query expansion techniques."
            )
        
        # Insight 2: Low diversity
        diversity = analysis_results.get("diversity", {}).get("diversity_score", 1.0)
        if diversity < 0.3:
            insights.append(
                f"Low diversity in results (score: {diversity:.2f}). "
                "Consider adding diversity re-ranking or filtering duplicate information."
            )
        
        # Insight 3: Duplicate content
        duplicate_count = analysis_results.get("diversity", {}).get("duplicate_count", 0)
        if duplicate_count > 0:
            insights.append(
                f"Found {duplicate_count} similar document pairs. "
                "Consider implementing deduplication in your retrieval pipeline."
            )
        
        # Insight 4: Metadata patterns
        common_values = analysis_results.get("metadata_patterns", {}).get("common_values", {})
        if common_values:
            fields = list(common_values.keys())
            if fields:
                insights.append(
                    f"Strong patterns found in metadata fields: {', '.join(fields[:3])}. "
                    "Consider adding metadata filtering to improve relevance."
                )
        
        # Insight 5: Keyword mismatch
        keywords = analysis_results.get("keywords", [])
        if not keywords:
            insights.append(
                "No significant keyword overlap between query and results. "
                "Consider using query expansion or semantic search capabilities."
            )
        
        # Insight 6: High variance in similarity
        variance = analysis_results.get("metrics", {}).get("variance", 0.0)
        if variance > 0.1:
            insights.append(
                f"High variance in result relevance (variance: {variance:.2f}). "
                "Consider re-ranking results or adjusting your retrieval model."
            )
        
        # Insight 7: No results or too few results
        result_count = analysis_results.get("result_count", 0)
        if result_count < 3:
            insights.append(
                f"Very few results ({result_count}). "
                "Consider expanding your corpus or using query relaxation techniques."
            )
        
        # Limit to max_insights
        return insights[:self.max_insights]
    
    def analyze(self, query: str, results: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Analyze retrieval results for a given query.
        
        Args:
            query: The query string that produced the results
            results: List of retrieval results, each containing at least id, score, and content
            **kwargs: Additional parameters for analysis
                - similarity_threshold: Override similarity_threshold
                - relevance_threshold: Override relevance_threshold
                - analyze_content: Override analyze_content
                - analyze_metadata: Override analyze_metadata
                - include_charts: Override include_charts
        
        Returns:
            Dict[str, Any]: Analysis results containing:
                - metrics: Similarity and score metrics
                - diversity: Diversity analysis results
                - keywords: Common keywords between query and results
                - metadata_patterns: Patterns in result metadata
                - insights: Actionable insights based on analysis
        """
        # Apply overrides from kwargs
        similarity_threshold = kwargs.get('similarity_threshold', self.similarity_threshold)
        relevance_threshold = kwargs.get('relevance_threshold', self.relevance_threshold)
        analyze_content = kwargs.get('analyze_content', self.analyze_content)
        analyze_metadata = kwargs.get('analyze_metadata', self.analyze_metadata)
        include_charts = kwargs.get('include_charts', self.include_charts)
        
        # Store original values to restore later
        orig_similarity_threshold = self.similarity_threshold
        orig_relevance_threshold = self.relevance_threshold
        orig_analyze_content = self.analyze_content
        orig_analyze_metadata = self.analyze_metadata
        orig_include_charts = self.include_charts
        
        # Apply temporary overrides
        self.similarity_threshold = similarity_threshold
        self.relevance_threshold = relevance_threshold
        self.analyze_content = analyze_content
        self.analyze_metadata = analyze_metadata
        self.include_charts = include_charts
        
        try:
            # Ensure results list is not empty
            if not results:
                return {
                    "result_count": 0,
                    "metrics": {},
                    "diversity": {"diversity_score": 1.0, "similar_pairs": []},
                    "keywords": [],
                    "metadata_patterns": {"fields": {}, "common_values": {}},
                    "insights": ["No results returned for the query."]
                }
            
            analysis_results = {
                "result_count": len(results)
            }
            
            # Calculate metrics
            if analyze_content:
                analysis_results["metrics"] = self._calculate_metrics(query, results)
                analysis_results["diversity"] = self._calculate_result_diversity(results)
                analysis_results["keywords"] = self._extract_common_keywords(query, results)
            
            # Analyze metadata patterns
            if analyze_metadata:
                analysis_results["metadata_patterns"] = self._analyze_metadata_patterns(results)
            
            # Generate insights
            analysis_results["insights"] = self._generate_insights(analysis_results)
            
            # Generate chart data if requested
            if include_charts:
                chart_data = {}
                
                # Similarity distribution chart
                if analyze_content and "metrics" in analysis_results:
                    similarities = [item["query_similarity"] for item in analysis_results["metrics"].get("query_similarity", [])]
                    chart_data["similarity_distribution"] = {
                        "type": "histogram",
                        "data": similarities,
                        "bins": min(10, len(similarities)) if similarities else 0
                    }
                
                analysis_results["chart_data"] = chart_data
            
            return analysis_results
            
        finally:
            # Restore original values
            self.similarity_threshold = orig_similarity_threshold
            self.relevance_threshold = orig_relevance_threshold
            self.analyze_content = orig_analyze_content
            self.analyze_metadata = orig_analyze_metadata
            self.include_charts = orig_include_charts
    
    def compare(
        self,
        query: str,
        results_sets: List[List[Dict[str, Any]]],
        names: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compare multiple sets of retrieval results for the same query.
        
        Args:
            query: The query string that produced the results
            results_sets: List of result sets to compare
            names: Optional names for each result set
            **kwargs: Additional parameters for comparison
        
        Returns:
            Dict[str, Any]: Comparison results
        """
        if not results_sets:
            return {"comparison": [], "best_performing": None, "insights": []}
        
        # Assign default names if not provided
        if not names or len(names) != len(results_sets):
            names = [f"System {i+1}" for i in range(len(results_sets))]
        
        # Analyze each result set
        analyses = []
        
        for i, results in enumerate(results_sets):
            analysis = self.analyze(query, results, **kwargs)
            analysis["name"] = names[i]
            analyses.append(analysis)
        
        # Collect comparison metrics
        comparison = []
        
        for analysis in analyses:
            metrics = {
                "name": analysis["name"],
                "result_count": analysis["result_count"],
                "average_similarity": analysis.get("metrics", {}).get("average_similarity", 0.0),
                "diversity_score": analysis.get("diversity", {}).get("diversity_score", 0.0),
                "duplicate_count": analysis.get("diversity", {}).get("duplicate_count", 0),
                "keyword_count": len(analysis.get("keywords", []))
            }
            comparison.append(metrics)
        
        # Determine best performing system
        # Simple scoring: (similarity + diversity) / 2
        best_index = 0
        best_score = 0.0
        
        for i, metrics in enumerate(comparison):
            score = (metrics.get("average_similarity", 0.0) + metrics.get("diversity_score", 0.0)) / 2
            
            if score > best_score:
                best_score = score
                best_index = i
        
        best_performing = comparison[best_index]["name"] if comparison else None
        
        # Generate comparative insights
        insights = []
        
        if len(comparison) > 1:
            # Diversity comparison
            max_diversity = max(item.get("diversity_score", 0.0) for item in comparison)
            min_diversity = min(item.get("diversity_score", 0.0) for item in comparison)
            
            if max_diversity - min_diversity > 0.2:
                most_diverse = next(item["name"] for item in comparison if item.get("diversity_score", 0.0) == max_diversity)
                least_diverse = next(item["name"] for item in comparison if item.get("diversity_score", 0.0) == min_diversity)
                
                insights.append(
                    f"{most_diverse} provides the most diverse results, while {least_diverse} has the least diversity. "
                    "Consider techniques from the more diverse system if diversity is important."
                )
            
            # Similarity comparison
            max_similarity = max(item.get("average_similarity", 0.0) for item in comparison)
            min_similarity = min(item.get("average_similarity", 0.0) for item in comparison)
            
            if max_similarity - min_similarity > 0.2:
                most_relevant = next(item["name"] for item in comparison if item.get("average_similarity", 0.0) == max_similarity)
                least_relevant = next(item["name"] for item in comparison if item.get("average_similarity", 0.0) == min_similarity)
                
                insights.append(
                    f"{most_relevant} provides the most relevant results, while {least_relevant} has the least relevance. "
                    "Consider the retrieval approach of the more relevant system."
                )
            
            # Result count comparison
            max_count = max(item.get("result_count", 0) for item in comparison)
            min_count = min(item.get("result_count", 0) for item in comparison)
            
            if max_count > min_count * 2:
                most_results = next(item["name"] for item in comparison if item.get("result_count", 0) == max_count)
                least_results = next(item["name"] for item in comparison if item.get("result_count", 0) == min_count)
                
                insights.append(
                    f"{most_results} returns significantly more results than {least_results}. "
                    "Consider adjusting retrieval parameters for more consistent result counts."
                )
        
        return {
            "comparison": comparison,
            "best_performing": best_performing,
            "insights": insights[:self.max_insights],
            "detailed_analyses": analyses
        }
    
    def evaluate(
        self,
        query: str,
        results: List[Dict[str, Any]],
        ground_truth: Union[List[str], List[Dict[str, Any]]],
        **kwargs
    ) -> Dict[str, float]:
        """
        Evaluate retrieval results against ground truth.
        
        Args:
            query: The query string that produced the results
            results: List of retrieval results
            ground_truth: Ground truth relevant documents or IDs
            **kwargs: Additional parameters for evaluation
                - k_values: List of k values for precision@k and recall@k
                - id_field: Field name to use for ID matching
                - relevance_threshold: Threshold for considering a document relevant
        
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        k_values = kwargs.get('k_values', [1, 3, 5, 10])
        id_field = kwargs.get('id_field', 'id')
        relevance_threshold = kwargs.get('relevance_threshold', self.relevance_threshold)
        
        if not results or not ground_truth:
            return {f"precision@{k}": 0.0 for k in k_values} | {f"recall@{k}": 0.0 for k in k_values} | {"ndcg@10": 0.0, "mrr": 0.0}
        
        # Extract ground truth IDs
        if isinstance(ground_truth[0], dict):
            truth_ids = [doc.get(id_field, "") for doc in ground_truth]
        else:
            truth_ids = [str(doc) for doc in ground_truth]
        
        # Extract result IDs
        result_ids = [doc.get(id_field, "") for doc in results]
        
        # Calculate precision@k and recall@k
        metrics = {}
        
        for k in k_values:
            if k <= 0:
                continue
                
            # Limit to actual results length or k, whichever is smaller
            actual_k = min(k, len(results))
            
            # Precision@k
            relevant_at_k = sum(1 for i in range(actual_k) if result_ids[i] in truth_ids)
            precision_at_k = relevant_at_k / actual_k if actual_k > 0 else 0.0
            metrics[f"precision@{k}"] = precision_at_k
            
            # Recall@k
            recall_at_k = relevant_at_k / len(truth_ids) if truth_ids else 0.0
            metrics[f"recall@{k}"] = recall_at_k
        
        # Calculate Mean Reciprocal Rank (MRR)
        first_relevant_ranks = []
        for truth_id in truth_ids:
            if truth_id in result_ids:
                rank = result_ids.index(truth_id) + 1
                first_relevant_ranks.append(1.0 / rank)
            else:
                first_relevant_ranks.append(0.0)
        
        metrics["mrr"] = sum(first_relevant_ranks) / len(first_relevant_ranks) if first_relevant_ranks else 0.0
        
        # Calculate Normalized Discounted Cumulative Gain (NDCG@10)
        # Simplification: binary relevance (1 if in ground truth, 0 otherwise)
        k = min(10, len(results))
        dcg = 0.0
        
        for i in range(k):
            if result_ids[i] in truth_ids:
                # log_2(i+2) is used because ranking is 1-based
                dcg += 1.0 / math.log2(i + 2)
        
        # Calculate ideal DCG (IDCG)
        idcg = 0.0
        for i in range(min(k, len(truth_ids))):
            idcg += 1.0 / math.log2(i + 2)
        
        metrics["ndcg@10"] = dcg / idcg if idcg > 0 else 0.0
        
        return metrics
    
    def get_insights(self, query: str, results: List[Dict[str, Any]], **kwargs) -> List[str]:
        """
        Get actionable insights about retrieval results.
        
        Args:
            query: The query string that produced the results
            results: List of retrieval results
            **kwargs: Additional parameters for insight generation
        
        Returns:
            List[str]: List of insights or suggestions
        """
        # Analyze results and extract insights
        analysis_results = self.analyze(query, results, **kwargs)
        return analysis_results.get("insights", []) 