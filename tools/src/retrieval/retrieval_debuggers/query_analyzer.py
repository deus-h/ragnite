"""
Query Analyzer

This module provides the QueryAnalyzer class for analyzing query performance and characteristics.
"""

import logging
import re
import time
from collections import Counter
from typing import List, Dict, Any, Optional, Union, Tuple, Set, Callable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from .base_debugger import BaseRetrievalDebugger

# Configure logging
logger = logging.getLogger(__name__)


class QueryAnalyzer(BaseRetrievalDebugger):
    """
    Query analyzer for analyzing query performance and characteristics.
    
    This class helps analyze and understand queries, their performance,
    and how they affect retrieval results. It provides insights into
    query complexity, key terms, processing time, and improvement opportunities.
    """
    
    def __init__(
        self,
        max_key_terms: int = 5,
        complexity_threshold: int = 8,
        min_term_length: int = 3,
        include_entity_recognition: bool = False,
        use_tfidf_weighting: bool = True,
        custom_stopwords: Optional[List[str]] = None,
        performance_metrics: bool = True,
        query_reformulation: bool = True,
        term_weighting_method: str = "tfidf",
        max_insights: int = 5
    ):
        """
        Initialize a QueryAnalyzer.
        
        Args:
            max_key_terms: Maximum number of key terms to extract
            complexity_threshold: Word count threshold for complex queries
            min_term_length: Minimum length of terms to consider
            include_entity_recognition: Whether to perform entity recognition
            use_tfidf_weighting: Whether to use TF-IDF for term importance
            custom_stopwords: Additional stopwords to filter out
            performance_metrics: Whether to track query performance metrics
            query_reformulation: Whether to suggest query reformulations
            term_weighting_method: Method for weighting terms ("tfidf", "count", "custom")
            max_insights: Maximum number of insights to generate
        """
        self.max_key_terms = max_key_terms
        self.complexity_threshold = complexity_threshold
        self.min_term_length = min_term_length
        self.include_entity_recognition = include_entity_recognition
        self.use_tfidf_weighting = use_tfidf_weighting
        self.custom_stopwords = set(custom_stopwords or [])
        self.performance_metrics = performance_metrics
        self.query_reformulation = query_reformulation
        self.term_weighting_method = term_weighting_method
        self.max_insights = max_insights
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = None
        self.default_stopwords = self._get_default_stopwords()
        
        # Initialize entity recognizer if enabled
        if self.include_entity_recognition:
            self._initialize_entity_recognizer()
    
    def _get_default_stopwords(self) -> Set[str]:
        """Get default stopwords set."""
        default_stopwords = {
            "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
            "which", "this", "that", "these", "those", "then", "just", "so", "than",
            "such", "when", "who", "how", "where", "why", "is", "are", "was", "were",
            "be", "been", "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "shall", "should", "may", "might", "must", "can", "could", "of",
            "for", "with", "about", "to", "from", "up", "down", "in", "out", "on", 
            "off", "over", "under", "again", "further", "then", "once", "here", "there",
            "all", "any", "both", "each", "few", "more", "most", "other", "some", "such"
        }
        return default_stopwords.union(self.custom_stopwords)
    
    def _initialize_entity_recognizer(self):
        """Initialize entity recognizer."""
        try:
            import spacy
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("Spacy model not found. Downloading en_core_web_sm...")
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], 
                              check=True, capture_output=True)
                self.nlp = spacy.load("en_core_web_sm")
        except ImportError:
            logger.warning("Spacy not installed. Entity recognition disabled.")
            self.include_entity_recognition = False
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess query for analysis."""
        if not query or not isinstance(query, str):
            return ""
        
        # Convert to lowercase
        query = query.lower()
        
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query).strip()
        
        return query
    
    def _extract_key_terms(self, query: str, corpus: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """Extract key terms from the query with their importance scores."""
        preprocessed_query = self._preprocess_query(query)
        
        if not preprocessed_query:
            return []
        
        # If no corpus provided, use a single-document corpus (the query itself)
        if not corpus:
            corpus = [preprocessed_query]
        
        terms = []
        if self.term_weighting_method == "tfidf":
            # Use TF-IDF for term importance
            vectorizer = TfidfVectorizer(
                stop_words=list(self.default_stopwords),
                min_df=1,
                max_df=0.9,
                ngram_range=(1, 2)  # Include unigrams and bigrams
            )
            try:
                tfidf_matrix = vectorizer.fit_transform(corpus)
                feature_names = vectorizer.get_feature_names_out()
                
                # Get query vector (if query is in corpus, it's the first document)
                query_vector = tfidf_matrix[0]
                
                # Convert to array of (term, score) tuples
                query_terms = []
                for i, score in enumerate(query_vector.toarray()[0]):
                    if score > 0:
                        term = feature_names[i]
                        # Filter out short terms and terms without letters
                        if len(term) >= self.min_term_length and any(c.isalpha() for c in term):
                            query_terms.append((term, score))
                
                # Sort by score and get top terms
                terms = sorted(query_terms, key=lambda x: x[1], reverse=True)[:self.max_key_terms]
            except Exception as e:
                logger.error(f"Error in TF-IDF term extraction: {str(e)}")
                
        elif self.term_weighting_method == "count":
            # Use simple count vectorizer
            vectorizer = CountVectorizer(
                stop_words=list(self.default_stopwords),
                ngram_range=(1, 2)
            )
            try:
                count_matrix = vectorizer.fit_transform([preprocessed_query])
                feature_names = vectorizer.get_feature_names_out()
                
                # Convert to array of (term, count) tuples
                query_terms = []
                for i, count in enumerate(count_matrix.toarray()[0]):
                    if count > 0:
                        term = feature_names[i]
                        # Filter out short terms and terms without letters
                        if len(term) >= self.min_term_length and any(c.isalpha() for c in term):
                            query_terms.append((term, float(count)))
                
                # Sort by count and get top terms
                terms = sorted(query_terms, key=lambda x: x[1], reverse=True)[:self.max_key_terms]
            except Exception as e:
                logger.error(f"Error in count-based term extraction: {str(e)}")
        else:
            # Basic tokenization and counting
            tokens = preprocessed_query.split()
            token_counts = Counter(tokens)
            
            # Filter tokens
            filtered_tokens = [
                (token, count) for token, count in token_counts.items()
                if token not in self.default_stopwords and 
                len(token) >= self.min_term_length and
                any(c.isalpha() for c in token)
            ]
            
            # Sort by count and get top terms
            terms = sorted(filtered_tokens, key=lambda x: x[1], reverse=True)[:self.max_key_terms]
        
        return terms
    
    def _extract_entities(self, query: str) -> List[Dict[str, Any]]:
        """Extract named entities from the query."""
        if not self.include_entity_recognition:
            return []
        
        try:
            doc = self.nlp(query)
            
            entities = []
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                })
            
            return entities
        except Exception as e:
            logger.error(f"Error in entity extraction: {str(e)}")
            return []
    
    def _analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """Analyze query complexity based on length, structure, etc."""
        if not query:
            return {
                "word_count": 0,
                "character_count": 0,
                "is_complex": False,
                "question_type": "unknown",
                "has_negation": False
            }
        
        # Basic metrics
        words = query.split()
        word_count = len(words)
        char_count = len(query)
        avg_word_length = sum(len(word) for word in words) / max(1, word_count)
        
        # Check for complex query indicators
        is_complex = word_count >= self.complexity_threshold
        
        # Determine question type
        question_words = {"what", "who", "where", "when", "why", "how", "which", "whose", "whom"}
        first_word = words[0].lower() if words else ""
        
        if query.endswith("?"):
            if first_word in question_words:
                question_type = f"{first_word}-question"
            else:
                question_type = "yes-no-question"
        elif first_word in question_words:
            question_type = f"{first_word}-query"
        elif any(word.lower() in question_words for word in words[:3]):
            question_type = "indirect-question"
        else:
            question_type = "statement"
        
        # Check for negation
        negation_words = {"not", "no", "never", "neither", "nor", "without", "cannot", "can't", "won't", "wouldn't", "shouldn't", "doesn't", "don't"}
        has_negation = any(word.lower() in negation_words for word in words)
        
        # Check for comparison
        comparison_words = {"versus", "vs", "compared", "better", "worse", "difference", "between", "similar", "against"}
        has_comparison = any(word.lower() in comparison_words for word in words)
        
        return {
            "word_count": word_count,
            "character_count": char_count,
            "average_word_length": avg_word_length,
            "is_complex": is_complex,
            "question_type": question_type,
            "has_negation": has_negation,
            "has_comparison": has_comparison
        }
    
    def _measure_query_performance(self, query: str, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Measure query processing performance."""
        if not self.performance_metrics:
            return {}
        
        # Simple timing-based metrics
        start_time = time.time()
        
        # Preprocess query (simulating processing time)
        _ = self._preprocess_query(query)
        
        # Extract key terms
        _ = self._extract_key_terms(query)
        
        # Extract entities if enabled
        if self.include_entity_recognition:
            _ = self._extract_entities(query)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Calculate ratio of result count to query complexity
        word_count = len(query.split())
        result_count = len(results)
        
        # Avoid division by zero
        complexity_ratio = result_count / max(1, word_count)
        
        return {
            "processing_time": processing_time,
            "result_count": result_count,
            "complexity_ratio": complexity_ratio
        }
    
    def _suggest_query_reformulations(self, query: str, key_terms: List[Tuple[str, float]]) -> List[str]:
        """Suggest potential query reformulations to improve retrieval."""
        if not self.query_reformulation or not query:
            return []
        
        suggestions = []
        
        # Extract terms for reformulation
        term_texts = [term[0] for term in key_terms[:3]] if key_terms else []
        
        # Simplification suggestion for complex queries
        complexity = self._analyze_query_complexity(query)
        if complexity["is_complex"]:
            if term_texts:
                suggestions.append(f"Simplify to focus on key terms: \"{' '.join(term_texts)}\"")
            else:
                suggestions.append("Consider simplifying your query by focusing on fewer key concepts")
        
        # Question transformation for non-questions
        if complexity["question_type"] not in ["what-question", "how-question", "why-question"]:
            if term_texts:
                # Create different question forms
                suggestions.append(f"Try a what-question: \"What is {' '.join(term_texts)}?\"")
                suggestions.append(f"Try a how-question: \"How does {' '.join(term_texts)} work?\"")
        
        # Handle negation
        if complexity["has_negation"]:
            suggestions.append("Consider removing negation and focusing on what you want instead of what you don't want")
        
        # Specificity suggestion for short queries
        if complexity["word_count"] < 4 and not complexity["has_comparison"]:
            suggestions.append("Add more specificity to your query to improve results")
        
        # Remove redundancy for repetitive queries
        word_count = len(set(query.lower().split()))
        total_words = len(query.split())
        if word_count < total_words * 0.7:  # significant repetition
            suggestions.append("Remove repetitive terms from your query")
        
        return suggestions[:self.max_insights]
    
    def _generate_insights(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate actionable insights based on query analysis."""
        insights = []
        
        # Key term insights
        key_terms = analysis_results.get("key_terms", [])
        if key_terms:
            top_terms = [term[0] for term in key_terms[:3]]
            insights.append(f"Query focuses on key concepts: {', '.join(top_terms)}")
        else:
            insights.append("Query lacks strong topical terms. Consider adding more specific keywords.")
        
        # Complexity insights
        complexity = analysis_results.get("complexity", {})
        if complexity.get("is_complex", False):
            insights.append(
                f"Query is relatively complex ({complexity.get('word_count', 0)} words). "
                "Consider breaking it into simpler queries for better results."
            )
        
        # Question type insights
        question_type = complexity.get("question_type", "unknown")
        if question_type == "statement":
            insights.append(
                "Query is phrased as a statement. "
                "Consider reformulating as a question for potentially better results."
            )
        
        # Entity insights
        entities = analysis_results.get("entities", [])
        if entities:
            entity_types = Counter(entity["label"] for entity in entities)
            most_common_type, count = entity_types.most_common(1)[0]
            if count > 1:
                insights.append(f"Query contains multiple {most_common_type} entities. Consider focusing on one at a time.")
        
        # Performance insights
        performance = analysis_results.get("performance", {})
        if performance:
            result_count = performance.get("result_count", 0)
            if result_count == 0:
                insights.append("Query returned no results. Try broadening terms or using synonyms.")
            elif result_count < 3:
                insights.append("Query returned very few results. Consider generalizing some terms.")
                
        # Add suggested reformulations if available
        reformulations = analysis_results.get("reformulations", [])
        insights.extend(reformulations)
        
        # Limit to max_insights
        return insights[:self.max_insights]
    
    def analyze(self, query: str, results: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Analyze a query and its retrieval results.
        
        Args:
            query: The query string to analyze
            results: List of retrieval results
            **kwargs: Additional parameters for analysis
                - corpus: Optional list of documents for term relevance
                - max_key_terms: Override max_key_terms
                - include_entity_recognition: Override include_entity_recognition
        
        Returns:
            Dict[str, Any]: Analysis results containing:
                - key_terms: Key terms in the query with importance scores
                - entities: Named entities in the query (if enabled)
                - complexity: Query complexity analysis
                - performance: Query performance metrics
                - reformulations: Suggested query reformulations
                - insights: Actionable insights based on analysis
        """
        # Apply overrides from kwargs
        corpus = kwargs.get('corpus', None)
        max_key_terms = kwargs.get('max_key_terms', self.max_key_terms)
        include_entity_recognition = kwargs.get('include_entity_recognition', self.include_entity_recognition)
        
        # Store original values to restore later
        orig_max_key_terms = self.max_key_terms
        orig_include_entity_recognition = self.include_entity_recognition
        
        # Apply temporary overrides
        self.max_key_terms = max_key_terms
        self.include_entity_recognition = include_entity_recognition
        
        try:
            if not query:
                return {
                    "key_terms": [],
                    "entities": [],
                    "complexity": {},
                    "performance": {},
                    "reformulations": [],
                    "insights": ["Empty query provided. Please enter a valid query."]
                }
            
            # Extract key terms
            key_terms = self._extract_key_terms(query, corpus)
            
            # Extract entities if enabled
            entities = self._extract_entities(query) if include_entity_recognition else []
            
            # Analyze query complexity
            complexity = self._analyze_query_complexity(query)
            
            # Measure query performance
            performance = self._measure_query_performance(query, results)
            
            # Suggest query reformulations
            reformulations = self._suggest_query_reformulations(query, key_terms)
            
            # Compile analysis results
            analysis_results = {
                "key_terms": key_terms,
                "entities": entities,
                "complexity": complexity,
                "performance": performance,
                "reformulations": reformulations
            }
            
            # Generate insights
            analysis_results["insights"] = self._generate_insights(analysis_results)
            
            return analysis_results
            
        finally:
            # Restore original values
            self.max_key_terms = orig_max_key_terms
            self.include_entity_recognition = orig_include_entity_recognition
    
    def compare(
        self,
        query: str,
        results_sets: List[List[Dict[str, Any]]],
        names: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compare the same query against multiple result sets.
        
        Args:
            query: The query string to analyze
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
        
        # Base query analysis (performed once)
        base_analysis = self.analyze(query, [], **kwargs)
        
        # Analyze performance for each result set
        performance_metrics = []
        
        for i, results in enumerate(results_sets):
            performance = self._measure_query_performance(query, results)
            
            metrics = {
                "name": names[i],
                "result_count": len(results),
                "processing_time": performance.get("processing_time", 0),
                "complexity_ratio": performance.get("complexity_ratio", 0)
            }
            performance_metrics.append(metrics)
        
        # Determine best performing system based on result count and processing time
        if performance_metrics:
            # Normalize metrics
            result_counts = [m["result_count"] for m in performance_metrics]
            max_result_count = max(result_counts) if result_counts else 1
            
            processing_times = [m["processing_time"] for m in performance_metrics]
            max_processing_time = max(processing_times) if processing_times else 1
            
            # Calculate overall score: higher result count and lower processing time is better
            scores = []
            for metrics in performance_metrics:
                normalized_result_count = metrics["result_count"] / max_result_count
                normalized_processing_time = 1 - (metrics["processing_time"] / max_processing_time)  # Invert so lower is better
                
                # Overall score (equal weighting)
                score = (normalized_result_count + normalized_processing_time) / 2
                scores.append((metrics["name"], score))
            
            # Find best performing system
            best_performing = max(scores, key=lambda x: x[1])[0] if scores else None
        else:
            best_performing = None
        
        # Generate comparative insights
        insights = []
        
        if len(performance_metrics) > 1:
            # Result count comparison
            max_count = max(m["result_count"] for m in performance_metrics)
            min_count = min(m["result_count"] for m in performance_metrics)
            
            if max_count > min_count * 2:  # One system returns at least twice as many results
                most_results = next(m["name"] for m in performance_metrics if m["result_count"] == max_count)
                least_results = next(m["name"] for m in performance_metrics if m["result_count"] == min_count)
                
                insights.append(
                    f"{most_results} returns significantly more results ({max_count}) than {least_results} ({min_count}). "
                    "Consider investigating the retrieval parameters of each system."
                )
            
            # Processing time comparison
            if processing_times:
                max_time = max(processing_times)
                min_time = min(processing_times)
                
                if max_time > min_time * 2:  # One system is at least twice as slow
                    slowest = next(m["name"] for m in performance_metrics if m["processing_time"] == max_time)
                    fastest = next(m["name"] for m in performance_metrics if m["processing_time"] == min_time)
                    
                    insights.append(
                        f"{fastest} processes the query faster ({min_time:.4f}s) than {slowest} ({max_time:.4f}s). "
                        "Consider optimizing the slower system if response time is critical."
                    )
        
        # Include base query insights
        query_insights = base_analysis.get("insights", [])
        insights.extend([f"Query analysis: {insight}" for insight in query_insights])
        
        return {
            "base_analysis": base_analysis,
            "performance_comparison": performance_metrics,
            "best_performing": best_performing,
            "insights": insights[:self.max_insights]
        }
    
    def evaluate(
        self,
        query: str,
        results: List[Dict[str, Any]],
        ground_truth: Union[List[str], List[Dict[str, Any]]],
        **kwargs
    ) -> Dict[str, float]:
        """
        Evaluate query effectiveness against ground truth.
        
        This method adapts the evaluation to focus on query characteristics
        rather than just retrieval metrics.
        
        Args:
            query: The query string to evaluate
            results: List of retrieval results
            ground_truth: Ground truth relevant documents or IDs
            **kwargs: Additional parameters for evaluation
        
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        # Start with standard retrieval metrics (using parent/helper class if available)
        metrics = {}
        
        # Extract result IDs
        if results and 'id' in results[0]:
            result_ids = [doc.get('id', '') for doc in results]
            
            # Extract ground truth IDs
            if ground_truth and isinstance(ground_truth[0], dict) and 'id' in ground_truth[0]:
                truth_ids = [doc.get('id', '') for doc in ground_truth]
            else:
                truth_ids = [str(doc) for doc in ground_truth]
            
            # Calculate basic precision@k for k=5,10
            for k in [5, 10]:
                if k <= len(results):
                    # Count relevant results in top k
                    relevant_at_k = sum(1 for i in range(k) if result_ids[i] in truth_ids)
                    precision_at_k = relevant_at_k / k
                    metrics[f"precision@{k}"] = precision_at_k
        
        # Query-specific metrics
        complexity = self._analyze_query_complexity(query)
        key_terms = self._extract_key_terms(query)
        
        # Term coverage: what fraction of query key terms appear in ground truth?
        if key_terms and ground_truth:
            # Extract content from ground truth if available
            truth_texts = []
            if isinstance(ground_truth[0], dict):
                for doc in ground_truth:
                    content = ""
                    for field in ['content', 'text', 'body', 'passage']:
                        if field in doc and isinstance(doc[field], str):
                            content = doc[field]
                            break
                    if content:
                        truth_texts.append(content)
            
            # Calculate term coverage if we have content
            if truth_texts:
                term_coverage = 0
                for term, _ in key_terms:
                    # Check if term appears in any ground truth document
                    if any(term.lower() in text.lower() for text in truth_texts):
                        term_coverage += 1
                
                if key_terms:
                    metrics["term_coverage"] = term_coverage / len(key_terms)
        
        # Query complexity score (normalized 0-1, higher means more complex)
        word_count = complexity.get("word_count", 0)
        max_expected_words = 20  # Normalize relative to this maximum
        metrics["complexity_score"] = min(1.0, word_count / max_expected_words)
        
        # Add query clarity score based on key term weights
        # Higher weights indicate more specific queries
        if key_terms:
            weights = [weight for _, weight in key_terms]
            avg_weight = sum(weights) / len(weights) if weights else 0
            # Normalize to 0-1 range (assuming weights are typically 0-10)
            metrics["query_clarity"] = min(1.0, avg_weight / 10)
        else:
            metrics["query_clarity"] = 0.0
        
        return metrics
    
    def get_insights(self, query: str, results: List[Dict[str, Any]], **kwargs) -> List[str]:
        """
        Get actionable insights about the query.
        
        Args:
            query: The query string to analyze
            results: List of retrieval results
            **kwargs: Additional parameters for insight generation
        
        Returns:
            List[str]: List of insights or suggestions
        """
        # Analyze query and extract insights
        analysis_results = self.analyze(query, results, **kwargs)
        return analysis_results.get("insights", []) 