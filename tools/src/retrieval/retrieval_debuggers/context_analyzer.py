"""
Context Analyzer

This module provides the ContextAnalyzer class for analyzing retrieved context.
"""

import logging
import re
import math
import statistics
from collections import Counter
from typing import List, Dict, Any, Optional, Union, Tuple, Set, Callable

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from .base_debugger import BaseRetrievalDebugger

# Configure logging
logger = logging.getLogger(__name__)


class ContextAnalyzer(BaseRetrievalDebugger):
    """
    Context analyzer for analyzing retrieved context quality and characteristics.
    
    This class helps analyze the content retrieved by RAG systems,
    evaluating relevance, information density, diversity, and other
    characteristics that impact generation quality.
    """
    
    def __init__(
        self,
        content_field: str = "content",
        metadata_fields: Optional[List[str]] = None,
        analyze_relevance: bool = True,
        analyze_diversity: bool = True,
        analyze_information: bool = True,
        analyze_readability: bool = True,
        similarity_threshold: float = 0.7,
        relevance_threshold: float = 0.5,
        similarity_method: str = "tfidf",
        readability_metrics: bool = True,
        sentiment_analysis: bool = False,
        custom_similarity_fn: Optional[Callable[[str, str], float]] = None,
        max_insights: int = 5
    ):
        """
        Initialize a ContextAnalyzer.
        
        Args:
            content_field: Field name for document content
            metadata_fields: List of metadata fields to analyze
            analyze_relevance: Whether to analyze relevance to the query
            analyze_diversity: Whether to analyze content diversity
            analyze_information: Whether to analyze information density
            analyze_readability: Whether to analyze readability
            similarity_threshold: Threshold for considering documents similar
            relevance_threshold: Threshold for considering content relevant
            similarity_method: Method to use for text similarity ("tfidf", "jaccard", "custom")
            readability_metrics: Whether to include readability metrics
            sentiment_analysis: Whether to perform sentiment analysis
            custom_similarity_fn: Custom function for text similarity
            max_insights: Maximum number of insights to generate
        """
        self.content_field = content_field
        self.metadata_fields = metadata_fields or ["source", "page", "category", "date", "author", "title"]
        self.analyze_relevance = analyze_relevance
        self.analyze_diversity = analyze_diversity
        self.analyze_information = analyze_information
        self.analyze_readability = analyze_readability
        self.similarity_threshold = similarity_threshold
        self.relevance_threshold = relevance_threshold
        self.similarity_method = similarity_method
        self.readability_metrics = readability_metrics
        self.sentiment_analysis = sentiment_analysis
        self.custom_similarity_fn = custom_similarity_fn
        self.max_insights = max_insights
        
        # Validate similarity method
        valid_methods = ["tfidf", "jaccard", "custom"]
        if self.similarity_method not in valid_methods:
            raise ValueError(f"Invalid similarity_method: {similarity_method}. Must be one of {valid_methods}")
        
        if self.similarity_method == "custom" and not self.custom_similarity_fn:
            raise ValueError("custom_similarity_fn must be provided when using 'custom' similarity method")
        
        # Initialize sentiment analyzer if enabled
        if self.sentiment_analysis:
            self._initialize_sentiment_analyzer()
    
    def _initialize_sentiment_analyzer(self):
        """Initialize sentiment analyzer."""
        try:
            import nltk
            from nltk.sentiment import SentimentIntensityAnalyzer
            
            try:
                nltk.data.find('sentiment/vader_lexicon.zip')
            except LookupError:
                logger.info("Downloading VADER lexicon for sentiment analysis")
                nltk.download('vader_lexicon', quiet=True)
                
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        except ImportError:
            logger.warning("NLTK not installed. Sentiment analysis disabled.")
            self.sentiment_analysis = False
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis."""
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s\.\?\!]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _extract_content(self, result: Dict[str, Any]) -> str:
        """Extract content from a result document."""
        # Try specified content field first
        if self.content_field in result and isinstance(result[self.content_field], str):
            return result[self.content_field]
        
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
        extracted_metadata = {}
        
        # If there's a dedicated metadata field
        if 'metadata' in result and isinstance(result['metadata'], dict):
            for field in self.metadata_fields:
                if field in result['metadata']:
                    extracted_metadata[field] = result['metadata'][field]
        
        # Also check top-level fields
        for field in self.metadata_fields:
            if field in result and field not in extracted_metadata:
                extracted_metadata[field] = result[field]
        
        return extracted_metadata
    
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
    
    def _analyze_content_relevance(self, query: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the relevance of retrieved content to the query."""
        if not self.analyze_relevance or not query or not results:
            return {"average_relevance": 0.0, "relevance_scores": []}
        
        # Extract content from each result
        contents = [self._extract_content(result) for result in results]
        
        # Calculate relevance scores (similarity between query and content)
        relevance_scores = []
        
        for i, content in enumerate(contents):
            if not content:
                continue
                
            # Calculate similarity between query and content
            relevance = self._calculate_text_similarity(query, content)
            
            # Store relevance info
            relevance_info = {
                "index": i,
                "id": results[i].get("id", str(i)),
                "relevance": relevance,
                "is_relevant": relevance >= self.relevance_threshold
            }
            
            # Add original score if available
            if "score" in results[i]:
                relevance_info["original_score"] = results[i]["score"]
                
            relevance_scores.append(relevance_info)
        
        # Calculate overall metrics
        if relevance_scores:
            relevance_values = [item["relevance"] for item in relevance_scores]
            average_relevance = sum(relevance_values) / len(relevance_values)
            relevant_count = sum(1 for item in relevance_scores if item["is_relevant"])
            relevant_percentage = relevant_count / len(relevance_scores) * 100
        else:
            average_relevance = 0.0
            relevant_count = 0
            relevant_percentage = 0.0
        
        return {
            "average_relevance": average_relevance,
            "relevant_count": relevant_count,
            "relevant_percentage": relevant_percentage,
            "relevance_scores": relevance_scores
        }
    
    def _analyze_content_diversity(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze diversity of the retrieved content."""
        if not self.analyze_diversity or not results:
            return {"diversity_score": 1.0, "similar_pairs": []}
        
        # Extract content from each result
        contents = [self._extract_content(result) for result in results]
        n_results = len(contents)
        
        if n_results <= 1:
            return {
                "diversity_score": 1.0,
                "similar_pairs": [],
                "duplicate_count": 0,
                "unique_information_percentage": 100.0
            }
        
        # Calculate pairwise similarities
        similarities = []
        similar_pairs = []
        
        for i in range(n_results):
            for j in range(i + 1, n_results):
                if not contents[i] or not contents[j]:
                    continue
                    
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
            
            # Calculate unique information percentage (estimate)
            # Higher similarity means more duplicate information
            unique_information = 1.0 - (sum(1 for sim in similarities if sim >= self.similarity_threshold) / len(similarities))
            unique_information_percentage = unique_information * 100
        else:
            diversity_score = 1.0
            unique_information_percentage = 100.0
            
        # Count duplicate/similar content
        duplicate_count = sum(1 for sim in similarities if sim >= self.similarity_threshold)
        
        return {
            "diversity_score": diversity_score,
            "similar_pairs": similar_pairs,
            "duplicate_count": duplicate_count,
            "unique_information_percentage": unique_information_percentage,
            "average_similarity": sum(similarities) / len(similarities) if similarities else 0.0
        }
    
    def _analyze_information_density(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the information density and quality of the retrieved content."""
        if not self.analyze_information or not results:
            return {}
        
        # Extract content from each result
        contents = [self._extract_content(result) for result in results]
        
        # Calculate various information density metrics
        content_metrics = []
        
        for i, content in enumerate(contents):
            if not content:
                continue
                
            # Get preprocessed content
            processed_content = self._preprocess_text(content)
            
            # Word count
            words = processed_content.split()
            word_count = len(words)
            
            # Sentence count (rough approximation)
            sentences = re.split(r'[.!?]+', content)
            sentences = [s for s in sentences if s.strip()]
            sentence_count = len(sentences)
            
            # Unique word count
            unique_words = set(words)
            unique_word_count = len(unique_words)
            
            # Unique-to-total word ratio (lexical diversity)
            lexical_diversity = unique_word_count / max(1, word_count)
            
            # Information density score (higher means more information)
            # Simple heuristic based on sentence length, lexical diversity
            avg_sentence_length = word_count / max(1, sentence_count)
            
            # Normalize metrics for scoring
            normalized_sentence_length = min(1.0, avg_sentence_length / 25.0)  # 25 words is considered a long sentence
            normalized_lexical_diversity = min(1.0, lexical_diversity * 2.0)  # 0.5 diversity is considered good
            
            # Combined information density score
            info_density_score = (normalized_sentence_length + normalized_lexical_diversity) / 2
            
            # Number of potential entities (capitalized non-sentence-initial words)
            entity_count = 0
            for j, word in enumerate(content.split()):
                if j > 0 and word and word[0].isupper():
                    entity_count += 1
            
            # Store metrics
            metrics = {
                "index": i,
                "id": results[i].get("id", str(i)),
                "word_count": word_count,
                "sentence_count": sentence_count,
                "unique_word_count": unique_word_count,
                "lexical_diversity": lexical_diversity,
                "avg_sentence_length": avg_sentence_length,
                "information_density_score": info_density_score,
                "potential_entity_count": entity_count
            }
            
            content_metrics.append(metrics)
        
        # Calculate aggregate metrics
        if content_metrics:
            avg_word_count = sum(item["word_count"] for item in content_metrics) / len(content_metrics)
            avg_info_density = sum(item["information_density_score"] for item in content_metrics) / len(content_metrics)
            avg_lexical_diversity = sum(item["lexical_diversity"] for item in content_metrics) / len(content_metrics)
            total_potential_entities = sum(item["potential_entity_count"] for item in content_metrics)
        else:
            avg_word_count = 0
            avg_info_density = 0
            avg_lexical_diversity = 0
            total_potential_entities = 0
        
        return {
            "content_metrics": content_metrics,
            "average_word_count": avg_word_count,
            "average_information_density": avg_info_density,
            "average_lexical_diversity": avg_lexical_diversity,
            "total_potential_entities": total_potential_entities
        }
    
    def _analyze_readability(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the readability of the retrieved content."""
        if not self.analyze_readability or not self.readability_metrics or not results:
            return {}
        
        # Extract content from each result
        contents = [self._extract_content(result) for result in results]
        
        # Calculate readability metrics
        readability_metrics = []
        
        for i, content in enumerate(contents):
            if not content or len(content) < 50:  # Skip very short content
                continue
                
            # Count sentences, words, syllables
            sentences = re.split(r'[.!?]+', content)
            sentences = [s for s in sentences if s.strip()]
            sentence_count = len(sentences)
            
            words = re.findall(r'\b\w+\b', content.lower())
            word_count = len(words)
            
            if sentence_count == 0 or word_count == 0:
                continue
            
            # Estimate syllables (very rough approximation)
            syllable_count = 0
            for word in words:
                syllable_count += max(1, len(re.findall(r'[aeiouy]+', word)))
            
            # Calculate Flesch Reading Ease
            # Higher scores = easier to read (90-100: Very easy, 0-30: Very difficult)
            try:
                flesch_reading_ease = 206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (syllable_count / word_count)
                flesch_reading_ease = max(0, min(100, flesch_reading_ease))  # Clamp to 0-100 range
            except ZeroDivisionError:
                flesch_reading_ease = 50  # Default to middle value
            
            # Calculate Flesch-Kincaid Grade Level
            # Corresponds to US grade level (e.g., 8.0 = 8th grade)
            try:
                flesch_kincaid_grade = 0.39 * (word_count / sentence_count) + 11.8 * (syllable_count / word_count) - 15.59
                flesch_kincaid_grade = max(0, min(18, flesch_kincaid_grade))  # Clamp to 0-18 range
            except ZeroDivisionError:
                flesch_kincaid_grade = 10  # Default to middle grade
            
            # Automated Readability Index (ARI)
            # Also corresponds to US grade level
            try:
                char_count = sum(len(word) for word in words)
                ari = 4.71 * (char_count / word_count) + 0.5 * (word_count / sentence_count) - 21.43
                ari = max(0, min(18, ari))  # Clamp to 0-18 range
            except ZeroDivisionError:
                ari = 10  # Default to middle grade
            
            # Store metrics
            metrics = {
                "index": i,
                "id": results[i].get("id", str(i)),
                "flesch_reading_ease": flesch_reading_ease,
                "flesch_kincaid_grade": flesch_kincaid_grade,
                "automated_readability_index": ari,
                "reading_difficulty": "Easy" if flesch_reading_ease >= 70 else 
                                      "Moderate" if flesch_reading_ease >= 50 else "Difficult"
            }
            
            readability_metrics.append(metrics)
        
        # Calculate aggregate metrics
        if readability_metrics:
            avg_reading_ease = sum(item["flesch_reading_ease"] for item in readability_metrics) / len(readability_metrics)
            avg_grade_level = sum(item["flesch_kincaid_grade"] for item in readability_metrics) / len(readability_metrics)
            difficulty_counts = Counter(item["reading_difficulty"] for item in readability_metrics)
            most_common_difficulty = difficulty_counts.most_common(1)[0][0] if difficulty_counts else "Unknown"
        else:
            avg_reading_ease = 50
            avg_grade_level = 10
            most_common_difficulty = "Unknown"
        
        return {
            "readability_metrics": readability_metrics,
            "average_reading_ease": avg_reading_ease,
            "average_grade_level": avg_grade_level,
            "most_common_difficulty": most_common_difficulty
        }
    
    def _analyze_sentiment(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the sentiment of the retrieved content."""
        if not self.sentiment_analysis or not hasattr(self, 'sentiment_analyzer') or not results:
            return {}
        
        # Extract content from each result
        contents = [self._extract_content(result) for result in results]
        
        # Calculate sentiment for each document
        sentiment_metrics = []
        
        for i, content in enumerate(contents):
            if not content:
                continue
                
            # Get sentiment scores
            try:
                sentiment_scores = self.sentiment_analyzer.polarity_scores(content)
                
                # Determine sentiment category
                compound_score = sentiment_scores['compound']
                if compound_score >= 0.05:
                    sentiment = "Positive"
                elif compound_score <= -0.05:
                    sentiment = "Negative"
                else:
                    sentiment = "Neutral"
                
                # Store metrics
                metrics = {
                    "index": i,
                    "id": results[i].get("id", str(i)),
                    "compound_score": compound_score,
                    "positive_score": sentiment_scores['pos'],
                    "negative_score": sentiment_scores['neg'],
                    "neutral_score": sentiment_scores['neu'],
                    "sentiment": sentiment
                }
                
                sentiment_metrics.append(metrics)
                
            except Exception as e:
                logger.error(f"Error in sentiment analysis: {str(e)}")
                continue
        
        # Calculate aggregate metrics
        if sentiment_metrics:
            avg_compound = sum(item["compound_score"] for item in sentiment_metrics) / len(sentiment_metrics)
            avg_positive = sum(item["positive_score"] for item in sentiment_metrics) / len(sentiment_metrics)
            avg_negative = sum(item["negative_score"] for item in sentiment_metrics) / len(sentiment_metrics)
            
            sentiment_counts = Counter(item["sentiment"] for item in sentiment_metrics)
            most_common_sentiment = sentiment_counts.most_common(1)[0][0] if sentiment_counts else "Unknown"
        else:
            avg_compound = 0
            avg_positive = 0
            avg_negative = 0
            most_common_sentiment = "Unknown"
        
        return {
            "sentiment_metrics": sentiment_metrics,
            "average_compound_score": avg_compound,
            "average_positive_score": avg_positive,
            "average_negative_score": avg_negative,
            "most_common_sentiment": most_common_sentiment
        }
    
    def _generate_insights(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate actionable insights based on analysis results."""
        insights = []
        
        # Relevance insights
        if "relevance" in analysis_results:
            relevance = analysis_results["relevance"]
            avg_relevance = relevance.get("average_relevance", 0.0)
            relevant_percentage = relevance.get("relevant_percentage", 0.0)
            
            if avg_relevance < self.relevance_threshold:
                insights.append(
                    f"Low relevance detected (average score: {avg_relevance:.2f}). "
                    "Consider using more specific queries or a different retrieval method."
                )
            
            if relevant_percentage < 50:
                insights.append(
                    f"Only {relevant_percentage:.0f}% of retrieved documents are relevant. "
                    "Consider increasing result quality through better filtering or reranking."
                )
        
        # Diversity insights
        if "diversity" in analysis_results:
            diversity = analysis_results["diversity"]
            diversity_score = diversity.get("diversity_score", 1.0)
            duplicate_count = diversity.get("duplicate_count", 0)
            
            if diversity_score < 0.5:
                insights.append(
                    f"Low content diversity detected (score: {diversity_score:.2f}). "
                    "Consider adding diversity reranking to reduce information redundancy."
                )
            
            if duplicate_count > 0:
                insights.append(
                    f"Found {duplicate_count} similar content pairs. "
                    "Consider implementing deduplication to improve information efficiency."
                )
        
        # Information density insights
        if "information" in analysis_results:
            information = analysis_results["information"]
            avg_density = information.get("average_information_density", 0.0)
            avg_word_count = information.get("average_word_count", 0)
            
            if avg_density < 0.4:
                insights.append(
                    f"Retrieved content has low information density (score: {avg_density:.2f}). "
                    "Consider retrieving more information-rich documents."
                )
            
            if avg_word_count < 50:
                insights.append(
                    "Retrieved content contains very short documents. "
                    "Consider adjusting chunking or retrieval strategy for more comprehensive information."
                )
        
        # Readability insights
        if "readability" in analysis_results:
            readability = analysis_results["readability"]
            avg_grade_level = readability.get("average_grade_level", 10)
            most_common_difficulty = readability.get("most_common_difficulty", "Unknown")
            
            if avg_grade_level > 12:
                insights.append(
                    f"Retrieved content is highly specialized (grade level: {avg_grade_level:.1f}). "
                    "Consider simplifying or providing additional context for complex terminology."
                )
                
            if most_common_difficulty == "Difficult":
                insights.append(
                    "Content is generally difficult to read. "
                    "Consider retrieving more accessible documents or providing simplification."
                )
        
        # Sentiment insights
        if "sentiment" in analysis_results:
            sentiment = analysis_results["sentiment"]
            most_common_sentiment = sentiment.get("most_common_sentiment", "Unknown")
            avg_negative = sentiment.get("average_negative_score", 0)
            
            if most_common_sentiment == "Negative" and avg_negative > 0.2:
                insights.append(
                    "Retrieved content has a predominantly negative sentiment. "
                    "Consider balancing with more neutral or positive information if appropriate."
                )
        
        # Limit to max_insights
        return insights[:self.max_insights]
    
    def analyze(self, query: str, results: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Analyze retrieved context for quality and characteristics.
        
        Args:
            query: The query string used for retrieval
            results: List of retrieval results
            **kwargs: Additional parameters for analysis
                - content_field: Override content_field
                - analyze_relevance: Override analyze_relevance
                - analyze_diversity: Override analyze_diversity
                - analyze_information: Override analyze_information
                - analyze_readability: Override analyze_readability
                - sentiment_analysis: Override sentiment_analysis
        
        Returns:
            Dict[str, Any]: Analysis results
        """
        # Apply overrides from kwargs
        content_field = kwargs.get('content_field', self.content_field)
        analyze_relevance = kwargs.get('analyze_relevance', self.analyze_relevance)
        analyze_diversity = kwargs.get('analyze_diversity', self.analyze_diversity)
        analyze_information = kwargs.get('analyze_information', self.analyze_information)
        analyze_readability = kwargs.get('analyze_readability', self.analyze_readability)
        sentiment_analysis = kwargs.get('sentiment_analysis', self.sentiment_analysis)
        
        # Store original values to restore later
        orig_content_field = self.content_field
        orig_analyze_relevance = self.analyze_relevance
        orig_analyze_diversity = self.analyze_diversity
        orig_analyze_information = self.analyze_information
        orig_analyze_readability = self.analyze_readability
        orig_sentiment_analysis = self.sentiment_analysis
        
        # Apply temporary overrides
        self.content_field = content_field
        self.analyze_relevance = analyze_relevance
        self.analyze_diversity = analyze_diversity
        self.analyze_information = analyze_information
        self.analyze_readability = analyze_readability
        self.sentiment_analysis = sentiment_analysis
        
        try:
            # Ensure results list is not empty
            if not results:
                return {
                    "result_count": 0,
                    "insights": ["No retrieved context to analyze."]
                }
            
            analysis_results = {
                "result_count": len(results)
            }
            
            # Analyze relevance to query
            if analyze_relevance and query:
                analysis_results["relevance"] = self._analyze_content_relevance(query, results)
            
            # Analyze content diversity
            if analyze_diversity:
                analysis_results["diversity"] = self._analyze_content_diversity(results)
            
            # Analyze information density
            if analyze_information:
                analysis_results["information"] = self._analyze_information_density(results)
            
            # Analyze readability
            if analyze_readability:
                analysis_results["readability"] = self._analyze_readability(results)
            
            # Analyze sentiment
            if sentiment_analysis:
                analysis_results["sentiment"] = self._analyze_sentiment(results)
            
            # Generate insights
            analysis_results["insights"] = self._generate_insights(analysis_results)
            
            return analysis_results
            
        finally:
            # Restore original values
            self.content_field = orig_content_field
            self.analyze_relevance = orig_analyze_relevance
            self.analyze_diversity = orig_analyze_diversity
            self.analyze_information = orig_analyze_information
            self.analyze_readability = orig_analyze_readability
            self.sentiment_analysis = orig_sentiment_analysis
    
    def compare(
        self,
        query: str,
        results_sets: List[List[Dict[str, Any]]],
        names: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compare context quality across multiple retrieval results for the same query.
        
        Args:
            query: The query string used for retrieval
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
                "result_count": analysis["result_count"]
            }
            
            # Add relevance metrics if available
            if "relevance" in analysis:
                metrics["average_relevance"] = analysis["relevance"].get("average_relevance", 0.0)
                metrics["relevant_percentage"] = analysis["relevance"].get("relevant_percentage", 0.0)
            
            # Add diversity metrics if available
            if "diversity" in analysis:
                metrics["diversity_score"] = analysis["diversity"].get("diversity_score", 0.0)
                metrics["duplicate_count"] = analysis["diversity"].get("duplicate_count", 0)
            
            # Add information metrics if available
            if "information" in analysis:
                metrics["information_density"] = analysis["information"].get("average_information_density", 0.0)
                metrics["average_word_count"] = analysis["information"].get("average_word_count", 0)
            
            # Add readability metrics if available
            if "readability" in analysis:
                metrics["reading_ease"] = analysis["readability"].get("average_reading_ease", 50)
                metrics["grade_level"] = analysis["readability"].get("average_grade_level", 10)
            
            comparison.append(metrics)
        
        # Calculate which system is best for different aspects
        best_systems = {}
        
        if all("average_relevance" in metrics for metrics in comparison):
            best_relevance_idx = max(range(len(comparison)), key=lambda i: comparison[i].get("average_relevance", 0))
            best_systems["relevance"] = comparison[best_relevance_idx]["name"]
        
        if all("diversity_score" in metrics for metrics in comparison):
            best_diversity_idx = max(range(len(comparison)), key=lambda i: comparison[i].get("diversity_score", 0))
            best_systems["diversity"] = comparison[best_diversity_idx]["name"]
        
        if all("information_density" in metrics for metrics in comparison):
            best_info_idx = max(range(len(comparison)), key=lambda i: comparison[i].get("information_density", 0))
            best_systems["information"] = comparison[best_info_idx]["name"]
        
        # Determine overall best system using a simple weighted score
        scores = []
        for metrics in comparison:
            score = 0.0
            components = 0
            
            if "average_relevance" in metrics:
                score += metrics["average_relevance"] * 0.4  # Weight: 40%
                components += 1
            
            if "diversity_score" in metrics:
                score += metrics["diversity_score"] * 0.3  # Weight: 30%
                components += 1
            
            if "information_density" in metrics:
                score += metrics["information_density"] * 0.3  # Weight: 30%
                components += 1
            
            # Normalize by number of components
            normalized_score = score / components if components > 0 else 0
            scores.append((metrics["name"], normalized_score))
        
        best_overall = max(scores, key=lambda x: x[1])[0] if scores else None
        
        # Generate insights
        insights = []
        
        if len(comparison) > 1:
            # Relevance comparison
            if all("average_relevance" in metrics for metrics in comparison):
                relevance_values = [metrics["average_relevance"] for metrics in comparison]
                max_relevance = max(relevance_values)
                min_relevance = min(relevance_values)
                
                if max_relevance - min_relevance > 0.2:  # Significant difference
                    most_relevant = best_systems["relevance"]
                    least_relevant = comparison[relevance_values.index(min_relevance)]["name"]
                    
                    insights.append(
                        f"{most_relevant} provides the most relevant context ({max_relevance:.2f}), while {least_relevant} ({min_relevance:.2f}) has the least relevance. "
                        "Consider the retrieval approach of the more relevant system."
                    )
            
            # Diversity comparison
            if all("diversity_score" in metrics for metrics in comparison):
                diversity_values = [metrics["diversity_score"] for metrics in comparison]
                max_diversity = max(diversity_values)
                min_diversity = min(diversity_values)
                
                if max_diversity - min_diversity > 0.2:  # Significant difference
                    most_diverse = best_systems["diversity"]
                    least_diverse = comparison[diversity_values.index(min_diversity)]["name"]
                    
                    insights.append(
                        f"{most_diverse} provides the most diverse context ({max_diversity:.2f}), while {least_diverse} ({min_diversity:.2f}) has the least diversity. "
                        "Consider techniques from the more diverse system if information variety is important."
                    )
            
            # Information density comparison
            if all("information_density" in metrics for metrics in comparison):
                info_values = [metrics["information_density"] for metrics in comparison]
                max_info = max(info_values)
                min_info = min(info_values)
                
                if max_info - min_info > 0.2:  # Significant difference
                    most_informative = best_systems["information"]
                    least_informative = comparison[info_values.index(min_info)]["name"]
                    
                    insights.append(
                        f"{most_informative} provides the most information-rich context ({max_info:.2f}), while {least_informative} ({min_info:.2f}) has less information density. "
                        "Consider the retrieval or chunking approach of the more informative system."
                    )
        
        return {
            "comparison": comparison,
            "best_systems": best_systems,
            "best_overall": best_overall,
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
        Evaluate context quality against ground truth.
        
        Args:
            query: The query string used for retrieval
            results: List of retrieval results
            ground_truth: Ground truth relevant documents or content
            **kwargs: Additional parameters for evaluation
                - content_overlap: Whether to evaluate content overlap (not just ID matching)
                - id_field: Field name to use for ID matching
                - relevance_threshold: Override relevance_threshold
        
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        content_overlap = kwargs.get('content_overlap', True)
        id_field = kwargs.get('id_field', 'id')
        relevance_threshold = kwargs.get('relevance_threshold', self.relevance_threshold)
        
        metrics = {}
        
        # If no results or ground truth, return empty metrics
        if not results or not ground_truth:
            return {
                "content_recall": 0.0,
                "content_precision": 0.0,
                "content_f1": 0.0,
                "content_overlap_percentage": 0.0
            }
        
        # ID-based evaluation (standard retrieval metrics)
        if all(isinstance(item, dict) and id_field in item for item in results):
            result_ids = [doc.get(id_field, "") for doc in results]
            
            # Get ground truth IDs
            if all(isinstance(item, dict) and id_field in item for item in ground_truth):
                truth_ids = [doc.get(id_field, "") for doc in ground_truth]
            elif all(isinstance(item, str) for item in ground_truth):
                truth_ids = [str(item) for item in ground_truth]
            else:
                truth_ids = []
            
            if truth_ids:
                # Calculate ID overlap metrics
                relevant_retrieved = sum(1 for rid in result_ids if rid in truth_ids)
                
                # Precision: What fraction of retrieved items are relevant?
                id_precision = relevant_retrieved / len(result_ids) if result_ids else 0.0
                
                # Recall: What fraction of relevant items are retrieved?
                id_recall = relevant_retrieved / len(truth_ids) if truth_ids else 0.0
                
                # F1: Harmonic mean of precision and recall
                id_f1 = 2 * (id_precision * id_recall) / (id_precision + id_recall) if (id_precision + id_recall) > 0 else 0.0
                
                metrics.update({
                    "id_precision": id_precision,
                    "id_recall": id_recall,
                    "id_f1": id_f1
                })
        
        # Content-based evaluation (semantic overlap)
        if content_overlap:
            # Extract content from results
            result_contents = [self._extract_content(result) for result in results]
            result_contents = [content for content in result_contents if content]
            
            # Extract content from ground truth
            truth_contents = []
            if all(isinstance(item, dict) for item in ground_truth):
                for item in ground_truth:
                    content = self._extract_content(item)
                    if content:
                        truth_contents.append(content)
            elif all(isinstance(item, str) for item in ground_truth):
                truth_contents = [item for item in ground_truth if item]
            
            if result_contents and truth_contents:
                # Calculate content overlap
                # For each ground truth content, find the best matching result content
                content_matches = []
                
                for truth in truth_contents:
                    best_match = 0.0
                    for result in result_contents:
                        sim = self._calculate_text_similarity(truth, result)
                        best_match = max(best_match, sim)
                    
                    content_matches.append((best_match, best_match >= relevance_threshold))
                
                # For each result, find the best matching ground truth
                result_matches = []
                
                for result in result_contents:
                    best_match = 0.0
                    for truth in truth_contents:
                        sim = self._calculate_text_similarity(truth, result)
                        best_match = max(best_match, sim)
                    
                    result_matches.append((best_match, best_match >= relevance_threshold))
                
                # Calculate content overlap metrics
                # Precision: What fraction of retrieved content is relevant to ground truth?
                content_precision = sum(1 for _, is_match in result_matches if is_match) / len(result_matches) if result_matches else 0.0
                
                # Recall: What fraction of ground truth content is covered by retrieved content?
                content_recall = sum(1 for _, is_match in content_matches if is_match) / len(content_matches) if content_matches else 0.0
                
                # F1: Harmonic mean of precision and recall
                content_f1 = 2 * (content_precision * content_recall) / (content_precision + content_recall) if (content_precision + content_recall) > 0 else 0.0
                
                # Average semantic overlap
                avg_overlap = sum(sim for sim, _ in content_matches) / len(content_matches) if content_matches else 0.0
                
                metrics.update({
                    "content_precision": content_precision,
                    "content_recall": content_recall,
                    "content_f1": content_f1,
                    "content_overlap_percentage": avg_overlap * 100
                })
        
        return metrics
    
    def get_insights(self, query: str, results: List[Dict[str, Any]], **kwargs) -> List[str]:
        """
        Get actionable insights about the retrieved context.
        
        Args:
            query: The query string used for retrieval
            results: List of retrieval results
            **kwargs: Additional parameters for insight generation
        
        Returns:
            List[str]: List of insights or suggestions
        """
        # Analyze context and extract insights
        analysis_results = self.analyze(query, results, **kwargs)
        return analysis_results.get("insights", []) 