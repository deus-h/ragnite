#!/usr/bin/env python3
"""
Unit tests for the ContextAnalyzer class.
"""

import unittest
from unittest.mock import patch, MagicMock

from tools.src.retrieval.retrieval_debuggers.context_analyzer import ContextAnalyzer


class TestContextAnalyzer(unittest.TestCase):
    """Test cases for the ContextAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = ContextAnalyzer(
            content_field="content",
            analyze_relevance=True,
            analyze_diversity=True,
            analyze_information=True,
            analyze_readability=True,
            similarity_threshold=0.7,
            relevance_threshold=0.5,
            similarity_method="tfidf",
            readability_metrics=True,
            sentiment_analysis=False
        )
        
        self.query = "What are the best practices for RAG with vector databases?"
        
        self.results = [
            {
                "id": "doc_1",
                "score": 0.89,
                "content": """
                Best practices for implementing RAG with vector databases include:
                
                1. Proper document chunking: Break documents into semantically meaningful chunks rather than arbitrary splits.
                
                2. High-quality embeddings: Use domain-specific embedding models when possible to better capture semantic relationships.
                
                3. Vector database optimization: Configure index parameters (like HNSW or IVF) based on your specific retrieval needs.
                
                4. Metadata filtering: Store and utilize metadata to enable more precise filtering during retrieval.
                
                5. Regular performance evaluation: Continuously assess retrieval quality using metrics like precision, recall, and relevance.
                """,
                "metadata": {
                    "source": "vector_db_guide.pdf",
                    "page": 24,
                    "category": "best_practices",
                    "date": "2023-09-15"
                }
            },
            {
                "id": "doc_2",
                "score": 0.76,
                "content": """
                For optimal RAG performance with vector databases, consider the following technical optimizations:
                
                - Select the appropriate index type based on your data size and query patterns
                - For smaller datasets (<1M vectors), HNSW often provides better recall
                - For larger datasets, IVF with proper clustering can be more efficient
                - Monitor query latency and adjust parameters accordingly
                - Use metadata filtering to narrow search space before vector similarity search
                - Configure distance metrics based on your embedding model (cosine vs euclidean)
                """,
                "metadata": {
                    "source": "vector_db_guide.pdf",
                    "page": 25,
                    "category": "best_practices",
                    "date": "2023-09-15"
                }
            },
            {
                "id": "doc_3",
                "score": 0.65,
                "content": """
                Vector databases are specialized database systems designed to store and efficiently search vector embeddings. These embeddings are high-dimensional numerical representations of data objects, created by machine learning models. Popular vector databases include Chroma, Qdrant, Pinecone, Weaviate, Milvus, and pgvector (PostgreSQL extension).
                """,
                "metadata": {
                    "source": "vector_db_comparison.pdf",
                    "page": 5,
                    "category": "introduction",
                    "date": "2023-08-10"
                }
            }
        ]

    def test_initialization(self):
        """Test ContextAnalyzer initialization."""
        self.assertEqual(self.analyzer.content_field, "content")
        self.assertTrue(self.analyzer.analyze_relevance)
        self.assertTrue(self.analyzer.analyze_diversity)
        self.assertTrue(self.analyzer.analyze_information)
        self.assertTrue(self.analyzer.analyze_readability)
        self.assertEqual(self.analyzer.similarity_threshold, 0.7)
        self.assertEqual(self.analyzer.relevance_threshold, 0.5)
        self.assertEqual(self.analyzer.similarity_method, "tfidf")
        self.assertTrue(self.analyzer.readability_metrics)
        self.assertFalse(self.analyzer.sentiment_analysis)

    @patch('tools.src.retrieval.retrieval_debuggers.context_analyzer.ContextAnalyzer._analyze_content_relevance')
    @patch('tools.src.retrieval.retrieval_debuggers.context_analyzer.ContextAnalyzer._analyze_content_diversity')
    @patch('tools.src.retrieval.retrieval_debuggers.context_analyzer.ContextAnalyzer._analyze_information_density')
    @patch('tools.src.retrieval.retrieval_debuggers.context_analyzer.ContextAnalyzer._analyze_readability')
    @patch('tools.src.retrieval.retrieval_debuggers.context_analyzer.ContextAnalyzer._generate_insights')
    def test_analyze(self, mock_insights, mock_readability, mock_information, mock_diversity, mock_relevance):
        """Test analyze method."""
        # Configure mocks
        mock_relevance.return_value = {"average_relevance": 0.8}
        mock_diversity.return_value = {"diversity_score": 0.7}
        mock_information.return_value = {"information_density": 0.6}
        mock_readability.return_value = {"grade_level": 12.0}
        mock_insights.return_value = ["Sample insight"]
        
        # Call analyze method
        result = self.analyzer.analyze(self.query, self.results)
        
        # Check if all analysis methods were called
        mock_relevance.assert_called_once()
        mock_diversity.assert_called_once()
        mock_information.assert_called_once()
        mock_readability.assert_called_once()
        mock_insights.assert_called_once()
        
        # Check result structure
        self.assertIn("result_count", result)
        self.assertIn("relevance", result)
        self.assertIn("diversity", result)
        self.assertIn("information", result)
        self.assertIn("readability", result)
        self.assertIn("insights", result)
        
        # Check values
        self.assertEqual(result["result_count"], 3)
        self.assertEqual(result["relevance"], {"average_relevance": 0.8})
        self.assertEqual(result["diversity"], {"diversity_score": 0.7})
        self.assertEqual(result["information"], {"information_density": 0.6})
        self.assertEqual(result["readability"], {"grade_level": 12.0})
        self.assertEqual(result["insights"], ["Sample insight"])
    
    @patch('tools.src.retrieval.retrieval_debuggers.context_analyzer.ContextAnalyzer.analyze')
    def test_compare(self, mock_analyze):
        """Test compare method."""
        # Configure mock
        mock_analyze.side_effect = [
            {
                "result_count": 3,
                "relevance": {"average_relevance": 0.8},
                "diversity": {"diversity_score": 0.7},
                "information": {"average_information_density": 0.6},
                "readability": {"average_grade_level": 12.0}
            },
            {
                "result_count": 2,
                "relevance": {"average_relevance": 0.9},
                "diversity": {"diversity_score": 0.8},
                "information": {"average_information_density": 0.7},
                "readability": {"average_grade_level": 10.0}
            }
        ]
        
        # Call compare method
        results_set2 = [self.results[0], self.results[1]]
        comparison = self.analyzer.compare(self.query, [self.results, results_set2], names=["System A", "System B"])
        
        # Check if analyze was called twice
        self.assertEqual(mock_analyze.call_count, 2)
        
        # Check comparison structure
        self.assertIn("comparison", comparison)
        self.assertIn("best_systems", comparison)
        self.assertIn("best_overall", comparison)
        self.assertIn("insights", comparison)
        
        # Check if comparison contains metrics for both systems
        comparison_metrics = comparison["comparison"]
        self.assertEqual(len(comparison_metrics), 2)
        self.assertEqual(comparison_metrics[0]["name"], "System A")
        self.assertEqual(comparison_metrics[1]["name"], "System B")
    
    @patch('tools.src.retrieval.retrieval_debuggers.context_analyzer.ContextAnalyzer._calculate_content_similarity')
    def test_evaluate(self, mock_similarity):
        """Test evaluate method."""
        # Configure mock
        mock_similarity.return_value = 0.75
        
        # Define ground truth
        ground_truth = [
            """Best practices for RAG with vector databases include proper document chunking, 
            using high-quality embeddings, optimizing vector database configuration, 
            implementing metadata filtering, and continuously evaluating performance.""",
            "doc_1"
        ]
        
        # Call evaluate method
        evaluation = self.analyzer.evaluate(self.query, self.results, ground_truth)
        
        # Check evaluation structure
        self.assertIn("precision", evaluation)
        self.assertIn("recall", evaluation)
        self.assertIn("f1_score", evaluation)
        self.assertIn("overlap_score", evaluation)
        
        # Check if similarity calculation was called
        self.assertGreater(mock_similarity.call_count, 0)
    
    @patch('tools.src.retrieval.retrieval_debuggers.context_analyzer.ContextAnalyzer.analyze')
    def test_get_insights(self, mock_analyze):
        """Test get_insights method."""
        # Configure mock
        mock_analyze.return_value = {
            "insights": ["Insight 1", "Insight 2", "Insight 3"]
        }
        
        # Call get_insights method
        insights = self.analyzer.get_insights(self.query, self.results)
        
        # Check if analyze was called
        mock_analyze.assert_called_once()
        
        # Check insights
        self.assertIsInstance(insights, list)
        self.assertEqual(len(insights), 3)
        self.assertEqual(insights[0], "Insight 1")
        self.assertEqual(insights[1], "Insight 2")
        self.assertEqual(insights[2], "Insight 3")
    
    def test_content_extraction(self):
        """Test content extraction from results."""
        contents = self.analyzer._extract_contents(self.results)
        
        # Check content extraction
        self.assertEqual(len(contents), 3)
        for content in contents:
            self.assertIsInstance(content, str)
            self.assertTrue(len(content) > 0)
    
    @patch('tools.src.retrieval.retrieval_debuggers.context_analyzer.ContextAnalyzer._calculate_text_relevance')
    def test_analyze_content_relevance(self, mock_relevance):
        """Test content relevance analysis."""
        # Configure mock
        mock_relevance.side_effect = [0.9, 0.8, 0.45]
        
        # Call analyze_content_relevance method
        relevance_metrics = self.analyzer._analyze_content_relevance(self.query, self.results)
        
        # Check relevance metrics structure
        self.assertIn("average_relevance", relevance_metrics)
        self.assertIn("relevant_count", relevance_metrics)
        self.assertIn("relevant_percentage", relevance_metrics)
        self.assertIn("relevance_scores", relevance_metrics)
        
        # Check values
        self.assertAlmostEqual(relevance_metrics["average_relevance"], 0.7166, places=3)
        self.assertEqual(relevance_metrics["relevant_count"], 2)
        self.assertAlmostEqual(relevance_metrics["relevant_percentage"], 66.67, places=2)
        self.assertEqual(len(relevance_metrics["relevance_scores"]), 3)


if __name__ == "__main__":
    unittest.main() 