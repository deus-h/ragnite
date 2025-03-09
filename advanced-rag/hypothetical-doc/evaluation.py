"""
HyDE Evaluation Module

This module provides tools to evaluate the quality and effectiveness of
hypothetical documents generated for HyDE (Hypothetical Document Embeddings).
"""

import logging
import time
from typing import List, Dict, Any, Optional, Union, Callable

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


class HyDEEvaluator:
    """
    Evaluator for HyDE-generated hypothetical documents.
    
    This evaluator assesses the quality and effectiveness of hypothetical
    documents generated for retrieval, helping optimize the HyDE pipeline.
    """
    
    def __init__(
        self,
        model_provider: Optional[Union[str, LLMProvider]] = None,
        reference_retriever: Optional[Callable] = None,
        hyde_retriever: Optional[Callable] = None,
        system_prompt: Optional[str] = None,
        evaluation_criteria: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the HyDE Evaluator.
        
        Args:
            model_provider: LLM provider for evaluation
            reference_retriever: Standard retriever for comparison
            hyde_retriever: HyDE-enhanced retriever
            system_prompt: Custom system prompt for evaluation
            evaluation_criteria: Custom weighting for evaluation criteria
        """
        if not MODELS_AVAILABLE:
            raise ImportError(
                "The models module is required. Make sure the 'tools.src.models' module is available."
            )
        
        # Set up the model provider for evaluation
        if model_provider is None:
            try:
                # Default to GPT-4 if not specified
                self.model_provider = get_model_provider("openai", model="gpt-4o")
                logger.info("Using default OpenAI GPT-4o model provider for evaluation")
            except Exception as e:
                logger.warning(f"Error initializing default model provider: {str(e)}")
                raise
        elif isinstance(model_provider, str):
            try:
                self.model_provider = get_model_provider(model_provider)
                logger.info(f"Using {model_provider} model provider for evaluation")
            except Exception as e:
                logger.warning(f"Error initializing model provider {model_provider}: {str(e)}")
                raise
        else:
            self.model_provider = model_provider
        
        # Set retrievers
        self.reference_retriever = reference_retriever
        self.hyde_retriever = hyde_retriever
        
        # Initialize default evaluation criteria if not provided
        if not evaluation_criteria:
            self.evaluation_criteria = {
                "relevance": 0.3,     # How relevant the document is to the query
                "factuality": 0.2,    # Factual accuracy of the document
                "completeness": 0.2,  # How comprehensive the document is
                "coherence": 0.15,    # Logical flow and organization
                "specificity": 0.15   # Contains specific, detailed information
            }
        else:
            self.evaluation_criteria = evaluation_criteria
        
        # Initialize default system prompt if not provided
        if not system_prompt:
            self.system_prompt = """
            You are an expert evaluator of document quality and relevance. Your task is to
            evaluate the quality of hypothetical documents generated to answer a given query.
            
            These hypothetical documents are used to improve retrieval in a RAG (Retrieval 
            Augmented Generation) system. They are not meant to directly answer the query,
            but rather to create an embedding that will help retrieve relevant documents.
            
            Evaluate the documents objectively based on the following criteria:
            1. Relevance: How well the document relates to the query
            2. Factuality: The accuracy of information contained in the document
            3. Completeness: How comprehensive the document is in covering aspects of the query
            4. Coherence: The logical flow, organization, and readability
            5. Specificity: The level of specific, detailed information provided
            
            For each criterion, assign a score from 1-10, where 1 is very poor and 10 is excellent.
            Provide brief justifications for each score.
            """
        else:
            self.system_prompt = system_prompt
    
    def evaluate_document_quality(self, 
                                query: str, 
                                document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the quality of a single hypothetical document.
        
        Args:
            query: The original query
            document: The hypothetical document with metadata
            
        Returns:
            Evaluation results with scores and feedback
        """
        # Create evaluation prompt
        evaluation_prompt = f"""
        Please evaluate the quality of the following hypothetical document
        generated in response to the query:
        
        Query: {query}
        
        Hypothetical Document (Perspective: {document.get('perspective', 'Unknown')}):
        {document.get('document', '')}
        
        Evaluate this document on the following criteria, scoring each from 1-10:
        
        1. Relevance: How well does the document relate to the query?
        2. Factuality: How accurate is the information in the document?
        3. Completeness: How comprehensive is the document in covering aspects of the query?
        4. Coherence: How logical is the flow, organization, and readability?
        5. Specificity: How specific and detailed is the information provided?
        
        For each criterion, provide:
        - Score (1-10)
        - Brief justification for the score
        
        Finally, provide an overall assessment and suggestions for improvement.
        Format your response as JSON with the following structure:
        
        {
            "relevance": {"score": 0, "justification": ""},
            "factuality": {"score": 0, "justification": ""},
            "completeness": {"score": 0, "justification": ""},
            "coherence": {"score": 0, "justification": ""},
            "specificity": {"score": 0, "justification": ""},
            "overall_score": 0,
            "assessment": "",
            "improvement_suggestions": []
        }
        """
        
        # Create messages for evaluation
        messages = [
            Message(role=Role.SYSTEM, content=self.system_prompt),
            Message(role=Role.USER, content=evaluation_prompt)
        ]
        
        # Get evaluation from model
        try:
            response = self.model_provider.generate(
                messages=messages,
                temperature=0.3,  # Low temperature for consistent evaluation
                response_format={"type": "json_object"}  # Request JSON response
            )
            
            # Parse the JSON response
            evaluation = response.get("content", "{}")
            
            # If the response is a string containing JSON, parse it
            if isinstance(evaluation, str):
                import json
                try:
                    evaluation = json.loads(evaluation)
                except Exception as e:
                    logger.error(f"Error parsing evaluation JSON: {str(e)}")
                    # Create a basic evaluation object as fallback
                    evaluation = {
                        "relevance": {"score": 0, "justification": "Error parsing evaluation"},
                        "factuality": {"score": 0, "justification": "Error parsing evaluation"},
                        "completeness": {"score": 0, "justification": "Error parsing evaluation"},
                        "coherence": {"score": 0, "justification": "Error parsing evaluation"},
                        "specificity": {"score": 0, "justification": "Error parsing evaluation"},
                        "overall_score": 0,
                        "assessment": "Error parsing evaluation",
                        "improvement_suggestions": ["Unable to provide suggestions due to parsing error"]
                    }
            
            # Add document metadata to evaluation
            evaluation["document_metadata"] = {
                "perspective": document.get("perspective", "Unknown"),
                "refinement_steps": document.get("refinement_steps", 0),
                "domain": document.get("domain", "general")
            }
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating document quality: {str(e)}")
            # Return a basic evaluation object as fallback
            return {
                "relevance": {"score": 0, "justification": "Evaluation failed"},
                "factuality": {"score": 0, "justification": "Evaluation failed"},
                "completeness": {"score": 0, "justification": "Evaluation failed"},
                "coherence": {"score": 0, "justification": "Evaluation failed"},
                "specificity": {"score": 0, "justification": "Evaluation failed"},
                "overall_score": 0,
                "assessment": f"Evaluation failed: {str(e)}",
                "improvement_suggestions": ["Unable to provide suggestions due to evaluation failure"],
                "document_metadata": {
                    "perspective": document.get("perspective", "Unknown"),
                    "refinement_steps": document.get("refinement_steps", 0),
                    "domain": document.get("domain", "general")
                }
            }
    
    def evaluate_all_documents(self, 
                             query: str, 
                             documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate the quality of all hypothetical documents for a query.
        
        Args:
            query: The original query
            documents: List of hypothetical documents with metadata
            
        Returns:
            Aggregate evaluation results for all documents
        """
        # Evaluate each document
        evaluations = []
        for doc in documents:
            evaluation = self.evaluate_document_quality(query, doc)
            evaluations.append(evaluation)
        
        # Calculate aggregate scores
        aggregate_scores = {}
        for criterion in self.evaluation_criteria.keys():
            scores = [eval.get(criterion, {}).get("score", 0) for eval in evaluations]
            if scores:
                aggregate_scores[criterion] = sum(scores) / len(scores)
            else:
                aggregate_scores[criterion] = 0
        
        # Calculate weighted overall score
        weighted_score = sum(
            aggregate_scores.get(criterion, 0) * weight
            for criterion, weight in self.evaluation_criteria.items()
        )
        
        # Prepare result
        result = {
            "query": query,
            "document_count": len(documents),
            "individual_evaluations": evaluations,
            "aggregate_scores": aggregate_scores,
            "weighted_overall_score": weighted_score,
        }
        
        return result
    
    def evaluate_retrieval_effectiveness(self,
                                       query: str,
                                       hyde_documents: List[Dict[str, Any]],
                                       reference_results: Optional[List[Dict[str, Any]]] = None,
                                       hyde_results: Optional[List[Dict[str, Any]]] = None,
                                       limit: int = 10) -> Dict[str, Any]:
        """
        Evaluate the effectiveness of HyDE retrieval compared to a reference retriever.
        
        Args:
            query: The original query
            hyde_documents: The hypothetical documents used for retrieval
            reference_results: Results from the reference retriever (optional)
            hyde_results: Results from the HyDE retriever (optional)
            limit: Maximum number of results to compare
            
        Returns:
            Evaluation of retrieval effectiveness
        """
        if not self.reference_retriever and not reference_results:
            raise ValueError("Either reference_retriever or reference_results must be provided")
        
        if not self.hyde_retriever and not hyde_results:
            raise ValueError("Either hyde_retriever or hyde_results must be provided")
        
        # Get reference results if not provided
        if not reference_results and self.reference_retriever:
            logger.info(f"Retrieving reference results for query: {query}")
            reference_results = self.reference_retriever(query=query, limit=limit)
        
        # Get HyDE results if not provided
        if not hyde_results and self.hyde_retriever:
            logger.info(f"Retrieving HyDE results for query: {query}")
            hyde_results = self.hyde_retriever(query=query, limit=limit)
        
        # Ensure results are limited to the specified number
        reference_results = reference_results[:limit]
        hyde_results = hyde_results[:limit]
        
        # Create evaluation prompt
        evaluation_prompt = f"""
        Please evaluate the effectiveness of two sets of retrieval results for the query:
        
        Query: {query}
        
        Set A (Reference Results):
        {self._format_results_for_evaluation(reference_results)}
        
        Set B (HyDE Results):
        {self._format_results_for_evaluation(hyde_results)}
        
        Evaluate these results based on:
        1. Relevance: How relevant are the retrieved documents to the query?
        2. Diversity: How diverse and comprehensive is the coverage?
        3. Precision: How precise are the top results?
        4. Novelty: Does either set provide unique valuable results?
        5. Overall Quality: Which set provides better overall retrieval?
        
        For each criterion, compare the two sets and explain which is better and why.
        Also identify any documents that appear in both sets.
        
        Format your response as JSON with the following structure:
        
        {
            "relevance": {
                "better_set": "A or B",
                "explanation": ""
            },
            "diversity": {
                "better_set": "A or B",
                "explanation": ""
            },
            "precision": {
                "better_set": "A or B",
                "explanation": ""
            },
            "novelty": {
                "better_set": "A or B",
                "explanation": ""
            },
            "overall_quality": {
                "better_set": "A or B",
                "explanation": ""
            },
            "overlapping_results": [
                {"id": "", "title": ""}
            ],
            "unique_valuable_results": {
                "set_a": [{"id": "", "reason": ""}],
                "set_b": [{"id": "", "reason": ""}]
            },
            "recommendation": ""
        }
        """
        
        # Create messages for evaluation
        messages = [
            Message(role=Role.SYSTEM, content=self.system_prompt),
            Message(role=Role.USER, content=evaluation_prompt)
        ]
        
        # Get evaluation from model
        try:
            response = self.model_provider.generate(
                messages=messages,
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            # Parse the JSON response
            evaluation = response.get("content", "{}")
            
            # If the response is a string containing JSON, parse it
            if isinstance(evaluation, str):
                import json
                try:
                    evaluation = json.loads(evaluation)
                except Exception as e:
                    logger.error(f"Error parsing retrieval evaluation JSON: {str(e)}")
                    evaluation = {
                        "error": "Failed to parse evaluation response",
                        "raw_response": evaluation[:1000]  # Include part of the raw response for debugging
                    }
            
            # Add metadata to evaluation
            evaluation["query"] = query
            evaluation["reference_count"] = len(reference_results)
            evaluation["hyde_count"] = len(hyde_results)
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating retrieval effectiveness: {str(e)}")
            # Return a basic evaluation object as fallback
            return {
                "error": f"Evaluation failed: {str(e)}",
                "query": query,
                "reference_count": len(reference_results),
                "hyde_count": len(hyde_results)
            }
    
    def _format_results_for_evaluation(self, results: List[Dict[str, Any]]) -> str:
        """
        Format retrieval results for evaluation.
        
        Args:
            results: List of retrieval results
            
        Returns:
            Formatted results as a string
        """
        formatted = []
        for i, result in enumerate(results):
            # Extract key information
            doc_id = result.get("id", f"doc_{i}")
            score = result.get("score", 0)
            title = result.get("title", "Untitled")
            content = result.get("content", "")
            
            # Truncate content if too long
            if len(content) > 500:
                content = content[:497] + "..."
            
            # Format the result
            formatted.append(
                f"Document {i+1} (ID: {doc_id}, Score: {score:.4f})\n"
                f"Title: {title}\n"
                f"Content: {content}\n"
            )
        
        return "\n".join(formatted)
    
    def evaluate_perspective_effectiveness(self, 
                                         queries: List[str],
                                         domains: Optional[List[str]] = None,
                                         perspectives_to_test: Optional[List[str]] = None,
                                         limit: int = 5) -> Dict[str, Any]:
        """
        Evaluate the effectiveness of different HyDE perspectives across multiple queries.
        
        Args:
            queries: List of test queries
            domains: List of domains to test (defaults to ["general"])
            perspectives_to_test: List of perspectives to evaluate
            limit: Maximum number of results per query
            
        Returns:
            Evaluation of perspective effectiveness across queries
        """
        if not domains:
            domains = ["general"]
        
        if not perspectives_to_test:
            # Default perspectives to test
            perspectives_to_test = [
                "expert", "educational", "summary", "example_based", "qa_format"
            ]
        
        if not self.hyde_retriever:
            raise ValueError("HyDE retriever must be provided for perspective evaluation")
        
        # Run evaluations for each query and domain
        evaluations = []
        
        for query in queries:
            for domain in domains:
                domain_results = {
                    "query": query,
                    "domain": domain,
                    "perspective_results": []
                }
                
                # Get reference results (standard retrieval)
                if self.reference_retriever:
                    reference_results = self.reference_retriever(query=query, limit=limit)
                else:
                    reference_results = []
                
                # Test each perspective
                for perspective in perspectives_to_test:
                    try:
                        # Generate a single hypothetical document for this perspective
                        hyde_result = self._generate_single_perspective_document(
                            query=query,
                            perspective=perspective,
                            domain=domain
                        )
                        
                        # Evaluate document quality
                        doc_evaluation = self.evaluate_document_quality(query, hyde_result)
                        
                        # Retrieve with this document
                        retrieval_results = self.hyde_retriever(
                            query=query, 
                            hypothetical_documents=[hyde_result],
                            limit=limit,
                            hyde_only=True  # Use only HyDE for this test
                        )
                        
                        # Evaluate retrieval effectiveness
                        if reference_results:
                            retrieval_evaluation = self.evaluate_retrieval_effectiveness(
                                query=query,
                                hyde_documents=[hyde_result],
                                reference_results=reference_results,
                                hyde_results=retrieval_results["results"],
                                limit=limit
                            )
                        else:
                            retrieval_evaluation = {"error": "No reference results available"}
                        
                        # Store results for this perspective
                        perspective_result = {
                            "perspective": perspective,
                            "document": hyde_result,
                            "document_evaluation": doc_evaluation,
                            "retrieval_evaluation": retrieval_evaluation
                        }
                        
                        domain_results["perspective_results"].append(perspective_result)
                        
                    except Exception as e:
                        logger.error(f"Error evaluating perspective {perspective}: {str(e)}")
                        domain_results["perspective_results"].append({
                            "perspective": perspective,
                            "error": str(e)
                        })
                
                evaluations.append(domain_results)
        
        # Analyze results to determine most effective perspectives
        analysis = self._analyze_perspective_effectiveness(evaluations)
        
        # Prepare final report
        report = {
            "evaluations": evaluations,
            "analysis": analysis
        }
        
        return report
    
    def _generate_single_perspective_document(self,
                                            query: str,
                                            perspective: str,
                                            domain: str = "general") -> Dict[str, Any]:
        """
        Generate a single hypothetical document for a specific perspective.
        
        Args:
            query: The query to generate for
            perspective: The perspective to use
            domain: The domain to use
            
        Returns:
            Generated document with metadata
        """
        # Import templates dynamically to avoid circular import
        try:
            from .templates import HyDETemplates
            templates = HyDETemplates.get_domain_templates(domain)
        except ImportError:
            # Fallback simple template
            templates = {
                "system_prompt": "Generate a hypothetical document to answer the query.",
                "perspective_prompts": {
                    perspective: "Generate a document for this query: {query}"
                },
                "settings": {"temperature": 0.7}
            }
        
        # Get the system prompt
        system_prompt = templates.get("system_prompt", "")
        
        # Get the perspective prompt
        prompt_template = templates.get("perspective_prompts", {}).get(
            perspective, "Generate a document for this query: {query}"
        )
        
        # Format the prompt
        prompt = prompt_template.format(query=query, domain=domain)
        
        # Create messages
        messages = [
            Message(role=Role.SYSTEM, content=system_prompt),
            Message(role=Role.USER, content=prompt)
        ]
        
        # Generate the document
        try:
            temperature = templates.get("settings", {}).get("temperature", 0.7)
            response = self.model_provider.generate(
                messages=messages,
                temperature=temperature,
                max_tokens=2048
            )
            
            document = response.get("content", "")
            
            # Create document with metadata
            result = {
                "query": query,
                "perspective": perspective,
                "document": document,
                "refinement_steps": 0,
                "domain": domain
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating perspective document: {str(e)}")
            # Return minimal document as fallback
            return {
                "query": query,
                "perspective": perspective,
                "document": f"Document generation failed: {str(e)}",
                "refinement_steps": 0,
                "domain": domain,
                "error": str(e)
            }
    
    def _analyze_perspective_effectiveness(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the effectiveness of different perspectives across evaluations.
        
        Args:
            evaluations: List of evaluation results
            
        Returns:
            Analysis of perspective effectiveness
        """
        # Track scores by perspective and domain
        perspective_scores = {}
        domain_perspective_scores = {}
        
        # Process all evaluations
        for eval_result in evaluations:
            query = eval_result.get("query", "")
            domain = eval_result.get("domain", "general")
            
            # Initialize domain tracking if needed
            if domain not in domain_perspective_scores:
                domain_perspective_scores[domain] = {}
            
            # Process each perspective result
            for perspective_result in eval_result.get("perspective_results", []):
                perspective = perspective_result.get("perspective", "")
                
                # Skip if there was an error
                if "error" in perspective_result:
                    continue
                
                # Get document quality score
                doc_eval = perspective_result.get("document_evaluation", {})
                doc_score = doc_eval.get("overall_score", 0)
                
                # Initialize perspective tracking if needed
                if perspective not in perspective_scores:
                    perspective_scores[perspective] = {
                        "document_quality_scores": [],
                        "retrieval_effectiveness": {
                            "wins": 0,
                            "total": 0
                        },
                        "queries": []
                    }
                
                if perspective not in domain_perspective_scores[domain]:
                    domain_perspective_scores[domain][perspective] = {
                        "document_quality_scores": [],
                        "retrieval_effectiveness": {
                            "wins": 0,
                            "total": 0
                        },
                        "queries": []
                    }
                
                # Add document quality score
                perspective_scores[perspective]["document_quality_scores"].append(doc_score)
                domain_perspective_scores[domain][perspective]["document_quality_scores"].append(doc_score)
                
                # Track this query
                perspective_scores[perspective]["queries"].append(query)
                domain_perspective_scores[domain][perspective]["queries"].append(query)
                
                # Check retrieval effectiveness
                retrieval_eval = perspective_result.get("retrieval_evaluation", {})
                overall_quality = retrieval_eval.get("overall_quality", {})
                better_set = overall_quality.get("better_set", "")
                
                if better_set:
                    perspective_scores[perspective]["retrieval_effectiveness"]["total"] += 1
                    domain_perspective_scores[domain][perspective]["retrieval_effectiveness"]["total"] += 1
                    
                    if better_set == "B":  # HyDE was better
                        perspective_scores[perspective]["retrieval_effectiveness"]["wins"] += 1
                        domain_perspective_scores[domain][perspective]["retrieval_effectiveness"]["wins"] += 1
        
        # Calculate average scores and win rates
        for perspective, data in perspective_scores.items():
            scores = data["document_quality_scores"]
            if scores:
                data["avg_document_quality"] = sum(scores) / len(scores)
            else:
                data["avg_document_quality"] = 0
            
            total = data["retrieval_effectiveness"]["total"]
            if total > 0:
                data["retrieval_win_rate"] = data["retrieval_effectiveness"]["wins"] / total
            else:
                data["retrieval_win_rate"] = 0
        
        # Calculate domain-specific average scores and win rates
        for domain, perspectives in domain_perspective_scores.items():
            for perspective, data in perspectives.items():
                scores = data["document_quality_scores"]
                if scores:
                    data["avg_document_quality"] = sum(scores) / len(scores)
                else:
                    data["avg_document_quality"] = 0
                
                total = data["retrieval_effectiveness"]["total"]
                if total > 0:
                    data["retrieval_win_rate"] = data["retrieval_effectiveness"]["wins"] / total
                else:
                    data["retrieval_win_rate"] = 0
        
        # Determine overall best perspectives
        best_perspectives = sorted(
            perspective_scores.keys(),
            key=lambda p: (
                perspective_scores[p].get("retrieval_win_rate", 0),
                perspective_scores[p].get("avg_document_quality", 0)
            ),
            reverse=True
        )
        
        # Determine domain-specific best perspectives
        domain_best_perspectives = {}
        for domain, perspectives in domain_perspective_scores.items():
            domain_best_perspectives[domain] = sorted(
                perspectives.keys(),
                key=lambda p: (
                    perspectives[p].get("retrieval_win_rate", 0),
                    perspectives[p].get("avg_document_quality", 0)
                ),
                reverse=True
            )
        
        # Create analysis report
        analysis = {
            "perspective_performance": perspective_scores,
            "domain_perspective_performance": domain_perspective_scores,
            "best_perspectives_overall": best_perspectives,
            "best_perspectives_by_domain": domain_best_perspectives,
            "recommendation": {
                "overall": best_perspectives[0] if best_perspectives else None,
                "by_domain": {d: p[0] if p else None for d, p in domain_best_perspectives.items()}
            }
        }
        
        return analysis 