"""
Self-RAG Engine

This module implements Self-RAG (Retrieval-Augmented Generation with Self-Reflection),
providing verification, attribution, and confidence scoring for generated responses.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Union, Callable, Tuple
import json

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


class SelfRAGEngine:
    """
    Self-RAG Engine that implements advanced RAG techniques with self-reflection capabilities.
    
    Key features:
    - Retrieval verification to assess relevance and usefulness of retrieved passages
    - Citation and attribution tracking for knowledge tracing
    - Confidence scoring for generated content
    - Answer revision based on verification results
    - Support for controllable generation (e.g., creativity vs. factuality)
    """
    
    def __init__(
        self,
        model_provider: Optional[Union[str, LLMProvider]] = None,
        retriever: Optional[Callable] = None,
        verification_threshold: float = 0.7,
        confidence_threshold: float = 0.6,
        system_prompt: Optional[str] = None,
        verification_prompt: Optional[str] = None,
        generation_prompt: Optional[str] = None,
        revision_prompt: Optional[str] = None,
        citation_format: str = "inline",  # "inline", "footnote", "endnote", "academic", "none"
        max_retrieval_count: int = 5,
        creativity_control: float = 0.3,  # 0.0 (factual) to 1.0 (creative)
        include_verification_details: bool = False,
    ):
        """
        Initialize the Self-RAG Engine.
        
        Args:
            model_provider: LLM provider for text generation and verification
            retriever: Function for retrieving passages from a knowledge base
            verification_threshold: Threshold for passage verification (relevance)
            confidence_threshold: Threshold for statement confidence in generation
            system_prompt: Custom system prompt for the main generation
            verification_prompt: Custom prompt for retrieval verification step
            generation_prompt: Custom prompt for the final answer generation
            revision_prompt: Custom prompt for answer revision step
            citation_format: Format for citations in the generated response
            max_retrieval_count: Maximum number of passages to retrieve
            creativity_control: Balance between factual accuracy and creativity
            include_verification_details: Whether to include details of the verification process
        """
        if not MODELS_AVAILABLE:
            raise ImportError(
                "The models module is required. Make sure the 'tools.src.models' module is available."
            )
        
        # Set up the model provider
        if model_provider is None:
            try:
                # Default to OpenAI GPT-4 if not specified
                self.model_provider = get_model_provider("openai", model="gpt-4o")
                logger.info("Using default OpenAI GPT-4o model provider for Self-RAG")
            except Exception as e:
                logger.warning(f"Error initializing default model provider: {str(e)}")
                raise
        elif isinstance(model_provider, str):
            try:
                self.model_provider = get_model_provider(model_provider)
                logger.info(f"Using {model_provider} model provider for Self-RAG")
            except Exception as e:
                logger.warning(f"Error initializing model provider {model_provider}: {str(e)}")
                raise
        else:
            self.model_provider = model_provider
        
        # Set retriever (can be updated later if None)
        self.retriever = retriever
        
        # Set thresholds and configuration
        self.verification_threshold = verification_threshold
        self.confidence_threshold = confidence_threshold
        self.citation_format = citation_format
        self.max_retrieval_count = max_retrieval_count
        self.creativity_control = creativity_control
        self.include_verification_details = include_verification_details
        
        # Initialize default prompts
        self._init_default_prompts()
        
        # Override with custom prompts if provided
        if system_prompt:
            self.system_prompt = system_prompt
        if verification_prompt:
            self.verification_prompt = verification_prompt
        if generation_prompt:
            self.generation_prompt = generation_prompt
        if revision_prompt:
            self.revision_prompt = revision_prompt
        
        logger.debug(f"Initialized Self-RAG Engine with thresholds: verification={verification_threshold}, confidence={confidence_threshold}")
    
    def _init_default_prompts(self):
        """Initialize default prompts for the Self-RAG pipeline."""
        
        # Main system prompt for the LLM
        self.system_prompt = """
        You are a Self-RAG assistant that carefully verifies information before responding. 
        You always aim to provide accurate, relevant, and helpful information while being transparent
        about your knowledge limitations. When you're uncertain, you admit it rather than making things up.
        
        When working with retrieved information:
        1. Evaluate the relevance and quality of each retrieved passage
        2. Use only verified and relevant passages to form your answer
        3. Provide citations for factual claims
        4. When information is missing, acknowledge the limitation
        
        Always maintain a balanced, neutral, and helpful tone.
        """
        
        # Prompt for verifying retrieved passages
        self.verification_prompt = """
        Please evaluate the relevance and usefulness of the following retrieved passages for answering the user's question.
        
        User question: {question}
        
        For each passage, provide:
        1. A relevance score from 0.0 to 1.0, where 1.0 means highly relevant and 0.0 means irrelevant
        2. A brief explanation of your scoring reasoning
        3. A judgment (Yes/No) on whether this passage should be used in the answer
        
        Retrieved passages:
        {passages}
        
        Return your evaluation in JSON format:
        ```json
        [
            {{
                "passage_id": "passage-1",
                "relevance_score": 0.95,
                "explanation": "This passage directly answers the user's question with specific details.",
                "use_in_answer": true
            }},
            ...
        ]
        ```
        """
        
        # Prompt for generating the final answer
        self.generation_prompt = """
        Based on the verified passages, answer the user's question: {question}
        
        Use only the information in these verified passages:
        {verified_passages}
        
        Guidelines for your answer:
        1. Be concise and directly address the question
        2. Cite your sources using {citation_format} citations
        3. If the verified passages don't contain enough information, acknowledge the limitations
        4. For each statement, include a confidence score [High/Medium/Low] if your confidence threshold is {confidence_threshold}
        5. Creativity level: {creativity_level} (adjust your response style accordingly)
        
        Your answer should follow this structure:
        1. Direct answer to the question
        2. Supporting details and explanation
        3. Conclusion or summary
        """
        
        # Prompt for revising the answer
        self.revision_prompt = """
        Please review and revise your answer based on the following feedback:
        
        Original answer:
        {original_answer}
        
        Revision feedback:
        {revision_feedback}
        
        Guidelines for revision:
        1. Address all the points in the feedback
        2. Maintain factual accuracy and citation quality
        3. Ensure the answer is clear, concise, and helpful
        4. Remove any statements not supported by the verified passages
        5. Adjust confidence levels as needed
        
        Provide your revised answer:
        """
    
    def set_retriever(self, retriever: Callable):
        """
        Set or update the retriever function.
        
        Args:
            retriever: Function for retrieving passages from a knowledge base
        """
        self.retriever = retriever
        logger.info("Retriever function updated")
    
    def _format_passages_for_verification(self, passages: List[Dict[str, Any]]) -> str:
        """
        Format retrieved passages for the verification step.
        
        Args:
            passages: List of retrieved passage dictionaries
            
        Returns:
            Formatted passages string for verification prompt
        """
        formatted_passages = []
        
        for i, passage in enumerate(passages):
            # Extract passage ID (or create one)
            passage_id = passage.get("id", f"passage-{i+1}")
            
            # Extract passage content
            content = passage.get("content", passage.get("text", ""))
            
            # Extract source information if available
            source = passage.get("source", passage.get("metadata", {}).get("source", "Unknown source"))
            
            # Format the passage
            formatted_passage = f"Passage {i+1} [ID: {passage_id}] from {source}:\n{content}\n\n"
            formatted_passages.append(formatted_passage)
        
        return "\n".join(formatted_passages)
    
    def _verify_passages(self, question: str, passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Verify the relevance and quality of retrieved passages.
        
        Args:
            question: User's question
            passages: List of retrieved passages
            
        Returns:
            List of verified passages with verification metadata
        """
        if not passages:
            logger.warning("No passages provided for verification")
            return []
        
        # Format passages for verification
        formatted_passages = self._format_passages_for_verification(passages)
        
        # Create verification prompt
        verification_content = self.verification_prompt.format(
            question=question,
            passages=formatted_passages
        )
        
        # Create messages for verification
        messages = [
            Message(role=Role.SYSTEM, content="You are a helpful assistant that evaluates the relevance of retrieved passages."),
            Message(role=Role.USER, content=verification_content)
        ]
        
        # Generate verification evaluation
        try:
            response = self.model_provider.generate(
                messages=messages,
                temperature=0.3,  # Low temperature for evaluation
                max_tokens=2048
            )
            
            # Extract and parse JSON output
            content = response.get("content", "")
            
            # Try to extract JSON from the output
            json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try without code blocks
                json_match = re.search(r'\[\s*{.*}\s*\]', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    logger.warning("Could not extract JSON from verification response")
                    return passages  # Return original passages as fallback
            
            # Parse the JSON evaluation
            try:
                evaluations = json.loads(json_str)
                
                # Link evaluations to original passages
                verified_passages = []
                
                for i, passage in enumerate(passages):
                    passage_id = passage.get("id", f"passage-{i+1}")
                    
                    # Find the corresponding evaluation
                    eval_entry = next(
                        (eval for eval in evaluations if eval.get("passage_id") == passage_id),
                        None
                    )
                    
                    # If not found by ID, try index match
                    if eval_entry is None and i < len(evaluations):
                        eval_entry = evaluations[i]
                    
                    # If still not found, use default values
                    if eval_entry is None:
                        eval_entry = {
                            "relevance_score": 0.5,
                            "explanation": "No specific evaluation provided.",
                            "use_in_answer": passage.get("score", 0) > self.verification_threshold
                        }
                    
                    # Add verification metadata to passage
                    verified_passage = passage.copy()
                    verified_passage["relevance_score"] = eval_entry.get("relevance_score", 0.5)
                    verified_passage["verification_explanation"] = eval_entry.get("explanation", "")
                    verified_passage["use_in_answer"] = eval_entry.get("use_in_answer", False)
                    
                    verified_passages.append(verified_passage)
                
                logger.info(f"Verification complete: {sum(1 for p in verified_passages if p['use_in_answer'])} of {len(verified_passages)} passages verified as useful")
                return verified_passages
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing verification JSON: {str(e)}")
                return passages  # Return original passages as fallback
            
        except Exception as e:
            logger.error(f"Error during passage verification: {str(e)}")
            return passages  # Return original passages as fallback
    
    def _format_verified_passages(self, passages: List[Dict[str, Any]]) -> str:
        """
        Format verified passages for the answer generation step.
        
        Args:
            passages: List of verified passage dictionaries
            
        Returns:
            Formatted verified passages string for generation prompt
        """
        # Filter passages that should be used in the answer
        useful_passages = [p for p in passages if p.get("use_in_answer", False)]
        
        if not useful_passages:
            logger.warning("No useful passages found for answer generation")
            # Include at least one passage with the highest relevance score
            if passages:
                sorted_passages = sorted(passages, key=lambda p: p.get("relevance_score", 0), reverse=True)
                useful_passages = [sorted_passages[0]]
        
        # Format the useful passages
        formatted_passages = []
        
        for i, passage in enumerate(useful_passages):
            # Extract passage content
            content = passage.get("content", passage.get("text", ""))
            
            # Extract source information
            source = passage.get("source", passage.get("metadata", {}).get("source", "Unknown source"))
            
            # Format with citation information
            formatted_passage = f"[{i+1}] Source: {source}\nContent: {content}\nRelevance: {passage.get('relevance_score', 0.0):.2f}\n\n"
            formatted_passages.append(formatted_passage)
        
        return "\n".join(formatted_passages)
    
    def _get_creativity_level_description(self) -> str:
        """
        Convert the creativity control value to a descriptive string.
        
        Returns:
            Description of the creativity level
        """
        if self.creativity_control <= 0.2:
            return "Very factual, stick closely to the verified information only"
        elif self.creativity_control <= 0.4:
            return "Mostly factual, minimal extrapolation beyond verified information"
        elif self.creativity_control <= 0.6:
            return "Balanced between factual information and helpful elaboration"
        elif self.creativity_control <= 0.8:
            return "Creative elaboration allowed, while maintaining factual correctness"
        else:
            return "Highly creative, can elaborate substantially while maintaining core factual accuracy"
    
    def _generate_answer(self, question: str, verified_passages: List[Dict[str, Any]]) -> str:
        """
        Generate a final answer based on verified passages.
        
        Args:
            question: User's question
            verified_passages: List of verified passage dictionaries
            
        Returns:
            Generated answer with citations
        """
        # Format verified passages
        formatted_passages = self._format_verified_passages(verified_passages)
        
        # Determine citation format instructions
        citation_format_instructions = {
            "inline": "Use numbered inline citations like [1], [2]",
            "footnote": "Use superscript numbers for footnote citations",
            "endnote": "Use numbered citations at the end of your answer",
            "academic": "Use academic citations in the format (Author, Year)",
            "none": "No formal citations needed, but mention sources naturally in your answer"
        }
        
        citation_instruction = citation_format_instructions.get(
            self.citation_format, "Use numbered inline citations like [1], [2]"
        )
        
        # Create generation prompt
        generation_content = self.generation_prompt.format(
            question=question,
            verified_passages=formatted_passages,
            citation_format=citation_instruction,
            confidence_threshold=self.confidence_threshold,
            creativity_level=self._get_creativity_level_description()
        )
        
        # Create messages for generation
        messages = [
            Message(role=Role.SYSTEM, content=self.system_prompt),
            Message(role=Role.USER, content=generation_content)
        ]
        
        # Generate answer
        try:
            response = self.model_provider.generate(
                messages=messages,
                temperature=0.5 + (self.creativity_control * 0.5),  # Adjust temperature based on creativity control
                max_tokens=2048
            )
            
            # Extract and return the generated answer
            answer = response.get("content", "")
            return answer
            
        except Exception as e:
            logger.error(f"Error during answer generation: {str(e)}")
            return f"I apologize, but I encountered an error while generating your answer. Error: {str(e)}"
    
    def _evaluate_answer(self, 
                         question: str, 
                         answer: str, 
                         verified_passages: List[Dict[str, Any]]
                         ) -> Dict[str, Any]:
        """
        Evaluate the generated answer for factual correctness and citation quality.
        
        Args:
            question: User's question
            answer: Generated answer
            verified_passages: Verified passages used for generation
            
        Returns:
            Evaluation results with revision feedback if needed
        """
        evaluation_prompt = f"""
        Please evaluate the following answer for factual correctness, citation quality, and completeness.
        
        Question: {question}
        
        Answer:
        {answer}
        
        The answer was generated based on these verified passages:
        {self._format_verified_passages(verified_passages)}
        
        Provide your evaluation as JSON with the following fields:
        1. factual_accuracy (0.0-1.0): Score for factual correctness
        2. citation_quality (0.0-1.0): Score for proper citation of claims
        3. completeness (0.0-1.0): Score for how completely the answer addresses the question
        4. needs_revision (true/false): Whether the answer needs revision
        5. revision_feedback: Specific feedback for revising the answer if needed
        
        Return evaluation in JSON format.
        """
        
        # Create messages for evaluation
        messages = [
            Message(role=Role.SYSTEM, content="You are a critical evaluator of generated answers."),
            Message(role=Role.USER, content=evaluation_prompt)
        ]
        
        # Generate evaluation
        try:
            response = self.model_provider.generate(
                messages=messages,
                temperature=0.3,  # Low temperature for evaluation
                max_tokens=1024
            )
            
            # Extract and parse JSON output
            content = response.get("content", "")
            
            # Try to extract JSON from the output
            json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try without code blocks
                json_match = re.search(r'{\s*"factual_accuracy".*}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    logger.warning("Could not extract JSON from evaluation response")
                    # Return a default evaluation
                    return {
                        "factual_accuracy": 0.8,
                        "citation_quality": 0.7,
                        "completeness": 0.8,
                        "needs_revision": False,
                        "revision_feedback": "No specific feedback."
                    }
            
            # Parse the JSON evaluation
            try:
                evaluation = json.loads(json_str)
                logger.info(f"Answer evaluation: factual_accuracy={evaluation.get('factual_accuracy', 0.0):.2f}, needs_revision={evaluation.get('needs_revision', False)}")
                return evaluation
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing evaluation JSON: {str(e)}")
                # Return a default evaluation
                return {
                    "factual_accuracy": 0.8,
                    "citation_quality": 0.7,
                    "completeness": 0.8,
                    "needs_revision": False,
                    "revision_feedback": "No specific feedback due to error."
                }
            
        except Exception as e:
            logger.error(f"Error during answer evaluation: {str(e)}")
            # Return a default evaluation
            return {
                "factual_accuracy": 0.8,
                "citation_quality": 0.7,
                "completeness": 0.8,
                "needs_revision": False,
                "revision_feedback": f"Error during evaluation: {str(e)}"
            }
    
    def _revise_answer(self, 
                       question: str, 
                       original_answer: str, 
                       revision_feedback: str, 
                       verified_passages: List[Dict[str, Any]]
                       ) -> str:
        """
        Revise the answer based on evaluation feedback.
        
        Args:
            question: User's question
            original_answer: Original generated answer
            revision_feedback: Feedback for revision
            verified_passages: Verified passages used for generation
            
        Returns:
            Revised answer
        """
        # Create revision prompt
        revision_content = self.revision_prompt.format(
            original_answer=original_answer,
            revision_feedback=revision_feedback
        )
        
        # Add context about verified passages
        revision_content += f"\n\nOriginal question: {question}\n\n"
        revision_content += f"Verified passages for reference:\n{self._format_verified_passages(verified_passages)}"
        
        # Create messages for revision
        messages = [
            Message(role=Role.SYSTEM, content=self.system_prompt),
            Message(role=Role.USER, content=revision_content)
        ]
        
        # Generate revised answer
        try:
            response = self.model_provider.generate(
                messages=messages,
                temperature=0.4,  # Lower temperature for revision
                max_tokens=2048
            )
            
            # Extract and return the revised answer
            revised_answer = response.get("content", "")
            return revised_answer
            
        except Exception as e:
            logger.error(f"Error during answer revision: {str(e)}")
            return original_answer  # Return original answer as fallback
    
    def generate(self, 
                 question: str, 
                 context: Optional[List[Dict[str, Any]]] = None, 
                 **kwargs) -> Dict[str, Any]:
        """
        Generate a verified and cited answer to the user's question.
        
        Args:
            question: User's question
            context: Optional pre-retrieved context passages
            **kwargs: Additional parameters for customization
                - retriever_kwargs: Dictionary of parameters for the retriever
                - verification_threshold: Override the default verification threshold
                - confidence_threshold: Override the default confidence threshold
                - creativity_control: Override the default creativity control
                - citation_format: Override the default citation format
                - include_verification_details: Override the default include_verification_details
                
        Returns:
            Dictionary containing:
                - answer: The final generated answer
                - verified_passages: The verified passages used in the answer
                - citations: Citation information for traceability
                - confidence: Overall confidence score
                - (optional) verification_details: Details of the verification process
        """
        # Override configuration parameters if provided
        verification_threshold = kwargs.get('verification_threshold', self.verification_threshold)
        confidence_threshold = kwargs.get('confidence_threshold', self.confidence_threshold)
        creativity_control = kwargs.get('creativity_control', self.creativity_control)
        citation_format = kwargs.get('citation_format', self.citation_format)
        include_verification_details = kwargs.get('include_verification_details', self.include_verification_details)
        
        # Get passages, either from provided context or from retriever
        passages = []
        if context is not None:
            passages = context
        elif self.retriever is not None:
            retriever_kwargs = kwargs.get('retriever_kwargs', {})
            try:
                passages = self.retriever(question, **retriever_kwargs)
                logger.info(f"Retrieved {len(passages)} passages")
            except Exception as e:
                logger.error(f"Error during retrieval: {str(e)}")
                return {
                    "answer": f"I apologize, but I encountered an error during information retrieval: {str(e)}",
                    "verified_passages": [],
                    "citations": [],
                    "confidence": 0.0
                }
        else:
            logger.warning("No context provided and no retriever available")
            return {
                "answer": "I don't have enough information to answer this question. Please provide context or configure a retriever.",
                "verified_passages": [],
                "citations": [],
                "confidence": 0.0
            }
        
        # Verify retrieved passages
        verified_passages = self._verify_passages(question, passages)
        
        # If no passages are verified as useful, return early
        useful_passages = [p for p in verified_passages if p.get("use_in_answer", False)]
        if not useful_passages and verified_passages:
            logger.warning("No passages verified as useful for answering the question")
            # Include highest scored passage if available
            sorted_passages = sorted(verified_passages, key=lambda p: p.get("relevance_score", 0), reverse=True)
            if sorted_passages[0].get("relevance_score", 0) < verification_threshold:
                return {
                    "answer": "I don't have enough relevant information to answer this question accurately.",
                    "verified_passages": verified_passages if include_verification_details else [],
                    "citations": [],
                    "confidence": 0.0
                }
        
        # Generate answer using verified passages
        answer = self._generate_answer(question, verified_passages)
        
        # Evaluate the answer
        evaluation = self._evaluate_answer(question, answer, verified_passages)
        
        # Revise the answer if needed
        if evaluation.get("needs_revision", False):
            logger.info("Answer needs revision, generating revised answer")
            revision_feedback = evaluation.get("revision_feedback", "Please improve factual accuracy and citations.")
            answer = self._revise_answer(question, answer, revision_feedback, verified_passages)
        
        # Extract citations from the answer
        citations = self._extract_citations(answer, verified_passages)
        
        # Calculate overall confidence
        confidence = self._calculate_confidence(evaluation, verified_passages)
        
        # Create the result
        result = {
            "answer": answer,
            "citations": citations,
            "confidence": confidence
        }
        
        # Include verification details if requested
        if include_verification_details:
            result["verified_passages"] = verified_passages
            result["evaluation"] = evaluation
        
        return result
    
    def _extract_citations(self, answer: str, verified_passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract citation information from the answer.
        
        Args:
            answer: Generated answer
            verified_passages: Verified passages used in the answer
            
        Returns:
            List of citation dictionaries
        """
        citations = []
        
        # Different citation formats require different extraction patterns
        if self.citation_format == "inline":
            # Match patterns like [1], [2], etc.
            citation_matches = re.finditer(r'\[(\d+)\]', answer)
            
            for match in citation_matches:
                citation_num = int(match.group(1))
                
                # Find the corresponding passage
                if 1 <= citation_num <= len(verified_passages):
                    passage = verified_passages[citation_num - 1]
                    
                    # Extract source information
                    source = passage.get("source", passage.get("metadata", {}).get("source", "Unknown source"))
                    
                    citations.append({
                        "citation_number": citation_num,
                        "source": source,
                        "passage_id": passage.get("id", f"passage-{citation_num}"),
                        "relevance_score": passage.get("relevance_score", 0.0)
                    })
        
        elif self.citation_format == "academic":
            # Match patterns like (Author, Year)
            citation_matches = re.finditer(r'\(([^)]+),\s*(\d{4})\)', answer)
            
            for i, match in enumerate(citation_matches):
                author = match.group(1)
                year = match.group(2)
                
                # Find a matching passage if possible
                matching_passage = None
                for passage in verified_passages:
                    source = passage.get("source", "")
                    if author.lower() in source.lower() and year in source:
                        matching_passage = passage
                        break
                
                if matching_passage:
                    citations.append({
                        "citation_number": i + 1,
                        "source": f"{author}, {year}",
                        "passage_id": matching_passage.get("id", f"passage-{i+1}"),
                        "relevance_score": matching_passage.get("relevance_score", 0.0)
                    })
                else:
                    # No matching passage found
                    citations.append({
                        "citation_number": i + 1,
                        "source": f"{author}, {year}",
                        "passage_id": None,
                        "relevance_score": None
                    })
        
        # If no citations were extracted or citation format is "none", create generic source list
        if not citations or self.citation_format == "none":
            for i, passage in enumerate(verified_passages):
                if passage.get("use_in_answer", False):
                    source = passage.get("source", passage.get("metadata", {}).get("source", "Unknown source"))
                    
                    citations.append({
                        "citation_number": i + 1,
                        "source": source,
                        "passage_id": passage.get("id", f"passage-{i+1}"),
                        "relevance_score": passage.get("relevance_score", 0.0)
                    })
        
        return citations
    
    def _calculate_confidence(self, 
                             evaluation: Dict[str, Any], 
                             verified_passages: List[Dict[str, Any]]
                             ) -> float:
        """
        Calculate overall confidence score for the answer.
        
        Args:
            evaluation: Evaluation results
            verified_passages: Verified passages used in the answer
            
        Returns:
            Confidence score (0.0-1.0)
        """
        # Extract evaluation scores
        factual_accuracy = evaluation.get("factual_accuracy", 0.7)
        citation_quality = evaluation.get("citation_quality", 0.7)
        completeness = evaluation.get("completeness", 0.7)
        
        # Calculate average passage relevance
        useful_passages = [p for p in verified_passages if p.get("use_in_answer", False)]
        if useful_passages:
            avg_relevance = sum(p.get("relevance_score", 0.0) for p in useful_passages) / len(useful_passages)
        else:
            avg_relevance = 0.0
        
        # Weighted calculation of confidence
        confidence = (
            0.4 * factual_accuracy +
            0.3 * citation_quality +
            0.2 * completeness +
            0.1 * avg_relevance
        )
        
        return min(1.0, max(0.0, confidence))  # Ensure value is between 0.0 and 1.0 