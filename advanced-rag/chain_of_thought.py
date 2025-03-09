"""
Chain-of-Thought Reasoning

This module implements Chain-of-Thought (CoT) reasoning for RAG systems,
allowing for explicit step-by-step reasoning processes for complex queries.
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


class ChainOfThoughtReasoner:
    """
    Chain-of-Thought Reasoner that implements explicit step-by-step reasoning
    for complex queries, with full transparency in the reasoning process.
    
    Similar to Grok's transparent reasoning, this component shows its work,
    making the reasoning process visible and auditable. It integrates retrieved
    information directly into the reasoning steps.
    """
    
    def __init__(
        self,
        model_provider: Optional[Union[str, LLMProvider]] = None,
        retriever: Optional[Callable] = None,
        decomposition_strategy: str = "auto",  # "auto", "sequential", "tree", "recursive"
        max_reasoning_steps: int = 5,
        max_retrieval_per_step: int = 3,
        system_prompt: Optional[str] = None,
        reasoning_prompt: Optional[str] = None,
        show_intermediate_steps: bool = True,
        reasoning_temperature: float = 0.4,
        verify_reasoning: bool = True,
    ):
        """
        Initialize the Chain-of-Thought Reasoner.
        
        Args:
            model_provider: LLM provider for reasoning generation
            retriever: Function for retrieving passages from a knowledge base
            decomposition_strategy: Strategy for decomposing complex questions
            max_reasoning_steps: Maximum number of reasoning steps allowed
            max_retrieval_per_step: Maximum number of passages to retrieve per step
            system_prompt: Custom system prompt for the reasoning process
            reasoning_prompt: Custom prompt for the reasoning steps
            show_intermediate_steps: Whether to include intermediate steps in the final answer
            reasoning_temperature: Temperature setting for the reasoning generation
            verify_reasoning: Whether to verify the reasoning process with a final check
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
                logger.info("Using default OpenAI GPT-4o model provider for Chain-of-Thought reasoning")
            except Exception as e:
                logger.warning(f"Error initializing default model provider: {str(e)}")
                raise
        elif isinstance(model_provider, str):
            try:
                self.model_provider = get_model_provider(model_provider)
                logger.info(f"Using {model_provider} model provider for Chain-of-Thought reasoning")
            except Exception as e:
                logger.warning(f"Error initializing model provider {model_provider}: {str(e)}")
                raise
        else:
            self.model_provider = model_provider
        
        # Set retriever (can be updated later if None)
        self.retriever = retriever
        
        # Set reasoning parameters
        self.decomposition_strategy = decomposition_strategy
        self.max_reasoning_steps = max_reasoning_steps
        self.max_retrieval_per_step = max_retrieval_per_step
        self.show_intermediate_steps = show_intermediate_steps
        self.reasoning_temperature = reasoning_temperature
        self.verify_reasoning = verify_reasoning
        
        # Initialize default prompts
        self._init_default_prompts()
        
        # Override with custom prompts if provided
        if system_prompt:
            self.system_prompt = system_prompt
        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt
        
        logger.debug(f"Initialized Chain-of-Thought Reasoner with strategy: {decomposition_strategy}")
    
    def _init_default_prompts(self):
        """Initialize default prompts for the Chain-of-Thought reasoning."""
        
        # Main system prompt for the LLM
        self.system_prompt = """
        You are a Chain-of-Thought reasoning assistant that breaks down complex problems into explicit steps.
        
        For each question, you will:
        1. Identify the key components and requirements of the question
        2. Plan a logical approach to answer the question
        3. Execute the plan step-by-step, showing your work clearly
        4. Use available information sources when necessary
        5. Summarize the findings after completing the reasoning steps
        
        When reasoning:
        - Be methodical and transparent about each step
        - Clearly communicate what you know and what you need to find out
        - For factual claims, use retrieved information and cite your sources
        - For logical deductions, explain your thought process
        - Acknowledge uncertainty when present
        
        Your goal is to make your reasoning process as clear and explicit as possible so that a human 
        could follow and verify your logical steps.
        """
        
        # Prompt for generating chain-of-thought reasoning
        self.reasoning_prompt = """
        Solve the following question by breaking it down into explicit reasoning steps. 
        Show your work clearly at each step.
        
        Question: {question}
        
        {context_prompt}
        
        Use the following format:
        
        **Step 1: [Short step description]**
        [Detailed reasoning for this step]
        [Information needed]
        [Conclusions from this step]
        
        **Step 2: [Short step description]**
        [Detailed reasoning for this step]
        [Information needed]
        [Conclusions from this step]
        
        ... (continue with as many steps as needed)
        
        **Final Answer:**
        [Complete answer to the original question, incorporating all reasoning steps]
        
        Remember to:
        1. Break the problem into logical steps
        2. Explain your reasoning at each step
        3. Retrieve and incorporate relevant information as needed
        4. Cite any sources used in your reasoning
        5. Provide a comprehensive final answer
        """
    
    def set_retriever(self, retriever: Callable):
        """
        Set or update the retriever function.
        
        Args:
            retriever: Function for retrieving passages from a knowledge base
        """
        self.retriever = retriever
        logger.info("Retriever function updated")
    
    def _decompose_question(self, question: str) -> List[Dict[str, Any]]:
        """
        Decompose a complex question into sub-questions or reasoning steps.
        
        Args:
            question: The complex question to decompose
            
        Returns:
            List of dictionaries with decomposed steps/questions
        """
        decomposition_prompt = f"""
        Break down the following complex question into a step-by-step reasoning process:
        
        Question: {question}
        
        The decomposition should identify:
        1. What are the key parts of this question?
        2. What information is needed to answer each part?
        3. What are the logical steps needed to solve this question?
        4. What order should these steps be executed in?
        
        Provide your decomposition in the following JSON format:
        ```json
        {{
            "overall_approach": "Brief description of the overall approach",
            "steps": [
                {{
                    "step_number": 1,
                    "description": "Short step description",
                    "reasoning": "Why this step is necessary",
                    "information_needed": "Information required for this step",
                    "retrieval_query": "Specific query to retrieve information (if needed)"
                }},
                ...
            ]
        }}
        ```
        """
        
        # Create messages for decomposition
        messages = [
            Message(role=Role.SYSTEM, content="You are a question decomposition expert that breaks complex questions into logical steps."),
            Message(role=Role.USER, content=decomposition_prompt)
        ]
        
        # Generate the decomposition
        try:
            response = self.model_provider.generate(
                messages=messages,
                temperature=0.3,  # Low temperature for analytical task
                max_tokens=2048
            )
            
            # Extract JSON from the response
            content = response.get("content", "")
            
            # Find JSON block in the response
            json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try without code blocks
                json_match = re.search(r'{\s*"overall_approach".*}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    logger.warning("Could not extract JSON from decomposition response")
                    # Create a simple default decomposition
                    return [{
                        "step_number": 1,
                        "description": "Answer the question directly",
                        "reasoning": "Question seems straightforward enough for direct reasoning",
                        "information_needed": question,
                        "retrieval_query": question
                    }]
            
            # Parse the JSON response
            try:
                decomposition = json.loads(json_str)
                
                logger.info(f"Question decomposed into {len(decomposition.get('steps', []))} steps")
                return decomposition.get("steps", [])
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing decomposition JSON: {str(e)}")
                # Create a simple default decomposition
                return [{
                    "step_number": 1,
                    "description": "Answer the question directly",
                    "reasoning": "Question seems straightforward enough for direct reasoning",
                    "information_needed": question,
                    "retrieval_query": question
                }]
                
        except Exception as e:
            logger.error(f"Error during question decomposition: {str(e)}")
            # Create a simple default decomposition
            return [{
                "step_number": 1,
                "description": "Answer the question directly",
                "reasoning": "Question seems straightforward enough for direct reasoning",
                "information_needed": question,
                "retrieval_query": question
            }]
    
    def _retrieve_for_step(self, step: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Retrieve information for a specific reasoning step.
        
        Args:
            step: The reasoning step dictionary
            
        Returns:
            List of retrieved passages for this step
        """
        if not self.retriever:
            logger.warning("No retriever available for information retrieval")
            return []
        
        # Extract the retrieval query
        query = step.get("retrieval_query", step.get("description", ""))
        
        try:
            # Perform the retrieval
            results = self.retriever(query, limit=self.max_retrieval_per_step)
            logger.info(f"Retrieved {len(results)} passages for step: {step.get('description', '')}")
            return results
        except Exception as e:
            logger.error(f"Error retrieving information for step: {str(e)}")
            return []
    
    def _format_retrieved_context(self, retrieved_passages: List[Dict[str, Any]]) -> str:
        """
        Format retrieved passages for inclusion in the reasoning prompt.
        
        Args:
            retrieved_passages: List of retrieved passages
            
        Returns:
            Formatted context string for the reasoning prompt
        """
        if not retrieved_passages:
            return "No additional information available for this step."
        
        context_parts = ["Here is some relevant information that may help with this question:"]
        
        for i, passage in enumerate(retrieved_passages):
            # Extract passage content
            content = passage.get("content", passage.get("text", ""))
            
            # Extract source information
            source = passage.get("source", passage.get("metadata", {}).get("source", "Unknown source"))
            
            # Format with citation information
            context_parts.append(f"[{i+1}] From {source}:\n{content}\n")
        
        return "\n".join(context_parts)
    
    def _execute_reasoning_step(self, 
                              question: str, 
                              step: Dict[str, Any], 
                              previous_steps: List[Dict[str, Any]] = None,
                              retrieved_info: List[Dict[str, Any]] = None
                              ) -> Dict[str, Any]:
        """
        Execute a single reasoning step in the chain-of-thought process.
        
        Args:
            question: The original question
            step: The current reasoning step
            previous_steps: List of previous reasoning steps and their results
            retrieved_info: Retrieved information for this step
            
        Returns:
            Updated step dictionary with reasoning results
        """
        previous_steps = previous_steps or []
        retrieved_info = retrieved_info or []
        
        # Construct the context for this step
        context_prompt = self._format_retrieved_context(retrieved_info)
        
        # If there are previous steps, include them
        if previous_steps:
            previous_reasoning = "Previous reasoning steps:\n\n"
            
            for prev_step in previous_steps:
                step_num = prev_step.get("step_number", 0)
                description = prev_step.get("description", "")
                reasoning = prev_step.get("reasoning_result", "")
                
                previous_reasoning += f"**Step {step_num}: {description}**\n{reasoning}\n\n"
            
            context_prompt = previous_reasoning + "\n" + context_prompt
        
        # Construct the reasoning prompt for this step
        step_prompt = f"""
        I'm solving this question: {question}
        
        Current step: {step.get('step_number', 0)}. {step.get('description', '')}
        
        For this step, I need to: {step.get('reasoning', '')}
        Information needed: {step.get('information_needed', '')}
        
        {context_prompt}
        
        Please provide detailed reasoning for this specific step. Focus only on this step, not the entire question yet.
        """
        
        # Create messages for reasoning
        messages = [
            Message(role=Role.SYSTEM, content=self.system_prompt),
            Message(role=Role.USER, content=step_prompt)
        ]
        
        # Generate the reasoning for this step
        try:
            response = self.model_provider.generate(
                messages=messages,
                temperature=self.reasoning_temperature,
                max_tokens=1024
            )
            
            # Extract the reasoning
            reasoning_result = response.get("content", "")
            
            # Update the step with the reasoning result
            updated_step = step.copy()
            updated_step["reasoning_result"] = reasoning_result
            updated_step["retrieved_info"] = retrieved_info
            
            return updated_step
            
        except Exception as e:
            logger.error(f"Error during reasoning step execution: {str(e)}")
            
            # Create a fallback response
            updated_step = step.copy()
            updated_step["reasoning_result"] = f"Error occurred during reasoning: {str(e)}"
            updated_step["error"] = str(e)
            
            return updated_step
    
    def _generate_final_answer(self, 
                              question: str, 
                              completed_steps: List[Dict[str, Any]]
                              ) -> str:
        """
        Generate a final answer based on all reasoning steps.
        
        Args:
            question: The original question
            completed_steps: List of completed reasoning steps with results
            
        Returns:
            Final answer text
        """
        # Construct the final answer prompt
        all_steps = "\n\n".join([
            f"**Step {step.get('step_number', i+1)}: {step.get('description', '')}**\n{step.get('reasoning_result', '')}"
            for i, step in enumerate(completed_steps)
        ])
        
        final_prompt = f"""
        Question: {question}
        
        I've completed the reasoning steps for this question:
        
        {all_steps}
        
        Based on these steps, please provide a comprehensive final answer to the original question.
        The answer should synthesize all the reasoning steps and present a clear conclusion.
        
        If the show_intermediate_steps parameter is set to {self.show_intermediate_steps}, {'include' if self.show_intermediate_steps else 'do not include'} the detailed reasoning steps in the final answer.
        """
        
        # Create messages for final answer
        messages = [
            Message(role=Role.SYSTEM, content=self.system_prompt),
            Message(role=Role.USER, content=final_prompt)
        ]
        
        # Generate the final answer
        try:
            response = self.model_provider.generate(
                messages=messages,
                temperature=0.3,  # Lower temperature for final answer synthesis
                max_tokens=2048
            )
            
            # Extract the final answer
            final_answer = response.get("content", "")
            
            return final_answer
            
        except Exception as e:
            logger.error(f"Error generating final answer: {str(e)}")
            
            # Create a fallback response
            fallback_answer = f"I attempted to solve this question through a step-by-step reasoning process, but encountered an error in generating the final answer: {str(e)}. "
            
            # Include the steps we were able to complete
            if self.show_intermediate_steps:
                fallback_answer += "\n\nHere are the reasoning steps I was able to complete:\n\n" + all_steps
            
            return fallback_answer
    
    def _verify_reasoning_and_answer(self, 
                                    question: str, 
                                    answer: str, 
                                    reasoning_steps: List[Dict[str, Any]]
                                    ) -> Dict[str, Any]:
        """
        Verify the reasoning process and final answer for correctness.
        
        Args:
            question: The original question
            answer: The generated final answer
            reasoning_steps: The completed reasoning steps
            
        Returns:
            Verification results with potential corrections
        """
        # Create a summary of the reasoning steps
        steps_summary = "\n".join([
            f"Step {step.get('step_number', i+1)}: {step.get('description', '')}"
            for i, step in enumerate(reasoning_steps)
        ])
        
        # Construct the verification prompt
        verification_prompt = f"""
        Please verify the following reasoning process and answer for correctness:
        
        Question: {question}
        
        Reasoning Steps:
        {steps_summary}
        
        Final Answer:
        {answer}
        
        Please evaluate:
        1. Are the reasoning steps logically sound and appropriate for the question?
        2. Does the final answer follow logically from the reasoning steps?
        3. Is the final answer factually correct based on the information provided?
        4. Are there any logical fallacies or errors in the reasoning?
        5. Are there any improvements that could be made to the reasoning or answer?
        
        Provide your verification as JSON with the following fields:
        - reasoning_correct (true/false): Whether the reasoning process is correct
        - answer_correct (true/false): Whether the final answer is correct
        - issues (array): List of any issues identified in the reasoning or answer
        - corrections (object): Suggested corrections for any issues
        - improved_answer (string): An improved version of the answer if needed
        
        Return verification in JSON format.
        """
        
        # Create messages for verification
        messages = [
            Message(role=Role.SYSTEM, content="You are a critical evaluator of reasoning processes."),
            Message(role=Role.USER, content=verification_prompt)
        ]
        
        # Generate the verification
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
                json_match = re.search(r'{\s*"reasoning_correct".*}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    logger.warning("Could not extract JSON from verification response")
                    # Return a default verification
                    return {
                        "reasoning_correct": True,
                        "answer_correct": True,
                        "issues": [],
                        "corrections": {},
                        "improved_answer": answer
                    }
            
            # Parse the JSON verification
            try:
                verification = json.loads(json_str)
                logger.info(f"Reasoning verification: reasoning_correct={verification.get('reasoning_correct', True)}, answer_correct={verification.get('answer_correct', True)}")
                return verification
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing verification JSON: {str(e)}")
                # Return a default verification
                return {
                    "reasoning_correct": True,
                    "answer_correct": True,
                    "issues": [f"Error parsing verification: {str(e)}"],
                    "corrections": {},
                    "improved_answer": answer
                }
                
        except Exception as e:
            logger.error(f"Error during reasoning verification: {str(e)}")
            # Return a default verification
            return {
                "reasoning_correct": True,
                "answer_correct": True,
                "issues": [f"Error during verification: {str(e)}"],
                "corrections": {},
                "improved_answer": answer
            }
    
    def reason(self, question: str, context: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Dict[str, Any]:
        """
        Perform chain-of-thought reasoning on a complex question.
        
        Args:
            question: The question to reason about
            context: Optional pre-retrieved context
            **kwargs: Additional parameters for customization
                - retriever_kwargs: Parameters for the retriever function
                - decomposition_strategy: Override the default decomposition strategy
                - max_reasoning_steps: Override the maximum reasoning steps
                - show_intermediate_steps: Override whether to show intermediate steps
                - reasoning_temperature: Override the reasoning temperature
                - verify_reasoning: Override whether to verify the reasoning
                
        Returns:
            Dictionary containing:
                - answer: The final answer to the question
                - reasoning_steps: The detailed reasoning steps
                - decomposition: The question decomposition
                - verification: The verification results (if enabled)
        """
        # Override configuration parameters if provided
        decomposition_strategy = kwargs.get('decomposition_strategy', self.decomposition_strategy)
        max_reasoning_steps = kwargs.get('max_reasoning_steps', self.max_reasoning_steps)
        show_intermediate_steps = kwargs.get('show_intermediate_steps', self.show_intermediate_steps)
        reasoning_temperature = kwargs.get('reasoning_temperature', self.reasoning_temperature)
        verify_reasoning = kwargs.get('verify_reasoning', self.verify_reasoning)
        
        # If context is provided, use it instead of retrieval
        use_context = context is not None
        
        try:
            # Step 1: Decompose the question
            decomposed_steps = self._decompose_question(question)
            
            # Limit to max_reasoning_steps
            if len(decomposed_steps) > max_reasoning_steps:
                logger.warning(f"Question decomposed into {len(decomposed_steps)} steps, limiting to {max_reasoning_steps}")
                decomposed_steps = decomposed_steps[:max_reasoning_steps]
            
            # Step 2: Execute each reasoning step
            completed_steps = []
            
            for step in decomposed_steps:
                # Retrieve information for this step if needed
                if use_context:
                    # Use provided context
                    step_context = context
                elif self.retriever:
                    # Use retriever to get information
                    step_context = self._retrieve_for_step(step)
                else:
                    # No context or retriever available
                    step_context = []
                
                # Execute the reasoning step
                completed_step = self._execute_reasoning_step(
                    question=question,
                    step=step,
                    previous_steps=completed_steps,
                    retrieved_info=step_context
                )
                
                completed_steps.append(completed_step)
            
            # Step 3: Generate the final answer
            final_answer = self._generate_final_answer(question, completed_steps)
            
            # Step 4: Verify the reasoning if enabled
            verification = None
            if verify_reasoning:
                verification = self._verify_reasoning_and_answer(
                    question=question,
                    answer=final_answer,
                    reasoning_steps=completed_steps
                )
                
                # If verification suggests an improved answer, use it
                if verification.get("answer_correct", True) is False and "improved_answer" in verification:
                    final_answer = verification.get("improved_answer")
            
            # Construct the result
            result = {
                "answer": final_answer,
                "reasoning_steps": completed_steps,
                "decomposition": {
                    "strategy": decomposition_strategy,
                    "steps": [{"step_number": step.get("step_number"), "description": step.get("description")} 
                             for step in decomposed_steps]
                }
            }
            
            # Include verification if available
            if verification:
                result["verification"] = verification
            
            return result
            
        except Exception as e:
            logger.error(f"Error during chain-of-thought reasoning: {str(e)}")
            
            # Return a fallback result
            return {
                "answer": f"I encountered an error while trying to reason about this question: {str(e)}",
                "error": str(e),
                "reasoning_steps": []
            } 