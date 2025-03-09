"""
Multi-Hop RAG Module

This module implements advanced multi-hop retrieval augmented generation,
which enables more complex reasoning by performing multiple rounds of
retrieval for different aspects of a query.
"""

import logging
import re
import json
import time
import networkx as nx
from typing import List, Dict, Any, Optional, Union, Callable, Tuple, Set

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


class MultiHopRAG:
    """
    Multi-Hop Retrieval Augmented Generation system.
    
    This class implements advanced RAG techniques that perform multiple
    hops of retrieval for complex queries, building a knowledge graph
    from retrieved information and constructing comprehensive answers.
    """
    
    def __init__(
        self,
        model_provider: Optional[Union[str, LLMProvider]] = None,
        retriever: Optional[Callable] = None,
        max_hops: int = 3,
        max_sub_questions: int = 5,
        max_contexts_per_hop: int = 3,
        system_prompt: Optional[str] = None,
        decomposition_prompt: Optional[str] = None,
        synthesis_prompt: Optional[str] = None,
        follow_up_prompt: Optional[str] = None,
        temperature: float = 0.3,
        use_dynamic_hops: bool = True,
        save_intermediate_results: bool = True,
        visualization_enabled: bool = False,
    ):
        """
        Initialize the Multi-Hop RAG system.
        
        Args:
            model_provider: LLM provider for generating reasoning and answers
            retriever: Function for retrieving passages from a knowledge base
            max_hops: Maximum number of retrieval hops to perform
            max_sub_questions: Maximum number of sub-questions to generate
            max_contexts_per_hop: Maximum number of contexts to retrieve per hop
            system_prompt: Custom system prompt for the LLM
            decomposition_prompt: Custom prompt for question decomposition
            synthesis_prompt: Custom prompt for synthesizing the final answer
            follow_up_prompt: Custom prompt for generating follow-up questions
            temperature: Temperature for LLM generation
            use_dynamic_hops: Whether to dynamically determine the number of hops needed
            save_intermediate_results: Whether to save intermediate results for debugging
            visualization_enabled: Whether to generate visualization data for the knowledge graph
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
                logger.info("Using default OpenAI GPT-4o model provider for Multi-Hop RAG")
            except Exception as e:
                logger.warning(f"Error initializing default model provider: {str(e)}")
                raise
        elif isinstance(model_provider, str):
            try:
                self.model_provider = get_model_provider(model_provider)
                logger.info(f"Using {model_provider} model provider for Multi-Hop RAG")
            except Exception as e:
                logger.warning(f"Error initializing model provider {model_provider}: {str(e)}")
                raise
        else:
            self.model_provider = model_provider
        
        # Set retriever (can be updated later if None)
        self.retriever = retriever
        
        # Set multi-hop parameters
        self.max_hops = max_hops
        self.max_sub_questions = max_sub_questions
        self.max_contexts_per_hop = max_contexts_per_hop
        self.temperature = temperature
        self.use_dynamic_hops = use_dynamic_hops
        self.save_intermediate_results = save_intermediate_results
        self.visualization_enabled = visualization_enabled
        
        # Initialize default prompts
        self._init_default_prompts()
        
        # Override with custom prompts if provided
        if system_prompt:
            self.system_prompt = system_prompt
        if decomposition_prompt:
            self.decomposition_prompt = decomposition_prompt
        if synthesis_prompt:
            self.synthesis_prompt = synthesis_prompt
        if follow_up_prompt:
            self.follow_up_prompt = follow_up_prompt
        
        # Knowledge graph for tracking retrieval hops and relationships
        self.knowledge_graph = None
        
        logger.debug(f"Initialized Multi-Hop RAG with max {max_hops} hops")
    
    def _init_default_prompts(self):
        """Initialize default prompts for the Multi-Hop RAG system."""
        
        # Main system prompt
        self.system_prompt = """
        You are an expert at answering complex questions through multiple steps
        of research and reasoning. Your goal is to provide comprehensive, accurate 
        answers by breaking down questions into logical parts and exploring each part
        systematically.
        
        When analyzing a question, first understand what information is needed to
        provide a complete answer. Then decompose the question into sub-questions
        that can be researched independently. After gathering information, synthesize
        it into a coherent answer that addresses all aspects of the original question.
        
        Be thorough, precise, and ensure that your reasoning is clear. When information
        is incomplete or uncertain, acknowledge this and suggest what additional
        information would be helpful.
        """
        
        # Prompt for question decomposition
        self.decomposition_prompt = """
        I need to answer a complex question by breaking it down into smaller, focused
        sub-questions. The question is:
        
        "{query}"
        
        Analyze this question and break it down into {max_sub_questions} or fewer logical 
        sub-questions that would help answer the main question thoroughly. 
        
        For each sub-question:
        1. Make it specific and focused on a particular aspect
        2. Ensure it's directly relevant to the main question
        3. Frame it in a way that can be answered through retrieval from a knowledge base
        
        Format your response as a JSON object with the following structure:
        
        {{
          "analysis": "Brief analysis of what the question is asking",
          "sub_questions": [
            {{
              "id": "SQ1",
              "question": "First sub-question",
              "rationale": "Why this sub-question is important"
            }},
            ...
          ],
          "dependencies": [
            ["SQ2", "SQ1"]  // Indicates SQ2 depends on the answer to SQ1
          ]
        }}
        
        Include any dependencies between sub-questions if applicable, where one sub-question
        depends on information from another.
        """
        
        # Prompt for synthesizing final answer
        self.synthesis_prompt = """
        I've researched a complex question through multiple retrieval steps.
        
        Original Question: "{query}"
        
        Here are the results of my research, showing sub-questions and the information retrieved:
        
        {research_summary}
        
        Based on all this information, synthesize a comprehensive answer to the original question.
        Your answer should:
        
        1. Address all aspects of the original question
        2. Integrate information from the different retrieval steps
        3. Present a logical flow of reasoning
        4. Acknowledge any limitations or uncertainties in the available information
        5. Be detailed and thorough
        
        In your answer, cite specific sources from the retrieved information where appropriate
        to support key points.
        """
        
        # Prompt for generating follow-up questions
        self.follow_up_prompt = """
        Based on the following sub-question and the information already retrieved:
        
        Sub-question: "{sub_question}"
        
        Retrieved information:
        {retrieved_info}
        
        Generate {num_follow_up} follow-up questions that would help gather additional
        relevant information not covered in the current retrieval. These questions should:
        
        1. Target specific information gaps in the current retrieval
        2. Help build a more complete understanding of the topic
        3. Be specific and focused enough for targeted retrieval
        4. Directly relate to answering the original sub-question
        
        Format your response as a JSON object with the following structure:
        
        {{
          "follow_up_questions": [
            {{
              "question": "First follow-up question",
              "rationale": "Why this follow-up question is needed"
            }},
            ...
          ]
        }}
        """
    
    def set_retriever(self, retriever: Callable):
        """
        Set or update the retriever function.
        
        Args:
            retriever: Function for retrieving passages from a knowledge base
        """
        self.retriever = retriever
        logger.info("Retriever function updated")
    
    def _decompose_question(self, query: str) -> Dict[str, Any]:
        """
        Decompose a complex query into sub-questions.
        
        Args:
            query: The user's query
            
        Returns:
            Dict containing analysis and sub-questions
        """
        # Format the decomposition prompt
        formatted_prompt = self.decomposition_prompt.format(
            query=query,
            max_sub_questions=self.max_sub_questions
        )
        
        # Create messages for question decomposition
        messages = [
            Message(role=Role.SYSTEM, content=self.system_prompt),
            Message(role=Role.USER, content=formatted_prompt)
        ]
        
        # Generate decomposition
        try:
            response = self.model_provider.generate(
                messages=messages,
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            
            # Extract the JSON content
            content = response.get("content", "{}")
            
            # Parse the JSON content
            if isinstance(content, str):
                try:
                    decomposition = json.loads(content)
                except json.JSONDecodeError:
                    logger.error(f"Error parsing decomposition JSON: {content[:100]}...")
                    # Extract JSON from text if possible
                    import re
                    json_match = re.search(r'({.*})', content, re.DOTALL)
                    if json_match:
                        try:
                            decomposition = json.loads(json_match.group(1))
                        except json.JSONDecodeError:
                            # Fallback to minimal structure
                            decomposition = {
                                "analysis": "Failed to parse JSON response",
                                "sub_questions": [],
                                "dependencies": []
                            }
                    else:
                        # Fallback to minimal structure
                        decomposition = {
                            "analysis": "Failed to parse JSON response",
                            "sub_questions": [],
                            "dependencies": []
                        }
            else:
                decomposition = content
            
            # Validate and ensure required fields
            if "sub_questions" not in decomposition:
                decomposition["sub_questions"] = []
            
            if "dependencies" not in decomposition:
                decomposition["dependencies"] = []
            
            if "analysis" not in decomposition:
                decomposition["analysis"] = "No analysis provided"
            
            # Log decomposition
            num_questions = len(decomposition.get("sub_questions", []))
            logger.info(f"Decomposed query into {num_questions} sub-questions")
            
            return decomposition
            
        except Exception as e:
            logger.error(f"Error decomposing question: {str(e)}")
            # Return a minimal decomposition as fallback
            return {
                "analysis": f"Error decomposing question: {str(e)}",
                "sub_questions": [{"id": "SQ1", "question": query, "rationale": "Direct query"}],
                "dependencies": []
            }
    
    def _retrieve_for_question(self, 
                             question: str, 
                             previous_contexts: Optional[List[Dict[str, Any]]] = None,
                             hop_number: int = 1) -> List[Dict[str, Any]]:
        """
        Retrieve information for a specific question.
        
        Args:
            question: The question to retrieve for
            previous_contexts: Previously retrieved contexts (for filtering)
            hop_number: The current hop number
            
        Returns:
            List of retrieved passages
        """
        if not self.retriever:
            raise ValueError("Retriever function is required")
        
        try:
            # Retrieve passages
            logger.info(f"Retrieving for question (hop {hop_number}): {question}")
            contexts = self.retriever(
                query=question,
                limit=self.max_contexts_per_hop
            )
            
            # Filter out duplicate content if previous contexts provided
            if previous_contexts:
                prev_content = {c.get("content", "") for c in previous_contexts}
                contexts = [c for c in contexts if c.get("content", "") not in prev_content]
            
            # Add hop metadata
            for context in contexts:
                context["hop"] = hop_number
                context["query"] = question
            
            logger.info(f"Retrieved {len(contexts)} contexts for hop {hop_number}")
            return contexts
            
        except Exception as e:
            logger.error(f"Error retrieving for question: {str(e)}")
            return []
    
    def _generate_follow_up_questions(self, 
                                   sub_question: Dict[str, Any],
                                   retrieved_contexts: List[Dict[str, Any]],
                                   num_follow_up: int = 2) -> List[Dict[str, Any]]:
        """
        Generate follow-up questions based on retrieved information.
        
        Args:
            sub_question: The original sub-question
            retrieved_contexts: Contexts retrieved for the sub-question
            num_follow_up: Number of follow-up questions to generate
            
        Returns:
            List of follow-up questions
        """
        # Format retrieved information
        retrieved_info = "\n\n".join([
            f"Source {i+1}:\n{ctx.get('content', '')}"
            for i, ctx in enumerate(retrieved_contexts)
        ])
        
        # Format the follow-up prompt
        formatted_prompt = self.follow_up_prompt.format(
            sub_question=sub_question.get("question", ""),
            retrieved_info=retrieved_info,
            num_follow_up=num_follow_up
        )
        
        # Create messages for follow-up generation
        messages = [
            Message(role=Role.SYSTEM, content=self.system_prompt),
            Message(role=Role.USER, content=formatted_prompt)
        ]
        
        # Generate follow-up questions
        try:
            response = self.model_provider.generate(
                messages=messages,
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            
            # Extract the JSON content
            content = response.get("content", "{}")
            
            # Parse the JSON content
            if isinstance(content, str):
                try:
                    follow_up_data = json.loads(content)
                except json.JSONDecodeError:
                    logger.error(f"Error parsing follow-up JSON: {content[:100]}...")
                    # Fallback to minimal structure
                    follow_up_data = {
                        "follow_up_questions": []
                    }
            else:
                follow_up_data = content
            
            # Extract the follow-up questions
            follow_up_questions = follow_up_data.get("follow_up_questions", [])
            
            # Add metadata and IDs
            for i, q in enumerate(follow_up_questions):
                q["id"] = f"{sub_question.get('id', 'SQ')}-FQ{i+1}"
                q["parent_id"] = sub_question.get('id', 'SQ')
                q["hop_level"] = 2  # Follow-ups are always at hop level 2
            
            logger.info(f"Generated {len(follow_up_questions)} follow-up questions")
            return follow_up_questions
            
        except Exception as e:
            logger.error(f"Error generating follow-up questions: {str(e)}")
            return []
    
    def _synthesize_answer(self, 
                         query: str, 
                         research_results: Dict[str, Any]) -> str:
        """
        Synthesize a final answer from research results.
        
        Args:
            query: The original query
            research_results: Results from multi-hop research
            
        Returns:
            Synthesized answer
        """
        # Format research summary
        research_summary = ""
        
        # Add sub-questions and their retrieved information
        for sq in research_results.get("sub_questions", []):
            sq_id = sq.get("id", "")
            sq_question = sq.get("question", "")
            
            research_summary += f"\n\nSUB-QUESTION {sq_id}: {sq_question}\n"
            research_summary += f"Rationale: {sq.get('rationale', '')}\n\n"
            
            # Add contexts for this sub-question
            contexts = research_results.get("contexts", {}).get(sq_id, [])
            if contexts:
                research_summary += "Retrieved Information:\n"
                for i, ctx in enumerate(contexts):
                    content = ctx.get("content", "")
                    source = ctx.get("source", "Unknown")
                    research_summary += f"Source {i+1} [{source}]: {content}\n\n"
            else:
                research_summary += "No information retrieved for this sub-question.\n"
            
            # Add follow-up questions and their contexts
            follow_ups = research_results.get("follow_up_questions", {}).get(sq_id, [])
            for fq in follow_ups:
                fq_id = fq.get("id", "")
                fq_question = fq.get("question", "")
                
                research_summary += f"\nFOLLOW-UP QUESTION {fq_id}: {fq_question}\n"
                research_summary += f"Rationale: {fq.get('rationale', '')}\n\n"
                
                # Add contexts for this follow-up question
                fq_contexts = research_results.get("contexts", {}).get(fq_id, [])
                if fq_contexts:
                    research_summary += "Retrieved Information:\n"
                    for i, ctx in enumerate(fq_contexts):
                        content = ctx.get("content", "")
                        source = ctx.get("source", "Unknown")
                        research_summary += f"Source {i+1} [{source}]: {content}\n\n"
                else:
                    research_summary += "No information retrieved for this follow-up question.\n"
        
        # Format the synthesis prompt
        formatted_prompt = self.synthesis_prompt.format(
            query=query,
            research_summary=research_summary
        )
        
        # Create messages for synthesis
        messages = [
            Message(role=Role.SYSTEM, content=self.system_prompt),
            Message(role=Role.USER, content=formatted_prompt)
        ]
        
        # Generate synthesized answer
        try:
            response = self.model_provider.generate(
                messages=messages,
                temperature=self.temperature
            )
            
            # Extract the synthesized answer
            answer = response.get("content", "")
            
            logger.info(f"Synthesized answer of {len(answer.split())} words")
            return answer
            
        except Exception as e:
            logger.error(f"Error synthesizing answer: {str(e)}")
            # Return a minimal answer as fallback
            return f"Unable to synthesize a complete answer due to an error: {str(e)}"
    
    def _build_knowledge_graph(self, research_results: Dict[str, Any]) -> nx.DiGraph:
        """
        Build a knowledge graph from research results for visualization.
        
        Args:
            research_results: Results from multi-hop research
            
        Returns:
            NetworkX directed graph
        """
        G = nx.DiGraph()
        
        # Add main query node
        query = research_results.get("query", "Main Query")
        G.add_node("QUERY", label=query, type="query")
        
        # Add sub-question nodes and connect to main query
        for sq in research_results.get("sub_questions", []):
            sq_id = sq.get("id", "")
            sq_question = sq.get("question", "")
            G.add_node(sq_id, label=sq_question, type="sub_question")
            G.add_edge("QUERY", sq_id, type="decomposes_to")
        
        # Add dependency edges between sub-questions
        for dep in research_results.get("dependencies", []):
            if len(dep) == 2:
                G.add_edge(dep[0], dep[1], type="depends_on")
        
        # Add follow-up question nodes and connect to parent sub-questions
        for sq_id, follow_ups in research_results.get("follow_up_questions", {}).items():
            for fq in follow_ups:
                fq_id = fq.get("id", "")
                fq_question = fq.get("question", "")
                G.add_node(fq_id, label=fq_question, type="follow_up")
                G.add_edge(sq_id, fq_id, type="follows_up")
        
        # Add context nodes and connect to respective questions
        for question_id, contexts in research_results.get("contexts", {}).items():
            for i, ctx in enumerate(contexts):
                ctx_id = f"{question_id}-CTX{i+1}"
                ctx_content = ctx.get("content", "")[:100] + "..."  # Truncated content
                ctx_source = ctx.get("source", "Unknown")
                
                G.add_node(ctx_id, label=ctx_content, source=ctx_source, type="context")
                G.add_edge(question_id, ctx_id, type="retrieves")
        
        # Add answer node
        answer = research_results.get("answer", "")[:100] + "..."
        G.add_node("ANSWER", label=answer, type="answer")
        G.add_edge("QUERY", "ANSWER", type="answers")
        
        return G
    
    def _get_visualization_data(self, G: nx.DiGraph) -> Dict[str, Any]:
        """
        Extract visualization data from the knowledge graph.
        
        Args:
            G: NetworkX directed graph
            
        Returns:
            Dictionary with visualization data
        """
        # Get node data
        nodes = []
        for node, data in G.nodes(data=True):
            node_data = {
                "id": node,
                "label": data.get("label", node),
                "type": data.get("type", "unknown")
            }
            
            # Add source if available
            if "source" in data:
                node_data["source"] = data["source"]
            
            nodes.append(node_data)
        
        # Get edge data
        edges = []
        for source, target, data in G.edges(data=True):
            edge_data = {
                "source": source,
                "target": target,
                "type": data.get("type", "connects")
            }
            edges.append(edge_data)
        
        # Extract hierarchical graph structure
        # Find the main query node
        query_node = None
        for node in nodes:
            if node["type"] == "query":
                query_node = node["id"]
                break
        
        # BFS from query node to build hierarchy
        hierarchy = {"id": query_node, "children": []}
        visited = {query_node}
        
        # Helper function to build tree
        def build_tree(node_id, tree):
            children = [target for source, target in G.edges() if source == node_id]
            for child in children:
                if child not in visited:
                    visited.add(child)
                    child_tree = {"id": child, "children": []}
                    tree["children"].append(child_tree)
                    build_tree(child, child_tree)
        
        # Build the tree
        if query_node:
            build_tree(query_node, hierarchy)
        
        return {
            "nodes": nodes,
            "edges": edges,
            "hierarchy": hierarchy
        }
    
    def answer(self, query: str) -> Dict[str, Any]:
        """
        Generate a comprehensive answer through multi-hop retrieval.
        
        Args:
            query: The user's query
            
        Returns:
            Dictionary with final answer and research details
        """
        if not self.retriever:
            raise ValueError("Retriever function is required for multi-hop retrieval")
        
        start_time = time.time()
        logger.info(f"Starting multi-hop RAG for query: {query}")
        
        # Initialize research results
        research_results = {
            "query": query,
            "sub_questions": [],
            "dependencies": [],
            "follow_up_questions": {},
            "contexts": {},
            "answer": "",
            "metrics": {
                "total_contexts": 0,
                "total_hops": 0
            }
        }
        
        try:
            # Step 1: Decompose the question
            decomposition = self._decompose_question(query)
            
            # Store decomposition results
            research_results["sub_questions"] = decomposition.get("sub_questions", [])
            research_results["dependencies"] = decomposition.get("dependencies", [])
            research_results["analysis"] = decomposition.get("analysis", "")
            
            # Track all retrieved contexts to avoid duplicates
            all_contexts = []
            
            # Step 2: Retrieve information for each sub-question
            for sq in research_results["sub_questions"]:
                sq_id = sq.get("id", "")
                sq_question = sq.get("question", "")
                
                # Initialize contexts for this sub-question
                research_results["contexts"][sq_id] = []
                
                # Retrieve information
                contexts = self._retrieve_for_question(
                    question=sq_question,
                    previous_contexts=all_contexts,
                    hop_number=1
                )
                
                # Store contexts
                research_results["contexts"][sq_id] = contexts
                all_contexts.extend(contexts)
                
                # Generate follow-up questions
                if contexts and len(contexts) > 0:
                    follow_ups = self._generate_follow_up_questions(
                        sub_question=sq,
                        retrieved_contexts=contexts
                    )
                    
                    if follow_ups:
                        research_results["follow_up_questions"][sq_id] = follow_ups
                        
                        # Step 3: Retrieve information for follow-up questions
                        for fq in follow_ups:
                            fq_id = fq.get("id", "")
                            fq_question = fq.get("question", "")
                            
                            # Initialize contexts for this follow-up question
                            research_results["contexts"][fq_id] = []
                            
                            # Retrieve information
                            fq_contexts = self._retrieve_for_question(
                                question=fq_question,
                                previous_contexts=all_contexts,
                                hop_number=2
                            )
                            
                            # Store contexts
                            research_results["contexts"][fq_id] = fq_contexts
                            all_contexts.extend(fq_contexts)
            
            # Count total contexts
            total_contexts = sum(len(ctx_list) for ctx_list in research_results["contexts"].values())
            research_results["metrics"]["total_contexts"] = total_contexts
            research_results["metrics"]["total_hops"] = 2 if any(research_results["follow_up_questions"]) else 1
            
            # Step 4: Synthesize final answer
            answer = self._synthesize_answer(query, research_results)
            research_results["answer"] = answer
            
            # Build knowledge graph if visualization is enabled
            if self.visualization_enabled:
                self.knowledge_graph = self._build_knowledge_graph(research_results)
                research_results["visualization"] = self._get_visualization_data(self.knowledge_graph)
            
            # Calculate and record metrics
            end_time = time.time()
            research_results["metrics"]["time_taken"] = end_time - start_time
            
            logger.info(f"Multi-hop RAG completed in {end_time - start_time:.2f} seconds")
            return research_results
            
        except Exception as e:
            logger.error(f"Error in multi-hop RAG: {str(e)}")
            
            # Add error information to results
            research_results["error"] = str(e)
            
            # Try to synthesize an answer with whatever information we have
            if any(research_results["contexts"]):
                try:
                    answer = self._synthesize_answer(query, research_results)
                    research_results["answer"] = answer
                except Exception as synth_error:
                    research_results["answer"] = f"Error synthesizing answer: {str(synth_error)}"
            else:
                research_results["answer"] = f"Unable to generate an answer due to an error: {str(e)}"
            
            # Calculate and record metrics even in case of error
            end_time = time.time()
            research_results["metrics"]["time_taken"] = end_time - start_time
            
            return research_results 