"""
Query Decomposer

This module provides the QueryDecomposer class for breaking down complex queries into simpler ones.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Union, Callable

from .base_processor import BaseQueryProcessor

# Configure logging
logger = logging.getLogger(__name__)


class QueryDecomposer(BaseQueryProcessor):
    """
    Query decomposer for breaking down complex queries into simpler ones.
    
    This class analyzes and decomposes complex, multi-faceted queries into a set of 
    simpler, more focused sub-queries to improve retrieval performance.
    """
    
    def __init__(
        self,
        decomposition_method: str = "rule",
        decomposition_rules: Optional[List[Dict[str, str]]] = None,
        custom_decomposer: Optional[Callable[[str], List[str]]] = None,
        llm_service: Optional[Any] = None, 
        llm_params: Optional[Dict[str, Any]] = None,
        merge_original: bool = True,
        max_subqueries: int = 5,
        min_subquery_length: int = 3
    ):
        """
        Initialize a QueryDecomposer.
        
        Args:
            decomposition_method: Method to use for query decomposition
                - "rule": Use predefined rules and patterns
                - "llm": Use a language model for decomposition
                - "custom": Use a custom decomposition function
            decomposition_rules: List of rules for "rule" method, each rule has:
                - 'pattern': Regex pattern to match
                - 'subqueries': List of templates for subqueries, using capture groups
            custom_decomposer: Custom function for query decomposition
            llm_service: LLM service for generating decompositions (for "llm" method)
            llm_params: Parameters for the LLM service
            merge_original: Whether to include the original query in the result
            max_subqueries: Maximum number of subqueries to generate
            min_subquery_length: Minimum length of subqueries (in words)
        """
        self.decomposition_method = decomposition_method.lower()
        self.decomposition_rules = decomposition_rules or []
        self.custom_decomposer = custom_decomposer
        self.llm_service = llm_service
        self.llm_params = llm_params or {}
        self.merge_original = merge_original
        self.max_subqueries = max_subqueries
        self.min_subquery_length = min_subquery_length
        
        # Validate decomposition method
        valid_methods = ["rule", "llm", "custom"]
        if self.decomposition_method not in valid_methods:
            raise ValueError(f"Invalid decomposition_method: {decomposition_method}. Must be one of {valid_methods}")
        
        # Validate requirements for chosen method
        if self.decomposition_method == "custom" and not self.custom_decomposer:
            raise ValueError("custom_decomposer function must be provided when using 'custom' decomposition method")
        
        if self.decomposition_method == "llm" and not self.llm_service:
            raise ValueError("llm_service must be provided when using 'llm' decomposition method")
            
        # Initialize default rules if none provided
        if self.decomposition_method == "rule" and not self.decomposition_rules:
            self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default decomposition rules."""
        self.decomposition_rules = [
            # Rule for "and" separations
            {
                'pattern': r'(.+)\s+and\s+(.+)',
                'subqueries': [r'\1', r'\2']
            },
            # Rule for "versus/vs" comparisons
            {
                'pattern': r'(.+)\s+(?:versus|vs\.?)\s+(.+)',
                'subqueries': [r'\1', r'\2', r'comparison of \1 and \2']
            },
            # Rule for "how to" with multiple steps
            {
                'pattern': r'how\s+to\s+(.+?)\s+(?:and|then|after)\s+(.+)',
                'subqueries': [r'how to \1', r'how to \2', r'steps for \1 and \2']
            },
            # Rule for multi-part questions
            {
                'pattern': r'(.+?)\??\s*(?:Also|Additionally|Furthermore|Moreover),?\s*(.+)',
                'subqueries': [r'\1', r'\2']
            },
            # Rule for "what is X and how does it Y"
            {
                'pattern': r'what\s+is\s+(.+?)\s+and\s+how\s+(?:does|do|can|could|would|should)\s+(?:it|they)\s+(.+)',
                'subqueries': [r'what is \1', r'how does \1 \2']
            },
            # Rule for list questions
            {
                'pattern': r'(?:what|list|name|tell\s+me)\s+(?:are|about)\s+(?:the|some|all)?\s*(.+?)\s+(?:of|for|in|related\s+to)\s+(.+)',
                'subqueries': [r'\1 \2', r'\2']
            },
            # Rule for definition and examples
            {
                'pattern': r'(?:what|define)\s+(?:is|are)\s+(.+?)\s+(?:and|with)?\s+(?:give|provide|show)\s+(?:me|some|an)?\s+examples',
                'subqueries': [r'what is \1', r'examples of \1']
            },
            # Rule for cause and effect
            {
                'pattern': r'(?:what|explain|describe)\s+(?:is|are)\s+(?:the)?\s*(?:causes|effects|impacts|consequences|results)\s+(?:of|from)\s+(.+)',
                'subqueries': [r'causes of \1', r'effects of \1', r'\1']
            }
        ]
    
    def _rule_based_decomposition(self, query: str) -> List[str]:
        """Decompose query using rule-based matching."""
        subqueries = []
        
        # Apply all rules in sequence
        for rule in self.decomposition_rules:
            pattern = rule.get('pattern', '')
            if not pattern:
                continue
                
            # Check if the pattern matches
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                # Apply subquery templates
                templates = rule.get('subqueries', [])
                for template in templates:
                    try:
                        # Replace capture groups in template
                        subquery = re.sub(pattern, template, query, flags=re.IGNORECASE)
                        subquery = subquery.strip()
                        
                        # Check if subquery is valid
                        if (subquery and subquery != query and 
                            len(subquery.split()) >= self.min_subquery_length and
                            subquery not in subqueries):
                            subqueries.append(subquery)
                    except Exception as e:
                        logger.error(f"Error applying subquery template: {str(e)}")
                        continue
                
                # If we found subqueries with this rule, stop (avoid over-decomposition)
                if subqueries:
                    break
                    
        # If no rules match, use heuristic decomposition
        if not subqueries:
            # Try to split on common separators
            for separator in [';', ',', '.', 'and', 'or', 'but', 'however', 'additionally']:
                if separator in query.lower() and not subqueries:
                    parts = re.split(fr'\s*{separator}\s*', query, flags=re.IGNORECASE)
                    for part in parts:
                        if (part.strip() and len(part.split()) >= self.min_subquery_length and
                            part.strip() not in subqueries):
                            subqueries.append(part.strip())
                
        return subqueries
    
    def _llm_based_decomposition(self, query: str) -> List[str]:
        """Decompose query using a language model."""
        if not self.llm_service:
            return []
        
        # Prepare prompt for the LLM
        prompt = (
            f"Break down the following complex query into {self.max_subqueries} or fewer simpler subqueries. "
            f"Each subquery should represent one aspect of the original query and should stand alone as a complete question. "
            f"Respond with a numbered list of subqueries, one per line.\n\n"
            f"Original query: {query}\n\n"
            f"Subqueries:"
        )
        
        try:
            # Call the LLM service
            response = self.llm_service.generate(
                prompt=prompt,
                **self.llm_params
            )
            
            # Parse the response
            if hasattr(response, 'text'):
                response_text = response.text
            elif isinstance(response, str):
                response_text = response
            else:
                response_text = str(response)
            
            # Extract subqueries
            subqueries = []
            
            # Look for numbered lines or bullet points
            lines = response_text.strip().split('\n')
            pattern = r'^\s*(?:\d+[\.\)]\s*|\-\s*|\*\s*)(.*)'
            
            for line in lines:
                match = re.match(pattern, line)
                if match:
                    subquery = match.group(1).strip()
                    if (subquery and len(subquery.split()) >= self.min_subquery_length and
                        subquery not in subqueries):
                        subqueries.append(subquery)
                elif line.strip() and len(line.split()) >= self.min_subquery_length:
                    # If line doesn't match pattern but is substantial, consider it a subquery
                    subqueries.append(line.strip())
                    
            return subqueries[:self.max_subqueries]
            
        except Exception as e:
            logger.error(f"Error decomposing query with LLM: {str(e)}")
            return []
    
    def _custom_decomposition(self, query: str) -> List[str]:
        """Decompose query using a custom function."""
        if not self.custom_decomposer:
            return []
        
        try:
            subqueries = self.custom_decomposer(query)
            if not isinstance(subqueries, list):
                logger.warning("Custom decomposer did not return a list. Using original query.")
                return []
                
            # Filter and validate subqueries
            valid_subqueries = []
            for subquery in subqueries:
                if (isinstance(subquery, str) and subquery.strip() and 
                    len(subquery.split()) >= self.min_subquery_length and
                    subquery.strip() not in valid_subqueries):
                    valid_subqueries.append(subquery.strip())
            
            return valid_subqueries[:self.max_subqueries]
        except Exception as e:
            logger.error(f"Error in custom query decomposition: {str(e)}")
            return []
    
    def process_query(self, query: str, **kwargs) -> List[str]:
        """
        Decompose a complex query into simpler subqueries.
        
        Args:
            query: The original query string
            **kwargs: Additional parameters that can override init params
                - decomposition_method: Override decomposition_method
                - decomposition_rules: Override decomposition_rules
                - merge_original: Override merge_original
                - max_subqueries: Override max_subqueries
                - min_subquery_length: Override min_subquery_length
        
        Returns:
            List[str]: List of subqueries, possibly including the original query
        """
        # Apply overrides from kwargs
        decomposition_method = kwargs.get('decomposition_method', self.decomposition_method)
        decomposition_rules = kwargs.get('decomposition_rules', self.decomposition_rules)
        merge_original = kwargs.get('merge_original', self.merge_original)
        max_subqueries = kwargs.get('max_subqueries', self.max_subqueries)
        min_subquery_length = kwargs.get('min_subquery_length', self.min_subquery_length)
        
        # Store original values to restore later
        orig_decomposition_method = self.decomposition_method
        orig_decomposition_rules = self.decomposition_rules
        orig_merge_original = self.merge_original
        orig_max_subqueries = self.max_subqueries
        orig_min_subquery_length = self.min_subquery_length
        
        # Apply temporary overrides
        self.decomposition_method = decomposition_method
        self.decomposition_rules = decomposition_rules
        self.merge_original = merge_original
        self.max_subqueries = max_subqueries
        self.min_subquery_length = min_subquery_length
        
        try:
            # Apply the selected decomposition method
            if decomposition_method == "rule":
                subqueries = self._rule_based_decomposition(query)
            elif decomposition_method == "llm":
                subqueries = self._llm_based_decomposition(query)
            elif decomposition_method == "custom":
                subqueries = self._custom_decomposition(query)
            else:
                subqueries = []
                
            # Include original query if requested and there are subqueries
            if subqueries and merge_original and query not in subqueries:
                # Put original query at the beginning
                subqueries = [query] + subqueries
                
            # If no subqueries were generated, return the original query
            if not subqueries:
                return [query]
                
            # Limit to max_subqueries
            return subqueries[:max_subqueries]
                
        finally:
            # Restore original values
            self.decomposition_method = orig_decomposition_method
            self.decomposition_rules = orig_decomposition_rules
            self.merge_original = orig_merge_original
            self.max_subqueries = orig_max_subqueries
            self.min_subquery_length = orig_min_subquery_length 