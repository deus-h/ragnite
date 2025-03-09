"""
Query Rewriter

This module provides the QueryRewriter class for rewriting queries to improve retrieval.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Union, Callable

from .base_processor import BaseQueryProcessor

# Configure logging
logger = logging.getLogger(__name__)


class QueryRewriter(BaseQueryProcessor):
    """
    Query rewriter for transforming queries to improve retrieval performance.
    
    This class transforms queries by applying various rewriting techniques such as
    removing noise, fixing grammar, adding contextual information, or using templates.
    """
    
    def __init__(
        self,
        rewrite_method: str = "basic",
        templates: Optional[List[str]] = None,
        custom_rewriter: Optional[Callable[[str], str]] = None,
        remove_stopwords: bool = False,
        fix_spelling: bool = False,
        add_prefix: Optional[str] = None,
        add_suffix: Optional[str] = None,
        llm_service: Optional[Any] = None,
        llm_params: Optional[Dict[str, Any]] = None,
        language: str = "english"
    ):
        """
        Initialize a QueryRewriter.
        
        Args:
            rewrite_method: Method to use for query rewriting
                - "basic": Apply basic text processing (remove noise, fix casing)
                - "template": Apply query templates
                - "llm": Use a language model for rewriting
                - "custom": Use a custom rewriting function
            templates: List of templates for "template" method (e.g., ["what is {}", "how to {}"])
            custom_rewriter: Custom function for query rewriting
            remove_stopwords: Whether to remove stopwords
            fix_spelling: Whether to attempt to fix spelling
            add_prefix: Optional prefix to add to the query
            add_suffix: Optional suffix to add to the query
            llm_service: LLM service for generating rewrites (for "llm" method)
            llm_params: Parameters for the LLM service
            language: Language for stopwords (if remove_stopwords is True)
        """
        self.rewrite_method = rewrite_method.lower()
        self.templates = templates or []
        self.custom_rewriter = custom_rewriter
        self.remove_stopwords = remove_stopwords
        self.fix_spelling = fix_spelling
        self.add_prefix = add_prefix
        self.add_suffix = add_suffix
        self.llm_service = llm_service
        self.llm_params = llm_params or {}
        self.language = language
        self.stopwords = set()
        self.spell_checker = None
        
        # Initialize components based on selected options
        if self.remove_stopwords:
            self._initialize_stopwords()
            
        if self.fix_spelling:
            self._initialize_spell_checker()
        
        # Validate rewrite method
        valid_methods = ["basic", "template", "llm", "custom"]
        if self.rewrite_method not in valid_methods:
            raise ValueError(f"Invalid rewrite_method: {rewrite_method}. Must be one of {valid_methods}")
        
        # Validate requirements for chosen method
        if self.rewrite_method == "custom" and not self.custom_rewriter:
            raise ValueError("custom_rewriter function must be provided when using 'custom' rewrite method")
        
        if self.rewrite_method == "template" and not self.templates:
            raise ValueError("templates must be provided when using 'template' rewrite method")
            
        if self.rewrite_method == "llm" and not self.llm_service:
            raise ValueError("llm_service must be provided when using 'llm' rewrite method")
    
    def _initialize_stopwords(self):
        """Initialize stopwords for the specified language."""
        try:
            from nltk.corpus import stopwords
            import nltk
            
            try:
                self.stopwords = set(stopwords.words(self.language))
            except LookupError:
                logger.info(f"Downloading stopwords for {self.language}")
                nltk.download('stopwords', quiet=True)
                self.stopwords = set(stopwords.words(self.language))
        except ImportError:
            logger.warning("NLTK not installed. Stopword removal disabled.")
            self.remove_stopwords = False
    
    def _initialize_spell_checker(self):
        """Initialize spell checker."""
        try:
            from spellchecker import SpellChecker
            self.spell_checker = SpellChecker(language=self.language)
        except ImportError:
            logger.warning("pyspellchecker not installed. Spell checking disabled.")
            self.fix_spelling = False
    
    def _basic_rewrite(self, query: str) -> str:
        """Apply basic rewriting techniques."""
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query.strip())
        
        # Remove stopwords if enabled
        if self.remove_stopwords and self.stopwords:
            words = query.split()
            query = ' '.join([word for word in words if word.lower() not in self.stopwords])
        
        # Fix spelling if enabled
        if self.fix_spelling and self.spell_checker:
            words = query.split()
            corrected_words = []
            
            for word in words:
                # Skip proper nouns (assumed to start with uppercase)
                if word and word[0].isupper():
                    corrected_words.append(word)
                else:
                    # Check and correct spelling
                    corrected_word = self.spell_checker.correction(word)
                    corrected_words.append(corrected_word if corrected_word else word)
            
            query = ' '.join(corrected_words)
        
        # Add prefix/suffix if provided
        if self.add_prefix:
            query = f"{self.add_prefix} {query}"
        
        if self.add_suffix:
            query = f"{query} {self.add_suffix}"
        
        return query
    
    def _template_rewrite(self, query: str) -> str:
        """Apply template-based rewriting."""
        # Select the best template based on query structure
        # For simplicity, just use the first template if no selection logic is implemented
        if not self.templates:
            return query
            
        # Basic template selection - choose based on query type
        query = query.strip()
        
        # Check if query already has a question structure
        if query.endswith('?'):
            # Already a question, use as is or match to specific question template
            for template in self.templates:
                if '{' in template and '}' in template:
                    # Simple check if template is for questions
                    if template.endswith('{}?'):
                        # Remove the question mark from the original query
                        cleaned_query = query[:-1]
                        return template.format(cleaned_query)
            
            # No matching question template found, return original
            return query
        
        # If it's a short, keyword-like query
        if len(query.split()) <= 3:
            for template in self.templates:
                if '{' in template and '}' in template:
                    # Look for definition or explanation templates for short queries
                    if "what is" in template.lower() or "define" in template.lower():
                        return template.format(query)
            
            # Default to first template if no specific match
            if self.templates:
                return self.templates[0].format(query)
        
        # For longer queries that are not questions
        for template in self.templates:
            if '{' in template and '}' in template:
                # Look for "how to" templates for action-oriented queries
                if "how to" in template.lower() and ("how" in query.lower() or 
                                                   any(word in query.lower() for word in ["create", "make", "build", "do", "use"])):
                    return template.format(query)
        
        # Default to first template if no specific match
        if self.templates:
            return self.templates[0].format(query)
        
        return query
    
    def _llm_rewrite(self, query: str) -> str:
        """Rewrite query using a language model."""
        if not self.llm_service:
            return query
        
        # Prepare prompt for the LLM
        prompt = (
            f"Rewrite the following search query to make it more effective for retrieval: \n\n"
            f"Original query: {query}\n\n"
            f"Rewritten query:"
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
            
            # Clean up the response
            rewritten_query = response_text.strip()
            
            # If the rewritten query is empty or too long, fall back to the original
            if not rewritten_query or len(rewritten_query) > len(query) * 3:
                return query
                
            return rewritten_query
            
        except Exception as e:
            logger.error(f"Error rewriting query with LLM: {str(e)}")
            return query
    
    def _custom_rewrite(self, query: str) -> str:
        """Rewrite query using a custom function."""
        if not self.custom_rewriter:
            return query
        
        try:
            rewritten_query = self.custom_rewriter(query)
            if not isinstance(rewritten_query, str):
                logger.warning("Custom rewriter did not return a string. Using original query.")
                return query
            
            return rewritten_query
        except Exception as e:
            logger.error(f"Error in custom query rewriting: {str(e)}")
            return query
    
    def process_query(self, query: str, **kwargs) -> str:
        """
        Rewrite a query to improve retrieval performance.
        
        Args:
            query: The original query string
            **kwargs: Additional parameters that can override init params
                - rewrite_method: Override rewrite_method
                - templates: Override templates
                - remove_stopwords: Override remove_stopwords
                - fix_spelling: Override fix_spelling
                - add_prefix: Override add_prefix
                - add_suffix: Override add_suffix
        
        Returns:
            str: Rewritten query
        """
        # Apply overrides from kwargs
        rewrite_method = kwargs.get('rewrite_method', self.rewrite_method)
        templates = kwargs.get('templates', self.templates)
        remove_stopwords = kwargs.get('remove_stopwords', self.remove_stopwords)
        fix_spelling = kwargs.get('fix_spelling', self.fix_spelling)
        add_prefix = kwargs.get('add_prefix', self.add_prefix)
        add_suffix = kwargs.get('add_suffix', self.add_suffix)
        
        # Store original values to restore later
        orig_rewrite_method = self.rewrite_method
        orig_templates = self.templates
        orig_remove_stopwords = self.remove_stopwords
        orig_fix_spelling = self.fix_spelling
        orig_add_prefix = self.add_prefix
        orig_add_suffix = self.add_suffix
        
        # Apply temporary overrides
        self.rewrite_method = rewrite_method
        self.templates = templates if templates is not None else self.templates
        self.remove_stopwords = remove_stopwords
        self.fix_spelling = fix_spelling
        self.add_prefix = add_prefix
        self.add_suffix = add_suffix
        
        try:
            # Apply the selected rewriting method
            if rewrite_method == "basic":
                return self._basic_rewrite(query)
            elif rewrite_method == "template":
                return self._template_rewrite(query)
            elif rewrite_method == "llm":
                return self._llm_rewrite(query)
            elif rewrite_method == "custom":
                return self._custom_rewrite(query)
            else:
                return query
                
        finally:
            # Restore original values
            self.rewrite_method = orig_rewrite_method
            self.templates = orig_templates
            self.remove_stopwords = orig_remove_stopwords
            self.fix_spelling = orig_fix_spelling
            self.add_prefix = orig_add_prefix
            self.add_suffix = orig_add_suffix 