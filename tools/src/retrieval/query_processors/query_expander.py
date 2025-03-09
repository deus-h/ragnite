"""
Query Expander

This module provides the QueryExpander class for expanding queries with synonyms and related terms.
"""

import logging
from typing import List, Dict, Any, Optional, Union, Callable
import re

from .base_processor import BaseQueryProcessor

# Configure logging
logger = logging.getLogger(__name__)


class QueryExpander(BaseQueryProcessor):
    """
    Query expander for expanding queries with synonyms and related terms.
    
    This class expands the original query with synonyms, related terms, or alternative phrasings
    to improve recall in retrieval systems.
    """
    
    def __init__(
        self,
        expansion_method: str = "synonym",
        max_expansions: int = 3,
        custom_expander: Optional[Callable[[str], List[str]]] = None,
        thesaurus: Optional[Dict[str, List[str]]] = None,
        llm_service: Optional[Any] = None,
        llm_params: Optional[Dict[str, Any]] = None,
        min_token_length: int = 4,
        exclude_stopwords: bool = True,
        language: str = "english"
    ):
        """
        Initialize a QueryExpander.
        
        Args:
            expansion_method: Method to use for query expansion
                - "synonym": Use synonyms from thesaurus or WordNet
                - "wordnet": Use WordNet for expansions
                - "llm": Use a language model for expansions
                - "custom": Use a custom expansion function
            max_expansions: Maximum number of expanded queries to generate
            custom_expander: Custom function for query expansion
            thesaurus: Dictionary mapping terms to their synonyms/related terms
            llm_service: LLM service for generating expansions (for "llm" method)
            llm_params: Parameters for the LLM service
            min_token_length: Minimum length of tokens to expand
            exclude_stopwords: Whether to exclude stopwords from expansion
            language: Language for stopwords (if exclude_stopwords is True)
        """
        self.expansion_method = expansion_method.lower()
        self.max_expansions = max_expansions
        self.custom_expander = custom_expander
        self.thesaurus = thesaurus or {}
        self.llm_service = llm_service
        self.llm_params = llm_params or {}
        self.min_token_length = min_token_length
        self.exclude_stopwords = exclude_stopwords
        self.language = language
        self.stopwords = set()
        
        # Initialize stopwords if needed
        if self.exclude_stopwords:
            self._initialize_stopwords()
        
        # Validate expansion method
        valid_methods = ["synonym", "wordnet", "llm", "custom"]
        if self.expansion_method not in valid_methods:
            raise ValueError(f"Invalid expansion_method: {expansion_method}. Must be one of {valid_methods}")
        
        # Validate requirements for chosen method
        if self.expansion_method == "custom" and not self.custom_expander:
            raise ValueError("custom_expander function must be provided when using 'custom' expansion method")
        
        if self.expansion_method == "llm" and not self.llm_service:
            raise ValueError("llm_service must be provided when using 'llm' expansion method")
        
        # Initialize WordNet if needed
        if self.expansion_method == "wordnet":
            self._initialize_wordnet()
    
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
            logger.warning("NLTK not installed. Stopword filtering disabled.")
            self.exclude_stopwords = False
    
    def _initialize_wordnet(self):
        """Initialize WordNet for synonym expansion."""
        try:
            from nltk.corpus import wordnet
            import nltk
            
            try:
                # Test if WordNet is available
                wordnet.synsets('test')
            except LookupError:
                logger.info("Downloading WordNet")
                nltk.download('wordnet', quiet=True)
                nltk.download('omw-1.4', quiet=True)
        except ImportError:
            logger.warning("NLTK not installed. Falling back to thesaurus-based expansion.")
            self.expansion_method = "synonym"
    
    def _get_wordnet_synonyms(self, word: str) -> List[str]:
        """Get synonyms for a word using WordNet."""
        from nltk.corpus import wordnet
        
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym != word and len(synonym) >= self.min_token_length:
                    synonyms.add(synonym)
        
        return list(synonyms)[:self.max_expansions]
    
    def _expand_with_llm(self, query: str) -> List[str]:
        """Expand query using a language model."""
        if not self.llm_service:
            return [query]
        
        # Prepare prompt for the LLM
        prompt = (
            f"Generate {self.max_expansions} alternative versions of the following query "
            f"to improve search results. Keep the meaning the same but use different words "
            f"and phrasings.\n\nOriginal query: {query}\n\nAlternative queries:"
        )
        
        try:
            # Call the LLM service
            response = self.llm_service.generate(
                prompt=prompt,
                **self.llm_params
            )
            
            # Parse the response
            expansions = []
            if hasattr(response, 'text'):
                response_text = response.text
            elif isinstance(response, str):
                response_text = response
            else:
                response_text = str(response)
            
            # Extract queries from the response
            lines = response_text.strip().split('\n')
            for line in lines:
                # Remove numbering and other formatting
                clean_line = re.sub(r'^\d+[.)\s]+', '', line).strip()
                if clean_line and clean_line != query and len(clean_line) > 0:
                    expansions.append(clean_line)
            
            return expansions[:self.max_expansions]
        except Exception as e:
            logger.error(f"Error expanding query with LLM: {str(e)}")
            return [query]
    
    def _expand_with_custom(self, query: str) -> List[str]:
        """Expand query using a custom function."""
        if not self.custom_expander:
            return [query]
        
        try:
            expansions = self.custom_expander(query)
            if not isinstance(expansions, list):
                logger.warning("Custom expander did not return a list. Using original query.")
                return [query]
            
            return expansions[:self.max_expansions]
        except Exception as e:
            logger.error(f"Error in custom query expansion: {str(e)}")
            return [query]
    
    def _expand_with_thesaurus(self, query: str) -> List[str]:
        """Expand query using the thesaurus."""
        tokens = re.findall(r'\b\w+\b', query.lower())
        expansions = [query]
        
        for token in tokens:
            # Skip short tokens and stopwords
            if len(token) < self.min_token_length or (self.exclude_stopwords and token in self.stopwords):
                continue
            
            # Find synonyms in thesaurus
            synonyms = self.thesaurus.get(token, [])
            
            # Create new queries by replacing the token with its synonyms
            for synonym in synonyms[:self.max_expansions]:
                # Replace whole word only
                new_query = re.sub(r'\b' + token + r'\b', synonym, query, flags=re.IGNORECASE)
                if new_query != query and new_query not in expansions:
                    expansions.append(new_query)
                
                if len(expansions) >= self.max_expansions + 1:  # +1 for original query
                    break
        
        return expansions
    
    def process_query(self, query: str, **kwargs) -> List[str]:
        """
        Expand a query with synonyms and related terms.
        
        Args:
            query: The original query string
            **kwargs: Additional parameters that can override init params
                - max_expansions: Override max_expansions
                - expansion_method: Override expansion_method
                - thesaurus: Override or supplement thesaurus
        
        Returns:
            List[str]: List of expanded queries, including the original
        """
        # Apply overrides from kwargs
        max_expansions = kwargs.get('max_expansions', self.max_expansions)
        expansion_method = kwargs.get('expansion_method', self.expansion_method)
        thesaurus_update = kwargs.get('thesaurus', {})
        
        if thesaurus_update:
            # Create a temporary updated thesaurus
            combined_thesaurus = dict(self.thesaurus)
            combined_thesaurus.update(thesaurus_update)
        else:
            combined_thesaurus = self.thesaurus
        
        # Store original values to restore later
        orig_max_expansions = self.max_expansions
        orig_expansion_method = self.expansion_method
        orig_thesaurus = self.thesaurus
        
        # Apply temporary overrides
        self.max_expansions = max_expansions
        self.expansion_method = expansion_method
        self.thesaurus = combined_thesaurus
        
        try:
            if expansion_method == "synonym":
                expansions = self._expand_with_thesaurus(query)
            elif expansion_method == "wordnet":
                # Get WordNet synonyms for each meaningful token
                tokens = re.findall(r'\b\w+\b', query.lower())
                expansions = [query]
                
                for token in tokens:
                    if len(token) < self.min_token_length or (self.exclude_stopwords and token in self.stopwords):
                        continue
                    
                    synonyms = self._get_wordnet_synonyms(token)
                    for synonym in synonyms:
                        # Replace whole word only
                        new_query = re.sub(r'\b' + token + r'\b', synonym, query, flags=re.IGNORECASE)
                        if new_query != query and new_query not in expansions:
                            expansions.append(new_query)
                        
                        if len(expansions) >= max_expansions + 1:  # +1 for original query
                            break
            elif expansion_method == "llm":
                expansions = self._expand_with_llm(query)
                if query not in expansions:
                    expansions = [query] + expansions
            elif expansion_method == "custom":
                expansions = self._expand_with_custom(query)
                if query not in expansions:
                    expansions = [query] + expansions
            else:
                expansions = [query]
            
            # Ensure we have at least the original query
            if not expansions:
                expansions = [query]
                
            # Limit to max_expansions
            return expansions[:max_expansions + 1]  # +1 for original query
        
        finally:
            # Restore original values
            self.max_expansions = orig_max_expansions
            self.expansion_method = orig_expansion_method
            self.thesaurus = orig_thesaurus 