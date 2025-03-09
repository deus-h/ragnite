"""
Query Translator

This module provides the QueryTranslator class for translating queries between languages.
"""

import logging
from typing import List, Dict, Any, Optional, Union, Callable

from .base_processor import BaseQueryProcessor

# Configure logging
logger = logging.getLogger(__name__)


class QueryTranslator(BaseQueryProcessor):
    """
    Query translator for translating queries between languages.
    
    This class provides functionality to translate queries from one language to another,
    enabling cross-lingual retrieval capabilities.
    """
    
    def __init__(
        self,
        source_language: str = "auto",
        target_language: str = "en",
        translation_method: str = "library",
        custom_translator: Optional[Callable[[str, str, str], str]] = None,
        llm_service: Optional[Any] = None,
        llm_params: Optional[Dict[str, Any]] = None,
        library_name: str = "googletrans",
        detect_language: bool = True,
        preserve_format: bool = True,
        cache_translations: bool = False
    ):
        """
        Initialize a QueryTranslator.
        
        Args:
            source_language: Source language code (ISO 639-1) or 'auto' for automatic detection
            target_language: Target language code (ISO 639-1)
            translation_method: Method to use for translation
                - "library": Use a translation library (googletrans, deep_translator, etc.)
                - "llm": Use a language model for translation
                - "custom": Use a custom translation function
            custom_translator: Custom function for translation
            llm_service: LLM service for translation (for "llm" method)
            llm_params: Parameters for the LLM service
            library_name: Name of the translation library to use
                - "googletrans": Use googletrans library
                - "deep_translator": Use deep_translator library
            detect_language: Whether to detect the source language if set to 'auto'
            preserve_format: Whether to preserve capitalization, punctuation, etc.
            cache_translations: Whether to cache translations to avoid redundant API calls
        """
        self.source_language = source_language
        self.target_language = target_language
        self.translation_method = translation_method.lower()
        self.custom_translator = custom_translator
        self.llm_service = llm_service
        self.llm_params = llm_params or {}
        self.library_name = library_name.lower()
        self.detect_language = detect_language
        self.preserve_format = preserve_format
        self.cache_translations = cache_translations
        
        # Initialize translation cache if enabled
        self.translation_cache = {} if cache_translations else None
        
        # Validate translation method
        valid_methods = ["library", "llm", "custom"]
        if self.translation_method not in valid_methods:
            raise ValueError(f"Invalid translation_method: {translation_method}. Must be one of {valid_methods}")
        
        # Validate requirements for chosen method
        if self.translation_method == "custom" and not self.custom_translator:
            raise ValueError("custom_translator function must be provided when using 'custom' translation method")
        
        if self.translation_method == "llm" and not self.llm_service:
            raise ValueError("llm_service must be provided when using 'llm' translation method")
        
        # Initialize translation library if needed
        if self.translation_method == "library":
            self._initialize_translation_library()
    
    def _initialize_translation_library(self):
        """Initialize the translation library."""
        self.translator = None
        
        try:
            if self.library_name == "googletrans":
                try:
                    from googletrans import Translator
                    self.translator = Translator()
                except ImportError:
                    logger.warning("googletrans not installed. Please install with: pip install googletrans==4.0.0-rc1")
            
            elif self.library_name == "deep_translator":
                try:
                    import deep_translator
                    # We'll initialize specific translators when needed
                    self.translator = "deep_translator"
                except ImportError:
                    logger.warning("deep_translator not installed. Please install with: pip install deep_translator")
            
            else:
                supported_libraries = ["googletrans", "deep_translator"]
                logger.warning(f"Unsupported library: {self.library_name}. Supported libraries: {supported_libraries}")
                logger.warning("Falling back to googletrans if available.")
                
                try:
                    from googletrans import Translator
                    self.translator = Translator()
                    self.library_name = "googletrans"
                except ImportError:
                    logger.warning("googletrans not installed. Please install with: pip install googletrans==4.0.0-rc1")
        
        except Exception as e:
            logger.error(f"Error initializing translation library: {str(e)}")
    
    def _get_language_code(self, language: str) -> str:
        """Convert language name to ISO 639-1 code if needed."""
        # Simple mapping of common language names to codes
        language_map = {
            "english": "en",
            "spanish": "es",
            "french": "fr",
            "german": "de",
            "italian": "it",
            "portuguese": "pt",
            "dutch": "nl",
            "russian": "ru",
            "chinese": "zh-cn",
            "japanese": "ja",
            "korean": "ko",
            "arabic": "ar",
            "hindi": "hi",
            "bengali": "bn",
            "turkish": "tr",
            "auto": "auto"
        }
        
        # Check if it's already a code (assuming 2-5 chars)
        if len(language) >= 2 and len(language) <= 5:
            return language.lower()
        
        # Otherwise, look up in the map
        return language_map.get(language.lower(), "en")
    
    def _detect_language(self, query: str) -> str:
        """Detect the language of the query."""
        if not query.strip():
            return self.source_language if self.source_language != "auto" else "en"
        
        try:
            if self.library_name == "googletrans" and self.translator:
                detection = self.translator.detect(query)
                return detection.lang
            
            elif self.library_name == "deep_translator":
                try:
                    from deep_translator import GoogleTranslator
                    return GoogleTranslator().detect(query)
                except Exception as e:
                    logger.error(f"Error detecting language with deep_translator: {str(e)}")
                    return "en"
            
            else:
                logger.warning("Language detection not available. Using default.")
                return "en"
        
        except Exception as e:
            logger.error(f"Error detecting language: {str(e)}")
            return "en"
    
    def _preserve_case(self, original: str, translated: str) -> str:
        """Preserve the case pattern of the original string in the translated string."""
        if not self.preserve_format or not original or not translated:
            return translated
        
        # If original is all uppercase, make translated all uppercase
        if original.isupper():
            return translated.upper()
        
        # If original is title case, make translated title case
        if original[0].isupper() and not all(c.isupper() for c in original[1:]):
            words_translated = translated.split()
            if words_translated:
                words_translated[0] = words_translated[0].capitalize()
                return ' '.join(words_translated)
        
        return translated
    
    def _library_translation(self, query: str, source_lang: str, target_lang: str) -> str:
        """Translate query using a translation library."""
        if not query.strip():
            return query
        
        # Check cache first if enabled
        cache_key = f"{query}_{source_lang}_{target_lang}"
        if self.cache_translations and cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
        
        try:
            translated_text = query  # Default to original if translation fails
            
            if self.library_name == "googletrans" and self.translator:
                if source_lang == "auto":
                    translation = self.translator.translate(query, dest=target_lang)
                else:
                    translation = self.translator.translate(query, src=source_lang, dest=target_lang)
                
                translated_text = translation.text
            
            elif self.library_name == "deep_translator":
                try:
                    from deep_translator import GoogleTranslator
                    
                    if source_lang == "auto":
                        translator = GoogleTranslator(target=target_lang)
                    else:
                        translator = GoogleTranslator(source=source_lang, target=target_lang)
                    
                    translated_text = translator.translate(query)
                
                except Exception as e:
                    logger.error(f"Error translating with deep_translator: {str(e)}")
            
            else:
                logger.warning("No translation library available. Returning original query.")
            
            # Preserve case if enabled
            if self.preserve_format:
                translated_text = self._preserve_case(query, translated_text)
            
            # Cache the result if enabled
            if self.cache_translations:
                self.translation_cache[cache_key] = translated_text
            
            return translated_text
        
        except Exception as e:
            logger.error(f"Error in library translation: {str(e)}")
            return query
    
    def _llm_translation(self, query: str, source_lang: str, target_lang: str) -> str:
        """Translate query using a language model."""
        if not query.strip() or not self.llm_service:
            return query
        
        # Check cache first if enabled
        cache_key = f"{query}_{source_lang}_{target_lang}"
        if self.cache_translations and cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
        
        # Get language names for better prompting
        source_lang_name = source_lang
        target_lang_name = target_lang
        
        # Map common language codes to full names for better prompting
        language_names = {
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "nl": "Dutch",
            "ru": "Russian",
            "zh-cn": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "ar": "Arabic",
            "hi": "Hindi",
            "bn": "Bengali",
            "tr": "Turkish",
            "auto": "the original language"
        }
        
        if source_lang in language_names:
            source_lang_name = language_names[source_lang]
        
        if target_lang in language_names:
            target_lang_name = language_names[target_lang]
        
        # Prepare prompt for the LLM
        prompt = (
            f"Translate the following query from {source_lang_name} to {target_lang_name}. "
            f"Preserve the meaning and intent of the original query as accurately as possible.\n\n"
            f"Query: {query}\n\n"
            f"Translation:"
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
            translated_text = response_text.strip()
            
            # If the response is empty or too long, fall back to the original
            if not translated_text or len(translated_text) > len(query) * 5:
                return query
            
            # Preserve case if enabled
            if self.preserve_format:
                translated_text = self._preserve_case(query, translated_text)
            
            # Cache the result if enabled
            if self.cache_translations:
                self.translation_cache[cache_key] = translated_text
            
            return translated_text
            
        except Exception as e:
            logger.error(f"Error in LLM translation: {str(e)}")
            return query
    
    def _custom_translation(self, query: str, source_lang: str, target_lang: str) -> str:
        """Translate query using a custom function."""
        if not query.strip() or not self.custom_translator:
            return query
        
        # Check cache first if enabled
        cache_key = f"{query}_{source_lang}_{target_lang}"
        if self.cache_translations and cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
        
        try:
            translated_text = self.custom_translator(query, source_lang, target_lang)
            
            if not isinstance(translated_text, str):
                logger.warning("Custom translator did not return a string. Using original query.")
                return query
            
            # Preserve case if enabled
            if self.preserve_format:
                translated_text = self._preserve_case(query, translated_text)
            
            # Cache the result if enabled
            if self.cache_translations:
                self.translation_cache[cache_key] = translated_text
            
            return translated_text
            
        except Exception as e:
            logger.error(f"Error in custom translation: {str(e)}")
            return query
    
    def process_query(self, query: str, **kwargs) -> str:
        """
        Translate a query from source language to target language.
        
        Args:
            query: The original query string
            **kwargs: Additional parameters that can override init params
                - source_language: Override source_language
                - target_language: Override target_language
                - translation_method: Override translation_method
                - preserve_format: Override preserve_format
        
        Returns:
            str: Translated query
        """
        if not query.strip():
            return query
        
        # Apply overrides from kwargs
        source_language = kwargs.get('source_language', self.source_language)
        target_language = kwargs.get('target_language', self.target_language)
        translation_method = kwargs.get('translation_method', self.translation_method)
        preserve_format = kwargs.get('preserve_format', self.preserve_format)
        
        # Store original values to restore later
        orig_source_language = self.source_language
        orig_target_language = self.target_language
        orig_translation_method = self.translation_method
        orig_preserve_format = self.preserve_format
        
        # Apply temporary overrides
        self.source_language = source_language
        self.target_language = target_language
        self.translation_method = translation_method
        self.preserve_format = preserve_format
        
        try:
            # Convert language names to codes if needed
            source_lang = self._get_language_code(source_language)
            target_lang = self._get_language_code(target_language)
            
            # Detect language if source is 'auto' and detection is enabled
            if source_lang == "auto" and self.detect_language:
                source_lang = self._detect_language(query)
            
            # If source and target languages are the same, return the original query
            if source_lang == target_lang:
                return query
            
            # Apply the selected translation method
            if translation_method == "library":
                return self._library_translation(query, source_lang, target_lang)
            elif translation_method == "llm":
                return self._llm_translation(query, source_lang, target_lang)
            elif translation_method == "custom":
                return self._custom_translation(query, source_lang, target_lang)
            else:
                return query
                
        finally:
            # Restore original values
            self.source_language = orig_source_language
            self.target_language = orig_target_language
            self.translation_method = orig_translation_method
            self.preserve_format = orig_preserve_format 