#!/usr/bin/env python3
"""
Query Translator Example

This script demonstrates how to use the QueryTranslator to translate queries
between different languages.
"""

import sys
import os

# Add the 'tools' directory to the Python path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.retrieval import get_query_processor

def main():
    """Demonstrate query translation using various methods."""
    
    print("🔍 Query Translator Example\n")
    
    # Example 1: Library-based translation
    print("Example 1: Library-based Translation (googletrans)")
    
    # Create a library-based translator
    try:
        translator = get_query_processor(
            processor_type="translator",
            translation_method="library",
            library_name="googletrans",
            source_language="auto",
            target_language="en",
            preserve_format=True
        )
        
        # Test queries in different languages
        test_queries = {
            "English": "What is Retrieval Augmented Generation?",
            "Spanish": "¿Cómo implementar búsqueda vectorial en mi aplicación?",
            "French": "Quels sont les avantages de l'apprentissage automatique?",
            "German": "Wie kann ich die Leistung meiner Datenbank verbessern?",
            "Chinese": "如何使用向量数据库来改进搜索结果?"
        }
        
        for language, query in test_queries.items():
            translated_query = translator.process_query(query)
            print(f"\nOriginal ({language}): {query}")
            print(f"Translated (English): {translated_query}")
            
    except Exception as e:
        print(f"\nLibrary-based translation error: {str(e)}")
        print("You may need to install googletrans: pip install googletrans==4.0.0-rc1")
    
    # Example 2: Library-based translation with deep_translator
    print("\n\nExample 2: Library-based Translation (deep_translator)")
    
    try:
        deep_translator = get_query_processor(
            processor_type="translator",
            translation_method="library",
            library_name="deep_translator",
            source_language="auto",
            target_language="en"
        )
        
        # Test with the same queries
        for language, query in test_queries.items():
            translated_query = deep_translator.process_query(query)
            print(f"\nOriginal ({language}): {query}")
            print(f"Translated (English): {translated_query}")
            
    except Exception as e:
        print(f"\nDeep translator error: {str(e)}")
        print("You may need to install deep_translator: pip install deep_translator")
    
    # Example 3: Translate to a specific language
    print("\n\nExample 3: Translate to Different Target Languages")
    
    english_query = "How can I implement vector search in my application?"
    
    try:
        # Create translators for different target languages
        target_languages = {
            "Spanish": "es",
            "French": "fr",
            "German": "de",
            "Japanese": "ja"
        }
        
        for language_name, language_code in target_languages.items():
            target_translator = get_query_processor(
                processor_type="translator",
                translation_method="library", 
                source_language="en",
                target_language=language_code
            )
            
            translated_query = target_translator.process_query(english_query)
            print(f"\nOriginal (English): {english_query}")
            print(f"Translated ({language_name}): {translated_query}")
            
    except Exception as e:
        print(f"\nTranslation error: {str(e)}")
    
    # Example 4: Custom translation function
    print("\n\nExample 4: Custom Translation Function")
    
    # Define a simple custom translator function
    # This is a very basic example - a real implementation would use actual translation services
    def custom_translator(query: str, source_lang: str, target_lang: str) -> str:
        # This is just a mock implementation for demonstration
        # In a real scenario, you would call an actual translation API or service
        
        # Simple dictionary of pre-translated phrases for demo purposes
        translations = {
            "hello": {
                "es": "hola",
                "fr": "bonjour",
                "de": "hallo"
            },
            "how are you": {
                "es": "¿cómo estás?",
                "fr": "comment ça va?",
                "de": "wie geht es dir?"
            },
            "thank you": {
                "es": "gracias",
                "fr": "merci",
                "de": "danke"
            },
            "goodbye": {
                "es": "adiós",
                "fr": "au revoir",
                "de": "auf wiedersehen"
            },
            "vector search": {
                "es": "búsqueda vectorial",
                "fr": "recherche vectorielle",
                "de": "Vektorsuche"
            }
        }
        
        # Check if any known phrase is in the query
        query_lower = query.lower()
        for phrase, lang_dict in translations.items():
            if phrase in query_lower and target_lang in lang_dict:
                # Replace the phrase with its translation
                return query.replace(phrase, lang_dict[target_lang])
        
        # If no match, return the original (in a real implementation, you would use a full translation service)
        print(f"  (Custom translator doesn't know how to translate this query to {target_lang})")
        return query
    
    # Create a custom translator
    custom_processor = get_query_processor(
        processor_type="translator",
        translation_method="custom",
        custom_translator=custom_translator,
        source_language="en",
        target_language="es"  # Default target is Spanish
    )
    
    # Test with a few queries
    custom_test_queries = [
        "Hello, how are you?",
        "I need information about vector search",
        "Thank you for your help, goodbye!",
        "This query won't be translated by our simple custom translator"
    ]
    
    for query in custom_test_queries:
        translated_query = custom_processor.process_query(query)
        print(f"\nOriginal: {query}")
        print(f"Translated: {translated_query}")
    
    # Example 5: Changing target language dynamically
    print("\n\nExample 5: Changing Target Language Dynamically")
    
    # Create a translator with default settings
    dynamic_translator = get_query_processor(
        processor_type="translator",
        translation_method="custom",  # Using our custom function for simplicity
        custom_translator=custom_translator,
        source_language="en",
        target_language="es"  # Default target is Spanish
    )
    
    query = "Thank you for your help with vector search"
    
    # Translate to different languages using kwargs to override settings
    for language_name, language_code in {
        "Spanish": "es", 
        "French": "fr", 
        "German": "de"
    }.items():
        translated_query = dynamic_translator.process_query(
            query, 
            target_language=language_code
        )
        print(f"\nOriginal: {query}")
        print(f"Translated to {language_name}: {translated_query}")
        
    print("\n🔮 Query translation complete!")


if __name__ == "__main__":
    main() 