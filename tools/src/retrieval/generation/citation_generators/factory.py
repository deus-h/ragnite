"""
Citation Generator Factory

This module provides a factory function for creating citation generator instances
based on type and configuration.
"""

from typing import Any, Dict, Optional, Union, Type
from .base_citation_generator import BaseCitationGenerator
from .academic_citation_generator import AcademicCitationGenerator
from .legal_citation_generator import LegalCitationGenerator
from .web_citation_generator import WebCitationGenerator
from .custom_citation_generator import CustomCitationGenerator

def get_citation_generator(
    generator_type: str,
    config: Optional[Dict[str, Any]] = None
) -> BaseCitationGenerator:
    """
    Factory function to create a citation generator of the specified type.
    
    Args:
        generator_type: Type of citation generator to create.
            Valid values are 'academic', 'legal', 'web', and 'custom'.
        config: Optional configuration dictionary for the generator.
            
    Returns:
        A citation generator instance of the requested type.
        
    Raises:
        ValueError: If the specified generator type is not supported.
    """
    generator_classes: Dict[str, Type[BaseCitationGenerator]] = {
        'academic': AcademicCitationGenerator,
        'legal': LegalCitationGenerator,
        'web': WebCitationGenerator,
        'custom': CustomCitationGenerator
    }
    
    # Normalize type for case-insensitive matching
    normalized_type = generator_type.lower()
    
    if normalized_type not in generator_classes:
        valid_types = ', '.join(generator_classes.keys())
        raise ValueError(
            f"Unsupported citation generator type: '{generator_type}'. "
            f"Valid types are: {valid_types}"
        )
    
    # Create and return the generator instance
    return generator_classes[normalized_type](config) 