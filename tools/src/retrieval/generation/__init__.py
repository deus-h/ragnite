"""
Generation tools for Retrieval-Augmented Generation systems.

This module includes tools for generating content based on retrieved information,
including prompt templates, context formatters, and hallucination detectors.
"""

# Import prompt templates
from .prompt_templates.base_prompt_template import BasePromptTemplate
from .prompt_templates.basic_prompt_template import BasicPromptTemplate
from .prompt_templates.few_shot_prompt_template import FewShotPromptTemplate
from .prompt_templates.chain_of_thought_prompt_template import ChainOfThoughtPromptTemplate
from .prompt_templates.structured_prompt_template import StructuredPromptTemplate
from .prompt_templates.factory import get_prompt_template

# Import context formatters
from .context_formatters.base_context_formatter import BaseContextFormatter
from .context_formatters.basic_context_formatter import BasicContextFormatter
from .context_formatters.metadata_enriched_formatter import MetadataEnrichedFormatter
from .context_formatters.source_attribution_formatter import SourceAttributionFormatter
from .context_formatters.hierarchical_context_formatter import HierarchicalContextFormatter
from .context_formatters.factory import get_context_formatter

# Import output parsers
from .output_parsers.base_output_parser import BaseOutputParser
from .output_parsers.json_output_parser import JSONOutputParser
from .output_parsers.xml_output_parser import XMLOutputParser
from .output_parsers.markdown_output_parser import MarkdownOutputParser
from .output_parsers.factory import get_output_parser

# Import hallucination detectors
from .hallucination_detectors.base_hallucination_detector import BaseHallucinationDetector
from .hallucination_detectors.hallucination_detection_result import HallucinationDetectionResult
from .hallucination_detectors.factual_consistency_detector import FactualConsistencyDetector
from .hallucination_detectors.source_verification_detector import SourceVerificationDetector
from .hallucination_detectors.contradiction_detector import ContradictionDetector
from .hallucination_detectors.uncertainty_detector import UncertaintyDetector
from .hallucination_detectors.factory import get_hallucination_detector

# Import citation generators
from .citation_generators.base_citation_generator import BaseCitationGenerator
from .citation_generators.academic_citation_generator import AcademicCitationGenerator
from .citation_generators.legal_citation_generator import LegalCitationGenerator
from .citation_generators.web_citation_generator import WebCitationGenerator
from .citation_generators.custom_citation_generator import CustomCitationGenerator
from .citation_generators.factory import get_citation_generator

__all__ = [
    # Prompt templates
    'BasePromptTemplate',
    'BasicPromptTemplate',
    'FewShotPromptTemplate',
    'ChainOfThoughtPromptTemplate',
    'StructuredPromptTemplate',
    'get_prompt_template',
    
    # Context formatters
    'BaseContextFormatter',
    'BasicContextFormatter',
    'MetadataEnrichedFormatter',
    'SourceAttributionFormatter',
    'HierarchicalContextFormatter',
    'get_context_formatter',
    
    # Output parsers
    'BaseOutputParser',
    'JSONOutputParser',
    'XMLOutputParser',
    'MarkdownOutputParser',
    'get_output_parser',
    
    # Hallucination detectors
    'BaseHallucinationDetector',
    'HallucinationDetectionResult',
    'FactualConsistencyDetector',
    'SourceVerificationDetector',
    'ContradictionDetector',
    'UncertaintyDetector',
    'get_hallucination_detector',
    
    # Citation generators
    'BaseCitationGenerator',
    'AcademicCitationGenerator',
    'LegalCitationGenerator',
    'WebCitationGenerator',
    'CustomCitationGenerator',
    'get_citation_generator',
] 