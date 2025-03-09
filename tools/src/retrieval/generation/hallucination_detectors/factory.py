"""
Hallucination Detector Factory

This module provides a factory function for creating hallucination detector instances.
"""

from typing import Any, Dict, List, Optional, Union

from .base_hallucination_detector import BaseHallucinationDetector
from .factual_consistency_detector import FactualConsistencyDetector
from .source_verification_detector import SourceVerificationDetector
from .contradiction_detector import ContradictionDetector
from .uncertainty_detector import UncertaintyDetector


def get_hallucination_detector(
    detector_type: str,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> BaseHallucinationDetector:
    """
    Factory function to create a hallucination detector instance based on the specified type.
    
    Args:
        detector_type: Type of detector to create. Options:
            - "factual_consistency": Detector for factual inconsistencies
            - "source_verification": Detector for verifying against sources
            - "contradiction": Detector for internal contradictions
            - "uncertainty": Detector for uncertain statements
        config: Optional configuration dictionary for the detector
        **kwargs: Additional keyword arguments to include in the config
        
    Returns:
        An instance of the specified hallucination detector
        
    Raises:
        ValueError: If an unsupported detector type is specified
    """
    # Combine config and kwargs
    if config is None:
        config = {}
    
    combined_config = {**config, **kwargs}
    
    # Create the appropriate detector based on type
    if detector_type.lower() == "factual_consistency":
        return FactualConsistencyDetector(combined_config)
    elif detector_type.lower() == "source_verification":
        return SourceVerificationDetector(combined_config)
    elif detector_type.lower() == "contradiction":
        return ContradictionDetector(combined_config)
    elif detector_type.lower() == "uncertainty":
        return UncertaintyDetector(combined_config)
    else:
        raise ValueError(f"Unsupported hallucination detector type: {detector_type}. "
                        f"Supported types are: factual_consistency, source_verification, "
                        f"contradiction, uncertainty.") 