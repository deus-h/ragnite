#!/usr/bin/env python3
"""
Hallucination Detectors Example

This script demonstrates the use of various hallucination detectors
for identifying potential false or unsupported information in LLM outputs.
"""

import json
import time
from typing import Any, Dict, List, Callable, Optional

# Import the hallucination detectors
from tools.src.retrieval import (
    FactualConsistencyDetector,
    SourceVerificationDetector,
    ContradictionDetector,
    UncertaintyDetector,
    get_hallucination_detector,
    HallucinationDetectionResult
)

# Sample texts for demonstration
SAMPLE_TEXTS = {
    "factual_errors": """
The Eiffel Tower was built in 1789 as a monument to the French Revolution.
It stands 150 meters tall and is made primarily of wood with a steel frame.
The tower was designed by Gustav Eiffel, who was also famous for designing
the Statue of Liberty in New York City.
    """,
    
    "contradictions": """
The solar system has eight planets. Mercury is the closest planet to the sun,
while Neptune is the farthest. Actually, there are nine planets in our solar
system, with Pluto being the smallest. Mars is the fourth planet from the sun
and has two moons. It is the only planet without any moons in our solar system.
    """,
    
    "uncertain": """
Einstein might have developed his theory of relativity around 1905, though I'm
not entirely sure about the exact date. It seems that his work on general relativity
came somewhat later, perhaps around 1915 or so. I believe he received the Nobel Prize
for his work on relativity, but I could be mistaken about that detail.
    """,
    
    "unsupported": """
According to the latest research, drinking coffee improves cognitive function by
37% and reduces the risk of Alzheimer's disease by 42%. The study conducted at
Harvard University in 2022 showed that individuals who consume 3-4 cups of coffee
daily have significantly better memory retention than non-coffee drinkers.
    """,
    
    "clean": """
The Pacific Ocean is the largest and deepest ocean on Earth, covering more than
60 million square miles and containing more than half of the free water on Earth.
It stretches from the Arctic in the north to the Antarctic in the south and is
bounded by Asia and Australia in the west and the Americas in the east.
    """
}

# Sample source documents for source verification
SAMPLE_SOURCES = [
    {
        "content": """
The Pacific Ocean is the largest and deepest ocean on Earth. It extends from the
Arctic Ocean in the north to the Southern Ocean in the south and is bounded by Asia
and Australia in the west and the Americas in the east. The Pacific covers 63.8 million
square miles (165.25 million square kilometers).
        """,
        "metadata": {
            "source": "Geography Textbook",
            "chapter": "Oceans of the World"
        }
    },
    {
        "content": """
Coffee is one of the most popular beverages worldwide. While some studies suggest
potential health benefits associated with moderate coffee consumption, the exact
effects on cognitive function and disease prevention remain areas of active research.
Current evidence does not support precise percentage improvements in cognitive function
or disease risk reduction from coffee consumption.
        """,
        "metadata": {
            "source": "Medical Journal",
            "title": "Effects of Caffeine on Human Health"
        }
    }
]

# Mock knowledge base for factual consistency checking
KNOWLEDGE_BASE = {
    "general": {
        "The Eiffel Tower was built in 1889": {
            "negation": "The Eiffel Tower was built in 1789",
            "confidence": 0.95,
            "severity": "high"
        },
        "The Eiffel Tower is 330 meters tall": {
            "negation": "The Eiffel Tower is 150 meters tall",
            "confidence": 0.9,
            "severity": "medium"
        },
        "The Eiffel Tower is made of wrought iron": {
            "negation": "The Eiffel Tower is made of wood",
            "confidence": 0.95,
            "severity": "high"
        },
        "Frederic Auguste Bartholdi designed the Statue of Liberty": {
            "negation": "Gustav Eiffel designed the Statue of Liberty",
            "confidence": 0.85,
            "severity": "medium"
        }
    }
}

# Mock LLM provider for verification
def mock_llm_provider(prompt: str) -> str:
    """
    Simulates an LLM response based on the prompt.
    In a real implementation, this would call an LLM API.
    
    Args:
        prompt: The prompt to send to the LLM
        
    Returns:
        A simulated LLM response
    """
    # Simple keyword-based response generation
    if "contradiction" in prompt.lower():
        if "sun" in prompt.lower() and "planet" in prompt.lower():
            return """
            Yes, these statements contradict each other. 
            
            The text first states there are eight planets, then later says there are nine.
            It also states Mars has two moons but then says it has no moons.
            
            Confidence: 0.9
            Reason: Clear numerical contradictions about planet count and Mars' moons
            """
    
    elif "verify" in prompt.lower() and "source" in prompt.lower():
        if "pacific" in prompt.lower() and "ocean" in prompt.lower():
            return """
            True, confidence: 0.85, reason: The statement about the Pacific Ocean being the
            largest ocean is directly supported by the source document, which states
            "The Pacific Ocean is the largest and deepest ocean on Earth."
            """
        elif "coffee" in prompt.lower() and "cognitive" in prompt.lower():
            return """
            False, confidence: 0.8, reason: The specific claims about coffee improving
            cognitive function by 37% and reducing Alzheimer's risk by 42% are not
            supported by the source document, which states that "the exact effects on
            cognitive function and disease prevention remain areas of active research."
            """
    
    elif "fact-check" in prompt.lower():
        if "eiffel tower" in prompt.lower() and "1789" in prompt.lower():
            return """
            {
              "is_hallucination": true,
              "confidence": 0.95,
              "reason": "The Eiffel Tower was completed in 1889, not 1789",
              "severity": "high"
            }
            """
    
    # Default response for anything else
    return """
    I cannot determine with confidence whether this statement is accurate,
    contradictory, or supported by the sources.
    
    Confidence: 0.5
    Reason: Insufficient information to make a determination
    """

# Helper function to print detection results
def print_detection_results(detector_name: str, results: Dict[str, Any]) -> None:
    """
    Print the hallucination detection results in a readable format.
    
    Args:
        detector_name: Name of the detector
        results: Detection results dictionary
    """
    print(f"\n=== {detector_name} Results ===")
    
    # Convert to HallucinationDetectionResult if it's a dict
    if isinstance(results, dict):
        result_obj = HallucinationDetectionResult.from_dict(results)
    else:
        result_obj = results
    
    print(f"Overall Score: {result_obj.score:.2f} (1.0 is best, 0.0 is worst)")
    print(f"Confidence: {result_obj.confidence:.2f}")
    print(f"Detection Type: {result_obj.detection_type}")
    
    if result_obj.has_hallucinations():
        print(f"Detected {len(result_obj.detected_hallucinations)} potential issues:")
        for i, h in enumerate(result_obj.detected_hallucinations, 1):
            print(f"  {i}. \"{h['text'][:70]}{'...' if len(h['text']) > 70 else ''}\"")
            print(f"     Reason: {h['reason']}")
            print(f"     Severity: {h['severity']}, Confidence: {h['confidence']:.2f}")
            if 'markers' in h:
                marker_str = ', '.join(h['markers'][:3])
                if len(h['markers']) > 3:
                    marker_str += f" and {len(h['markers']) - 3} more"
                print(f"     Markers: {marker_str}")
            print()
    else:
        print("No hallucinations detected.")
    
    print(f"Explanation: {result_obj.explanation}")
    print("="*50)

def run_factual_consistency_example():
    """Demonstrate using the FactualConsistencyDetector."""
    print("\n" + "="*50)
    print("Factual Consistency Detector Example")
    print("="*50)
    
    # Create a detector with knowledge base method
    detector = FactualConsistencyDetector({
        "method": "knowledge_base",
        "knowledge_base": KNOWLEDGE_BASE
    })
    
    # Generate format information
    print("Detection Type:", detector.get_detection_type())
    print()
    
    # Detect hallucinations in the sample text
    try:
        start_time = time.time()
        results = detector.detect(SAMPLE_TEXTS["factual_errors"])
        detection_time = time.time() - start_time
        
        print(f"Detection completed in {detection_time:.4f} seconds")
        print_detection_results("Factual Consistency", results)
    except Exception as e:
        print(f"Error in factual consistency detection: {e}")
    
    # Create a detector with LLM method
    llm_detector = FactualConsistencyDetector({
        "method": "llm",
        "llm_provider": mock_llm_provider
    })
    
    # Detect hallucinations using LLM
    try:
        results = llm_detector.detect(SAMPLE_TEXTS["factual_errors"])
        print_detection_results("Factual Consistency (LLM)", results)
    except Exception as e:
        print(f"Error in LLM-based factual consistency detection: {e}")

def run_source_verification_example():
    """Demonstrate using the SourceVerificationDetector."""
    print("\n" + "="*50)
    print("Source Verification Detector Example")
    print("="*50)
    
    # Create a detector with similarity method
    detector = SourceVerificationDetector({
        "method": "keyword",  # Using keyword since we don't have a real embedder
        "min_keywords_overlap": 0.3
    })
    
    # Detection with supported text
    print("\nChecking text with support in sources:")
    try:
        start_time = time.time()
        results = detector.detect(
            SAMPLE_TEXTS["clean"],
            source_documents=SAMPLE_SOURCES
        )
        detection_time = time.time() - start_time
        
        print(f"Detection completed in {detection_time:.4f} seconds")
        print_detection_results("Source Verification (Supported)", results)
    except Exception as e:
        print(f"Error in source verification: {e}")
    
    # Detection with unsupported text
    print("\nChecking text without support in sources:")
    try:
        results = detector.detect(
            SAMPLE_TEXTS["unsupported"],
            source_documents=SAMPLE_SOURCES
        )
        print_detection_results("Source Verification (Unsupported)", results)
    except Exception as e:
        print(f"Error in source verification: {e}")
    
    # Create a detector with LLM method
    llm_detector = SourceVerificationDetector({
        "method": "llm",
        "llm_provider": mock_llm_provider
    })
    
    # Detect hallucinations using LLM
    try:
        results = llm_detector.detect(
            SAMPLE_TEXTS["unsupported"],
            source_documents=SAMPLE_SOURCES
        )
        print_detection_results("Source Verification (LLM)", results)
    except Exception as e:
        print(f"Error in LLM-based source verification: {e}")

def run_contradiction_example():
    """Demonstrate using the ContradictionDetector."""
    print("\n" + "="*50)
    print("Contradiction Detector Example")
    print("="*50)
    
    # Create a detector with rule-based method
    detector = ContradictionDetector({
        "method": "rule"
    })
    
    # Detect contradictions in text with contradictions
    print("\nChecking text with contradictions:")
    try:
        start_time = time.time()
        results = detector.detect(SAMPLE_TEXTS["contradictions"])
        detection_time = time.time() - start_time
        
        print(f"Detection completed in {detection_time:.4f} seconds")
        print_detection_results("Contradiction (With Contradictions)", results)
    except Exception as e:
        print(f"Error in contradiction detection: {e}")
    
    # Detect contradictions in clean text
    print("\nChecking text without contradictions:")
    try:
        results = detector.detect(SAMPLE_TEXTS["clean"])
        print_detection_results("Contradiction (Clean Text)", results)
    except Exception as e:
        print(f"Error in contradiction detection: {e}")
    
    # Create a detector with LLM method
    llm_detector = ContradictionDetector({
        "method": "llm",
        "llm_provider": mock_llm_provider
    })
    
    # Detect contradictions using LLM
    try:
        results = llm_detector.detect(SAMPLE_TEXTS["contradictions"])
        print_detection_results("Contradiction (LLM)", results)
    except Exception as e:
        print(f"Error in LLM-based contradiction detection: {e}")

def run_uncertainty_example():
    """Demonstrate using the UncertaintyDetector."""
    print("\n" + "="*50)
    print("Uncertainty Detector Example")
    print("="*50)
    
    # Create a detector with default configuration
    detector = UncertaintyDetector()
    
    # Detect uncertainty in text with uncertainty markers
    print("\nChecking text with uncertainty markers:")
    try:
        start_time = time.time()
        results = detector.detect(SAMPLE_TEXTS["uncertain"])
        detection_time = time.time() - start_time
        
        print(f"Detection completed in {detection_time:.4f} seconds")
        print_detection_results("Uncertainty (Uncertain Text)", results)
    except Exception as e:
        print(f"Error in uncertainty detection: {e}")
    
    # Detect uncertainty in clean text
    print("\nChecking text without uncertainty markers:")
    try:
        results = detector.detect(SAMPLE_TEXTS["clean"])
        print_detection_results("Uncertainty (Clean Text)", results)
    except Exception as e:
        print(f"Error in uncertainty detection: {e}")
    
    # Detect with custom configuration
    custom_detector = UncertaintyDetector({
        "threshold": 0.3,  # Lower threshold to catch more subtle uncertainty
        "uncertainty_markers": [
            "uncertain", "unclear", "unknown", "not sure",
            "might", "could", "perhaps", "possibly"
        ]
    })
    
    print("\nChecking with custom detector (lower threshold):")
    try:
        results = custom_detector.detect(SAMPLE_TEXTS["uncertain"])
        print_detection_results("Uncertainty (Custom Config)", results)
    except Exception as e:
        print(f"Error in custom uncertainty detection: {e}")

def run_factory_function_example():
    """Demonstrate using the get_hallucination_detector factory function."""
    print("\n" + "="*50)
    print("Factory Function Example")
    print("="*50)
    
    # Create different detectors using the factory
    detectors = [
        (
            "factual_consistency",
            get_hallucination_detector(
                "factual_consistency",
                method="knowledge_base",
                knowledge_base=KNOWLEDGE_BASE
            ),
            SAMPLE_TEXTS["factual_errors"]
        ),
        (
            "source_verification",
            get_hallucination_detector(
                "source_verification",
                method="keyword"
            ),
            (SAMPLE_TEXTS["unsupported"], SAMPLE_SOURCES)
        ),
        (
            "contradiction",
            get_hallucination_detector(
                "contradiction"
            ),
            SAMPLE_TEXTS["contradictions"]
        ),
        (
            "uncertainty",
            get_hallucination_detector(
                "uncertainty",
                threshold=0.4
            ),
            SAMPLE_TEXTS["uncertain"]
        )
    ]
    
    # Test each detector
    for name, detector, test_data in detectors:
        print(f"\n{name.title()} Detector (via factory):")
        try:
            # Handle source verification differently
            if name == "source_verification":
                text, sources = test_data
                results = detector.detect(text, source_documents=sources)
            else:
                results = detector.detect(test_data)
                
            print(f"  Success! Detected {len(results['detected_hallucinations'])} issues")
            print(f"  Score: {results['score']:.2f}, Confidence: {results['confidence']:.2f}")
        except Exception as e:
            print(f"  Error: {e}")

def run_combined_detection_example():
    """Demonstrate combining multiple hallucination detectors."""
    print("\n" + "="*50)
    print("Combined Hallucination Detection Example")
    print("="*50)
    
    # Create all detectors
    detectors = {
        "factual": FactualConsistencyDetector({
            "method": "knowledge_base",
            "knowledge_base": KNOWLEDGE_BASE
        }),
        "source": SourceVerificationDetector({
            "method": "keyword"
        }),
        "contradiction": ContradictionDetector(),
        "uncertainty": UncertaintyDetector()
    }
    
    # Function to combine results
    def combine_detection_results(
        all_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Combine results from multiple detectors.
        
        Args:
            all_results: Dictionary mapping detector names to results
            
        Returns:
            Combined results dictionary
        """
        # Convert to HallucinationDetectionResult objects
        result_objects = [
            HallucinationDetectionResult.from_dict(result) 
            for result in all_results.values()
        ]
        
        # Combine using the built-in method
        combined = HallucinationDetectionResult.combine_results(result_objects)
        
        return combined.to_dict()
    
    # Text to analyze
    test_text = """
    The Earth is probably flat, though I'm not entirely sure. It might be spherical,
    but the evidence is unclear. The Earth was formed in 1750 when a giant meteor
    hit the sun, breaking off a piece that became our planet. Actually, the Earth
    formed billions of years ago when cosmic dust accumulated into a planetary body.
    """
    
    # Run all detectors
    results = {}
    for name, detector in detectors.items():
        if name == "source":
            # Source verification needs source documents
            results[name] = detector.detect(test_text, source_documents=[])
        else:
            results[name] = detector.detect(test_text)
    
    # Print individual results
    for name, result in results.items():
        print(f"\n{name.title()} Detection Results:")
        print(f"  Score: {result['score']:.2f}, Confidence: {result['confidence']:.2f}")
        print(f"  Detected Issues: {len(result['detected_hallucinations'])}")
    
    # Combine results
    combined_results = combine_detection_results(results)
    print("\nCombined Detection Results:")
    print_detection_results("Combined Detectors", combined_results)

def main():
    """Run all hallucination detector examples."""
    print("Hallucination Detectors - Examples")
    print("\nThis script demonstrates the various hallucination detectors for identifying false or unsupported information in LLM outputs.")
    
    # Run individual examples
    run_factual_consistency_example()
    run_source_verification_example()
    run_contradiction_example()
    run_uncertainty_example()
    run_factory_function_example()
    run_combined_detection_example()
    
    print("\n" + "="*50)
    print("All examples completed!")
    print("="*50)

if __name__ == "__main__":
    main() 