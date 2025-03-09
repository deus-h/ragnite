# Hallucination Detectors

This directory contains documentation for the hallucination detectors in the RAG Research project. These detectors identify potential false, unsupported, or uncertain information in language model outputs.

## Overview

Hallucination detectors address a critical challenge in RAG systems: ensuring the factual accuracy and reliability of generated content. They analyze text to identify:

1. Factual inconsistencies with known facts or general knowledge
2. Statements not supported by the source documents
3. Internal contradictions within the generated text
4. Uncertain or speculative language indicating low confidence

By detecting hallucinations, these tools help improve the reliability of RAG systems and provide transparency about the confidence level of generated information.

## Available Hallucination Detectors

### FactualConsistencyDetector

The `FactualConsistencyDetector` identifies statements that contradict widely accepted facts or general knowledge.

Features:
- Multiple detection methods: knowledge base lookup, LLM-based fact checking, and NLI models
- Extracts individual statements for fine-grained analysis
- Provides detailed explanations of factual inconsistencies
- Adjustable thresholds for different sensitivity levels
- See [factual_consistency_detector.md](./factual_consistency_detector.md) for detailed documentation

### SourceVerificationDetector

The `SourceVerificationDetector` checks if the generated text is supported by the source documents used for retrieval.

Features:
- Multiple verification methods: semantic similarity, entailment, keyword matching, and LLM-based verification
- Identifies statements that lack support in the retrieved context
- Provides relevance scores between generated content and source documents
- Works with any document structure through a flexible API
- See [source_verification_detector.md](./source_verification_detector.md) for detailed documentation

### ContradictionDetector

The `ContradictionDetector` identifies internal contradictions within the generated text.

Features:
- Multiple detection methods: rule-based patterns, NLI models, and LLM-based detection
- Detects contradictory statements about the same entities or concepts
- Identifies logical inconsistencies within the text
- Includes default contradiction patterns covering common contradiction types
- See [contradiction_detector.md](./contradiction_detector.md) for detailed documentation

### UncertaintyDetector

The `UncertaintyDetector` identifies statements containing hedging, vague language, or uncertainty markers.

Features:
- Detects hedging phrases, uncertainty markers, speculative language, and probability expressions
- Calculates uncertainty scores based on the prevalence and type of markers
- Customizable dictionaries of uncertainty indicators
- Adjustable thresholds for different sensitivity levels
- See [uncertainty_detector.md](./uncertainty_detector.md) for detailed documentation

## Usage

To use a hallucination detector, create an instance with your configuration and then call the `detect` method:

```python
from tools.src.retrieval import FactualConsistencyDetector, ParsingError

# Create a detector with knowledge base method
detector = FactualConsistencyDetector({
    "method": "knowledge_base",
    "knowledge_base": {
        "general": {
            "The Earth orbits the Sun": {
                "negation": "The Sun orbits the Earth",
                "confidence": 0.95,
                "severity": "high"
            }
        }
    }
})

# Detect hallucinations in text
text = "The Sun orbits the Earth, completing one orbit every 24 hours."
result = detector.detect(text)

# Check if hallucinations were detected
if result["detected_hallucinations"]:
    print(f"Detected {len(result['detected_hallucinations'])} hallucinations:")
    for h in result["detected_hallucinations"]:
        print(f"- {h['text']}")
        print(f"  Reason: {h['reason']}")
        print(f"  Severity: {h['severity']}")
else:
    print("No hallucinations detected.")

# Get overall hallucination score (1.0 is best, 0.0 is worst)
print(f"Overall score: {result['score']:.2f}")
```

## Factory Function

The `get_hallucination_detector` factory function provides a simplified interface for creating detector instances:

```python
from tools.src.retrieval import get_hallucination_detector

# Create a factual consistency detector
factual_detector = get_hallucination_detector(
    detector_type="factual_consistency",
    method="llm",
    llm_provider=my_llm_function
)

# Create a source verification detector
source_detector = get_hallucination_detector(
    detector_type="source_verification",
    method="similarity",
    embedder=my_embedding_function,
    similarity_threshold=0.75
)

# Create a contradiction detector
contradiction_detector = get_hallucination_detector(
    detector_type="contradiction"
)

# Create an uncertainty detector
uncertainty_detector = get_hallucination_detector(
    detector_type="uncertainty",
    threshold=0.6
)
```

## Choosing a Detector

Different detectors are suitable for different use cases:

- **FactualConsistencyDetector**: Use when you need to verify statements against known facts, especially for objective domains like science, history, or geography.

- **SourceVerificationDetector**: Use when you need to ensure the generated text is supported by the retrieved documents, which is particularly important for evidence-based responses.

- **ContradictionDetector**: Use when you need to check for logical coherence and consistency within the generated text, especially for longer responses.

- **UncertaintyDetector**: Use when you need to identify speculative or hedged statements that may indicate low confidence, which is useful for distinguishing between facts and opinions.

## Combining Detectors

For comprehensive hallucination detection, you can combine multiple detectors:

```python
from tools.src.retrieval import (
    FactualConsistencyDetector,
    SourceVerificationDetector,
    ContradictionDetector,
    UncertaintyDetector,
    HallucinationDetectionResult
)

# Create all detectors
detectors = {
    "factual": FactualConsistencyDetector(),
    "source": SourceVerificationDetector(),
    "contradiction": ContradictionDetector(),
    "uncertainty": UncertaintyDetector()
}

# Run all detectors on the same text
text = "The Earth is probably flat, though I'm not entirely sure."
results = {}

for name, detector in detectors.items():
    if name == "source":
        # Source verification needs source documents
        results[name] = detector.detect(text, source_documents=documents)
    else:
        results[name] = detector.detect(text)

# Convert to HallucinationDetectionResult objects
result_objects = [
    HallucinationDetectionResult.from_dict(result) 
    for result in results.values()
]

# Combine using the built-in method
combined = HallucinationDetectionResult.combine_results(result_objects)

# Access combined results
print(f"Combined score: {combined.score:.2f}")
print(f"Detected issues: {len(combined.detected_hallucinations)}")
print(f"Explanation: {combined.explanation}")
```

## Error Handling

All hallucination detectors handle exceptions gracefully, providing useful error messages and fallback behaviors:

```python
try:
    results = detector.detect(text)
except Exception as e:
    print(f"Error in hallucination detection: {e}")
    # Fallback behavior or error handling
```

## Examples

See the [examples directory](../../../../examples/retrieval/generation/hallucination_detectors/) for example scripts demonstrating the use of hallucination detectors.

## Advanced Configuration

### FactualConsistencyDetector

```python
detector = FactualConsistencyDetector({
    "method": "knowledge_base",  # Options: "knowledge_base", "llm", "nli"
    "knowledge_base": my_knowledge_base_dict,
    "llm_provider": my_llm_function,
    "nli_model": my_nli_model,
    "threshold": 0.7,
    "external_api_key": "my_api_key",
    "external_api_url": "https://api.example.com/factcheck"
})
```

### SourceVerificationDetector

```python
detector = SourceVerificationDetector({
    "method": "similarity",  # Options: "similarity", "entailment", "llm", "keyword"
    "similarity_threshold": 0.6,
    "entailment_threshold": 0.7,
    "min_keywords_overlap": 0.3,
    "embedder": my_embedding_function,
    "llm_provider": my_llm_function,
    "nli_model": my_nli_model
})
```

### ContradictionDetector

```python
detector = ContradictionDetector({
    "method": "rule",  # Options: "rule", "nli", "llm"
    "nli_model": my_nli_model,
    "llm_provider": my_llm_function,
    "contradiction_rules": my_custom_rules,
    "threshold": 0.7,
    "max_distance": 10  # Maximum distance between statements to check for contradictions
})
```

### UncertaintyDetector

```python
detector = UncertaintyDetector({
    "uncertainty_markers": ["uncertain", "unclear", "unknown", "not clear"],
    "hedging_phrases": ["seems to be", "appears to be", "might be"],
    "speculative_phrases": ["I think", "I believe", "I suspect"],
    "probability_markers": ["sometimes", "occasionally", "often"],
    "threshold": 0.5,
    "severity_weights": {
        "hedging": 0.7,
        "uncertainty": 0.8,
        "speculation": 0.6,
        "probability": 0.5
    }
})
```

## Custom Detectors

You can create custom hallucination detectors by extending the `BaseHallucinationDetector` class:

```python
from tools.src.retrieval import BaseHallucinationDetector

class CustomHallucinationDetector(BaseHallucinationDetector):
    """Custom hallucination detector for specific domain."""
    
    def __init__(self, config=None):
        super().__init__(config)
        # Initialize your custom detector
        
    def detect(self, generated_text, source_documents=None, **kwargs):
        """Implement your custom detection logic."""
        # Detection logic here
        return {
            "score": 0.9,
            "detected_hallucinations": [...],
            "explanation": "Custom detection results",
            "confidence": 0.8,
            "detection_type": self.get_detection_type()
        }
        
    def get_detection_type(self):
        """Return your detection type."""
        return "custom_detection"
``` 