"""
Factual Consistency Detector

This module defines the FactualConsistencyDetector that identifies statements in the
generated text that are factually inconsistent with general knowledge or accepted facts.
"""

import re
import json
import requests
from typing import Any, Dict, List, Optional, Union, Callable
from .base_hallucination_detector import BaseHallucinationDetector, HallucinationDetectionResult


class FactualConsistencyDetector(BaseHallucinationDetector):
    """
    Detector for identifying factually inconsistent statements in generated text.
    
    This detector analyzes generated text to identify statements that contradict
    widely accepted facts or general knowledge. It can use a variety of methods,
    including knowledge base lookups, LLM-based fact-checking, or NLI models.
    
    Attributes:
        model_name (str): Name of the model used for factual consistency checking.
        method (str): Method used for detection ("knowledge_base", "llm", "nli").
        knowledge_base (Dict[str, Any], optional): Knowledge base for fact verification.
        llm_provider (Callable, optional): Function for LLM-based fact verification.
        nli_model (Any, optional): NLI model for entailment-based fact verification.
        threshold (float): Confidence threshold for considering a statement hallucinatory.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the factual consistency detector.
        
        Args:
            config: Configuration dictionary that may include:
                - model_name: Name of the model to use (default: "auto")
                - method: Detection method ("knowledge_base", "llm", "nli") (default: "llm")
                - knowledge_base: Knowledge base dictionary for fact verification
                - llm_provider: Callable function for LLM-based fact verification
                - threshold: Confidence threshold (default: 0.7)
                - external_api_key: API key for external fact-checking services
                - external_api_url: URL for external fact-checking services
        """
        super().__init__(config)
        self.model_name = self.config.get("model_name", "auto")
        self.method = self.config.get("method", "llm")
        self.knowledge_base = self.config.get("knowledge_base", {})
        self.llm_provider = self.config.get("llm_provider")
        self.nli_model = self.config.get("nli_model")
        self.threshold = self.config.get("threshold", 0.7)
        self.external_api_key = self.config.get("external_api_key")
        self.external_api_url = self.config.get("external_api_url")
        
        # Initialize the appropriate method
        if self.method == "knowledge_base" and not self.knowledge_base:
            raise ValueError("Knowledge base method requires a knowledge_base")
        if self.method == "llm" and not self.llm_provider:
            # Use default LLM if not provided (placeholder)
            self.llm_provider = self._default_llm_provider
        if self.method == "nli" and not self.nli_model:
            # Load default NLI model if not provided (placeholder)
            self.nli_model = self._load_default_nli_model()
    
    def detect(self, 
               generated_text: str, 
               source_documents: Optional[List[Dict[str, Any]]] = None, 
               **kwargs) -> Dict[str, Any]:
        """
        Detect factually inconsistent statements in the generated text.
        
        Args:
            generated_text: The text generated by the language model.
            source_documents: Optional list of source documents (not directly used for 
                factual consistency checking, but may be used for context).
            **kwargs: Additional arguments:
                - facts: Optional list of known facts to verify against
                - domain: Optional domain context (e.g., "medical", "legal", "science")
            
        Returns:
            A dictionary containing detection results.
        """
        # Extract statements to check
        statements = self._extract_statements(generated_text)
        facts = kwargs.get("facts", [])
        domain = kwargs.get("domain", "general")
        
        # Detect factual inconsistencies using the appropriate method
        if self.method == "knowledge_base":
            results = self._check_with_knowledge_base(statements, domain)
        elif self.method == "llm":
            results = self._check_with_llm(statements, facts, domain)
        elif self.method == "nli":
            results = self._check_with_nli(statements, facts)
        elif self.method == "external_api":
            results = self._check_with_external_api(statements)
        else:
            raise ValueError(f"Unsupported method: {self.method}")
        
        # Process results
        detected_hallucinations = []
        overall_score = 1.0  # Start with perfect score (no hallucinations)
        
        for result in results:
            if result["is_hallucination"]:
                # Add to the detected hallucinations list
                detected_hallucinations.append({
                    "text": result["statement"],
                    "span": result["span"],
                    "reason": result["reason"],
                    "severity": result["severity"],
                    "confidence": result["confidence"]
                })
                
                # Update the overall score (lower is worse)
                severity_factor = {
                    "low": 0.9,
                    "medium": 0.7,
                    "high": 0.5
                }.get(result["severity"], 0.8)
                overall_score *= severity_factor
        
        # Ensure score is in [0, 1] range
        overall_score = max(0.0, min(1.0, overall_score))
        
        # Generate explanation
        explanation = self._generate_explanation(detected_hallucinations)
        
        # Calculate confidence in results
        if not detected_hallucinations:
            confidence = 0.8  # Default confidence when no hallucinations found
        else:
            confidence = sum(h["confidence"] for h in detected_hallucinations) / len(detected_hallucinations)
        
        # Create result object
        result = HallucinationDetectionResult(
            score=overall_score,
            detected_hallucinations=detected_hallucinations,
            explanation=explanation,
            confidence=confidence,
            detection_type=self.get_detection_type()
        )
        
        return result.to_dict()
    
    def get_detection_type(self) -> str:
        """
        Get the type of hallucination detection performed.
        
        Returns:
            The string "factual_consistency"
        """
        return "factual_consistency"
    
    def _extract_statements(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract individual statements from the generated text for fact-checking.
        
        This is a simplified implementation that splits text into sentences and
        considers each sentence a statement. More sophisticated NLP could be used.
        
        Args:
            text: The generated text to analyze
            
        Returns:
            List of dictionaries with statement text and character spans
        """
        statements = []
        
        # Simple sentence splitting using regex
        # In a production system, use a more robust sentence tokenizer
        sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
        spans = list(re.finditer(sentence_pattern, text))
        
        start_idx = 0
        for span in spans:
            end_idx = span.start() + 1  # Include the punctuation
            sentence = text[start_idx:end_idx].strip()
            
            if sentence:
                statements.append({
                    "statement": sentence,
                    "span": (start_idx, end_idx)
                })
            
            start_idx = span.end()
        
        # Don't forget the last sentence if it doesn't end with punctuation
        if start_idx < len(text):
            sentence = text[start_idx:].strip()
            if sentence:
                statements.append({
                    "statement": sentence,
                    "span": (start_idx, len(text))
                })
        
        return statements
    
    def _check_with_knowledge_base(self, 
                                 statements: List[Dict[str, Any]], 
                                 domain: str) -> List[Dict[str, Any]]:
        """
        Check statements against a knowledge base.
        
        Args:
            statements: List of statements to check
            domain: Domain context for knowledge base lookup
            
        Returns:
            List of results with hallucination flags and reasons
        """
        results = []
        
        kb = self.knowledge_base.get(domain, {})
        if not kb and domain != "general":
            kb = self.knowledge_base.get("general", {})
        
        for statement_info in statements:
            statement = statement_info["statement"]
            result = {
                "statement": statement,
                "span": statement_info["span"],
                "is_hallucination": False,
                "confidence": 0.0,
                "reason": "",
                "severity": "none"
            }
            
            # Simple keyword-based lookup in knowledge base
            # In a real system, this would use more sophisticated semantic matching
            statement_lower = statement.lower()
            for fact, fact_info in kb.items():
                fact_lower = fact.lower()
                
                # Check for contradictions
                if fact_info.get("negation", "") and fact_info["negation"].lower() in statement_lower:
                    result["is_hallucination"] = True
                    result["confidence"] = fact_info.get("confidence", 0.8)
                    result["reason"] = f"Contradicts known fact: {fact}"
                    result["severity"] = fact_info.get("severity", "medium")
                    break
            
            results.append(result)
        
        return results
    
    def _check_with_llm(self, 
                       statements: List[Dict[str, Any]], 
                       facts: List[str], 
                       domain: str) -> List[Dict[str, Any]]:
        """
        Check statements using an LLM-based fact verification.
        
        Args:
            statements: List of statements to check
            facts: List of known facts to verify against
            domain: Domain context for fact verification
            
        Returns:
            List of results with hallucination flags and reasons
        """
        results = []
        
        for statement_info in statements:
            statement = statement_info["statement"]
            
            # Construct prompt for LLM
            prompt = self._construct_fact_checking_prompt(statement, facts, domain)
            
            # Get response from LLM provider
            if self.llm_provider:
                llm_response = self.llm_provider(prompt)
                result = self._parse_llm_response(llm_response, statement_info)
                results.append(result)
            else:
                # Default to no hallucination if no LLM provider
                results.append({
                    "statement": statement,
                    "span": statement_info["span"],
                    "is_hallucination": False,
                    "confidence": 0.5,
                    "reason": "No LLM provider available for verification",
                    "severity": "none"
                })
        
        return results
    
    def _check_with_nli(self, 
                       statements: List[Dict[str, Any]], 
                       facts: List[str]) -> List[Dict[str, Any]]:
        """
        Check statements using Natural Language Inference models.
        
        Args:
            statements: List of statements to check
            facts: List of known facts to verify against
            
        Returns:
            List of results with hallucination flags and reasons
        """
        results = []
        
        for statement_info in statements:
            statement = statement_info["statement"]
            
            # Default values
            is_hallucination = False
            confidence = 0.5
            reason = ""
            severity = "none"
            
            # Check if we have an NLI model
            if self.nli_model:
                # Check statement against each fact
                contradictions = []
                for fact in facts:
                    # This is a placeholder - in a real implementation,
                    # we would run the statement and fact through an NLI model
                    # and get entailment, contradiction, or neutral prediction
                    result = self._run_nli_model(statement, fact)
                    
                    if result["label"] == "contradiction":
                        contradictions.append({
                            "fact": fact,
                            "confidence": result["confidence"]
                        })
                
                # If we found contradictions, mark as hallucination
                if contradictions:
                    max_confidence_contradiction = max(contradictions, key=lambda x: x["confidence"])
                    is_hallucination = True
                    confidence = max_confidence_contradiction["confidence"]
                    reason = f"Contradicts: {max_confidence_contradiction['fact']}"
                    severity = "medium" if confidence > 0.8 else "low"
            
            results.append({
                "statement": statement,
                "span": statement_info["span"],
                "is_hallucination": is_hallucination,
                "confidence": confidence,
                "reason": reason,
                "severity": severity
            })
        
        return results
    
    def _check_with_external_api(self, statements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Check statements using an external fact-checking API.
        
        Args:
            statements: List of statements to check
            
        Returns:
            List of results with hallucination flags and reasons
        """
        results = []
        
        if not self.external_api_url or not self.external_api_key:
            # Fall back to default result if no API configured
            for statement_info in statements:
                results.append({
                    "statement": statement_info["statement"],
                    "span": statement_info["span"],
                    "is_hallucination": False,
                    "confidence": 0.5,
                    "reason": "No external API configured for verification",
                    "severity": "none"
                })
            return results
        
        for statement_info in statements:
            statement = statement_info["statement"]
            
            try:
                # Prepare the API request
                headers = {
                    "Authorization": f"Bearer {self.external_api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "statement": statement,
                    "options": {
                        "detailed": True,
                        "domain": self.config.get("domain", "general")
                    }
                }
                
                # Make the API request
                response = requests.post(
                    self.external_api_url,
                    headers=headers,
                    data=json.dumps(data),
                    timeout=10  # 10 second timeout
                )
                
                if response.status_code == 200:
                    api_result = response.json()
                    
                    # Parse the API response
                    results.append({
                        "statement": statement,
                        "span": statement_info["span"],
                        "is_hallucination": api_result.get("is_hallucination", False),
                        "confidence": api_result.get("confidence", 0.5),
                        "reason": api_result.get("reason", ""),
                        "severity": api_result.get("severity", "none")
                    })
                else:
                    # API request failed
                    results.append({
                        "statement": statement,
                        "span": statement_info["span"],
                        "is_hallucination": False,
                        "confidence": 0.5,
                        "reason": f"API error: {response.status_code}",
                        "severity": "none"
                    })
            
            except Exception as e:
                # Handle API exceptions
                results.append({
                    "statement": statement,
                    "span": statement_info["span"],
                    "is_hallucination": False,
                    "confidence": 0.5,
                    "reason": f"API exception: {str(e)}",
                    "severity": "none"
                })
        
        return results
    
    def _construct_fact_checking_prompt(self, 
                                      statement: str, 
                                      facts: List[str], 
                                      domain: str) -> str:
        """
        Construct a prompt for LLM-based fact checking.
        
        Args:
            statement: Statement to check
            facts: Known facts to verify against
            domain: Domain context
            
        Returns:
            Formatted prompt for the LLM
        """
        prompt = f"Please fact-check the following statement within the {domain} domain:\n\n"
        prompt += f"Statement: \"{statement}\"\n\n"
        
        if facts:
            prompt += "Known facts:\n"
            for i, fact in enumerate(facts, 1):
                prompt += f"{i}. {fact}\n"
        
        prompt += "\nIs the statement factually accurate? If not, explain why it's inaccurate.\n"
        prompt += "Respond in the following JSON format:\n"
        prompt += "{\n"
        prompt += '  "is_hallucination": true/false,\n'
        prompt += '  "confidence": 0.0-1.0,\n'
        prompt += '  "reason": "explanation of why",\n'
        prompt += '  "severity": "none/low/medium/high"\n'
        prompt += "}"
        
        return prompt
    
    def _parse_llm_response(self, 
                          llm_response: str, 
                          statement_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the LLM response to extract fact-checking results.
        
        Args:
            llm_response: Response from the LLM
            statement_info: Original statement information
            
        Returns:
            Parsed result dictionary
        """
        # Default values
        result = {
            "statement": statement_info["statement"],
            "span": statement_info["span"],
            "is_hallucination": False,
            "confidence": 0.5,
            "reason": "",
            "severity": "none"
        }
        
        # Try to extract JSON from the response
        try:
            # Find JSON in the response
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed_json = json.loads(json_str)
                
                # Update result with parsed values
                result["is_hallucination"] = parsed_json.get("is_hallucination", False)
                result["confidence"] = parsed_json.get("confidence", 0.5)
                result["reason"] = parsed_json.get("reason", "")
                result["severity"] = parsed_json.get("severity", "none")
        except Exception as e:
            # If JSON parsing fails, try to extract information with regex
            hallucination_match = re.search(r'(is|not) (factually accurate|accurate|correct|true)', llm_response, re.IGNORECASE)
            if hallucination_match and "not" in hallucination_match.group(0).lower():
                result["is_hallucination"] = True
                result["confidence"] = 0.7
                
                # Try to extract reason
                reason_match = re.search(r'(?:reason|because|explanation|why)[\s:]+([^\n]+)', llm_response, re.IGNORECASE)
                if reason_match:
                    result["reason"] = reason_match.group(1).strip()
                else:
                    result["reason"] = "Unable to extract specific reason from LLM response"
                
                # Default to medium severity
                result["severity"] = "medium"
        
        return result
    
    def _run_nli_model(self, statement: str, fact: str) -> Dict[str, Any]:
        """
        Run an NLI model to determine if statement contradicts, entails, or is neutral to fact.
        
        This is a placeholder method. In a real implementation, this would use an actual NLI model.
        
        Args:
            statement: Statement to check
            fact: Fact to check against
            
        Returns:
            Dictionary with NLI prediction results
        """
        # This is a simplified placeholder implementation
        if self.nli_model:
            # In a real implementation, run the NLI model here
            # For now, return a default neutral result
            return {
                "label": "neutral",
                "confidence": 0.5
            }
        
        # Very simple keyword-based detection as fallback
        statement_lower = statement.lower()
        fact_lower = fact.lower()
        
        # Look for negation words followed by key phrases from the fact
        negation_words = ["not", "no", "never", "isn't", "aren't", "wasn't", "weren't", "doesn't", "don't", "didn't"]
        
        # Extract key phrases from the fact (simplified)
        key_phrases = [word for word in fact_lower.split() if len(word) > 4]
        
        # Check for contradictions
        for negation in negation_words:
            if negation in statement_lower:
                for phrase in key_phrases:
                    if phrase in statement_lower:
                        # Potential contradiction found
                        return {
                            "label": "contradiction",
                            "confidence": 0.6
                        }
        
        # Check for entailment (simplified)
        shared_words = sum(1 for word in key_phrases if word in statement_lower)
        if shared_words >= len(key_phrases) // 2:
            return {
                "label": "entailment",
                "confidence": 0.6
            }
        
        # Default to neutral
        return {
            "label": "neutral",
            "confidence": 0.5
        }
    
    def _load_default_nli_model(self):
        """
        Load a default NLI model.
        
        This is a placeholder method. In a real implementation, this would load
        an actual NLI model like RoBERTa for NLI.
        
        Returns:
            None (placeholder)
        """
        # In a real implementation, load an NLI model here
        # For example:
        # from transformers import AutoModelForSequenceClassification, AutoTokenizer
        # model_name = "roberta-large-mnli"
        # tokenizer = AutoTokenizer.from_pretrained(model_name)
        # model = AutoModelForSequenceClassification.from_pretrained(model_name)
        # return {"model": model, "tokenizer": tokenizer}
        return None
    
    def _default_llm_provider(self, prompt: str) -> str:
        """
        Default LLM provider that returns a placeholder response.
        
        This is used when no LLM provider is configured. In a real system,
        this would be replaced with an actual LLM API call.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            A placeholder response
        """
        # This is a placeholder. In a real system, this would call an actual LLM API.
        return """
        {
            "is_hallucination": false,
            "confidence": 0.5,
            "reason": "Unable to verify without LLM provider",
            "severity": "none"
        }
        """
    
    def _generate_explanation(self, hallucinations: List[Dict[str, Any]]) -> str:
        """
        Generate a human-readable explanation of detection results.
        
        Args:
            hallucinations: List of detected hallucinations
            
        Returns:
            A formatted explanation string
        """
        if not hallucinations:
            return "No factual inconsistencies detected in the generated text."
        
        explanation = f"Detected {len(hallucinations)} factual inconsistencies:\n"
        
        for i, h in enumerate(hallucinations, 1):
            explanation += f"{i}. \"{h['text']}\" - {h['reason']} "
            explanation += f"(Severity: {h['severity']}, Confidence: {h['confidence']:.2f})\n"
        
        return explanation.strip()
    
    def __repr__(self) -> str:
        """Return a string representation of the detector."""
        return f"FactualConsistencyDetector(method={self.method}, model={self.model_name})" 