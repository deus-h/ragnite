"""
Source Verification Detector

This module defines the SourceVerificationDetector that checks if generated text
is supported by the source documents used for retrieval.
"""

import re
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from .base_hallucination_detector import BaseHallucinationDetector, HallucinationDetectionResult


class SourceVerificationDetector(BaseHallucinationDetector):
    """
    Detector for verifying generated content against source documents.
    
    This detector analyzes whether the generated text is supported by the provided
    source documents, identifying statements that may be hallucinations because
    they lack support in the retrieved context.
    
    Attributes:
        method (str): Method used for verification ("similarity", "entailment", "llm", "keyword").
        similarity_threshold (float): Threshold for considering text supported by similarity.
        entailment_threshold (float): Threshold for considering text supported by entailment.
        min_keywords_overlap (float): Minimum keyword overlap ratio for support.
        embedder (Callable): Function for generating text embeddings.
        llm_provider (Callable): Function for LLM-based verification.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the source verification detector.
        
        Args:
            config: Configuration dictionary that may include:
                - method: Verification method ("similarity", "entailment", "llm", "keyword")
                - similarity_threshold: Threshold for similarity-based verification (default: 0.6)
                - entailment_threshold: Threshold for entailment-based verification (default: 0.7)
                - min_keywords_overlap: Threshold for keyword-based verification (default: 0.3)
                - embedder: Function for generating text embeddings
                - llm_provider: Function for LLM-based verification
                - nli_model: Model for entailment-based verification
        """
        super().__init__(config)
        self.method = self.config.get("method", "similarity")
        self.similarity_threshold = self.config.get("similarity_threshold", 0.6)
        self.entailment_threshold = self.config.get("entailment_threshold", 0.7)
        self.min_keywords_overlap = self.config.get("min_keywords_overlap", 0.3)
        self.embedder = self.config.get("embedder")
        self.llm_provider = self.config.get("llm_provider")
        self.nli_model = self.config.get("nli_model")
        
        # Validate configuration based on the method
        if self.method == "similarity" and not self.embedder:
            self.embedder = self._default_embedder
            
        if self.method == "entailment" and not self.nli_model:
            # In a real implementation, we would load a default NLI model
            # For now, we'll fall back to a simpler method
            self.method = "keyword"
            
        if self.method == "llm" and not self.llm_provider:
            self.llm_provider = self._default_llm_provider
    
    def detect(self, 
               generated_text: str, 
               source_documents: Optional[List[Dict[str, Any]]] = None, 
               **kwargs) -> Dict[str, Any]:
        """
        Detect statements in the generated text that aren't supported by source documents.
        
        Args:
            generated_text: The text generated by the language model.
            source_documents: List of source documents used for retrieval.
                Each document should be a dictionary with at least 'content' and 'metadata' fields.
            **kwargs: Additional arguments:
                - extract_claims: Whether to extract specific claims for verification
                - ignore_keywords: List of keywords/phrases to ignore in verification
                
        Returns:
            A dictionary containing detection results.
        """
        # Check if we have source documents
        if not source_documents:
            # If no source documents, everything is potentially a hallucination
            return HallucinationDetectionResult(
                score=0.0,  # Worst possible score
                detected_hallucinations=[{
                    "text": generated_text,
                    "span": (0, len(generated_text)),
                    "reason": "No source documents provided for verification",
                    "severity": "high",
                    "confidence": 0.9
                }],
                explanation="Cannot verify text - no source documents provided",
                confidence=0.9,
                detection_type=self.get_detection_type()
            ).to_dict()
        
        # Extract source document contents
        source_contents = [doc.get("content", "") for doc in source_documents if "content" in doc]
        if not source_contents:
            return HallucinationDetectionResult(
                score=0.0,
                detected_hallucinations=[{
                    "text": generated_text,
                    "span": (0, len(generated_text)),
                    "reason": "Source documents have no content",
                    "severity": "high",
                    "confidence": 0.9
                }],
                explanation="Cannot verify text - source documents have no content",
                confidence=0.9,
                detection_type=self.get_detection_type()
            ).to_dict()
        
        # Extract claims/statements from generated text
        extract_claims = kwargs.get("extract_claims", True)
        if extract_claims:
            statements = self._extract_statements(generated_text)
        else:
            # Use the entire text as a single statement
            statements = [{
                "statement": generated_text,
                "span": (0, len(generated_text))
            }]
        
        # Verify statements against source documents
        if self.method == "similarity":
            verification_results = self._verify_with_similarity(statements, source_contents)
        elif self.method == "entailment":
            verification_results = self._verify_with_entailment(statements, source_contents)
        elif self.method == "llm":
            verification_results = self._verify_with_llm(statements, source_contents)
        elif self.method == "keyword":
            verification_results = self._verify_with_keywords(statements, source_contents)
        else:
            raise ValueError(f"Unsupported verification method: {self.method}")
        
        # Process results
        detected_hallucinations = []
        for statement, result in zip(statements, verification_results):
            if not result["is_supported"]:
                detected_hallucinations.append({
                    "text": statement["statement"],
                    "span": statement["span"],
                    "reason": result["reason"],
                    "severity": result["severity"],
                    "confidence": result["confidence"]
                })
        
        # Calculate overall score
        if not detected_hallucinations:
            score = 1.0  # Perfect score
            confidence = 0.8
            explanation = "All statements are supported by source documents."
        else:
            # Start with perfect score and reduce for each hallucination
            score = 1.0
            for h in detected_hallucinations:
                severity_factor = {
                    "low": 0.9,
                    "medium": 0.7,
                    "high": 0.5
                }.get(h["severity"], 0.8)
                score *= severity_factor
            
            # Ensure score is in [0, 1] range
            score = max(0.0, min(1.0, score))
            
            # Calculate confidence as average of individual confidences
            confidence = sum(h["confidence"] for h in detected_hallucinations) / len(detected_hallucinations)
            
            # Generate explanation
            explanation = self._generate_explanation(detected_hallucinations)
        
        # Create result object
        result = HallucinationDetectionResult(
            score=score,
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
            The string "source_verification"
        """
        return "source_verification"
    
    def _extract_statements(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract individual statements from the generated text for verification.
        
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
    
    def _verify_with_similarity(self, 
                              statements: List[Dict[str, Any]], 
                              source_contents: List[str]) -> List[Dict[str, Any]]:
        """
        Verify statements using semantic similarity with source documents.
        
        Args:
            statements: List of statements to verify
            source_contents: List of source document contents
            
        Returns:
            List of verification results
        """
        results = []
        
        # Check if we have an embedder
        if not self.embedder:
            # Fall back to keyword verification if no embedder
            return self._verify_with_keywords(statements, source_contents)
        
        # Combine all source contents into chunks
        source_chunks = []
        for content in source_contents:
            # Simple chunking by paragraphs
            chunks = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]
            source_chunks.extend(chunks)
        
        # If no chunks, return unsupported
        if not source_chunks:
            return [{"is_supported": False, "confidence": 0.8, 
                    "reason": "Empty source documents", "severity": "high"}
                   for _ in statements]
        
        # Generate embeddings for source chunks
        try:
            source_embeddings = [self.embedder(chunk) for chunk in source_chunks]
        except Exception as e:
            # If embedding fails, fall back to keyword verification
            return self._verify_with_keywords(statements, source_contents)
        
        # Verify each statement
        for statement_info in statements:
            statement = statement_info["statement"]
            
            try:
                # Generate embedding for the statement
                statement_embedding = self.embedder(statement)
                
                # Calculate similarities with all source chunks
                similarities = []
                for source_embedding in source_embeddings:
                    similarity = self._cosine_similarity(statement_embedding, source_embedding)
                    similarities.append(similarity)
                
                # Find the maximum similarity
                max_similarity = max(similarities)
                max_chunk_idx = similarities.index(max_similarity)
                
                # Check if the statement is supported
                if max_similarity >= self.similarity_threshold:
                    # Statement is supported
                    results.append({
                        "is_supported": True,
                        "confidence": max_similarity,
                        "reason": f"Supported by source (similarity: {max_similarity:.2f})",
                        "severity": "none",
                        "source_chunk": source_chunks[max_chunk_idx]
                    })
                else:
                    # Statement is not supported
                    results.append({
                        "is_supported": False,
                        "confidence": 1.0 - max_similarity,
                        "reason": f"Not sufficiently supported by sources (max similarity: {max_similarity:.2f})",
                        "severity": "medium" if max_similarity < 0.3 else "low",
                        "source_chunk": source_chunks[max_chunk_idx] if max_similarity > 0.3 else None
                    })
            except Exception as e:
                # Error in similarity calculation
                results.append({
                    "is_supported": False,
                    "confidence": 0.5,
                    "reason": f"Error in similarity calculation: {str(e)}",
                    "severity": "medium"
                })
        
        return results
    
    def _verify_with_entailment(self, 
                              statements: List[Dict[str, Any]], 
                              source_contents: List[str]) -> List[Dict[str, Any]]:
        """
        Verify statements using natural language inference/entailment with source documents.
        
        Args:
            statements: List of statements to verify
            source_contents: List of source document contents
            
        Returns:
            List of verification results
        """
        results = []
        
        # Check if we have an NLI model
        if not self.nli_model:
            # Fall back to similarity or keyword verification
            if self.embedder:
                return self._verify_with_similarity(statements, source_contents)
            else:
                return self._verify_with_keywords(statements, source_contents)
        
        # Combine all source contents into chunks
        source_chunks = []
        for content in source_contents:
            # Simple chunking by paragraphs
            chunks = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]
            source_chunks.extend(chunks)
        
        # If no chunks, return unsupported
        if not source_chunks:
            return [{"is_supported": False, "confidence": 0.8, 
                    "reason": "Empty source documents", "severity": "high"}
                   for _ in statements]
        
        # Verify each statement
        for statement_info in statements:
            statement = statement_info["statement"]
            
            # Check statement against each source chunk
            best_entailment = {"label": "contradiction", "score": 0.0, "chunk": ""}
            
            for chunk in source_chunks:
                # This is a placeholder. In a real implementation, this would use an actual NLI model
                # to determine if the chunk entails the statement
                result = self._run_nli(chunk, statement)
                
                if result["label"] == "entailment" and result["score"] > best_entailment["score"]:
                    best_entailment = {"label": "entailment", "score": result["score"], "chunk": chunk}
                elif result["label"] == "neutral" and best_entailment["label"] != "entailment" and result["score"] > best_entailment["score"]:
                    best_entailment = {"label": "neutral", "score": result["score"], "chunk": chunk}
            
            # Check if the statement is supported
            if best_entailment["label"] == "entailment" and best_entailment["score"] >= self.entailment_threshold:
                # Statement is supported
                results.append({
                    "is_supported": True,
                    "confidence": best_entailment["score"],
                    "reason": f"Entailed by source (confidence: {best_entailment['score']:.2f})",
                    "severity": "none",
                    "source_chunk": best_entailment["chunk"]
                })
            elif best_entailment["label"] == "neutral":
                # Statement is neutral (not directly contradicted)
                severity = "low" if best_entailment["score"] > 0.7 else "medium"
                results.append({
                    "is_supported": False,
                    "confidence": best_entailment["score"],
                    "reason": f"Not explicitly supported by sources (neutral)",
                    "severity": severity,
                    "source_chunk": best_entailment["chunk"]
                })
            else:
                # Statement is contradicted or not supported
                results.append({
                    "is_supported": False,
                    "confidence": best_entailment["score"],
                    "reason": f"Contradicted by or absent from sources",
                    "severity": "high",
                    "source_chunk": best_entailment["chunk"] if best_entailment["chunk"] else None
                })
        
        return results
    
    def _verify_with_llm(self, 
                       statements: List[Dict[str, Any]], 
                       source_contents: List[str]) -> List[Dict[str, Any]]:
        """
        Verify statements using an LLM to compare with source documents.
        
        Args:
            statements: List of statements to verify
            source_contents: List of source document contents
            
        Returns:
            List of verification results
        """
        results = []
        
        # Check if we have an LLM provider
        if not self.llm_provider:
            # Fall back to another method
            if self.embedder:
                return self._verify_with_similarity(statements, source_contents)
            else:
                return self._verify_with_keywords(statements, source_contents)
        
        # Combine source contents (limit to avoid too long prompts)
        combined_sources = "\n\n".join(source_contents)
        if len(combined_sources) > 10000:  # Arbitrary limit to avoid too long prompts
            combined_sources = combined_sources[:10000] + "..."
        
        # Verify each statement
        for statement_info in statements:
            statement = statement_info["statement"]
            
            # Construct prompt for LLM
            prompt = self._construct_verification_prompt(statement, combined_sources)
            
            try:
                # Get response from LLM provider
                llm_response = self.llm_provider(prompt)
                
                # Parse the response
                if "true" in llm_response.lower() and "supported" in llm_response.lower():
                    # Statement is supported
                    confidence = 0.8  # Default confidence
                    
                    # Try to extract confidence
                    confidence_match = re.search(r'confidence[:\s]+([0-9.]+)', llm_response, re.IGNORECASE)
                    if confidence_match:
                        try:
                            confidence = float(confidence_match.group(1))
                            if confidence > 1.0:  # Normalize if on a different scale
                                confidence /= 10.0 if confidence <= 10.0 else 100.0
                        except ValueError:
                            pass
                    
                    results.append({
                        "is_supported": True,
                        "confidence": confidence,
                        "reason": "Supported according to LLM verification",
                        "severity": "none"
                    })
                else:
                    # Statement is not supported
                    confidence = 0.7  # Default confidence
                    
                    # Try to extract confidence
                    confidence_match = re.search(r'confidence[:\s]+([0-9.]+)', llm_response, re.IGNORECASE)
                    if confidence_match:
                        try:
                            confidence = float(confidence_match.group(1))
                            if confidence > 1.0:  # Normalize if on a different scale
                                confidence /= 10.0 if confidence <= 10.0 else 100.0
                        except ValueError:
                            pass
                    
                    # Try to extract reason
                    reason = "Not supported according to LLM verification"
                    reason_match = re.search(r'reason[:\s]+([^\n]+)', llm_response, re.IGNORECASE)
                    if reason_match:
                        reason = reason_match.group(1).strip()
                    
                    # Determine severity
                    severity = "medium"  # Default severity
                    if "contradicts" in llm_response.lower() or "contradicted" in llm_response.lower():
                        severity = "high"
                    elif "partially" in llm_response.lower() or "somewhat" in llm_response.lower():
                        severity = "low"
                    
                    results.append({
                        "is_supported": False,
                        "confidence": confidence,
                        "reason": reason,
                        "severity": severity
                    })
            except Exception as e:
                # Error in LLM verification
                results.append({
                    "is_supported": False,
                    "confidence": 0.5,
                    "reason": f"Error in LLM verification: {str(e)}",
                    "severity": "medium"
                })
        
        return results
    
    def _verify_with_keywords(self, 
                            statements: List[Dict[str, Any]], 
                            source_contents: List[str]) -> List[Dict[str, Any]]:
        """
        Verify statements using keyword matching with source documents.
        
        Args:
            statements: List of statements to verify
            source_contents: List of source document contents
            
        Returns:
            List of verification results
        """
        results = []
        
        # Combine all source contents
        combined_sources = " ".join(source_contents).lower()
        
        # Verify each statement
        for statement_info in statements:
            statement = statement_info["statement"]
            
            # Extract keywords from statement
            keywords = self._extract_keywords(statement)
            
            if not keywords:
                # No keywords found, consider not supported
                results.append({
                    "is_supported": False,
                    "confidence": 0.6,
                    "reason": "No significant keywords found for verification",
                    "severity": "low"
                })
                continue
            
            # Count how many keywords are found in sources
            found_keywords = [keyword for keyword in keywords if keyword.lower() in combined_sources]
            overlap_ratio = len(found_keywords) / len(keywords)
            
            # Check if statement is supported based on keyword overlap
            if overlap_ratio >= self.min_keywords_overlap:
                results.append({
                    "is_supported": True,
                    "confidence": min(0.5 + overlap_ratio / 2, 0.9),  # Scale to 0.5-0.9 range
                    "reason": f"Supported based on keyword overlap ({len(found_keywords)}/{len(keywords)} keywords)",
                    "severity": "none",
                    "keywords": keywords,
                    "found_keywords": found_keywords
                })
            else:
                # Determine severity based on overlap
                if overlap_ratio == 0:
                    severity = "high"
                elif overlap_ratio < 0.2:
                    severity = "medium"
                else:
                    severity = "low"
                
                results.append({
                    "is_supported": False,
                    "confidence": 0.6 + (1.0 - overlap_ratio) / 4,  # Scale to 0.6-0.85 range
                    "reason": f"Insufficient keyword overlap ({len(found_keywords)}/{len(keywords)} keywords)",
                    "severity": severity,
                    "keywords": keywords,
                    "found_keywords": found_keywords
                })
        
        return results
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract significant keywords from text for matching.
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            List of keywords
        """
        # This is a simplified implementation
        # In a real system, use NLP techniques like named entity recognition,
        # POS tagging, and keyword extraction algorithms
        
        # Remove common stop words
        stop_words = {
            "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
            "has", "have", "had", "be", "been", "being", "in", "on", "at", "by",
            "for", "with", "about", "against", "between", "into", "through",
            "during", "before", "after", "above", "below", "to", "from", "up",
            "down", "of", "off", "over", "under", "again", "further", "then",
            "once", "here", "there", "when", "where", "why", "how", "all", "any",
            "both", "each", "few", "more", "most", "other", "some", "such", "no",
            "nor", "not", "only", "own", "same", "so", "than", "too", "very",
            "s", "t", "can", "will", "just", "don", "should", "now", "d", "ll",
            "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn",
            "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn",
            "shan", "shouldn", "wasn", "weren", "won", "wouldn", "i", "you", "he",
            "she", "it", "we", "they", "what", "which", "who", "whom", "this", "that"
        }
        
        # Tokenize and filter
        words = []
        for word in re.findall(r'\b\w+\b', text.lower()):
            if (
                word not in stop_words and 
                len(word) > 2 and  # Skip very short words
                not word.isdigit()  # Skip pure numbers
            ):
                words.append(word)
        
        # Get named entities (a simplified approach)
        # Look for capitalized words that aren't at the start of sentences
        named_entities = []
        for match in re.finditer(r'(?<!^)(?<!\. )[A-Z][a-zA-Z]+', text):
            named_entities.append(match.group(0))
        
        # Add multi-word phrases (simplified)
        phrases = []
        for match in re.finditer(r'\b([A-Z][a-zA-Z]+ [A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b', text):
            phrases.append(match.group(0))
        
        # Combine all keywords
        all_keywords = words + named_entities + phrases
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in all_keywords:
            if keyword.lower() not in seen:
                seen.add(keyword.lower())
                unique_keywords.append(keyword)
        
        return unique_keywords
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity value
        """
        # Convert to numpy arrays if they aren't already
        a_array = np.array(a)
        b_array = np.array(b)
        
        # Calculate cosine similarity
        dot_product = np.dot(a_array, b_array)
        norm_a = np.linalg.norm(a_array)
        norm_b = np.linalg.norm(b_array)
        
        # Handle zero division
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return dot_product / (norm_a * norm_b)
    
    def _run_nli(self, premise: str, hypothesis: str) -> Dict[str, Any]:
        """
        Run NLI model to determine if premise entails, contradicts, or is neutral to hypothesis.
        
        This is a placeholder method. In a real implementation, this would use an actual NLI model.
        
        Args:
            premise: The source text (context)
            hypothesis: The statement to verify
            
        Returns:
            Dictionary with NLI prediction results
        """
        # This is a simplified placeholder implementation
        if self.nli_model:
            # In a real implementation, run the NLI model here
            # For now, return a default neutral result
            return {
                "label": "neutral",
                "score": 0.5
            }
        
        # Very simple keyword-based matching as fallback
        premise_lower = premise.lower()
        hypothesis_lower = hypothesis.lower()
        
        # Extract keywords from hypothesis
        keywords = self._extract_keywords(hypothesis)
        if not keywords:
            return {"label": "neutral", "score": 0.5}
        
        # Count how many keywords are found in premise
        found_keywords = [keyword for keyword in keywords if keyword.lower() in premise_lower]
        overlap_ratio = len(found_keywords) / len(keywords)
        
        # Look for negation words near keywords
        negation_words = ["not", "no", "never", "isn't", "aren't", "wasn't", "weren't", "doesn't", "don't", "didn't"]
        
        # Check for contradictions
        for keyword in keywords:
            for negation in negation_words:
                if negation + " " + keyword.lower() in premise_lower or keyword.lower() + " " + negation in premise_lower:
                    # Potential contradiction found
                    return {
                        "label": "contradiction",
                        "score": 0.6
                    }
        
        # Check for entailment based on keyword overlap
        if overlap_ratio >= 0.7:
            return {
                "label": "entailment",
                "score": 0.5 + overlap_ratio / 2  # Scale to 0.5-1.0 range
            }
        elif overlap_ratio >= 0.3:
            return {
                "label": "neutral",
                "score": 0.4 + overlap_ratio / 2  # Scale to 0.4-0.65 range
            }
        else:
            return {
                "label": "contradiction",
                "score": 0.3 + (1.0 - overlap_ratio) / 2  # Scale to 0.3-0.8 range
            }
    
    def _construct_verification_prompt(self, statement: str, source_content: str) -> str:
        """
        Construct a prompt for LLM-based source verification.
        
        Args:
            statement: Statement to verify
            source_content: Source content to verify against
            
        Returns:
            Formatted prompt for the LLM
        """
        prompt = "Please verify if the following statement is supported by the provided source information.\n\n"
        prompt += f"Statement to verify: \"{statement}\"\n\n"
        prompt += "Source information:\n```\n"
        prompt += source_content
        prompt += "\n```\n\n"
        prompt += "Is the statement supported by the source information? Answer with:\n"
        prompt += "- True/False (is the statement supported)\n"
        prompt += "- Confidence (0.0-1.0)\n"
        prompt += "- Reason (why you think it is or isn't supported)\n"
        
        return prompt
    
    def _default_embedder(self, text: str) -> List[float]:
        """
        Default text embedder that returns a placeholder embedding.
        
        This is used when no embedder is configured. In a real system,
        this would be replaced with an actual text embedding model.
        
        Args:
            text: The text to embed
            
        Returns:
            A placeholder embedding vector
        """
        # This is a placeholder. In a real system, this would use a real embedding model.
        # Generate a pseudorandom vector based on the text content
        import hashlib
        
        # Get hash of text
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Convert hash to a list of floats
        vector = []
        for i in range(0, len(text_hash), 2):
            if i + 1 < len(text_hash):
                # Convert two hex digits to a float between -1 and 1
                hex_pair = text_hash[i:i+2]
                value = int(hex_pair, 16) / 255.0 * 2.0 - 1.0
                vector.append(value)
        
        # Normalize the vector
        norm = np.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        
        return vector
    
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
        # Check if the prompt contains both "statement" and "source"
        statement_match = re.search(r'Statement to verify: "([^"]+)"', prompt)
        if not statement_match:
            return "Unable to find statement in prompt. False, confidence: 0.5, reason: Invalid prompt format."
        
        statement = statement_match.group(1).lower()
        
        # Extract source content
        source_match = re.search(r'Source information:\n```\n(.*?)\n```', prompt, re.DOTALL)
        if not source_match:
            return "Unable to find source information in prompt. False, confidence: 0.5, reason: Invalid prompt format."
        
        source = source_match.group(1).lower()
        
        # Simple keyword matching
        keywords = self._extract_keywords(statement)
        if not keywords:
            return "False, confidence: 0.6, reason: No significant keywords found for verification."
        
        # Count how many keywords are found in sources
        found_keywords = [keyword for keyword in keywords if keyword.lower() in source]
        overlap_ratio = len(found_keywords) / len(keywords) if keywords else 0
        
        if overlap_ratio >= 0.7:
            return f"True, confidence: {0.5 + overlap_ratio/2:.1f}, reason: Strong keyword overlap between statement and source."
        elif overlap_ratio >= 0.3:
            return f"False, confidence: {0.6 + (1-overlap_ratio)/4:.1f}, reason: Partial keyword overlap, but insufficient support."
        else:
            return f"False, confidence: 0.8, reason: Minimal or no keyword overlap between statement and source."
    
    def _generate_explanation(self, hallucinations: List[Dict[str, Any]]) -> str:
        """
        Generate a human-readable explanation of detection results.
        
        Args:
            hallucinations: List of detected hallucinations
            
        Returns:
            A formatted explanation string
        """
        if not hallucinations:
            return "All statements are supported by the provided source documents."
        
        explanation = f"Detected {len(hallucinations)} statements not supported by sources:\n"
        
        for i, h in enumerate(hallucinations, 1):
            explanation += f"{i}. \"{h['text']}\" - {h['reason']} "
            explanation += f"(Severity: {h['severity']}, Confidence: {h['confidence']:.2f})\n"
        
        return explanation.strip()
    
    def __repr__(self) -> str:
        """Return a string representation of the detector."""
        return f"SourceVerificationDetector(method={self.method})" 