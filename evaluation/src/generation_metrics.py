"""
Generation Metrics

This module provides metrics for evaluating the generation component of RAG systems.
"""

import re
import spacy
import numpy as np
from typing import List, Dict, Any, Union, Optional, Set, Tuple
from collections import Counter
import torch
from sentence_transformers import SentenceTransformer
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


class GenerationMetrics:
    """
    A class for evaluating the generation performance of RAG systems.
    """
    
    def __init__(
        self,
        use_semantic_similarity: bool = True,
        semantic_model_name: str = "all-MiniLM-L6-v2",
        use_spacy: bool = True,
        spacy_model: str = "en_core_web_md",
        faithfulness_threshold: float = 0.6,
    ):
        """
        Initialize the generation metrics calculator.
        
        Args:
            use_semantic_similarity: Whether to use semantic similarity
            semantic_model_name: Name of the sentence transformer model to use
            use_spacy: Whether to use spaCy for text processing
            spacy_model: Name of the spaCy model to use
            faithfulness_threshold: Threshold for considering a statement faithful
        """
        self.faithfulness_threshold = faithfulness_threshold
        
        # Initialize models if needed
        if use_semantic_similarity:
            try:
                self.semantic_model = SentenceTransformer(semantic_model_name)
                self.use_semantic_similarity = True
            except Exception as e:
                print(f"Warning: Could not load sentence transformer model: {e}")
                self.semantic_model = None
                self.use_semantic_similarity = False
        else:
            self.semantic_model = None
            self.use_semantic_similarity = False
        
        if use_spacy:
            try:
                self.nlp = spacy.load(spacy_model)
                self.use_spacy = True
            except Exception as e:
                print(f"Warning: Could not load spaCy model: {e}")
                self.nlp = None
                self.use_spacy = False
        else:
            self.nlp = None
            self.use_spacy = False
        
        # Initialize ROUGE
        try:
            self.rouge = Rouge()
            self.use_rouge = True
        except Exception as e:
            print(f"Warning: Could not initialize ROUGE: {e}")
            self.rouge = None
            self.use_rouge = False
    
    def faithfulness(
        self,
        generated_text: str,
        retrieved_contexts: List[str],
        method: str = "sentence_similarity",
    ) -> float:
        """
        Calculate faithfulness score, measuring how well the generation is supported
        by the retrieved contexts.
        
        Args:
            generated_text: The generated text to evaluate
            retrieved_contexts: List of retrieved context texts
            method: Method to use for faithfulness calculation
                ('sentence_similarity', 'entity_overlap', 'fact_coverage')
            
        Returns:
            Faithfulness score (0 to 1)
        """
        if not generated_text or not retrieved_contexts:
            return 0.0
        
        if method == "sentence_similarity" and self.use_semantic_similarity:
            return self._faithfulness_semantic(generated_text, retrieved_contexts)
        elif method == "entity_overlap" and self.use_spacy:
            return self._faithfulness_entity_overlap(generated_text, retrieved_contexts)
        else:
            # Fall back to token overlap
            return self._faithfulness_token_overlap(generated_text, retrieved_contexts)
    
    def _faithfulness_semantic(self, generated_text: str, retrieved_contexts: str) -> float:
        """Compute faithfulness using semantic similarity between sentences."""
        # Split generated text into sentences
        if self.use_spacy:
            doc = self.nlp(generated_text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        else:
            # Simple sentence splitting fallback
            sentences = re.split(r'[.!?]\s+', generated_text)
            sentences = [s.strip() + '.' for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        # Combine all contexts into one text
        combined_context = " ".join(retrieved_contexts)
        
        # Calculate similarity between each sentence and the context
        supported_sentences = 0
        
        # Encode the combined context once
        context_embedding = self.semantic_model.encode([combined_context])[0]
        
        # Encode all sentences at once for efficiency
        sentence_embeddings = self.semantic_model.encode(sentences)
        
        for sent_embedding in sentence_embeddings:
            # Calculate cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                torch.tensor(sent_embedding).unsqueeze(0),
                torch.tensor(context_embedding).unsqueeze(0),
                dim=1
            ).item()
            
            if similarity >= self.faithfulness_threshold:
                supported_sentences += 1
        
        return supported_sentences / len(sentences)
    
    def _faithfulness_entity_overlap(self, generated_text: str, retrieved_contexts: List[str]) -> float:
        """Compute faithfulness based on entity overlap."""
        # Extract entities from generated text
        gen_doc = self.nlp(generated_text)
        gen_entities = set([ent.text.lower() for ent in gen_doc.ents])
        
        if not gen_entities:
            return 0.0
        
        # Extract entities from contexts
        context_entities = set()
        for context in retrieved_contexts:
            context_doc = self.nlp(context)
            context_entities.update([ent.text.lower() for ent in context_doc.ents])
        
        # Calculate entity overlap
        if not context_entities:
            return 0.0
            
        overlap = len(gen_entities.intersection(context_entities))
        return overlap / len(gen_entities)
    
    def _faithfulness_token_overlap(self, generated_text: str, retrieved_contexts: List[str]) -> float:
        """Simple token overlap for faithfulness calculation."""
        # Tokenize generated text
        gen_tokens = set(generated_text.lower().split())
        
        # Remove common words
        common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "of", "for", "with", "by", "is", "are", "was", "were"}
        gen_tokens = gen_tokens - common_words
        
        if not gen_tokens:
            return 0.0
        
        # Tokenize contexts
        context_tokens = set()
        for context in retrieved_contexts:
            context_tokens.update(context.lower().split())
        
        context_tokens = context_tokens - common_words
        
        if not context_tokens:
            return 0.0
            
        # Calculate token overlap
        overlap = len(gen_tokens.intersection(context_tokens))
        return overlap / len(gen_tokens)
    
    def answer_relevance(
        self,
        query: str,
        answer: str,
        method: str = "semantic",
    ) -> float:
        """
        Calculate how relevant the answer is to the query.
        
        Args:
            query: The query text
            answer: The generated answer
            method: Method to use for relevance calculation ('semantic', 'overlap')
            
        Returns:
            Relevance score (0 to 1)
        """
        if not query or not answer:
            return 0.0
        
        if method == "semantic" and self.use_semantic_similarity:
            # Use semantic similarity
            query_embed = self.semantic_model.encode([query])[0]
            answer_embed = self.semantic_model.encode([answer])[0]
            
            similarity = torch.nn.functional.cosine_similarity(
                torch.tensor(query_embed).unsqueeze(0),
                torch.tensor(answer_embed).unsqueeze(0),
                dim=1
            ).item()
            
            return similarity
        else:
            # Fall back to token overlap
            if self.use_spacy:
                query_doc = self.nlp(query)
                answer_doc = self.nlp(answer)
                
                query_tokens = set([token.lemma_.lower() for token in query_doc 
                                   if not token.is_stop and not token.is_punct])
                answer_tokens = set([token.lemma_.lower() for token in answer_doc 
                                    if not token.is_stop and not token.is_punct])
            else:
                query_tokens = set(query.lower().split())
                answer_tokens = set(answer.lower().split())
            
            if not query_tokens or not answer_tokens:
                return 0.0
                
            # Calculate Jaccard similarity
            overlap = len(query_tokens.intersection(answer_tokens))
            union = len(query_tokens.union(answer_tokens))
            
            return overlap / union
    
    def detect_hallucinations(
        self,
        generated_text: str,
        retrieved_contexts: List[str],
    ) -> Dict[str, Any]:
        """
        Detect potential hallucinations in the generated text.
        
        Args:
            generated_text: The generated text to evaluate
            retrieved_contexts: List of retrieved context texts
            
        Returns:
            Dictionary with hallucination metrics
        """
        if not generated_text or not retrieved_contexts:
            return {
                "has_hallucinations": False,
                "hallucination_score": 0.0,
                "potential_hallucinations": []
            }
        
        # Split generated text into sentences
        if self.use_spacy:
            doc = self.nlp(generated_text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        else:
            # Simple sentence splitting fallback
            sentences = re.split(r'[.!?]\s+', generated_text)
            sentences = [s.strip() + '.' for s in sentences if s.strip()]
        
        if not sentences:
            return {
                "has_hallucinations": False,
                "hallucination_score": 0.0,
                "potential_hallucinations": []
            }
        
        # Combine all contexts into one text
        combined_context = " ".join(retrieved_contexts)
        
        potential_hallucinations = []
        supported_sentences = 0
        
        # Process generated text differently depending on available models
        if self.use_semantic_similarity and self.semantic_model:
            # Encode the combined context once
            context_embedding = self.semantic_model.encode([combined_context])[0]
            
            # Encode all sentences at once for efficiency
            sentence_embeddings = self.semantic_model.encode(sentences)
            
            for i, sent_embedding in enumerate(sentence_embeddings):
                # Calculate cosine similarity
                similarity = torch.nn.functional.cosine_similarity(
                    torch.tensor(sent_embedding).unsqueeze(0),
                    torch.tensor(context_embedding).unsqueeze(0),
                    dim=1
                ).item()
                
                if similarity < self.faithfulness_threshold:
                    potential_hallucinations.append({
                        "sentence": sentences[i],
                        "support_score": similarity
                    })
                else:
                    supported_sentences += 1
        
        else:
            # Fall back to token overlap for hallucination detection
            context_tokens = set(combined_context.lower().split())
            
            for sentence in sentences:
                sentence_tokens = set(sentence.lower().split())
                overlap_ratio = len(sentence_tokens.intersection(context_tokens)) / len(sentence_tokens) if sentence_tokens else 0
                
                if overlap_ratio < 0.5:  # Simple threshold for token overlap
                    potential_hallucinations.append({
                        "sentence": sentence,
                        "support_score": overlap_ratio
                    })
                else:
                    supported_sentences += 1
        
        # Calculate overall hallucination score
        hallucination_score = len(potential_hallucinations) / len(sentences) if sentences else 0.0
        
        return {
            "has_hallucinations": hallucination_score > 0.0,
            "hallucination_score": hallucination_score,
            "supported_ratio": supported_sentences / len(sentences) if sentences else 0.0,
            "potential_hallucinations": potential_hallucinations
        }
    
    def factuality(
        self,
        generated_text: str,
        reference_text: Optional[str] = None,
        retrieved_contexts: Optional[List[str]] = None,
    ) -> float:
        """
        Evaluate factuality of generated text.
        
        Args:
            generated_text: The generated text to evaluate
            reference_text: Optional reference text for comparison
            retrieved_contexts: Optional list of retrieved context texts
            
        Returns:
            Factuality score (0 to 1)
        """
        if not generated_text:
            return 0.0
        
        # If we have a reference text, compare against it
        if reference_text:
            if self.use_rouge and self.rouge:
                try:
                    # Calculate ROUGE-L score as an approximation of factual overlap
                    scores = self.rouge.get_scores(generated_text, reference_text)[0]
                    return scores['rouge-l']['f']
                except Exception as e:
                    print(f"Warning: ROUGE calculation failed: {e}")
            
            # Fall back to BLEU if ROUGE fails
            try:
                reference_tokens = reference_text.split()
                generated_tokens = generated_text.split()
                
                smoothing = SmoothingFunction().method1
                bleu_score = sentence_bleu([reference_tokens], generated_tokens, smoothing_function=smoothing)
                
                return bleu_score
            except Exception as e:
                print(f"Warning: BLEU calculation failed: {e}")
        
        # If we have retrieved contexts but no reference, use faithfulness as proxy for factuality
        if retrieved_contexts:
            return self.faithfulness(generated_text, retrieved_contexts)
        
        # If neither reference nor contexts are available, we can't assess factuality
        return 0.0
    
    def conciseness(self, generated_text: str, query: str) -> float:
        """
        Evaluate the conciseness of the generated text.
        
        Args:
            generated_text: The generated text to evaluate
            query: The original query
            
        Returns:
            Conciseness score (0 to 1)
        """
        if not generated_text or not query:
            return 0.0
        
        # Simple length-based conciseness
        query_length = len(query.split())
        answer_length = len(generated_text.split())
        
        # Calculate a conciseness score based on length ratio
        # Ideal answer should be 1-3x the query length
        if answer_length <= query_length:
            # Too short
            return 0.5
        elif answer_length <= 3 * query_length:
            # Ideal length
            return 1.0
        else:
            # Longer answers get reduced scores
            return max(0.0, 1.0 - (answer_length - 3 * query_length) / (10 * query_length))
    
    def coherence(self, generated_text: str) -> float:
        """
        Evaluate the coherence of the generated text.
        
        Args:
            generated_text: The generated text to evaluate
            
        Returns:
            Coherence score (0 to 1)
        """
        if not generated_text:
            return 0.0
        
        # Simple sentence-level coherence based on language model embeddings
        if self.use_semantic_similarity and self.semantic_model:
            # Split into sentences
            if self.use_spacy:
                doc = self.nlp(generated_text)
                sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            else:
                sentences = re.split(r'[.!?]\s+', generated_text)
                sentences = [s.strip() + '.' for s in sentences if s.strip()]
            
            if len(sentences) <= 1:
                return 1.0  # Single sentence is coherent by definition
            
            # Calculate embeddings for all sentences
            sentence_embeddings = self.semantic_model.encode(sentences)
            
            # Calculate cosine similarity between adjacent sentences
            similarity_scores = []
            for i in range(len(sentences) - 1):
                similarity = torch.nn.functional.cosine_similarity(
                    torch.tensor(sentence_embeddings[i]).unsqueeze(0),
                    torch.tensor(sentence_embeddings[i+1]).unsqueeze(0),
                    dim=1
                ).item()
                similarity_scores.append(similarity)
            
            # Average similarity as coherence score
            return sum(similarity_scores) / len(similarity_scores)
        else:
            # Fall back to a very simplistic coherence metric
            return min(1.0, max(0.0, 0.5 + 0.1 * len(generated_text.split()) / 100))
    
    def evaluate_all(
        self,
        queries: List[str],
        generated_texts: List[str],
        retrieved_contexts_list: List[List[str]],
        reference_texts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate generation using multiple metrics.
        
        Args:
            queries: List of query texts
            generated_texts: List of generated texts to evaluate
            retrieved_contexts_list: List of retrieved contexts for each query
            reference_texts: Optional list of reference texts
            
        Returns:
            Dictionary of evaluation results
        """
        results = {
            "faithfulness": [],
            "answer_relevance": [],
            "hallucination": [],
            "factuality": [],
            "conciseness": [],
            "coherence": [],
            "overall_scores": {}
        }
        
        # Make sure reference_texts is the same length if provided
        if reference_texts is not None and len(reference_texts) != len(queries):
            reference_texts = None
        
        # Calculate metrics for each query
        for i, (query, generated_text, retrieved_contexts) in enumerate(zip(queries, generated_texts, retrieved_contexts_list)):
            # Get reference text if available
            reference_text = reference_texts[i] if reference_texts else None
            
            # Calculate metrics
            faithfulness_score = self.faithfulness(generated_text, retrieved_contexts)
            relevance_score = self.answer_relevance(query, generated_text)
            hallucination_results = self.detect_hallucinations(generated_text, retrieved_contexts)
            factuality_score = self.factuality(generated_text, reference_text, retrieved_contexts)
            conciseness_score = self.conciseness(generated_text, query)
            coherence_score = self.coherence(generated_text)
            
            # Store individual results
            results["faithfulness"].append({
                "query": query,
                "score": faithfulness_score
            })
            
            results["answer_relevance"].append({
                "query": query,
                "score": relevance_score
            })
            
            results["hallucination"].append({
                "query": query,
                "has_hallucinations": hallucination_results["has_hallucinations"],
                "hallucination_score": hallucination_results["hallucination_score"],
                "potential_hallucinations": hallucination_results["potential_hallucinations"][:3]  # Limit to 3 examples
            })
            
            results["factuality"].append({
                "query": query,
                "score": factuality_score
            })
            
            results["conciseness"].append({
                "query": query,
                "score": conciseness_score
            })
            
            results["coherence"].append({
                "query": query,
                "score": coherence_score
            })
        
        # Calculate overall scores
        if results["faithfulness"]:
            results["overall_scores"]["faithfulness"] = sum(item["score"] for item in results["faithfulness"]) / len(results["faithfulness"])
        
        if results["answer_relevance"]:
            results["overall_scores"]["answer_relevance"] = sum(item["score"] for item in results["answer_relevance"]) / len(results["answer_relevance"])
        
        if results["hallucination"]:
            results["overall_scores"]["hallucination"] = sum(item["hallucination_score"] for item in results["hallucination"]) / len(results["hallucination"])
        
        if results["factuality"]:
            results["overall_scores"]["factuality"] = sum(item["score"] for item in results["factuality"]) / len(results["factuality"])
        
        if results["conciseness"]:
            results["overall_scores"]["conciseness"] = sum(item["score"] for item in results["conciseness"]) / len(results["conciseness"])
        
        if results["coherence"]:
            results["overall_scores"]["coherence"] = sum(item["score"] for item in results["coherence"]) / len(results["coherence"])
        
        # Overall aggregated score (weighted average)
        weights = {
            "faithfulness": 0.3,
            "answer_relevance": 0.25,
            "hallucination": 0.2,
            "factuality": 0.15,
            "coherence": 0.1,
            "conciseness": 0.0  # Not included in overall score
        }
        
        overall_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric == "hallucination":
                # Invert hallucination score since lower is better
                if metric in results["overall_scores"]:
                    overall_score += (1.0 - results["overall_scores"][metric]) * weight
                    total_weight += weight
            else:
                if metric in results["overall_scores"]:
                    overall_score += results["overall_scores"][metric] * weight
                    total_weight += weight
        
        if total_weight > 0:
            results["overall_scores"]["overall"] = overall_score / total_weight
        
        return results 