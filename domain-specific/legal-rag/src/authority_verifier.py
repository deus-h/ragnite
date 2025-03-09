"""
Legal Authority Verifier

This module verifies legal claims against authoritative sources, assesses the level of
legal authority, and provides information about the strength of legal evidence.
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple

from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate

class LegalAuthorityVerifier:
    """
    Verifies legal claims against authoritative sources and assesses the level
    of legal authority for those claims.
    """

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        confidence_threshold: float = 0.7,
        strict_mode: bool = False
    ):
        """
        Initialize the legal authority verifier.

        Args:
            model_name (str): The name of the LLM to use
            temperature (float): Temperature setting for the LLM
            confidence_threshold (float): Threshold for claim verification confidence
            strict_mode (bool): If True, requires higher confidence and more support
        """
        self.model_name = model_name
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold
        self.strict_mode = strict_mode

        # Set up LLM
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        
        # Legal authority hierarchy
        self.authority_levels = {
            "primary_binding": 5,  # Supreme Court, statutes in jurisdiction
            "primary_persuasive": 4,  # Out-of-jurisdiction high courts
            "secondary_authoritative": 3,  # Restatements, major treatises
            "secondary_persuasive": 2,  # Law reviews, respected commentaries
            "tertiary": 1,  # Legal dictionaries, general reference
            "non_legal": 0   # Non-legal sources
        }
        
        # Prompts for legal claim extraction and verification
        self.claim_extraction_prompt = ChatPromptTemplate.from_template(
            """You are a legal expert tasked with identifying specific legal claims from the text below.
            A legal claim is an assertion about what the law is, how it should be interpreted, 
            or how it applies to a specific situation.
            
            For each legal claim, extract:
            1. The exact claim statement
            2. The legal domain it falls under (e.g., constitutional law, contract law, etc.)
            3. Any specific legal concepts or principles mentioned
            4. Any authorities cited in support of the claim
            
            Present each claim separately with the above information.
            
            TEXT:
            {text}
            
            LEGAL CLAIMS:"""
        )
        
        self.verification_prompt = ChatPromptTemplate.from_template(
            """You are a legal expert tasked with verifying a legal claim against provided legal authorities.
            
            LEGAL CLAIM:
            {claim}
            
            LEGAL AUTHORITIES PROVIDED:
            {context}
            
            Please analyze whether and to what extent the claim is supported by the provided authorities.
            
            For each authority that relates to the claim:
            1. Identify the authority and its level (primary binding, primary persuasive, secondary, etc.)
            2. Explain how it supports or contradicts the claim
            3. Note any limitations or qualifications to its support
            
            Then provide:
            1. An overall assessment of whether the claim is:
               - Fully supported: Strong, directly applicable authorities support the claim
               - Partially supported: Some support exists but with significant limitations or qualifications
               - Unsupported: No substantial support in the provided authorities
               - Contradicted: Authorities directly oppose the claim
            
            2. A confidence score from 0.0 to 1.0 reflecting how well supported the claim is
            
            3. A suggested revision of the claim if needed to make it more accurate
            
            Present your analysis in a structured format with clear headings.
            
            VERIFICATION ANALYSIS:"""
        )
        
        self.authority_assessment_prompt = ChatPromptTemplate.from_template(
            """You are a legal expert tasked with assessing the weight of legal authority for the following sources.
            For each source, classify it according to this hierarchy:
            
            - Primary Binding: Controlling law in the relevant jurisdiction (e.g., US Supreme Court for federal issues)
            - Primary Persuasive: Primary authority from other jurisdictions (e.g., other circuit courts)
            - Secondary Authoritative: Restatements, major treatises, uniform laws
            - Secondary Persuasive: Law reviews, respected commentaries 
            - Tertiary: Legal dictionaries, encyclopedias, general reference works
            - Non-legal: Sources without legal authority
            
            For each authority, indicate:
            1. Its classification in the hierarchy
            2. The jurisdiction it applies to (if relevant)
            3. Its recency/currency
            4. Any particular weight it should be given (e.g., unanimous Supreme Court decision)
            
            LEGAL AUTHORITIES:
            {authorities}
            
            AUTHORITY ASSESSMENT:"""
        )

    def extract_legal_claims(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract legal claims from text.

        Args:
            text (str): The text to analyze

        Returns:
            List[Dict[str, Any]]: A list of extracted legal claims
        """
        # Create the extraction chain
        extraction_chain = LLMChain(llm=self.llm, prompt=self.claim_extraction_prompt)
        
        # Run the extraction
        result = extraction_chain.run(text=text)
        
        # Process the result into structured claims
        claims = []
        current_claim = {}
        
        for line in result.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("Claim") or (current_claim and len(current_claim) > 0 and line[0].isdigit() and line[1] == '.'):
                # Save previous claim if it exists
                if current_claim and "statement" in current_claim:
                    claims.append(current_claim)
                    
                # Start a new claim
                current_claim = {"statement": ""}
                
                # Try to extract the claim statement
                if ":" in line:
                    current_claim["statement"] = line.split(":", 1)[1].strip()
            elif "Domain:" in line or "Legal Domain:" in line:
                current_claim["domain"] = line.split(":", 1)[1].strip()
            elif "Concepts:" in line or "Legal Concepts:" in line:
                current_claim["concepts"] = [c.strip() for c in line.split(":", 1)[1].split(",")]
            elif "Authorities:" in line or "Cited Authorities:" in line:
                authorities = line.split(":", 1)[1].strip()
                if authorities:
                    current_claim["cited_authorities"] = [a.strip() for a in authorities.split(",")]
                else:
                    current_claim["cited_authorities"] = []
            elif current_claim and "statement" in current_claim:
                # Append to the current claim statement if it's a continuation
                if not any(key in line.lower() for key in ["domain:", "concepts:", "authorities:"]):
                    current_claim["statement"] += " " + line
        
        # Add the last claim if it exists
        if current_claim and "statement" in current_claim:
            claims.append(current_claim)
            
        # Add unique identifier to each claim
        for i, claim in enumerate(claims):
            claim["id"] = f"claim_{i+1}"
            
        return claims

    def verify_legal_claim(self, claim: Dict[str, Any], context_docs: List[Document]) -> Dict[str, Any]:
        """
        Verify a legal claim against provided documents.

        Args:
            claim (Dict[str, Any]): The claim to verify
            context_docs (List[Document]): Documents to verify against

        Returns:
            Dict[str, Any]: Verification results
        """
        # Prepare the context from the documents
        context_text = "\n\n".join([f"SOURCE {i+1}:\n{doc.page_content}" for i, doc in enumerate(context_docs)])
        
        # Create the verification chain
        verification_chain = LLMChain(llm=self.llm, prompt=self.verification_prompt)
        
        # Run the verification
        claim_text = claim["statement"]
        if "domain" in claim:
            claim_text += f"\nDomain: {claim['domain']}"
        if "concepts" in claim:
            claim_text += f"\nConcepts: {', '.join(claim['concepts'])}"
        if "cited_authorities" in claim:
            claim_text += f"\nCited Authorities: {', '.join(claim['cited_authorities'])}"
            
        result = verification_chain.run(claim=claim_text, context=context_text)
        
        # Parse the verification result
        verification_result = {
            "claim_id": claim["id"],
            "claim": claim["statement"],
            "raw_analysis": result,
            "sources": []
        }
        
        # Extract verification status
        if "fully supported" in result.lower():
            verification_result["status"] = "fully_supported"
        elif "partially supported" in result.lower():
            verification_result["status"] = "partially_supported"
        elif "contradicted" in result.lower():
            verification_result["status"] = "contradicted"
        else:
            verification_result["status"] = "unsupported"
            
        # Extract confidence score
        confidence_match = re.search(r"confidence score:?\s*(0\.\d+|1\.0|1)", result.lower())
        if confidence_match:
            verification_result["confidence"] = float(confidence_match.group(1))
        else:
            # Default confidence based on status
            if verification_result["status"] == "fully_supported":
                verification_result["confidence"] = 0.9
            elif verification_result["status"] == "partially_supported":
                verification_result["confidence"] = 0.6
            elif verification_result["status"] == "contradicted":
                verification_result["confidence"] = 0.2
            else:
                verification_result["confidence"] = 0.1
                
        # Extract revised claim if available
        revised_match = re.search(r"suggested revision:?\s*(.+?)(?:\n\n|\Z)", result, re.DOTALL | re.IGNORECASE)
        if revised_match:
            verification_result["revised_claim"] = revised_match.group(1).strip()
            
        # Extract supporting sources
        for i, doc in enumerate(context_docs):
            if f"SOURCE {i+1}" in result:
                source_info = {
                    "document_id": getattr(doc, "id", f"doc_{i+1}"),
                    "source": doc.metadata.get("source", "Unknown source"),
                    "supports_claim": "supports" in result.lower() and f"SOURCE {i+1}" in result
                }
                verification_result["sources"].append(source_info)
                
        return verification_result

    def assess_authority_levels(self, verified_claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Assess the level of legal authority for verified claims.

        Args:
            verified_claims (List[Dict[str, Any]]): Claims with verification info

        Returns:
            List[Dict[str, Any]]: Claims with authority assessment
        """
        for claim in verified_claims:
            # Skip claims with no sources
            if not claim.get("sources"):
                claim["authority_level"] = "no_authority"
                claim["authority_score"] = 0.0
                continue
                
            # Collect all authorities supporting this claim
            authorities = []
            for source in claim["sources"]:
                if source.get("supports_claim", False):
                    # Try to extract authority information from source metadata
                    if "metadata" in source:
                        if "jurisdiction" in source["metadata"]:
                            authorities.append(f"{source['source']} ({source['metadata']['jurisdiction']})")
                        else:
                            authorities.append(source["source"])
                    else:
                        authorities.append(source["source"])
            
            if not authorities:
                claim["authority_level"] = "no_authority"
                claim["authority_score"] = 0.0
                continue
                
            # Use LLM to assess authority levels
            authorities_text = "\n".join(authorities)
            assessment_chain = LLMChain(llm=self.llm, prompt=self.authority_assessment_prompt)
            assessment = assessment_chain.run(authorities=authorities_text)
            
            # Try to extract the highest authority level mentioned
            authority_levels = []
            for level_name, level_score in self.authority_levels.items():
                if level_name.replace("_", " ") in assessment.lower():
                    authority_levels.append((level_name, level_score))
            
            if authority_levels:
                # Get the highest authority level
                highest_authority = max(authority_levels, key=lambda x: x[1])
                claim["authority_level"] = highest_authority[0]
                claim["authority_score"] = highest_authority[1] / 5.0  # Normalize to 0-1
            else:
                claim["authority_level"] = "unknown"
                claim["authority_score"] = 0.3
                
            # Add the raw assessment
            claim["authority_assessment"] = assessment
            
        return verified_claims

    def verify_legal_content(self, text: str, context_docs: List[Document]) -> Dict[str, Any]:
        """
        Extract and verify legal claims from text against provided documents.

        Args:
            text (str): The text containing legal claims
            context_docs (List[Document]): Documents to verify against

        Returns:
            Dict[str, Any]: Verification results
        """
        # Extract claims
        claims = self.extract_legal_claims(text)
        
        if not claims:
            return {
                "verified": False,
                "reason": "No legal claims identified",
                "overall_confidence": 0.0,
                "claims": []
            }
            
        # Verify each claim
        verified_claims = []
        for claim in claims:
            verification = self.verify_legal_claim(claim, context_docs)
            verified_claims.append(verification)
            
        # Assess authority levels
        verified_claims = self.assess_authority_levels(verified_claims)
        
        # Calculate overall verification status
        confidence_scores = [claim["confidence"] for claim in verified_claims]
        authority_scores = [claim.get("authority_score", 0.0) for claim in verified_claims]
        
        # Combine confidence and authority
        combined_scores = [(c * 0.7 + a * 0.3) for c, a in zip(confidence_scores, authority_scores)]
        overall_confidence = sum(combined_scores) / len(combined_scores) if combined_scores else 0.0
        
        # Determine if the text is verified based on confidence threshold
        threshold = self.confidence_threshold
        if self.strict_mode:
            threshold += 0.1
            
        return {
            "verified": overall_confidence >= threshold,
            "overall_confidence": overall_confidence,
            "claims": verified_claims,
            "verification_threshold": threshold
        }

    def generate_verification_report(self, verification_result: Dict[str, Any]) -> str:
        """
        Generate a human-readable report of the verification results.

        Args:
            verification_result (Dict[str, Any]): Verification results

        Returns:
            str: A formatted report
        """
        report = []
        report.append("# Legal Verification Report")
        report.append("")
        
        # Overall verification
        status = "✓ Verified" if verification_result["verified"] else "✗ Not Verified"
        report.append(f"## Overall: {status}")
        report.append(f"Confidence: {verification_result['overall_confidence']:.2f}")
        report.append("")
        
        # List of claims
        report.append("## Claims Analysis")
        
        for i, claim in enumerate(verification_result["claims"]):
            report.append(f"### Claim {i+1}: {claim['claim']}")
            
            # Status and confidence
            status_map = {
                "fully_supported": "✓ Fully Supported",
                "partially_supported": "⚠ Partially Supported",
                "unsupported": "✗ Unsupported",
                "contradicted": "✗ Contradicted"
            }
            status = status_map.get(claim["status"], "? Unknown")
            report.append(f"**Status**: {status}")
            report.append(f"**Confidence**: {claim['confidence']:.2f}")
            
            # Authority level
            if "authority_level" in claim:
                authority_level = claim["authority_level"].replace("_", " ").title()
                report.append(f"**Authority Level**: {authority_level}")
            
            # Supporting sources
            if claim.get("sources"):
                report.append("\n**Supporting Sources**:")
                for source in claim["sources"]:
                    if source.get("supports_claim", False):
                        report.append(f"- {source['source']}")
            
            # Revised claim
            if "revised_claim" in claim:
                report.append("\n**Suggested Revision**:")
                report.append(f"> {claim['revised_claim']}")
                
            report.append("")
            
        return "\n".join(report)

    def generate_citation_footnotes(self, verification_result: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
        """
        Generate footnotes with citations for verified legal content.

        Args:
            verification_result (Dict[str, Any]): Verification results

        Returns:
            Dict[str, List[Dict[str, str]]]: Citations for each claim
        """
        citations = {}
        
        for claim in verification_result["claims"]:
            claim_id = claim["claim_id"]
            claim_citations = []
            
            for source in claim.get("sources", []):
                if source.get("supports_claim", False):
                    citation = {
                        "text": source["source"],
                        "confidence": claim["confidence"]
                    }
                    claim_citations.append(citation)
                    
            if claim_citations:
                citations[claim_id] = claim_citations
                
        return citations


def verify_legal_text(text: str, context_docs: List[Document]) -> Dict[str, Any]:
    """
    Helper function to verify legal text against context documents.

    Args:
        text (str): The legal text to verify
        context_docs (List[Document]): Documents to verify against

    Returns:
        Dict[str, Any]: Verification results
    """
    verifier = LegalAuthorityVerifier()
    return verifier.verify_legal_content(text, context_docs)


def generate_legal_report(verification_result: Dict[str, Any]) -> str:
    """
    Helper function to generate a legal verification report.

    Args:
        verification_result (Dict[str, Any]): Verification results

    Returns:
        str: A formatted report
    """
    verifier = LegalAuthorityVerifier()
    return verifier.generate_verification_report(verification_result) 