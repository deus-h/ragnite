"""
Medical Fact Verifier

This module provides functionality to verify medical facts, extract claims,
assess claim validity, and generate citations for medical information.
"""

import re
from typing import List, Dict, Any, Optional, Tuple, Union

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.docstore.document import Document


class MedicalFactVerifier:
    """
    A component that verifies medical facts and provides citations.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        confidence_threshold: float = 0.7,
        strict_mode: bool = False
    ):
        """
        Initialize the medical fact verifier.
        
        Args:
            model_name: Name of the LLM to use.
            temperature: Temperature for the LLM.
            confidence_threshold: Threshold for confidence score to determine if a claim is verifiable.
            strict_mode: If True, only include claims that can be verified with high confidence.
        """
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.confidence_threshold = confidence_threshold
        self.strict_mode = strict_mode
        
        # Track metrics
        self.verified_claims = 0
        self.unverified_claims = 0
        self.modified_claims = 0
        
        # Define the prompt template for claim extraction
        self.claim_extraction_template = PromptTemplate(
            input_variables=["text"],
            template="""
            You are a medical expert tasked with identifying medical claims from text.
            
            Text: {text}
            
            Extract all medical claims from the text. A medical claim is a statement that asserts something about:
            - Disease causes, symptoms, or progression
            - Treatment efficacy or side effects
            - Diagnostic procedures or criteria
            - Epidemiological statistics
            - Physiological mechanisms
            - Recommendations for clinical practice
            
            For each claim, provide:
            1. The exact claim (verbatim from the text)
            2. The type of claim (diagnosis, treatment, prognosis, etc.)
            
            Format your response as a numbered list of claims:
            1. Claim: [exact text] | Type: [type]
            2. Claim: [exact text] | Type: [type]
            
            Only extract factual claims, not opinions or hypotheticals.
            """,
        )
        
        self.claim_extraction_chain = LLMChain(llm=self.llm, prompt=self.claim_extraction_template)
        
        # Define the prompt template for claim verification
        self.claim_verification_template = PromptTemplate(
            input_variables=["claim", "claim_type", "context"],
            template="""
            You are a medical professional verifying the accuracy of medical claims based on provided context.
            
            Claim to verify: {claim}
            Type of claim: {claim_type}
            
            Context from medical literature:
            {context}
            
            For the above claim, provide:
            
            1. Verification Status (choose one):
               - VERIFIED: The claim is fully supported by the context
               - PARTIALLY VERIFIED: The claim is partially supported, with some elements not covered in the context
               - UNVERIFIED: The context doesn't provide clear evidence about this claim
               - CONTRADICTED: The context contradicts this claim
            
            2. Confidence Score (0.0 to 1.0):
               - 0.0 means no confidence in verification
               - 1.0 means complete confidence in verification
            
            3. Evidence:
               Quote the specific portions of the context that support or contradict the claim
            
            4. Citation:
               Provide a citation that would be appropriate for this claim, based on the context
            
            5. Corrected Claim (if needed):
               If the claim is partially verified or contradicted, provide a corrected version
            
            Format:
            Verification Status: [status]
            Confidence Score: [score]
            Evidence: [evidence]
            Citation: [citation]
            Corrected Claim: [corrected version if needed, otherwise "No correction needed"]
            """,
        )
        
        self.claim_verification_chain = LLMChain(llm=self.llm, prompt=self.claim_verification_template)
        
        # Define the prompt template for evidence assessment
        self.evidence_assessment_template = PromptTemplate(
            input_variables=["verification_results"],
            template="""
            You are a medical expert assessing the level of evidence for medical claims.
            
            Below are verification results for several medical claims:
            
            {verification_results}
            
            For each verification result, assign an appropriate level of evidence classification:
            
            1. Level I: Evidence from systematic review or meta-analysis of RCTs
            2. Level II: Evidence from well-designed RCTs
            3. Level III: Evidence from well-designed non-randomized controlled trials
            4. Level IV: Evidence from well-designed case-control or cohort studies
            5. Level V: Evidence from systematic reviews of descriptive and qualitative studies
            6. Level VI: Evidence from single descriptive or qualitative studies
            7. Level VII: Evidence from the opinion of authorities and/or reports of expert committees
            8. Insufficient: The evidence provided isn't sufficient to determine a level
            
            Format your response as a numbered list corresponding to each verification result:
            1. Evidence Level: [level] | Rationale: [brief explanation]
            2. Evidence Level: [level] | Rationale: [brief explanation]
            
            If there's not enough information to determine a level for a particular claim, state that.
            """,
        )
        
        self.evidence_assessment_chain = LLMChain(llm=self.llm, prompt=self.evidence_assessment_template)

    def extract_claims(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract medical claims from text.
        
        Args:
            text: The text to extract claims from.
            
        Returns:
            List of dictionaries containing claims and their types.
        """
        # Get claims from the LLM
        extraction_result = self.claim_extraction_chain.run(text=text)
        
        # Parse the result
        claims = []
        
        # Look for numbered claims like "1. Claim: X | Type: Y"
        claim_pattern = r'(\d+)\.\s+Claim:\s+(.*?)\s+\|\s+Type:\s+(.*?)(?=\n\d+\.|$)'
        for match in re.finditer(claim_pattern, extraction_result, re.DOTALL):
            claim_num = match.group(1)
            claim_text = match.group(2).strip()
            claim_type = match.group(3).strip()
            
            if claim_text:
                claims.append({
                    "claim": claim_text,
                    "type": claim_type,
                    "verified": False,
                    "confidence": 0.0,
                    "evidence": "",
                    "citation": "",
                    "corrected_claim": "",
                    "evidence_level": ""
                })
        
        return claims

    def verify_claim(self, claim: Dict[str, Any], context_docs: List[Document]) -> Dict[str, Any]:
        """
        Verify a medical claim against provided context documents.
        
        Args:
            claim: The claim to verify.
            context_docs: The context documents to verify against.
            
        Returns:
            The updated claim dictionary with verification information.
        """
        # Combine document texts into a single context string
        context = "\n\n".join([
            f"Source: {doc.metadata.get('source', 'Unknown')}\n" +
            (f"Section: {doc.metadata.get('section_name', 'Unknown')}\n" if 'section_name' in doc.metadata else "") +
            doc.page_content 
            for doc in context_docs
        ])
        
        # Get verification from the LLM
        verification_result = self.claim_verification_chain.run(
            claim=claim["claim"],
            claim_type=claim["type"],
            context=context
        )
        
        # Parse the result
        verification_status = "UNVERIFIED"
        confidence_score = 0.0
        evidence = ""
        citation = ""
        corrected_claim = ""
        
        # Extract each field
        for line in verification_result.strip().split("\n"):
            if line.startswith("Verification Status:"):
                verification_status = line.replace("Verification Status:", "").strip()
            elif line.startswith("Confidence Score:"):
                try:
                    confidence_score = float(line.replace("Confidence Score:", "").strip())
                except ValueError:
                    confidence_score = 0.0
            elif line.startswith("Evidence:"):
                evidence = line.replace("Evidence:", "").strip()
            elif line.startswith("Citation:"):
                citation = line.replace("Citation:", "").strip()
            elif line.startswith("Corrected Claim:"):
                corrected_claim = line.replace("Corrected Claim:", "").strip()
        
        # Update metrics
        if verification_status == "VERIFIED" or verification_status == "PARTIALLY VERIFIED":
            self.verified_claims += 1
        else:
            self.unverified_claims += 1
            
        if corrected_claim and corrected_claim != "No correction needed":
            self.modified_claims += 1
        
        # Update the claim dictionary
        claim.update({
            "verified": verification_status == "VERIFIED",
            "verification_status": verification_status,
            "confidence": confidence_score,
            "evidence": evidence,
            "citation": citation,
            "corrected_claim": corrected_claim if corrected_claim != "No correction needed" else ""
        })
        
        return claim

    def assess_evidence_levels(self, verified_claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Assess the level of evidence for verified claims.
        
        Args:
            verified_claims: List of claims that have been verified.
            
        Returns:
            The updated claims with evidence level information.
        """
        if not verified_claims:
            return []
        
        # Format the verification results for the LLM
        verification_results = "\n\n".join([
            f"Claim {i+1}: {claim['claim']}\n" +
            f"Verification Status: {claim['verification_status']}\n" +
            f"Evidence: {claim['evidence']}\n" +
            f"Citation: {claim['citation']}"
            for i, claim in enumerate(verified_claims)
        ])
        
        # Get evidence assessment from the LLM
        assessment_result = self.evidence_assessment_chain.run(verification_results=verification_results)
        
        # Parse the result
        evidence_pattern = r'(\d+)\.\s+Evidence Level:\s+(.*?)\s+\|\s+Rationale:\s+(.*?)(?=\n\d+\.|$)'
        for match in re.finditer(evidence_pattern, assessment_result, re.DOTALL):
            claim_num = int(match.group(1)) - 1
            evidence_level = match.group(2).strip()
            rationale = match.group(3).strip()
            
            if 0 <= claim_num < len(verified_claims):
                verified_claims[claim_num]["evidence_level"] = evidence_level
                verified_claims[claim_num]["evidence_rationale"] = rationale
        
        return verified_claims

    def verify_text(self, text: str, context_docs: List[Document]) -> Dict[str, Any]:
        """
        Verify all medical claims in a text against provided context documents.
        
        Args:
            text: The text to verify.
            context_docs: The context documents to verify against.
            
        Returns:
            Dictionary containing verification results.
        """
        # Extract claims
        claims = self.extract_claims(text)
        
        # Verify each claim
        for i, claim in enumerate(claims):
            claims[i] = self.verify_claim(claim, context_docs)
        
        # Filter claims based on confidence threshold and strict mode
        if self.strict_mode:
            verified_claims = [claim for claim in claims if claim["confidence"] >= self.confidence_threshold]
        else:
            verified_claims = claims
        
        # Assess evidence levels
        assessed_claims = self.assess_evidence_levels(verified_claims)
        
        return {
            "original_text": text,
            "claims": assessed_claims,
            "total_claims": len(claims),
            "verified_claims": self.verified_claims,
            "unverified_claims": self.unverified_claims,
            "modified_claims": self.modified_claims,
            "overall_confidence": sum(claim["confidence"] for claim in assessed_claims) / len(assessed_claims) if assessed_claims else 0.0
        }

    def generate_cited_text(self, verification_result: Dict[str, Any]) -> str:
        """
        Generate text with proper citations based on verification results.
        
        Args:
            verification_result: The result from verify_text.
            
        Returns:
            Text with citations.
        """
        original_text = verification_result["original_text"]
        claims = verification_result["claims"]
        
        # If no claims, return the original text
        if not claims:
            return original_text
        
        # Replace claims with cited or corrected versions
        cited_text = original_text
        
        # Sort claims by their position in the text (to replace from end to start to avoid index issues)
        sorted_claims = sorted(
            [(claim, original_text.find(claim["claim"])) for claim in claims if original_text.find(claim["claim"]) != -1],
            key=lambda x: x[1],
            reverse=True
        )
        
        for claim, position in sorted_claims:
            if position == -1:
                continue
                
            claim_text = claim["claim"]
            end_position = position + len(claim_text)
            
            if claim["verified"] and claim["confidence"] >= self.confidence_threshold:
                # For verified claims, add a citation
                cited_version = f"{claim_text} [{claim['citation']}]"
                cited_text = cited_text[:position] + cited_version + cited_text[end_position:]
            elif claim["corrected_claim"]:
                # For unverified claims with a correction, replace with the corrected version
                cited_text = cited_text[:position] + f"{claim['corrected_claim']} [Corrected based on available evidence]" + cited_text[end_position:]
        
        return cited_text


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify medical claims in text.")
    parser.add_argument("--text", type=str, required=True, help="Text to verify.")
    parser.add_argument("--context", type=str, required=True, help="Context to verify against.")
    args = parser.parse_args()
    
    verifier = MedicalFactVerifier()
    
    # Create a simple document for testing
    context_doc = Document(page_content=args.context, metadata={"source": "test"})
    
    result = verifier.verify_text(args.text, [context_doc])
    
    print("Original Text:")
    print(result["original_text"])
    print("\nExtracted Claims:")
    for i, claim in enumerate(result["claims"]):
        print(f"\nClaim {i+1}: {claim['claim']}")
        print(f"Type: {claim['type']}")
        print(f"Verification Status: {claim['verification_status']}")
        print(f"Confidence: {claim['confidence']}")
        if claim['evidence']:
            print(f"Evidence: {claim['evidence']}")
        if claim['citation']:
            print(f"Citation: {claim['citation']}")
        if claim['corrected_claim']:
            print(f"Corrected Claim: {claim['corrected_claim']}")
        if claim['evidence_level']:
            print(f"Evidence Level: {claim['evidence_level']}")
    
    print("\nCited Text:")
    print(verifier.generate_cited_text(result)) 