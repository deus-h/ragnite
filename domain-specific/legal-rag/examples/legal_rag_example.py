"""
Legal RAG Example

This script demonstrates the Legal RAG implementation for various legal research
and document analysis scenarios.
"""

import os
import sys
import json
from typing import List, Dict, Any
import logging

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.legal_rag_pipeline import LegalRAG
from src.legal_chunker import process_legal_document
from src.citation_extractor import extract_citations_from_text, standardize_citations_in_text
from src.authority_verifier import verify_legal_text, generate_legal_report

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_legal_document(filename: str, content: str) -> str:
    """
    Create a sample legal document for testing.
    
    Args:
        filename (str): The filename to create
        content (str): The content to write to the file
        
    Returns:
        str: The path to the created file
    """
    # Create the directory if it doesn't exist
    os.makedirs("../data/raw/samples", exist_ok=True)
    
    # Write the content to the file
    file_path = os.path.join("../data/raw/samples", filename)
    with open(file_path, "w") as f:
        f.write(content)
        
    return file_path

def run_legal_rag_example():
    """Run the Legal RAG example with various scenarios."""
    
    # Create sample legal documents
    create_sample_legal_document("miranda_v_arizona.txt", """
    Miranda v. Arizona, 384 U.S. 436 (1966)
    
    FACTS:
    Ernesto Miranda was arrested for kidnapping and rape. During police interrogation, he confessed to the crimes. 
    He was not informed of his right to counsel or his right to remain silent. At trial, the prosecution used his confession 
    as evidence, and Miranda was convicted.
    
    PROCEDURAL HISTORY:
    Miranda appealed his conviction to the Arizona Supreme Court, which affirmed the conviction. 
    The U.S. Supreme Court granted certiorari.
    
    ISSUE:
    Whether statements obtained from a defendant during custodial interrogation without full warning of constitutional 
    rights are admissible in a criminal trial?
    
    HOLDING:
    The Supreme Court held that the prosecution may not use statements stemming from custodial interrogation of the 
    defendant unless it demonstrates the use of procedural safeguards effective to secure the privilege against 
    self-incrimination.
    
    REASONING:
    The Court noted that custodial interrogation contains inherently compelling pressures which work to undermine the 
    individual's will to resist and to compel him to speak where he would not otherwise do so freely. To combat these 
    pressures and to permit a full opportunity to exercise the privilege against self-incrimination, the accused must be 
    adequately and effectively apprised of his rights.
    
    The Court established what are now known as the Miranda rights:
    1. The right to remain silent
    2. That anything said can and will be used against the individual in court
    3. The right to consult with counsel prior to questioning and to have counsel present during questioning
    4. That if the individual cannot afford an attorney, one will be appointed for him
    
    CONCLUSION:
    The judgment of the Arizona Supreme Court was reversed, and the case was remanded for further proceedings.
    """)
    
    create_sample_legal_document("us_constitution_4th_amendment.txt", """
    Fourth Amendment to the United States Constitution
    
    The right of the people to be secure in their persons, houses, papers, and effects, against unreasonable searches and 
    seizures, shall not be violated, and no Warrants shall issue, but upon probable cause, supported by Oath or affirmation, 
    and particularly describing the place to be searched, and the persons or things to be seized.
    
    INTERPRETATION:
    The Fourth Amendment prohibits unreasonable searches and seizures and requires that search warrants be supported by 
    probable cause. It was adopted in response to the abusive use of general warrants by British authorities during the 
    colonial period.
    
    KEY CASES:
    - Katz v. United States, 389 U.S. 347 (1967): Established that the Fourth Amendment protects people, not places, and 
      introduced the "reasonable expectation of privacy" test.
    - Terry v. Ohio, 392 U.S. 1 (1968): Established the legality of "stop and frisk" procedures under certain circumstances.
    - United States v. Leon, 468 U.S. 897 (1984): Established the "good faith exception" to the exclusionary rule.
    """)
    
    create_sample_legal_document("contract_example.txt", """
    SERVICES AGREEMENT
    
    This Services Agreement (the "Agreement") is entered into as of January 1, 2023 (the "Effective Date") by and between 
    ABC Corporation, a Delaware corporation with its principal place of business at 123 Main Street, Anytown, USA 
    ("Company"), and XYZ Consulting, LLC, a California limited liability company with its principal place of business at 
    456 Oak Avenue, Othertown, USA ("Consultant").
    
    WHEREAS, Company desires to engage Consultant to provide certain services, and Consultant is willing to provide such 
    services to Company;
    
    NOW, THEREFORE, in consideration of the mutual covenants and agreements hereinafter set forth, the parties agree as follows:
    
    ARTICLE 1. SERVICES
    
    1.1 Services. Consultant shall provide to Company the services described in Exhibit A attached hereto (the "Services").
    
    1.2 Performance Standards. Consultant shall perform the Services in a professional manner and in accordance with the 
    industry standards and shall comply with all applicable laws in the performance of the Services.
    
    ARTICLE 2. COMPENSATION
    
    2.1 Fees. Company shall pay to Consultant the fees set forth in Exhibit B attached hereto.
    
    2.2 Expenses. Company shall reimburse Consultant for reasonable expenses incurred in connection with the Services, 
    provided that such expenses are approved in advance by Company.
    
    ARTICLE 3. TERM AND TERMINATION
    
    3.1 Term. This Agreement shall commence on the Effective Date and shall continue until December 31, 2023, unless 
    earlier terminated in accordance with this Agreement.
    
    3.2 Termination. Either party may terminate this Agreement upon 30 days' written notice to the other party.
    
    ARTICLE 4. CONFIDENTIALITY
    
    4.1 Confidential Information. Consultant acknowledges that in the course of providing the Services, Consultant may 
    receive confidential information of Company. Consultant shall maintain the confidentiality of all such information 
    and shall not disclose it to any third party without Company's prior written consent.
    
    IN WITNESS WHEREOF, the parties have executed this Agreement as of the Effective Date.
    
    ABC CORPORATION
    By: ____________________
    Name: John Smith
    Title: CEO
    
    XYZ CONSULTING, LLC
    By: ____________________
    Name: Jane Doe
    Title: Managing Partner
    """)
    
    # Initialize the Legal RAG pipeline
    legal_rag = LegalRAG(
        vector_store_path="../data/processed/legal_vector_store",
        verify_authority=True
    )
    
    # Ingest the sample documents
    logger.info("Ingesting sample legal documents...")
    legal_rag.ingest_directory("../data/raw/samples")
    
    # Example scenarios
    scenarios = [
        {
            "name": "Miranda Rights Query",
            "query": "What are the Miranda rights and when must they be given?",
            "description": "Basic legal research query about Miranda rights"
        },
        {
            "name": "Fourth Amendment Analysis",
            "query": "What constitutes an unreasonable search under the Fourth Amendment?",
            "description": "Constitutional law analysis with citation to key cases"
        },
        {
            "name": "Contract Termination",
            "query": "What are the termination provisions in the services agreement between ABC Corporation and XYZ Consulting?",
            "description": "Contract analysis query"
        },
        {
            "name": "Legal Citation Query",
            "query": "What did the Supreme Court hold in Miranda v. Arizona, 384 U.S. 436 (1966)?",
            "description": "Query with specific legal citation"
        },
        {
            "name": "Jurisdictional Query",
            "query": "How does the exclusionary rule apply to Fourth Amendment violations in federal courts?",
            "description": "Query with jurisdictional component"
        }
    ]
    
    # Run each scenario
    for scenario in scenarios:
        logger.info(f"\n\nRunning scenario: {scenario['name']}")
        logger.info(f"Query: {scenario['query']}")
        
        # Query the Legal RAG system
        result = legal_rag.query(scenario['query'])
        
        # Print the results
        print("\n" + "="*80)
        print(f"SCENARIO: {scenario['name']}")
        print(f"QUERY: {scenario['query']}")
        print("="*80)
        print("\nRESPONSE:")
        print(result['response'])
        
        print("\nCITATIONS:")
        for citation in result['citations']:
            print(f"- {citation['text']} ({citation['type']})")
            
        print(f"\nCONFIDENCE SCORE: {result['confidence_score']:.2f}")
        
        if 'verification' in result:
            print("\nVERIFICATION:")
            print(f"Verified: {result['verification']['verified']}")
            print("Claims:")
            for claim in result['verification']['claims']:
                print(f"- {claim['claim']} ({claim['status']}, confidence: {claim['confidence']:.2f})")
                
        print("\n" + "="*80 + "\n")
        
    # Example of document analysis
    logger.info("\n\nPerforming document analysis on Miranda v. Arizona...")
    analysis = legal_rag.analyze_legal_document("../data/raw/samples/miranda_v_arizona.txt")
    
    print("\n" + "="*80)
    print("DOCUMENT ANALYSIS: Miranda v. Arizona")
    print("="*80)
    print(f"\nDocument Type: {analysis['document_type']}")
    print(f"\nSummary:\n{analysis['summary']}")
    
    print("\nLegal Concepts:")
    for concept in analysis['legal_concepts']:
        print(f"- {concept['name']}: {concept['definition']} (Domain: {concept['domain']})")
        
    print("\nCitations:")
    for citation in analysis['citations'][:5]:  # Show first 5 citations
        print(f"- {citation['standardized']}")
        
    print("\nDocument Structure:")
    for section, content in analysis['structure']['sections'].items():
        print(f"- {section}")
        
    print("\n" + "="*80 + "\n")
    
    # Example of citation extraction
    logger.info("\n\nDemonstrating citation extraction...")
    text_with_citations = """
    The Supreme Court's decision in Miranda v. Arizona, 384 U.S. 436 (1966), established the principle that 
    statements made by a criminal suspect during custodial interrogation are inadmissible unless certain 
    procedural safeguards are followed. This principle was later refined in Edwards v. Arizona, 451 U.S. 477 (1981), 
    which held that once a suspect invokes the right to counsel, interrogation must cease until counsel is provided.
    
    The Fourth Amendment's protection against unreasonable searches was addressed in Katz v. United States, 389 U.S. 347 (1967),
    which established the "reasonable expectation of privacy" test. This test was further developed in California v. Ciraolo, 
    476 U.S. 207 (1986), regarding aerial surveillance.
    
    For statutory interpretation, courts often look to 1 U.S.C. § 1, which provides definitions for the United States Code.
    Regulations such as 24 C.F.R. § 100.204 implement the Fair Housing Act's reasonable accommodation requirements.
    """
    
    citations = extract_citations_from_text(text_with_citations)
    standardized_text = standardize_citations_in_text(text_with_citations)
    
    print("\n" + "="*80)
    print("CITATION EXTRACTION EXAMPLE")
    print("="*80)
    print("\nExtracted Citations:")
    for citation in citations:
        print(f"- {citation['standardized']} (Type: {citation['type']})")
        if 'metadata' in citation and citation['metadata']:
            print(f"  Metadata: {citation['metadata']}")
            
    print("\nStandardized Text:")
    print(standardized_text)
    print("\n" + "="*80 + "\n")
    
    logger.info("Legal RAG example completed successfully!")

if __name__ == "__main__":
    run_legal_rag_example() 