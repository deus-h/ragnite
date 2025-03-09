#!/usr/bin/env python3
"""
Multi-Domain RAG Example

This script demonstrates how to combine multiple domain-specific RAG systems
to answer queries that span across different domains.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add the parent directories to the Python path to import the domain packages
domain_specific_dir = Path(__file__).parent.parent
sys.path.append(str(domain_specific_dir))

# Import domain-specific RAG implementations
# Note: You need to have each domain implementation installed
try:
    from code_rag.src.code_rag import CodeRAG
except ImportError:
    print("Warning: CodeRAG not found. Code-related queries will be skipped.")
    CodeRAG = None

try:
    from medical_rag.src.medical_rag import MedicalRAG
except ImportError:
    print("Warning: MedicalRAG not found. Medical-related queries will be skipped.")
    MedicalRAG = None

try:
    from legal_rag.src.legal_rag import LegalRAG
except ImportError:
    print("Warning: LegalRAG not found. Legal-related queries will be skipped.")
    LegalRAG = None

try:
    from scientific_rag.src.scientific_rag import ScientificRAG
except ImportError:
    print("Warning: ScientificRAG not found. Scientific-related queries will be skipped.")
    ScientificRAG = None


class MultiDomainRAG:
    """
    A meta-RAG system that combines multiple domain-specific RAG systems
    to handle queries across domains.
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        use_code_rag: bool = True,
        use_medical_rag: bool = True,
        use_legal_rag: bool = True,
        use_scientific_rag: bool = True,
        db_directory: str = "./multi_domain_rag_db",
    ):
        """
        Initialize the multi-domain RAG system.
        
        Args:
            openai_api_key: OpenAI API key for models and embeddings
            use_code_rag: Whether to include the CodeRAG system
            use_medical_rag: Whether to include the MedicalRAG system
            use_legal_rag: Whether to include the LegalRAG system
            use_scientific_rag: Whether to include the ScientificRAG system
            db_directory: Base directory for storing vector databases
        """
        self.db_directory = db_directory
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        
        # Dictionary to store domain-specific RAG systems
        self.rag_systems = {}
        
        # Initialize code RAG if available and requested
        if use_code_rag and CodeRAG is not None:
            self.rag_systems["code"] = CodeRAG(
                openai_api_key=self.openai_api_key,
                db_directory=os.path.join(db_directory, "code_rag_db"),
            )
        
        # Initialize medical RAG if available and requested
        if use_medical_rag and MedicalRAG is not None:
            self.rag_systems["medical"] = MedicalRAG(
                openai_api_key=self.openai_api_key,
                db_directory=os.path.join(db_directory, "medical_rag_db"),
            )
        
        # Initialize legal RAG if available and requested
        if use_legal_rag and LegalRAG is not None:
            self.rag_systems["legal"] = LegalRAG(
                openai_api_key=self.openai_api_key,
                db_directory=os.path.join(db_directory, "legal_rag_db"),
            )
        
        # Initialize scientific RAG if available and requested
        if use_scientific_rag and ScientificRAG is not None:
            self.rag_systems["scientific"] = ScientificRAG(
                use_openai=True,
                openai_api_key=self.openai_api_key,
                db_directory=os.path.join(db_directory, "scientific_rag_db"),
            )
        
        print(f"Initialized MultiDomainRAG with domains: {list(self.rag_systems.keys())}")
    
    def classify_query_domain(self, query: str) -> List[str]:
        """
        Classify a query to determine which domain(s) it belongs to.
        
        Args:
            query: The user query to classify
            
        Returns:
            List of domain names that the query might belong to
        """
        # This is a simple rule-based classification
        # In a production system, you would use a more sophisticated classifier
        
        domains = []
        
        # Check for code-related keywords
        code_keywords = ["code", "program", "function", "class", "bug", "error", "algorithm", 
                         "api", "library", "framework", "syntax", "compiler", "programming"]
        if any(keyword in query.lower() for keyword in code_keywords) and "code" in self.rag_systems:
            domains.append("code")
        
        # Check for medical-related keywords
        medical_keywords = ["disease", "patient", "treatment", "symptom", "diagnosis", "drug", 
                            "medical", "health", "clinical", "doctor", "hospital", "vaccine"]
        if any(keyword in query.lower() for keyword in medical_keywords) and "medical" in self.rag_systems:
            domains.append("medical")
        
        # Check for legal-related keywords
        legal_keywords = ["law", "legal", "court", "regulation", "compliance", "contract", 
                          "lawsuit", "attorney", "judge", "plaintiff", "defendant", "statute"]
        if any(keyword in query.lower() for keyword in legal_keywords) and "legal" in self.rag_systems:
            domains.append("legal")
        
        # Check for scientific-related keywords
        scientific_keywords = ["research", "experiment", "theory", "hypothesis", "data", "analysis", 
                              "scientist", "paper", "journal", "laboratory", "publication"]
        if any(keyword in query.lower() for keyword in scientific_keywords) and "scientific" in self.rag_systems:
            domains.append("scientific")
        
        # If no domains matched, use all available domains
        if not domains:
            domains = list(self.rag_systems.keys())
        
        return domains
    
    def query(
        self,
        query: str,
        num_results: int = 3,
        auto_classify: bool = True,
        domains: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Query the multi-domain RAG system.
        
        Args:
            query: The user query
            num_results: Number of results to retrieve per domain
            auto_classify: Whether to automatically classify the query into domains
            domains: Specific domains to query (overrides auto_classify)
            
        Returns:
            Dictionary with answers from each domain and combined results
        """
        # Determine which domains to query
        if domains is None and auto_classify:
            domains = self.classify_query_domain(query)
        elif domains is None:
            domains = list(self.rag_systems.keys())
        
        print(f"Querying domains: {domains}")
        
        # Query each specified domain
        domain_results = {}
        all_sources = []
        
        for domain in domains:
            if domain in self.rag_systems:
                try:
                    print(f"Querying {domain} domain...")
                    result = self.rag_systems[domain].query(query, num_results=num_results)
                    domain_results[domain] = result
                    
                    # Add domain label to sources
                    for source in result["sources"]:
                        source["domain"] = domain
                        all_sources.append(source)
                except Exception as e:
                    print(f"Error querying {domain} domain: {str(e)}")
        
        # Combine results from different domains
        # In a production system, you would use a more sophisticated synthesis approach
        combined_answer = self._synthesize_answers(query, domain_results)
        
        return {
            "query": query,
            "domains_queried": domains,
            "domain_results": domain_results,
            "combined_answer": combined_answer,
            "all_sources": all_sources,
        }
    
    def _synthesize_answers(self, query: str, domain_results: Dict[str, Any]) -> str:
        """
        Synthesize answers from multiple domains into a cohesive response.
        
        Args:
            query: The original query
            domain_results: Results from each domain
            
        Returns:
            A combined answer that integrates insights from all domains
        """
        if not domain_results:
            return "No results found in any domain."
        
        synthesis = f"Query: {query}\n\n"
        synthesis += "Here are insights from different domains:\n\n"
        
        for domain, result in domain_results.items():
            synthesis += f"--- {domain.upper()} DOMAIN ---\n"
            synthesis += result["answer"]
            synthesis += "\n\n"
        
        synthesis += "--- SYNTHESIS ---\n"
        synthesis += "The question spans multiple domains. "
        
        # Add domain-specific syntheses
        domains = list(domain_results.keys())
        if "code" in domains and "scientific" in domains:
            synthesis += "This involves both software development and scientific research considerations. "
        
        if "medical" in domains and "legal" in domains:
            synthesis += "This has both medical and legal implications. "
        
        # Conclude with a general statement
        synthesis += "Consider consulting domain experts for the most accurate guidance."
        
        return synthesis
    
    def ingest_document(self, file_path: str, domain: str) -> bool:
        """
        Ingest a document into a specific domain RAG system.
        
        Args:
            file_path: Path to the document
            domain: Domain to ingest the document into
            
        Returns:
            True if ingestion was successful, False otherwise
        """
        if domain not in self.rag_systems:
            print(f"Error: Domain '{domain}' not available.")
            return False
        
        try:
            print(f"Ingesting {file_path} into {domain} domain...")
            self.rag_systems[domain].ingest_document(file_path)
            return True
        except Exception as e:
            print(f"Error ingesting document: {str(e)}")
            return False


def demo():
    """Run a demonstration of the MultiDomainRAG system."""
    # Create the multi-domain RAG system
    # In this demo, we'll just use the systems that are available
    multi_rag = MultiDomainRAG()
    
    if not multi_rag.rag_systems:
        print("No domain-specific RAG systems were initialized. Please install at least one.")
        return
    
    # Example queries that span multiple domains
    example_queries = [
        "How can machine learning be applied to analyze medical imaging data?",
        "What are the legal implications of open-source AI models in healthcare?",
        "How do I implement data privacy measures in medical software applications?",
        "What statistical methods are commonly used in clinical trials?",
    ]
    
    # Process each query
    for i, query in enumerate(example_queries, 1):
        print(f"\n\n{'=' * 80}")
        print(f"QUERY {i}: {query}")
        print(f"{'=' * 80}\n")
        
        # Determine domains automatically
        domains = multi_rag.classify_query_domain(query)
        print(f"Automatic domain classification: {domains}")
        
        # Get results
        results = multi_rag.query(query, domains=domains)
        
        # Print the combined answer
        print("\nCOMBINED ANSWER:")
        print(results["combined_answer"])
        
        # Print the sources (shortened for readability)
        print("\nSOURCES:")
        for i, source in enumerate(results["all_sources"][:3], 1):  # Show max 3 sources
            print(f"{i}. [{source['domain'].upper()}] {source['metadata'].get('source', 'Unknown')}:")
            content_preview = source['content'][:100] + "..." if len(source['content']) > 100 else source['content']
            print(f"   {content_preview}")


if __name__ == "__main__":
    demo() 