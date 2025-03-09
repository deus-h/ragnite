"""
Legal Citation Extractor

This module provides functionality to extract, parse, and standardize legal citations
from text. It can identify various citation formats for cases, statutes, and regulations.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
import json

class CitationExtractor:
    """
    A class that extracts and parses legal citations from text, with support
    for various citation formats.
    """

    def __init__(self):
        """Initialize the citation extractor with regex patterns for different citation types."""
        # Case citation patterns
        self.case_citation_patterns = [
            # US Reports: "410 U.S. 113 (1973)" (Roe v. Wade)
            r"(\d+)\s+U\.S\.\s+(\d+)(?:\s+\((\d{4})\))?",
            
            # Supreme Court Reporter: "93 S.Ct. 705" or "93 S. Ct. 705"
            r"(\d+)\s+S\.\s*Ct\.\s+(\d+)(?:\s+\((\d{4})\))?",
            
            # Federal Reporter: "347 F.2d 394" or "347 F. 2d 394"
            r"(\d+)\s+F\.\s*(\d)d\s+(\d+)(?:\s+\((\d{4}|\d{4}-\d{2})\))?",
            
            # Federal Supplement: "765 F. Supp. 2d 1123"
            r"(\d+)\s+F\.\s*Supp\.\s+(?:(\d)d\s+)?(\d+)(?:\s+\(([A-Za-z\.\s]+)?\s*(\d{4})\))?",
            
            # State/Regional Reports: "123 N.Y.2d 456" or "123 Cal.App.4th 456"
            r"(\d+)\s+([A-Za-z]+\.(?:[A-Za-z]+\.)*(?:\d[a-z]{2})?)\s+(\d+)(?:\s+\((\d{4})\))?"
        ]
        
        # Statute citation patterns
        self.statute_citation_patterns = [
            # US Code: "42 U.S.C. § 1983" or "42 USC § 1983" or "42 U.S.C. §§ 1983-1984"
            r"(\d+)\s+U\.?S\.?C\.?\s+(?:§+|ss\.?|section\s+)?\s*(\d+(?:(?:\-|\–|\—|\&)?\d+)*)(?:\([a-zA-Z0-9]+\))?",
            
            # CFR: "24 C.F.R. § 100.204" or "24 CFR § 100.204"
            r"(\d+)\s+C\.?F\.?R\.?\s+(?:§+|ss\.?|section\s+)?\s*(\d+\.\d+)(?:\([a-zA-Z0-9]+\))?",
            
            # State statutes: "Cal. Penal Code § 422" or "N.Y. Gen. Bus. Law § 349"
            r"([A-Za-z]+\.)\s+([A-Za-z]+\.(?:\s+[A-Za-z]+\.)*)\s+(?:§+|ss\.?|section\s+)?\s*(\d+(?:\.\d+)*)(?:\([a-zA-Z0-9]+\))?"
        ]
        
        # Regulation citation patterns
        self.regulation_citation_patterns = [
            # Federal Register: "83 Fed. Reg. 12,345"
            r"(\d+)\s+Fed\.\s*Reg\.\s+([0-9,]+)",
            
            # Federal regulations (e.g., Treasury Regulations): "Treas. Reg. § 1.1001-1"
            r"([A-Za-z]+\.)\s+Reg\.\s+(?:§+|ss\.?|section\s+)?\s*(\d+\.\d+(?:-\d+)?)"
        ]
        
        # Case name pattern: "Smith v. Jones"
        self.case_name_pattern = r"([A-Za-z\s'\.]+)\s+v\.\s+([A-Za-z\s'\.]+)"

    def extract_citations(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract all legal citations from text.
        
        Args:
            text (str): The legal text to analyze
            
        Returns:
            List[Dict[str, Any]]: A list of extracted citations with metadata
        """
        citations = []
        
        # Extract case citations
        for pattern in self.case_citation_patterns:
            for match in re.finditer(pattern, text):
                citation_text = match.group(0)
                start_pos = match.start()
                end_pos = match.end()
                
                # Look for case name in the vicinity (up to 150 chars before citation)
                case_name = None
                pre_text = text[max(0, start_pos - 150):start_pos]
                case_match = re.search(self.case_name_pattern, pre_text)
                if case_match:
                    case_name = case_match.group(0).strip()
                
                citation_data = {
                    "type": "case",
                    "citation": citation_text,
                    "start_pos": start_pos,
                    "end_pos": end_pos,
                    "case_name": case_name
                }
                
                # Add specific reporter information based on pattern
                if "U.S." in citation_text:
                    citation_data["reporter"] = "US Reports"
                    citation_data["volume"] = match.group(1)
                    citation_data["page"] = match.group(2)
                    if len(match.groups()) > 2 and match.group(3):
                        citation_data["year"] = match.group(3)
                elif "S.Ct." in citation_text or "S. Ct." in citation_text:
                    citation_data["reporter"] = "Supreme Court Reporter"
                    citation_data["volume"] = match.group(1)
                    citation_data["page"] = match.group(2)
                    if len(match.groups()) > 2 and match.group(3):
                        citation_data["year"] = match.group(3)
                elif "F." in citation_text and ("2d" in citation_text or "3d" in citation_text):
                    citation_data["reporter"] = "Federal Reporter"
                    citation_data["volume"] = match.group(1)
                    citation_data["series"] = match.group(2)
                    citation_data["page"] = match.group(3)
                    if len(match.groups()) > 3 and match.group(4):
                        citation_data["year"] = match.group(4)
                elif "F. Supp." in citation_text or "F.Supp." in citation_text:
                    citation_data["reporter"] = "Federal Supplement"
                    citation_data["volume"] = match.group(1)
                    citation_data["page"] = match.group(3)
                    if len(match.groups()) > 3 and match.group(4):
                        citation_data["court"] = match.group(4)
                    if len(match.groups()) > 4 and match.group(5):
                        citation_data["year"] = match.group(5)
                else:
                    # State/regional citation
                    citation_data["reporter"] = match.group(2)
                    citation_data["volume"] = match.group(1)
                    citation_data["page"] = match.group(3)
                    if len(match.groups()) > 3 and match.group(4):
                        citation_data["year"] = match.group(4)
                
                citations.append(citation_data)
        
        # Extract statute citations
        for pattern in self.statute_citation_patterns:
            for match in re.finditer(pattern, text):
                citation_text = match.group(0)
                start_pos = match.start()
                end_pos = match.end()
                
                citation_data = {
                    "type": "statute",
                    "citation": citation_text,
                    "start_pos": start_pos,
                    "end_pos": end_pos
                }
                
                # Add specific statute information based on pattern
                if "U.S.C." in citation_text or "USC" in citation_text:
                    citation_data["code"] = "U.S.C."
                    citation_data["title"] = match.group(1)
                    citation_data["section"] = match.group(2)
                elif "C.F.R." in citation_text or "CFR" in citation_text:
                    citation_data["code"] = "C.F.R."
                    citation_data["title"] = match.group(1)
                    citation_data["section"] = match.group(2)
                else:
                    # State statutes
                    citation_data["code"] = f"{match.group(1)} {match.group(2)}"
                    citation_data["section"] = match.group(3)
                
                citations.append(citation_data)
        
        # Extract regulation citations
        for pattern in self.regulation_citation_patterns:
            for match in re.finditer(pattern, text):
                citation_text = match.group(0)
                start_pos = match.start()
                end_pos = match.end()
                
                citation_data = {
                    "type": "regulation",
                    "citation": citation_text,
                    "start_pos": start_pos,
                    "end_pos": end_pos
                }
                
                if "Fed. Reg." in citation_text:
                    citation_data["source"] = "Federal Register"
                    citation_data["volume"] = match.group(1)
                    citation_data["page"] = match.group(2).replace(",", "")
                else:
                    citation_data["source"] = f"{match.group(1)} Regulations"
                    citation_data["section"] = match.group(2)
                
                citations.append(citation_data)
        
        # Sort citations by their position in the text
        citations.sort(key=lambda x: x["start_pos"])
        
        return citations

    def standardize_citation(self, citation: Dict[str, Any]) -> str:
        """
        Convert a citation to a standardized format.
        
        Args:
            citation (Dict[str, Any]): The citation data
            
        Returns:
            str: A standardized citation string
        """
        if citation["type"] == "case":
            if "case_name" in citation and citation["case_name"]:
                prefix = f"{citation['case_name']}, "
            else:
                prefix = ""
                
            if "reporter" in citation:
                if citation["reporter"] == "US Reports":
                    return f"{prefix}{citation['volume']} U.S. {citation['page']}{' (' + citation['year'] + ')' if 'year' in citation else ''}"
                elif citation["reporter"] == "Supreme Court Reporter":
                    return f"{prefix}{citation['volume']} S.Ct. {citation['page']}{' (' + citation['year'] + ')' if 'year' in citation else ''}"
                elif citation["reporter"] == "Federal Reporter":
                    return f"{prefix}{citation['volume']} F.{citation['series']}d {citation['page']}{' (' + citation['year'] + ')' if 'year' in citation else ''}"
                elif citation["reporter"] == "Federal Supplement":
                    court_year = ""
                    if "court" in citation and "year" in citation:
                        court_year = f" ({citation['court']} {citation['year']})"
                    elif "year" in citation:
                        court_year = f" ({citation['year']})"
                    return f"{prefix}{citation['volume']} F. Supp.{' ' + citation['series'] + 'd' if 'series' in citation else ''} {citation['page']}{court_year}"
                else:
                    # State/regional reporter
                    return f"{prefix}{citation['volume']} {citation['reporter']} {citation['page']}{' (' + citation['year'] + ')' if 'year' in citation else ''}"
        
        elif citation["type"] == "statute":
            if "code" in citation:
                if citation["code"] == "U.S.C.":
                    return f"{citation['title']} U.S.C. § {citation['section']}"
                elif citation["code"] == "C.F.R.":
                    return f"{citation['title']} C.F.R. § {citation['section']}"
                else:
                    return f"{citation['code']} § {citation['section']}"
        
        elif citation["type"] == "regulation":
            if "source" in citation:
                if citation["source"] == "Federal Register":
                    return f"{citation['volume']} Fed. Reg. {citation['page']}"
                else:
                    return f"{citation['source']} § {citation['section']}"
        
        # If we can't standardize, return the original citation
        return citation["citation"]

    def get_citation_metadata(self, citation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get additional metadata for a citation that might be useful for retrieval.
        
        Args:
            citation (Dict[str, Any]): The citation data
            
        Returns:
            Dict[str, Any]: Additional metadata about the citation
        """
        metadata = {}
        
        if citation["type"] == "case":
            # Case metadata
            if "case_name" in citation and citation["case_name"]:
                # Extract plaintiff and defendant from case name
                case_name_match = re.match(self.case_name_pattern, citation["case_name"])
                if case_name_match:
                    metadata["plaintiff"] = case_name_match.group(1).strip()
                    metadata["defendant"] = case_name_match.group(2).strip()
            
            if "reporter" in citation:
                metadata["reporter"] = citation["reporter"]
                
                # Add court information based on reporter
                if citation["reporter"] == "US Reports":
                    metadata["court"] = "Supreme Court of the United States"
                    metadata["jurisdiction"] = "federal"
                elif citation["reporter"] == "Supreme Court Reporter":
                    metadata["court"] = "Supreme Court of the United States"
                    metadata["jurisdiction"] = "federal"
                elif citation["reporter"] == "Federal Reporter":
                    metadata["court"] = "U.S. Court of Appeals"
                    metadata["jurisdiction"] = "federal"
                elif citation["reporter"] == "Federal Supplement":
                    metadata["court"] = "U.S. District Court"
                    metadata["jurisdiction"] = "federal"
                    if "court" in citation:
                        metadata["specific_court"] = citation["court"]
                elif any(state in citation["reporter"] for state in ["N.Y.", "Cal.", "Ill.", "Tex.", "Fla.", "Pa.", "Ohio", "Mass."]):
                    metadata["jurisdiction"] = "state"
                    
            if "year" in citation:
                metadata["year"] = citation["year"]
                
        elif citation["type"] == "statute":
            # Statute metadata
            if "code" in citation:
                if citation["code"] == "U.S.C.":
                    metadata["jurisdiction"] = "federal"
                    
                    # Determine subject matter based on U.S.C. title
                    if citation["title"] == "5":
                        metadata["subject"] = "administrative law"
                    elif citation["title"] == "11":
                        metadata["subject"] = "bankruptcy"
                    elif citation["title"] == "15":
                        metadata["subject"] = "commerce and trade"
                    elif citation["title"] == "17":
                        metadata["subject"] = "copyright"
                    elif citation["title"] == "26":
                        metadata["subject"] = "tax"
                    elif citation["title"] == "28":
                        metadata["subject"] = "judiciary and judicial procedure"
                    elif citation["title"] == "29":
                        metadata["subject"] = "labor"
                    elif citation["title"] == "35":
                        metadata["subject"] = "patents"
                    elif citation["title"] == "42":
                        metadata["subject"] = "public health and welfare"
                    elif citation["title"] == "47":
                        metadata["subject"] = "telecommunications"
                
                elif "Cal." in citation["code"]:
                    metadata["jurisdiction"] = "california"
                elif "N.Y." in citation["code"]:
                    metadata["jurisdiction"] = "new york"
                elif "Tex." in citation["code"]:
                    metadata["jurisdiction"] = "texas"
                # Add more state jurisdictions as needed
        
        elif citation["type"] == "regulation":
            # Regulation metadata
            if "source" in citation:
                if citation["source"] == "Federal Register":
                    metadata["jurisdiction"] = "federal"
                elif "Treas." in citation["source"]:
                    metadata["jurisdiction"] = "federal"
                    metadata["subject"] = "tax"
                elif "SEC" in citation["source"]:
                    metadata["jurisdiction"] = "federal"
                    metadata["subject"] = "securities"
                elif "FDA" in citation["source"]:
                    metadata["jurisdiction"] = "federal"
                    metadata["subject"] = "food and drug"
                elif "EPA" in citation["source"]:
                    metadata["jurisdiction"] = "federal"
                    metadata["subject"] = "environmental"
                # Add more regulatory agencies as needed
        
        return metadata

    def extract_and_process_citations(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract citations from text and add standardized forms and metadata.
        
        Args:
            text (str): The legal text to analyze
            
        Returns:
            List[Dict[str, Any]]: A list of processed citations
        """
        citations = self.extract_citations(text)
        
        for citation in citations:
            citation["standardized"] = self.standardize_citation(citation)
            citation["metadata"] = self.get_citation_metadata(citation)
            
            # Extract surrounding context
            context_start = max(0, citation["start_pos"] - 100)
            context_end = min(len(text), citation["end_pos"] + 100)
            citation["context"] = text[context_start:context_end]
        
        return citations

    def replace_with_standardized_citations(self, text: str) -> str:
        """
        Replace all citations in a text with their standardized forms.
        
        Args:
            text (str): The legal text to process
            
        Returns:
            str: Text with standardized citations
        """
        citations = self.extract_citations(text)
        
        # Sort citations in reverse order to replace from end to beginning
        # (to avoid index issues as we replace)
        citations.sort(key=lambda x: x["start_pos"], reverse=True)
        
        result = text
        for citation in citations:
            standardized = self.standardize_citation(citation)
            result = result[:citation["start_pos"]] + standardized + result[citation["end_pos"]:]
        
        return result

    def extract_citation_graph(self, documents: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Create a citation graph showing relationships between legal documents.
        
        Args:
            documents (List[Dict[str, Any]]): A list of documents with text and metadata
            
        Returns:
            Dict[str, List[str]]: A graph representation of citation relationships
        """
        citation_graph = {}
        
        # Create a mapping of case names/citations to document identifiers
        doc_identifiers = {}
        for doc in documents:
            if "case_name" in doc["metadata"]:
                doc_identifiers[doc["metadata"]["case_name"]] = doc["id"]
            if "citation" in doc["metadata"]:
                doc_identifiers[doc["metadata"]["citation"]] = doc["id"]
        
        # Extract citations from each document and build the graph
        for doc in documents:
            doc_id = doc["id"]
            citations = self.extract_citations(doc["text"])
            
            cited_docs = []
            for citation in citations:
                if "case_name" in citation and citation["case_name"] in doc_identifiers:
                    cited_docs.append(doc_identifiers[citation["case_name"]])
                elif citation["citation"] in doc_identifiers:
                    cited_docs.append(doc_identifiers[citation["citation"]])
                    
            citation_graph[doc_id] = cited_docs
        
        return citation_graph


def extract_citations_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Helper function to extract citations from a text string.
    
    Args:
        text (str): The legal text to analyze
        
    Returns:
        List[Dict[str, Any]]: A list of extracted and processed citations
    """
    extractor = CitationExtractor()
    return extractor.extract_and_process_citations(text)


def standardize_citations_in_text(text: str) -> str:
    """
    Helper function to standardize all citations in a text string.
    
    Args:
        text (str): The legal text to process
        
    Returns:
        str: Text with standardized citations
    """
    extractor = CitationExtractor()
    return extractor.replace_with_standardized_citations(text) 