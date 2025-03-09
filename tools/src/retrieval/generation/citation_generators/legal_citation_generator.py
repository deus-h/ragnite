"""
Legal Citation Generator

This module provides the LegalCitationGenerator class for generating citations
for legal sources such as cases, statutes, regulations, and law review articles.
"""

from typing import Any, Dict, List, Optional, Union
from .base_citation_generator import BaseCitationGenerator

class LegalCitationGenerator(BaseCitationGenerator):
    """
    Legal citation generator for legal sources.
    
    Generates citations for legal sources including cases, statutes, regulations,
    law review articles, and other legal materials in various citation styles
    such as Bluebook, ALWD, and jurisdiction-specific formats.
    
    Attributes:
        config (Dict[str, Any]): Configuration options for the generator.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the legal citation generator.
        
        Args:
            config: Optional configuration dictionary, may include:
                - default_style: Default citation style to use (default "BLUEBOOK").
                - include_url: Whether to include URLs (default True).
                - jurisdiction: Default jurisdiction for jurisdiction-specific formats.
                - court_abbreviations: Dictionary of court name abbreviations.
        """
        default_config = {
            'default_style': 'BLUEBOOK',
            'include_url': True,
            'jurisdiction': 'US',
            'court_abbreviations': {
                'Supreme Court of the United States': 'U.S.',
                'United States Court of Appeals': 'F.',
                'United States District Court': 'F. Supp.',
                'California Supreme Court': 'Cal.',
                'New York Court of Appeals': 'N.Y.',
                # Add more as needed
            }
        }
        
        if config:
            default_config.update(config)
            
        super().__init__(default_config)
    
    def generate_citation(self, source: Dict[str, Any], style: Optional[str] = None) -> str:
        """
        Generate a legal citation for the given source.
        
        Args:
            source: Dictionary containing source metadata including:
                - type: "case", "statute", "regulation", "article", etc.
                - title/name: Title or name of the source
                - volume: Volume number for reporters
                - reporter: Reporter name
                - year: Publication year
                - court: Court name
                - jurisdiction: Jurisdiction
                - citation: Official citation (if available)
                - etc. (source type specific fields)
            style: Citation style (BLUEBOOK, ALWD, etc.) If None,
                uses the default_style from config.
                
        Returns:
            A properly formatted citation string.
        """
        if not style:
            style = self.config.get('default_style', 'BLUEBOOK')
        
        # Check that source has minimum required fields
        if not self.validate_source(source, style):
            missing = self.get_missing_fields(source, style)
            return f"[Incomplete citation - missing: {', '.join(missing)}]"
        
        # Generate citation based on style and source type
        source_type = source.get('type', '').lower()
        
        if style.upper() == 'BLUEBOOK':
            return self._generate_bluebook_citation(source, source_type)
        elif style.upper() == 'ALWD':
            return self._generate_alwd_citation(source, source_type)
        else:
            # Fallback to Bluebook if style is not supported
            return self._generate_bluebook_citation(source, source_type)
    
    def generate_citations(self, sources: List[Dict[str, Any]], style: Optional[str] = None) -> List[str]:
        """
        Generate legal citations for multiple sources.
        
        Args:
            sources: List of source dictionaries.
            style: Citation style (BLUEBOOK, ALWD, etc.).
            
        Returns:
            List of citation strings in the same order as the input sources.
        """
        return [self.generate_citation(source, style) for source in sources]
    
    def get_supported_styles(self) -> List[str]:
        """
        Get a list of citation styles supported by this generator.
        
        Returns:
            List of supported citation style names.
        """
        return ["BLUEBOOK", "ALWD"]
    
    def get_required_fields(self, style: Optional[str] = None) -> List[str]:
        """
        Get a list of required fields for generating legal citations.
        
        Different legal citation styles and source types require different
        fields, but some core fields are always required.
        
        Args:
            style: Optional citation style to get required fields for.
                
        Returns:
            List of required field names.
        """
        # Core fields required for all styles, but depend on the source type
        return ["type"]
    
    def _get_required_fields_for_type(self, source_type: str) -> List[str]:
        """
        Get a list of required fields for a specific source type.
        
        Args:
            source_type: Type of legal source.
                
        Returns:
            List of required field names for the source type.
        """
        if source_type == "case":
            return ["name", "reporter", "volume", "page", "court", "year"]
        elif source_type == "statute":
            return ["title", "code", "section", "year"]
        elif source_type == "regulation":
            return ["title", "code", "section", "year"]
        elif source_type == "article":
            return ["title", "authors", "journal", "volume", "year", "page"]
        elif source_type == "treatise":
            return ["title", "authors", "edition", "publisher", "year"]
        else:
            return ["title"]  # Generic fallback
    
    def validate_source(self, source: Dict[str, Any], style: Optional[str] = None) -> bool:
        """
        Validate that the source contains all required fields for legal citation generation.
        
        Args:
            source: Source dictionary to validate.
            style: Optional citation style to validate against.
            
        Returns:
            True if the source is valid, False otherwise.
        """
        if "type" not in source or not source["type"]:
            return False
            
        source_type = source.get("type", "").lower()
        required_fields = self._get_required_fields_for_type(source_type)
        
        for field in required_fields:
            if field not in source or not source[field]:
                return False
                
        return True
    
    def get_missing_fields(self, source: Dict[str, Any], style: Optional[str] = None) -> List[str]:
        """
        Get a list of required fields that are missing from the source.
        
        Args:
            source: Source dictionary to check.
            style: Optional citation style to check against.
            
        Returns:
            List of missing field names.
        """
        if "type" not in source or not source["type"]:
            return ["type"]
            
        source_type = source.get("type", "").lower()
        required_fields = self._get_required_fields_for_type(source_type)
        
        return [field for field in required_fields if field not in source or not source[field]]
    
    def _abbreviate_court(self, court_name: str) -> str:
        """
        Get the abbreviation for a court name.
        
        Args:
            court_name: Full court name.
            
        Returns:
            Abbreviated court name if available, otherwise the original name.
        """
        court_abbreviations = self.config.get("court_abbreviations", {})
        
        # Try exact match
        if court_name in court_abbreviations:
            return court_abbreviations[court_name]
        
        # Try partial match
        for court, abbrev in court_abbreviations.items():
            if court.lower() in court_name.lower():
                return abbrev
        
        # If no match found, return the original name
        return court_name
    
    def _format_parties(self, parties: Dict[str, str]) -> str:
        """
        Format party names for case citations.
        
        Args:
            parties: Dictionary with "plaintiff" and "defendant" keys.
            
        Returns:
            Formatted party string, e.g., "Plaintiff v. Defendant"
        """
        plaintiff = parties.get("plaintiff", "")
        defendant = parties.get("defendant", "")
        
        if plaintiff and defendant:
            return f"{plaintiff} v. {defendant}"
        elif plaintiff:
            return plaintiff
        elif defendant:
            return defendant
        else:
            return ""
    
    def _generate_bluebook_citation(self, source: Dict[str, Any], source_type: str) -> str:
        """
        Generate a Bluebook style citation.
        
        Args:
            source: Source dictionary.
            source_type: Type of legal source.
            
        Returns:
            Bluebook formatted citation string.
        """
        if source_type == "case":
            # Case citation
            case_name = source.get("name", "")
            # If we have plaintiff/defendant instead of a name
            if not case_name and "parties" in source:
                case_name = self._format_parties(source.get("parties", {}))
            
            volume = source.get("volume", "")
            reporter = source.get("reporter", "")
            page = source.get("page", "")
            court = source.get("court", "")
            year = source.get("year", "")
            
            # Format the primary citation
            primary_citation = f"{volume} {reporter} {page}"
            
            # Format the court and year in parentheses
            court_abbrev = self._abbreviate_court(court)
            parenthetical = f" ({court_abbrev} {year})" if court_abbrev and year else f" ({year})" if year else ""
            
            # Format the pinpoint citation if available
            pincite = f", {source.get('pincite', '')}" if source.get("pincite") else ""
            
            # Format parallel citations if available
            parallel_citations = ""
            if "parallel_citations" in source and source["parallel_citations"]:
                parallel_citations = ", " + ", ".join(source["parallel_citations"])
            
            return f"{case_name}, {primary_citation}{parallel_citations}{pincite}{parenthetical}."
            
        elif source_type == "statute":
            # Statute citation
            title = source.get("title", "")
            code = source.get("code", "")
            section = source.get("section", "")
            year = source.get("year", "")
            
            # Basic format: Title # Code § Section (Year)
            title_part = f"{title} " if title else ""
            year_part = f" ({year})" if year else ""
            
            return f"{title_part}{code} § {section}{year_part}."
            
        elif source_type == "regulation":
            # Regulation citation
            title = source.get("title", "")
            code = source.get("code", "")
            section = source.get("section", "")
            year = source.get("year", "")
            
            # Basic format: Title # C.F.R. § Section (Year)
            year_part = f" ({year})" if year else ""
            
            return f"{title} {code} § {section}{year_part}."
            
        elif source_type == "article":
            # Law review article
            authors = source.get("authors", [])
            author_text = ""
            
            if authors:
                if len(authors) == 1:
                    author = authors[0]
                    first_name = author.get("first_name", "")
                    last_name = author.get("last_name", "")
                    author_text = f"{first_name} {last_name}"
                elif len(authors) > 1:
                    author_text = "et al."
            
            title = source.get("title", "")
            journal = source.get("journal", "")
            volume = source.get("volume", "")
            page = source.get("page", "")
            year = source.get("year", "")
            
            # Basic format: Author, Title, Volume Journal Page (Year)
            year_part = f" ({year})" if year else ""
            
            return f"{author_text}, {title}, {volume} {journal} {page}{year_part}."
            
        elif source_type == "treatise":
            # Treatise/book citation
            authors = source.get("authors", [])
            author_text = ""
            
            if authors:
                if len(authors) == 1:
                    author = authors[0]
                    first_name = author.get("first_name", "")
                    last_name = author.get("last_name", "")
                    author_text = f"{first_name} {last_name}"
                elif len(authors) > 1:
                    author_text = "et al."
            
            title = source.get("title", "")
            edition = source.get("edition", "")
            volume = source.get("volume", "")
            publisher = source.get("publisher", "")
            year = source.get("year", "")
            
            # Basic format: Author, Title Volume (Edition Year)
            volume_part = f"{volume} " if volume else ""
            edition_part = f"{edition} ed. " if edition else ""
            year_part = f"{year}" if year else ""
            parenthetical = f" ({edition_part}{year_part})" if edition_part or year_part else ""
            
            return f"{author_text}, {volume_part}{title}{parenthetical}."
            
        else:
            # Generic fallback
            title = source.get("title", "")
            authors = source.get("authors", [])
            author_text = ""
            
            if authors:
                if len(authors) == 1:
                    author = authors[0]
                    first_name = author.get("first_name", "")
                    last_name = author.get("last_name", "")
                    author_text = f"{first_name} {last_name}, "
                elif len(authors) > 1:
                    author_text = "et al., "
            
            publication = source.get("publication", "")
            date = source.get("date", "")
            url = source.get("url", "")
            url_part = f", {url}" if url and self.config.get("include_url", True) else ""
            
            return f"{author_text}{title}, {publication} ({date}){url_part}."
    
    def _generate_alwd_citation(self, source: Dict[str, Any], source_type: str) -> str:
        """
        Generate an ALWD (Association of Legal Writing Directors) style citation.
        
        Args:
            source: Source dictionary.
            source_type: Type of legal source.
            
        Returns:
            ALWD formatted citation string.
        """
        # ALWD is similar to Bluebook with some differences
        if source_type == "case":
            # Case citation
            case_name = source.get("name", "")
            # If we have plaintiff/defendant instead of a name
            if not case_name and "parties" in source:
                case_name = self._format_parties(source.get("parties", {}))
            
            volume = source.get("volume", "")
            reporter = source.get("reporter", "")
            page = source.get("page", "")
            court = source.get("court", "")
            year = source.get("year", "")
            
            # Format the primary citation
            primary_citation = f"{volume} {reporter} {page}"
            
            # Format the court and year in parentheses
            court_abbrev = self._abbreviate_court(court)
            parenthetical = f" ({court_abbrev} {year})" if court_abbrev and year else f" ({year})" if year else ""
            
            # Format the pinpoint citation if available
            pincite = f", {source.get('pincite', '')}" if source.get("pincite") else ""
            
            # Format parallel citations if available
            parallel_citations = ""
            if "parallel_citations" in source and source["parallel_citations"]:
                parallel_citations = ", " + ", ".join(source["parallel_citations"])
            
            return f"{case_name}, {primary_citation}{parallel_citations}{pincite}{parenthetical}."
            
        elif source_type == "statute":
            # Statute citation - ALWD uses similar format to Bluebook
            title = source.get("title", "")
            code = source.get("code", "")
            section = source.get("section", "")
            year = source.get("year", "")
            
            # Basic format: Title # Code § Section (Year)
            title_part = f"{title} " if title else ""
            year_part = f" ({year})" if year else ""
            
            return f"{title_part}{code} § {section}{year_part}."
            
        elif source_type == "regulation":
            # Regulation citation
            title = source.get("title", "")
            code = source.get("code", "")
            section = source.get("section", "")
            year = source.get("year", "")
            
            # Basic format: Title # C.F.R. § Section (Year)
            year_part = f" ({year})" if year else ""
            
            return f"{title} {code} § {section}{year_part}."
            
        elif source_type == "article":
            # Law review article
            authors = source.get("authors", [])
            author_text = ""
            
            if authors:
                if len(authors) == 1:
                    author = authors[0]
                    first_name = author.get("first_name", "")
                    last_name = author.get("last_name", "")
                    author_text = f"{first_name} {last_name}"
                elif len(authors) > 1:
                    first_author = authors[0]
                    first_name = first_author.get("first_name", "")
                    last_name = first_author.get("last_name", "")
                    author_text = f"{first_name} {last_name} et al."
            
            title = source.get("title", "")
            journal = source.get("journal", "")
            volume = source.get("volume", "")
            page = source.get("page", "")
            year = source.get("year", "")
            
            # Basic format: Author, Title, Volume Journal Page (Year)
            year_part = f" ({year})" if year else ""
            
            return f"{author_text}, {title}, {volume} {journal} {page}{year_part}."
            
        elif source_type == "treatise":
            # Treatise/book citation
            authors = source.get("authors", [])
            author_text = ""
            
            if authors:
                if len(authors) == 1:
                    author = authors[0]
                    first_name = author.get("first_name", "")
                    last_name = author.get("last_name", "")
                    author_text = f"{first_name} {last_name}"
                elif len(authors) > 1:
                    first_author = authors[0]
                    first_name = first_author.get("first_name", "")
                    last_name = first_author.get("last_name", "")
                    author_text = f"{first_name} {last_name} et al."
            
            title = source.get("title", "")
            edition = source.get("edition", "")
            volume = source.get("volume", "")
            publisher = source.get("publisher", "")
            year = source.get("year", "")
            
            # Basic format: Author, Title Volume (Edition Year)
            volume_part = f"{volume} " if volume else ""
            edition_part = f"{edition} ed. " if edition else ""
            year_part = f"{year}" if year else ""
            parenthetical = f" ({edition_part}{year_part})" if edition_part or year_part else ""
            
            return f"{author_text}, {volume_part}{title}{parenthetical}."
            
        else:
            # Generic fallback - similar to Bluebook
            return self._generate_bluebook_citation(source, source_type) 