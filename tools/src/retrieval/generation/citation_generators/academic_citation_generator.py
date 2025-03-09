"""
Academic Citation Generator

This module provides the AcademicCitationGenerator class for generating citations
for academic sources like journal articles, books, and conference papers.
"""

from typing import Any, Dict, List, Optional, Union
from .base_citation_generator import BaseCitationGenerator

class AcademicCitationGenerator(BaseCitationGenerator):
    """
    Academic citation generator for scholarly sources.
    
    Generates citations for academic sources like journal articles, books,
    conference papers, dissertations/theses, and book chapters in various styles
    including APA, MLA, Chicago, and Harvard.
    
    Attributes:
        config (Dict[str, Any]): Configuration options for the generator.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the academic citation generator.
        
        Args:
            config: Optional configuration dictionary, may include:
                - default_style: Default citation style to use.
                - include_doi: Whether to include DOIs in citations (default True).
                - include_url: Whether to include URLs (default True).
                - abbreviate_journal: Whether to abbreviate journal names (default False).
        """
        default_config = {
            'default_style': 'APA',
            'include_doi': True,
            'include_url': True,
            'abbreviate_journal': False
        }
        
        if config:
            default_config.update(config)
            
        super().__init__(default_config)
    
    def generate_citation(self, source: Dict[str, Any], style: Optional[str] = None) -> str:
        """
        Generate an academic citation for the given source.
        
        Args:
            source: Dictionary containing source metadata including:
                - type: "article", "book", "conference", "thesis", "chapter"
                - title: Title of the work
                - authors: List of author dictionaries with first_name, last_name
                - year: Publication year
                - journal/publisher: Journal name or publisher
                - volume, issue, pages: For articles
                - doi: Digital Object Identifier
                - url: Web address
                - accessed: Access date for online sources
            style: Citation style (APA, MLA, Chicago, Harvard). If None,
                uses the default_style from config.
                
        Returns:
            A properly formatted citation string.
        """
        if not style:
            style = self.config.get('default_style', 'APA')
        
        # Check that source has minimum required fields
        if not self.validate_source(source, style):
            missing = self.get_missing_fields(source, style)
            return f"[Incomplete citation - missing: {', '.join(missing)}]"
        
        # Generate citation based on style and source type
        source_type = source.get('type', '').lower()
        
        if style.upper() == 'APA':
            return self._generate_apa_citation(source, source_type)
        elif style.upper() == 'MLA':
            return self._generate_mla_citation(source, source_type)
        elif style.upper() == 'CHICAGO':
            return self._generate_chicago_citation(source, source_type)
        elif style.upper() == 'HARVARD':
            return self._generate_harvard_citation(source, source_type)
        else:
            # Fallback to APA if style is not supported
            return self._generate_apa_citation(source, source_type)
    
    def generate_citations(self, sources: List[Dict[str, Any]], style: Optional[str] = None) -> List[str]:
        """
        Generate academic citations for multiple sources.
        
        Args:
            sources: List of source dictionaries.
            style: Citation style (APA, MLA, Chicago, Harvard).
            
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
        return ["APA", "MLA", "CHICAGO", "HARVARD"]
    
    def get_required_fields(self, style: Optional[str] = None) -> List[str]:
        """
        Get a list of required fields for generating academic citations.
        
        Different academic citation styles and source types require different
        fields, but some core fields are always required.
        
        Args:
            style: Optional citation style to get required fields for.
                
        Returns:
            List of required field names.
        """
        # Core fields required for all styles
        core_fields = ["type", "title", "authors"]
        
        # Add style-specific requirements
        if not style or style.upper() == 'APA':
            return core_fields + ["year"]
        
        # All styles generally need these basics
        return core_fields
    
    def _generate_apa_citation(self, source: Dict[str, Any], source_type: str) -> str:
        """
        Generate an APA style citation.
        
        Args:
            source: Source dictionary.
            source_type: Type of academic source.
            
        Returns:
            APA formatted citation string.
        """
        # Format authors
        authors = source.get("authors", [])
        author_text = self.format_author_names(authors, "APA") if authors else ""
        
        # Format date
        year = source.get("year", "")
        month = source.get("month", "")
        day = source.get("day", "")
        date = {"year": year, "month": month, "day": day}
        date_text = f"({self.format_date(date, 'APA')})" if year else "(n.d.)"
        
        # Format title
        title = source.get("title", "")
        
        if source_type == "article":
            # Journal article
            journal = source.get("journal", "")
            volume = source.get("volume", "")
            issue = source.get("issue", "")
            pages = source.get("pages", "")
            doi = source.get("doi", "")
            
            # Journal formatting
            journal_part = f"{journal}"
            if self.config.get("abbreviate_journal", False) and source.get("journal_abbrev"):
                journal_part = source.get("journal_abbrev", "")
            
            # Volume/issue formatting
            volume_issue = ""
            if volume:
                volume_issue = f", {volume}"
                if issue:
                    volume_issue += f"({issue})"
            
            # Pages
            pages_part = f", {pages}" if pages else ""
            
            # DOI
            doi_part = f". https://doi.org/{doi}" if doi and self.config.get("include_doi", True) else ""
            
            return f"{author_text} {date_text}. {title}. {journal_part}{volume_issue}{pages_part}{doi_part}"
            
        elif source_type == "book":
            # Book
            publisher = source.get("publisher", "")
            location = source.get("location", "")
            edition = source.get("edition", "")
            
            # Handle location and publisher
            location_publisher = ""
            if location and publisher:
                location_publisher = f". {location}: {publisher}"
            elif publisher:
                location_publisher = f". {publisher}"
                
            # Edition
            edition_part = f" ({edition} ed.)" if edition else ""
            
            return f"{author_text} {date_text}. {title}{edition_part}{location_publisher}"
            
        elif source_type == "conference":
            # Conference paper
            conference = source.get("conference", "")
            location = source.get("location", "")
            
            conference_part = f". In {conference}"
            location_part = f", {location}" if location else ""
            
            return f"{author_text} {date_text}. {title}{conference_part}{location_part}"
            
        elif source_type == "thesis":
            # Thesis or dissertation
            thesis_type = source.get("thesis_type", "Doctoral dissertation")
            university = source.get("university", "")
            
            university_part = f", {university}" if university else ""
            
            return f"{author_text} {date_text}. {title} [{thesis_type}{university_part}]"
            
        elif source_type == "chapter":
            # Book chapter
            editors = source.get("editors", [])
            editor_text = self.format_author_names(editors, "APA") if editors else ""
            book_title = source.get("book_title", "")
            publisher = source.get("publisher", "")
            location = source.get("location", "")
            pages = source.get("pages", "")
            
            # Editor text
            editor_part = f"In {editor_text} (Ed.)" if editor_text else ""
            
            # Book title
            book_title_part = f", {book_title}" if book_title else ""
            
            # Pages
            pages_part = f" (pp. {pages})" if pages else ""
            
            # Publisher info
            location_publisher = ""
            if location and publisher:
                location_publisher = f". {location}: {publisher}"
            elif publisher:
                location_publisher = f". {publisher}"
                
            return f"{author_text} {date_text}. {title}. {editor_part}{book_title_part}{pages_part}{location_publisher}"
            
        else:
            # Generic fallback
            source_info = source.get("source", "")
            url = source.get("url", "")
            url_part = f". Retrieved from {url}" if url and self.config.get("include_url", True) else ""
            
            return f"{author_text} {date_text}. {title}. {source_info}{url_part}"
    
    def _generate_mla_citation(self, source: Dict[str, Any], source_type: str) -> str:
        """
        Generate an MLA style citation.
        
        Args:
            source: Source dictionary.
            source_type: Type of academic source.
            
        Returns:
            MLA formatted citation string.
        """
        # Format authors
        authors = source.get("authors", [])
        author_text = self.format_author_names(authors, "MLA") if authors else ""
        
        # Format title
        title = source.get("title", "")
        
        # Year
        year = source.get("year", "")
        
        if source_type == "article":
            # Journal article
            journal = source.get("journal", "")
            volume = source.get("volume", "")
            issue = source.get("issue", "")
            pages = source.get("pages", "")
            year = source.get("year", "")
            
            # MLA uses quotes for article titles and italics for journals
            journal_part = f'{journal}'
            if self.config.get("abbreviate_journal", False) and source.get("journal_abbrev"):
                journal_part = source.get("journal_abbrev", "")
            
            # Volume and issue
            volume_issue = ""
            if volume:
                volume_issue = f", vol. {volume}"
                if issue:
                    volume_issue += f", no. {issue}"
            
            # Pages
            pages_part = f", pp. {pages}" if pages else ""
            
            # Year
            year_part = f", {year}" if year else ""
            
            # DOI or URL (MLA prefers URLs in most cases)
            url = source.get("url", "")
            doi = source.get("doi", "")
            
            online_part = ""
            if url and self.config.get("include_url", True):
                online_part = f". {url}"
            elif doi and self.config.get("include_doi", True):
                online_part = f". DOI: {doi}"
            
            return f"{author_text}. \"{title}.\" {journal_part}{volume_issue}{year_part}{pages_part}{online_part}"
            
        elif source_type == "book":
            # Book
            publisher = source.get("publisher", "")
            year = source.get("year", "")
            
            # MLA uses italics for book titles
            publisher_year = ""
            if publisher and year:
                publisher_year = f", {publisher}, {year}"
            elif publisher:
                publisher_year = f", {publisher}"
            elif year:
                publisher_year = f", {year}"
                
            return f"{author_text}. {title}{publisher_year}."
            
        elif source_type == "conference":
            # Conference paper
            conference = source.get("conference", "")
            location = source.get("location", "")
            year = source.get("year", "")
            
            conference_part = f" {conference}"
            location_year = ""
            if location and year:
                location_year = f", {location}, {year}"
            elif location:
                location_year = f", {location}"
            elif year:
                location_year = f", {year}"
                
            return f"{author_text}. \"{title}.\" {conference_part}{location_year}."
            
        elif source_type == "thesis":
            # Thesis or dissertation
            thesis_type = source.get("thesis_type", "Dissertation")
            university = source.get("university", "")
            year = source.get("year", "")
            
            university_year = ""
            if university and year:
                university_year = f", {university}, {year}"
            elif university:
                university_year = f", {university}"
                
            return f"{author_text}. \"{title}.\" {thesis_type}{university_year}."
            
        elif source_type == "chapter":
            # Book chapter
            book_title = source.get("book_title", "")
            editors = source.get("editors", [])
            editor_text = self.format_author_names(editors, "MLA") if editors else ""
            publisher = source.get("publisher", "")
            year = source.get("year", "")
            pages = source.get("pages", "")
            
            # Editor information
            editor_part = f", edited by {editor_text}" if editor_text else ""
            
            # Publisher and year
            publisher_year = ""
            if publisher and year:
                publisher_year = f", {publisher}, {year}"
            elif publisher:
                publisher_year = f", {publisher}"
            elif year:
                publisher_year = f", {year}"
                
            # Pages
            pages_part = f", pp. {pages}" if pages else ""
            
            return f"{author_text}. \"{title}.\" {book_title}{editor_part}{publisher_year}{pages_part}."
            
        else:
            # Generic fallback
            source_info = source.get("source", "")
            year = source.get("year", "")
            url = source.get("url", "")
            
            year_part = f", {year}" if year else ""
            url_part = f". {url}" if url and self.config.get("include_url", True) else ""
            
            return f"{author_text}. \"{title}.\" {source_info}{year_part}{url_part}"
    
    def _generate_chicago_citation(self, source: Dict[str, Any], source_type: str) -> str:
        """
        Generate a Chicago style citation.
        
        Args:
            source: Source dictionary.
            source_type: Type of academic source.
            
        Returns:
            Chicago formatted citation string.
        """
        # Format authors
        authors = source.get("authors", [])
        author_text = self.format_author_names(authors, "CHICAGO") if authors else ""
        
        # Format title
        title = source.get("title", "")
        
        # Format date
        year = source.get("year", "")
        month = source.get("month", "")
        day = source.get("day", "")
        date = {"year": year, "month": month, "day": day}
        date_text = self.format_date(date, "CHICAGO") if year else "n.d."
        
        if source_type == "article":
            # Journal article
            journal = source.get("journal", "")
            volume = source.get("volume", "")
            issue = source.get("issue", "")
            pages = source.get("pages", "")
            
            # Journal formatting (Chicago uses italics)
            journal_part = f"{journal}"
            if self.config.get("abbreviate_journal", False) and source.get("journal_abbrev"):
                journal_part = source.get("journal_abbrev", "")
            
            # Volume/issue formatting
            volume_issue = ""
            if volume:
                volume_issue = f" {volume}"
                if issue:
                    volume_issue += f", no. {issue}"
            
            # Pages
            pages_part = f": {pages}" if pages else ""
            
            # Date (year) in Chicago style
            year_part = f" ({year})" if year else ""
            
            # DOI or URL
            doi = source.get("doi", "")
            url = source.get("url", "")
            
            online_part = ""
            if doi and self.config.get("include_doi", True):
                online_part = f". https://doi.org/{doi}"
            elif url and self.config.get("include_url", True):
                online_part = f". {url}"
            
            return f"{author_text}. \"{title}.\" {journal_part}{volume_issue}{year_part}{pages_part}{online_part}."
            
        elif source_type == "book":
            # Book
            publisher = source.get("publisher", "")
            location = source.get("location", "")
            
            # Chicago puts location: publisher
            location_publisher = ""
            if location and publisher:
                location_publisher = f". {location}: {publisher}"
            elif publisher:
                location_publisher = f". {publisher}"
                
            return f"{author_text}. {title}{location_publisher}, {date_text}."
            
        elif source_type == "conference":
            # Conference paper
            conference = source.get("conference", "")
            location = source.get("location", "")
            
            conference_part = f". Paper presented at {conference}"
            location_part = f", {location}" if location else ""
            date_part = f", {date_text}" if date_text != "n.d." else ""
            
            return f"{author_text}. \"{title}\"{conference_part}{location_part}{date_part}."
            
        elif source_type == "thesis":
            # Thesis or dissertation
            thesis_type = source.get("thesis_type", "PhD diss.")
            university = source.get("university", "")
            
            university_part = f", {university}" if university else ""
            date_part = f", {date_text}" if date_text != "n.d." else ""
            
            return f"{author_text}. \"{title}.\" {thesis_type}{university_part}{date_part}."
            
        elif source_type == "chapter":
            # Book chapter
            book_title = source.get("book_title", "")
            editors = source.get("editors", [])
            editor_text = self.format_author_names(editors, "CHICAGO") if editors else ""
            publisher = source.get("publisher", "")
            location = source.get("location", "")
            pages = source.get("pages", "")
            
            # Editor text
            editor_part = f", edited by {editor_text}" if editor_text else ""
            
            # Publisher info
            location_publisher = ""
            if location and publisher:
                location_publisher = f". {location}: {publisher}"
            elif publisher:
                location_publisher = f". {publisher}"
                
            # Pages
            pages_part = f", {pages}" if pages else ""
            
            return f"{author_text}. \"{title}.\" In {book_title}{editor_part}{pages_part}{location_publisher}, {date_text}."
            
        else:
            # Generic fallback
            source_info = source.get("source", "")
            url = source.get("url", "")
            url_part = f". {url}" if url and self.config.get("include_url", True) else ""
            
            return f"{author_text}. \"{title}.\" {source_info}, {date_text}{url_part}."
    
    def _generate_harvard_citation(self, source: Dict[str, Any], source_type: str) -> str:
        """
        Generate a Harvard style citation.
        
        Args:
            source: Source dictionary.
            source_type: Type of academic source.
            
        Returns:
            Harvard formatted citation string.
        """
        # Format authors
        authors = source.get("authors", [])
        author_text = ""
        if authors:
            # Harvard uses last names and initials
            author_list = []
            for author in authors:
                last_name = author.get("last_name", "")
                first_initial = author.get("first_name", "")[0] if author.get("first_name") else ""
                middle_initial = author.get("middle_name", "")[0] if author.get("middle_name") else ""
                
                if middle_initial:
                    author_list.append(f"{last_name}, {first_initial}.{middle_initial}.")
                else:
                    author_list.append(f"{last_name}, {first_initial}.")
            
            if len(author_list) == 1:
                author_text = author_list[0]
            elif len(author_list) == 2:
                author_text = f"{author_list[0]} and {author_list[1]}"
            elif len(author_list) > 2:
                author_text = f"{author_list[0]} et al."
        
        # Year
        year = source.get("year", "")
        year_text = f"({year})" if year else "(n.d.)"
        
        # Title
        title = source.get("title", "")
        
        if source_type == "article":
            # Journal article
            journal = source.get("journal", "")
            volume = source.get("volume", "")
            issue = source.get("issue", "")
            pages = source.get("pages", "")
            
            # Journal formatting
            journal_part = f"{journal}"
            if self.config.get("abbreviate_journal", False) and source.get("journal_abbrev"):
                journal_part = source.get("journal_abbrev", "")
            
            # Volume and issue
            volume_issue = ""
            if volume:
                volume_issue = f", {volume}"
                if issue:
                    volume_issue += f"({issue})"
            
            # Pages
            pages_part = f", pp. {pages}" if pages else ""
            
            # DOI
            doi = source.get("doi", "")
            doi_part = f". DOI: {doi}" if doi and self.config.get("include_doi", True) else ""
            
            return f"{author_text} {year_text}, '{title}', {journal_part}{volume_issue}{pages_part}{doi_part}."
            
        elif source_type == "book":
            # Book
            publisher = source.get("publisher", "")
            location = source.get("location", "")
            edition = source.get("edition", "")
            
            # Edition
            edition_part = f", {edition} edn" if edition else ""
            
            # Publisher info
            location_publisher = ""
            if location and publisher:
                location_publisher = f", {location}: {publisher}"
            elif publisher:
                location_publisher = f", {publisher}"
                
            return f"{author_text} {year_text}, {title}{edition_part}{location_publisher}."
            
        elif source_type == "conference":
            # Conference paper
            conference = source.get("conference", "")
            location = source.get("location", "")
            
            conference_part = f", {conference}"
            location_part = f", {location}" if location else ""
            
            return f"{author_text} {year_text}, '{title}'{conference_part}{location_part}."
            
        elif source_type == "thesis":
            # Thesis or dissertation
            thesis_type = source.get("thesis_type", "PhD thesis")
            university = source.get("university", "")
            
            university_part = f", {university}" if university else ""
            
            return f"{author_text} {year_text}, '{title}', {thesis_type}{university_part}."
            
        elif source_type == "chapter":
            # Book chapter
            book_title = source.get("book_title", "")
            editors = source.get("editors", [])
            
            # Format editors
            editor_text = ""
            if editors:
                # Harvard uses 'ed. by' or 'eds. by'
                editor_names = []
                for editor in editors:
                    last_name = editor.get("last_name", "")
                    first_initial = editor.get("first_name", "")[0] if editor.get("first_name") else ""
                    editor_names.append(f"{last_name}, {first_initial}.")
                
                if len(editor_names) == 1:
                    editor_text = f", ed. by {editor_names[0]}"
                else:
                    editor_text = f", eds. by {' and '.join(editor_names)}"
            
            publisher = source.get("publisher", "")
            location = source.get("location", "")
            pages = source.get("pages", "")
            
            # Book title
            book_part = f"in {book_title}"
            
            # Publisher info
            location_publisher = ""
            if location and publisher:
                location_publisher = f", {location}: {publisher}"
            elif publisher:
                location_publisher = f", {publisher}"
                
            # Pages
            pages_part = f", pp. {pages}" if pages else ""
            
            return f"{author_text} {year_text}, '{title}', {book_part}{editor_text}{location_publisher}{pages_part}."
            
        else:
            # Generic fallback
            source_info = source.get("source", "")
            url = source.get("url", "")
            accessed = source.get("accessed", "")
            
            url_part = ""
            if url and self.config.get("include_url", True):
                url_part = f". Available at: {url}"
                if accessed:
                    url_part += f" [Accessed {accessed}]"
            
            return f"{author_text} {year_text}, '{title}'{source_info}{url_part}." 