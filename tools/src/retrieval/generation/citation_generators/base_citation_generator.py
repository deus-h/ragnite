"""
Base Citation Generator

This module defines the BaseCitationGenerator abstract class that all citation generators must implement.
Citation generators create properly formatted citations for various types of sources.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, TypeVar, Generic

T = TypeVar('T')

class BaseCitationGenerator(ABC):
    """
    Abstract base class for citation generators.
    
    Citation generators create properly formatted citations for various types of sources,
    including academic works, legal documents, web resources, and more. They handle
    different citation styles and formats based on the source type and metadata.
    
    Attributes:
        config (Dict[str, Any]): Configuration options for the citation generator.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the citation generator.
        
        Args:
            config: Optional configuration dictionary for the generator.
        """
        self.config = config or {}
    
    @abstractmethod
    def generate_citation(self, 
                         source: Dict[str, Any], 
                         style: Optional[str] = None) -> str:
        """
        Generate a citation for the given source.
        
        Args:
            source: Dictionary containing source metadata such as title, author, date, etc.
            style: Optional citation style (e.g., "APA", "MLA", "Chicago", "Bluebook").
                If None, the default style configured for the generator is used.
            
        Returns:
            A properly formatted citation string.
        """
        pass
    
    @abstractmethod
    def generate_citations(self, 
                          sources: List[Dict[str, Any]], 
                          style: Optional[str] = None) -> List[str]:
        """
        Generate citations for multiple sources.
        
        Args:
            sources: List of source dictionaries.
            style: Optional citation style.
            
        Returns:
            List of citation strings in the same order as the input sources.
        """
        pass
    
    @abstractmethod
    def get_supported_styles(self) -> List[str]:
        """
        Get a list of citation styles supported by this generator.
        
        Returns:
            List of supported citation style names.
        """
        pass
    
    @abstractmethod
    def get_required_fields(self, style: Optional[str] = None) -> List[str]:
        """
        Get a list of required fields for generating citations.
        
        Args:
            style: Optional citation style to get required fields for.
                If None, returns fields required for all supported styles.
            
        Returns:
            List of required field names.
        """
        pass
    
    def validate_source(self, source: Dict[str, Any], style: Optional[str] = None) -> bool:
        """
        Validate that the source contains all required fields for citation generation.
        
        Args:
            source: Source dictionary to validate.
            style: Optional citation style to validate against.
            
        Returns:
            True if the source is valid, False otherwise.
        """
        required_fields = self.get_required_fields(style)
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
        required_fields = self.get_required_fields(style)
        return [field for field in required_fields if field not in source or not source[field]]
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration of the citation generator.
        
        Returns:
            A dictionary containing the current configuration.
        """
        return self.config.copy()
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Update the configuration of the citation generator.
        
        Args:
            config: A dictionary containing configuration options to update.
        """
        self.config.update(config)
    
    def format_author_names(self, authors: List[Dict[str, str]], style: str) -> str:
        """
        Format author names according to the specified citation style.
        
        Args:
            authors: List of author dictionaries, each containing "first_name",
                "middle_name" (optional), and "last_name".
            style: Citation style to format according to.
            
        Returns:
            Formatted author string.
        """
        if not authors:
            return ""
        
        # Default formatting (APA-like)
        if style.upper() == "APA" or style == "default":
            if len(authors) == 1:
                author = authors[0]
                last_name = author.get("last_name", "")
                first_initial = author.get("first_name", "")[0] if author.get("first_name") else ""
                middle_initial = author.get("middle_name", "")[0] if author.get("middle_name") else ""
                
                if middle_initial:
                    return f"{last_name}, {first_initial}. {middle_initial}."
                else:
                    return f"{last_name}, {first_initial}."
            elif len(authors) == 2:
                author1 = self.format_author_names([authors[0]], style)
                author2 = self.format_author_names([authors[1]], style)
                return f"{author1} & {author2}"
            else:
                first_authors = [self.format_author_names([author], style) for author in authors[:-1]]
                last_author = self.format_author_names([authors[-1]], style)
                return f"{', '.join(first_authors)}, & {last_author}"
        
        elif style.upper() == "MLA":
            if len(authors) == 1:
                author = authors[0]
                last_name = author.get("last_name", "")
                first_name = author.get("first_name", "")
                middle_name = author.get("middle_name", "")
                
                if middle_name:
                    return f"{last_name}, {first_name} {middle_name}"
                else:
                    return f"{last_name}, {first_name}"
            elif len(authors) == 2:
                author1 = self.format_author_names([authors[0]], style)
                author2_last = authors[1].get("last_name", "")
                author2_first = authors[1].get("first_name", "")
                author2_middle = authors[1].get("middle_name", "")
                
                if author2_middle:
                    author2 = f"{author2_first} {author2_middle} {author2_last}"
                else:
                    author2 = f"{author2_first} {author2_last}"
                
                return f"{author1} and {author2}"
            else:
                return f"{authors[0].get('last_name', '')}, et al."
        
        elif style.upper() == "CHICAGO":
            if len(authors) == 1:
                author = authors[0]
                last_name = author.get("last_name", "")
                first_name = author.get("first_name", "")
                middle_name = author.get("middle_name", "")
                
                if middle_name:
                    return f"{last_name}, {first_name} {middle_name}"
                else:
                    return f"{last_name}, {first_name}"
            elif len(authors) <= 3:
                names = []
                for i, author in enumerate(authors):
                    if i == 0:
                        # First author is last name first
                        last_name = author.get("last_name", "")
                        first_name = author.get("first_name", "")
                        middle_name = author.get("middle_name", "")
                        
                        if middle_name:
                            names.append(f"{last_name}, {first_name} {middle_name}")
                        else:
                            names.append(f"{last_name}, {first_name}")
                    else:
                        # Subsequent authors are first name first
                        last_name = author.get("last_name", "")
                        first_name = author.get("first_name", "")
                        middle_name = author.get("middle_name", "")
                        
                        if middle_name:
                            names.append(f"{first_name} {middle_name} {last_name}")
                        else:
                            names.append(f"{first_name} {last_name}")
                
                if len(names) == 2:
                    return f"{names[0]} and {names[1]}"
                else:
                    return f"{names[0]}, {names[1]}, and {names[2]}"
            else:
                return f"{authors[0].get('last_name', '')}, et al."
        
        # Default fallback for other styles
        return ", ".join([f"{a.get('last_name', '')}, {a.get('first_name', '')}" for a in authors])
    
    def format_date(self, date: Dict[str, int], style: str) -> str:
        """
        Format a date according to the specified citation style.
        
        Args:
            date: Dictionary with year, month (1-12), and day fields.
            style: Citation style to format according to.
            
        Returns:
            Formatted date string.
        """
        year = date.get("year", "")
        month = date.get("month", "")
        day = date.get("day", "")
        
        if not year:
            return "n.d."  # No date
        
        if style.upper() == "APA":
            if month and day:
                month_names = [
                    "January", "February", "March", "April", "May", "June",
                    "July", "August", "September", "October", "November", "December"
                ]
                return f"{year}, {month_names[month-1]} {day}"
            elif month:
                month_names = [
                    "January", "February", "March", "April", "May", "June",
                    "July", "August", "September", "October", "November", "December"
                ]
                return f"{year}, {month_names[month-1]}"
            else:
                return str(year)
        
        elif style.upper() == "MLA":
            return str(year)
        
        elif style.upper() == "CHICAGO":
            if month and day:
                month_names = [
                    "January", "February", "March", "April", "May", "June",
                    "July", "August", "September", "October", "November", "December"
                ]
                return f"{month_names[month-1]} {day}, {year}"
            elif month:
                month_names = [
                    "January", "February", "March", "April", "May", "June",
                    "July", "August", "September", "October", "November", "December"
                ]
                return f"{month_names[month-1]} {year}"
            else:
                return str(year)
        
        # Default format
        if month and day:
            return f"{year}-{month:02d}-{day:02d}"
        elif month:
            return f"{year}-{month:02d}"
        else:
            return str(year)
    
    def __repr__(self) -> str:
        """
        Return a string representation of the citation generator.
        
        Returns:
            String representation.
        """
        return f"{self.__class__.__name__}(styles={self.get_supported_styles()})" 