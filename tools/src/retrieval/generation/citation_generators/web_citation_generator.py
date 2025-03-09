"""
Web Citation Generator

This module provides the WebCitationGenerator class for generating citations
for web resources such as websites, blog posts, social media, and online articles.
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from .base_citation_generator import BaseCitationGenerator

class WebCitationGenerator(BaseCitationGenerator):
    """
    Web citation generator for online resources.
    
    Generates citations for web-based sources including websites, blog posts,
    online articles, social media content, and other digital resources in
    various styles.
    
    Attributes:
        config (Dict[str, Any]): Configuration options for the generator.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the web citation generator.
        
        Args:
            config: Optional configuration dictionary, may include:
                - default_style: Default citation style to use.
                - include_access_date: Whether to include access date (default True).
                - access_date_format: Format for access dates.
                - include_url: Whether to include URLs (default True).
                - default_publisher: Default publisher to use if none provided.
        """
        default_config = {
            'default_style': 'APA',
            'include_access_date': True,
            'access_date_format': '%B %d, %Y',
            'include_url': True,
            'default_publisher': 'Web'
        }
        
        if config:
            default_config.update(config)
            
        super().__init__(default_config)
    
    def generate_citation(self, source: Dict[str, Any], style: Optional[str] = None) -> str:
        """
        Generate a web citation for the given source.
        
        Args:
            source: Dictionary containing source metadata including:
                - type: "webpage", "article", "blog", "social", etc.
                - title: Title of the web resource
                - authors: List of author dictionaries with first_name, last_name
                - date: Publication or last update date
                - site_name: Name of the website or platform
                - url: Web address
                - accessed: Access date (if none, current date may be used)
            style: Citation style (APA, MLA, Chicago, etc.) If None,
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
        Generate web citations for multiple sources.
        
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
        Get a list of required fields for generating web citations.
        
        Different web citation styles and source types require different
        fields, but some core fields are always required.
        
        Args:
            style: Optional citation style to get required fields for.
                
        Returns:
            List of required field names.
        """
        # Core fields required for all styles
        core_fields = ["title", "url"]
        
        # Add style-specific requirements
        if not style or style.upper() in ['APA', 'CHICAGO', 'HARVARD']:
            return core_fields + ["site_name"]
        
        # MLA requires author or site name
        return core_fields
    
    def _format_access_date(self, date_str: Optional[str] = None) -> str:
        """
        Format the access date according to the configured format.
        
        Args:
            date_str: Date string in ISO format (YYYY-MM-DD),
                if None, current date is used.
            
        Returns:
            Formatted date string.
        """
        format_str = self.config.get('access_date_format', '%B %d, %Y')
        
        if date_str:
            try:
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                return date_obj.strftime(format_str)
            except ValueError:
                # If date cannot be parsed, return it as is
                return date_str
        else:
            # Use current date if none provided
            return datetime.now().strftime(format_str)
    
    def _generate_apa_citation(self, source: Dict[str, Any], source_type: str) -> str:
        """
        Generate an APA style citation for a web resource.
        
        Args:
            source: Source dictionary.
            source_type: Type of web resource.
            
        Returns:
            APA formatted citation string.
        """
        # Format authors
        authors = source.get("authors", [])
        author_text = self.format_author_names(authors, "APA") if authors else ""
        
        # Group name as author for organizations
        group_author = source.get("group_author", "")
        if not author_text and group_author:
            author_text = group_author
        
        # Format date
        date = source.get("date", "")
        date_text = f"({date})" if date else "(n.d.)"
        
        # Format title
        title = source.get("title", "")
        
        # Format site name
        site_name = source.get("site_name", "")
        if not site_name:
            site_name = self.config.get("default_publisher", "Web")
        
        # Format URL
        url = source.get("url", "")
        url_part = f" {url}" if url and self.config.get("include_url", True) else ""
        
        # Format access date
        accessed = source.get("accessed", "")
        access_part = ""
        if self.config.get("include_access_date", True):
            access_date = self._format_access_date(accessed)
            access_part = f" Retrieved {access_date}, from"
        
        if source_type == "webpage":
            # APA webpage citation
            if author_text:
                return f"{author_text} {date_text}. {title}. {site_name}.{access_part}{url_part}"
            else:
                return f"{title}. {date_text}. {site_name}.{access_part}{url_part}"
            
        elif source_type == "article":
            # Online article
            # Same format as webpage for APA
            if author_text:
                return f"{author_text} {date_text}. {title}. {site_name}.{access_part}{url_part}"
            else:
                return f"{title}. {date_text}. {site_name}.{access_part}{url_part}"
                
        elif source_type == "blog":
            # Blog post
            # Add [Blog post] after title
            return f"{author_text} {date_text}. {title} [Blog post]. {site_name}.{access_part}{url_part}"
            
        elif source_type == "social":
            # Social media post
            platform = source.get("platform", "social media")
            
            # Format differently based on platform
            if platform.lower() == "twitter" or platform.lower() == "x":
                username = source.get("username", "")
                username_part = f" [@{username}]" if username else ""
                
                return f"{author_text}{username_part}. {date_text}. {title} [{platform}].{access_part}{url_part}"
            else:
                return f"{author_text}. {date_text}. {title} [{platform} post].{access_part}{url_part}"
                
        else:
            # Generic web citation
            if author_text:
                return f"{author_text} {date_text}. {title}. {site_name}.{access_part}{url_part}"
            else:
                return f"{title}. {date_text}. {site_name}.{access_part}{url_part}"
    
    def _generate_mla_citation(self, source: Dict[str, Any], source_type: str) -> str:
        """
        Generate an MLA style citation for a web resource.
        
        Args:
            source: Source dictionary.
            source_type: Type of web resource.
            
        Returns:
            MLA formatted citation string.
        """
        # Format authors
        authors = source.get("authors", [])
        author_text = self.format_author_names(authors, "MLA") if authors else ""
        
        # Group name as author for organizations
        group_author = source.get("group_author", "")
        if not author_text and group_author:
            author_text = group_author
        
        # Format title
        title = source.get("title", "")
        
        # Format site name
        site_name = source.get("site_name", "")
        if not site_name:
            site_name = self.config.get("default_publisher", "Web")
        
        # Format date
        date = source.get("date", "")
        
        # Format URL
        url = source.get("url", "")
        url_part = f" {url}" if url and self.config.get("include_url", True) else ""
        
        # Format access date
        accessed = source.get("accessed", "")
        access_part = ""
        if self.config.get("include_access_date", True):
            access_date = self._format_access_date(accessed)
            access_part = f" Accessed {access_date}."
        
        if source_type == "webpage":
            # MLA webpage citation
            if author_text:
                return f"{author_text}. \"{title}.\" {site_name}, {date if date else 'n.d.'},{url_part}.{access_part}"
            else:
                return f"\"{title}.\" {site_name}, {date if date else 'n.d.'},{url_part}.{access_part}"
            
        elif source_type == "article":
            # Online article
            # MLA online article adds "online" designator
            if author_text:
                return f"{author_text}. \"{title}.\" {site_name}, {date if date else 'n.d.'}, online,{url_part}.{access_part}"
            else:
                return f"\"{title}.\" {site_name}, {date if date else 'n.d.'}, online,{url_part}.{access_part}"
                
        elif source_type == "blog":
            # Blog post
            # Specify as a blog
            if author_text:
                return f"{author_text}. \"{title}.\" {site_name}, {date if date else 'n.d.'}, blog,{url_part}.{access_part}"
            else:
                return f"\"{title}.\" {site_name}, {date if date else 'n.d.'}, blog,{url_part}.{access_part}"
                
        elif source_type == "social":
            # Social media post
            platform = source.get("platform", "social media")
            username = source.get("username", "")
            username_part = f" (@{username})" if username else ""
            
            if author_text:
                return f"{author_text}{username_part}. \"{title}.\" {platform}, {date if date else 'n.d.'},{url_part}.{access_part}"
            else:
                return f"\"{title}.\" {platform}, {date if date else 'n.d.'},{url_part}.{access_part}"
                
        else:
            # Generic web citation
            if author_text:
                return f"{author_text}. \"{title}.\" {site_name}, {date if date else 'n.d.'},{url_part}.{access_part}"
            else:
                return f"\"{title}.\" {site_name}, {date if date else 'n.d.'},{url_part}.{access_part}"
    
    def _generate_chicago_citation(self, source: Dict[str, Any], source_type: str) -> str:
        """
        Generate a Chicago style citation for a web resource.
        
        Args:
            source: Source dictionary.
            source_type: Type of web resource.
            
        Returns:
            Chicago formatted citation string.
        """
        # Format authors
        authors = source.get("authors", [])
        author_text = self.format_author_names(authors, "CHICAGO") if authors else ""
        
        # Group name as author for organizations
        group_author = source.get("group_author", "")
        if not author_text and group_author:
            author_text = group_author
        
        # Format title
        title = source.get("title", "")
        
        # Format site name
        site_name = source.get("site_name", "")
        if not site_name:
            site_name = self.config.get("default_publisher", "Web")
        
        # Format date
        date = source.get("date", "")
        
        # Format URL
        url = source.get("url", "")
        url_part = f"{url}" if url and self.config.get("include_url", True) else ""
        
        # Format access date
        accessed = source.get("accessed", "")
        access_part = ""
        if self.config.get("include_access_date", True):
            access_date = self._format_access_date(accessed)
            if url_part:
                access_part = f"accessed {access_date}, "
        
        if source_type == "webpage":
            # Chicago webpage citation
            if author_text:
                return f"{author_text}. \"{title}.\" {site_name}. {date if date else 'n.d.'}. {access_part}{url_part}."
            else:
                return f"\"{title}.\" {site_name}. {date if date else 'n.d.'}. {access_part}{url_part}."
            
        elif source_type == "article":
            # Online article
            if author_text:
                return f"{author_text}. \"{title}.\" {site_name}, {date if date else 'n.d.'}. {access_part}{url_part}."
            else:
                return f"\"{title}.\" {site_name}, {date if date else 'n.d.'}. {access_part}{url_part}."
                
        elif source_type == "blog":
            # Blog post
            if author_text:
                return f"{author_text}. \"{title}.\" {site_name} (blog), {date if date else 'n.d.'}. {access_part}{url_part}."
            else:
                return f"\"{title}.\" {site_name} (blog), {date if date else 'n.d.'}. {access_part}{url_part}."
                
        elif source_type == "social":
            # Social media post
            platform = source.get("platform", "social media")
            username = source.get("username", "")
            username_part = f" (@{username})" if username else ""
            
            if author_text:
                return f"{author_text}{username_part}. \"{title}.\" {platform}, {date if date else 'n.d.'}. {access_part}{url_part}."
            else:
                return f"\"{title}.\" {platform}, {date if date else 'n.d.'}. {access_part}{url_part}."
                
        else:
            # Generic web citation
            if author_text:
                return f"{author_text}. \"{title}.\" {site_name}, {date if date else 'n.d.'}. {access_part}{url_part}."
            else:
                return f"\"{title}.\" {site_name}, {date if date else 'n.d.'}. {access_part}{url_part}."
    
    def _generate_harvard_citation(self, source: Dict[str, Any], source_type: str) -> str:
        """
        Generate a Harvard style citation for a web resource.
        
        Args:
            source: Source dictionary.
            source_type: Type of web resource.
            
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
        
        # Group name as author for organizations
        group_author = source.get("group_author", "")
        if not author_text and group_author:
            author_text = group_author
        
        # Format date
        date = source.get("date", "")
        year_text = f"({date})" if date else "(n.d.)"
        
        # Format title
        title = source.get("title", "")
        
        # Format site name
        site_name = source.get("site_name", "")
        if not site_name:
            site_name = self.config.get("default_publisher", "Web")
        
        # Format URL
        url = source.get("url", "")
        url_part = f"Available at: {url}" if url and self.config.get("include_url", True) else ""
        
        # Format access date
        accessed = source.get("accessed", "")
        access_part = ""
        if self.config.get("include_access_date", True) and url_part:
            access_date = self._format_access_date(accessed)
            access_part = f" [Accessed {access_date}]"
        
        if source_type == "webpage":
            # Harvard webpage citation
            if author_text:
                return f"{author_text} {year_text}. {title}. [Online] {site_name}. {url_part}{access_part}."
            else:
                return f"{site_name} {year_text}. {title}. [Online]. {url_part}{access_part}."
            
        elif source_type == "article":
            # Online article
            if author_text:
                return f"{author_text} {year_text}. {title}. [Online] {site_name}. {url_part}{access_part}."
            else:
                return f"{site_name} {year_text}. {title}. [Online]. {url_part}{access_part}."
                
        elif source_type == "blog":
            # Blog post
            if author_text:
                return f"{author_text} {year_text}. {title}. [Blog] {site_name}. {url_part}{access_part}."
            else:
                return f"{site_name} {year_text}. {title}. [Blog]. {url_part}{access_part}."
                
        elif source_type == "social":
            # Social media post
            platform = source.get("platform", "social media")
            username = source.get("username", "")
            username_part = f" ({username})" if username else ""
            
            if author_text:
                return f"{author_text}{username_part} {year_text}. {title}. [{platform}]. {url_part}{access_part}."
            else:
                return f"{platform} {year_text}. {title}. {url_part}{access_part}."
                
        else:
            # Generic web citation
            if author_text:
                return f"{author_text} {year_text}. {title}. [Online] {site_name}. {url_part}{access_part}."
            else:
                return f"{site_name} {year_text}. {title}. [Online]. {url_part}{access_part}." 