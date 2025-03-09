"""
Custom Citation Generator

This module provides the CustomCitationGenerator class for generating citations
based on user-defined templates for any type of source.
"""

from typing import Any, Dict, List, Optional, Union, Callable
import re
import json
from datetime import datetime
from .base_citation_generator import BaseCitationGenerator

class CustomCitationGenerator(BaseCitationGenerator):
    """
    Custom citation generator using user-defined templates.
    
    Allows users to define custom citation formats through templates with
    placeholders for source fields, enabling citation generation for any
    type of source in any desired format.
    
    Attributes:
        config (Dict[str, Any]): Configuration options for the generator.
        templates (Dict[str, str]): Citation templates for different styles.
        field_formatters (Dict[str, Callable]): Functions to format specific fields.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the custom citation generator.
        
        Args:
            config: Optional configuration dictionary, may include:
                - default_style: Default template style to use.
                - templates: Dict mapping style names to template strings.
                - date_format: Format for dates.
                - field_formatters: Dict mapping field names to formatter functions.
        """
        default_config = {
            'default_style': 'default',
            'templates': {
                'default': '{authors}. ({year}). {title}. {source}.'
            },
            'date_format': '%B %d, %Y',
        }
        
        if config:
            default_config.update(config)
            
        super().__init__(default_config)
        
        # Extract templates from config
        self.templates = self.config.get('templates', {})
        
        # Set up default field formatters
        self.field_formatters = {
            'authors': self._format_authors,
            'date': self._format_date,
            'accessed': self._format_access_date,
            'url': self._format_url,
            'pages': self._format_pages
        }
        
        # Update with any custom formatters
        if 'field_formatters' in self.config:
            self.field_formatters.update(self.config['field_formatters'])
    
    def generate_citation(self, source: Dict[str, Any], style: Optional[str] = None) -> str:
        """
        Generate a custom citation using the template for the given style.
        
        Args:
            source: Dictionary containing source metadata.
            style: Template style to use. If None, uses the default_style from config.
                
        Returns:
            A properly formatted citation string.
        """
        if not style:
            style = self.config.get('default_style', 'default')
        
        # Get the template for the specified style
        if style not in self.templates:
            # Fallback to default if style not found
            style = 'default'
            if style not in self.templates:
                return f"[Error: No template available for style '{style}']"
        
        template = self.templates[style]
        
        # Check required fields
        if not self.validate_source(source, style):
            missing = self.get_missing_fields(source, style)
            return f"[Incomplete citation - missing: {', '.join(missing)}]"
        
        # Process the template by replacing placeholders with source data
        return self._process_template(template, source)
    
    def generate_citations(self, sources: List[Dict[str, Any]], style: Optional[str] = None) -> List[str]:
        """
        Generate custom citations for multiple sources.
        
        Args:
            sources: List of source dictionaries.
            style: Template style to use.
            
        Returns:
            List of citation strings in the same order as the input sources.
        """
        return [self.generate_citation(source, style) for source in sources]
    
    def get_supported_styles(self) -> List[str]:
        """
        Get a list of citation styles (template names) supported by this generator.
        
        Returns:
            List of supported template style names.
        """
        return list(self.templates.keys())
    
    def get_required_fields(self, style: Optional[str] = None) -> List[str]:
        """
        Get a list of required fields for the given style's template.
        
        Analyzes the template to determine which fields are required.
        
        Args:
            style: Optional template style to get required fields for.
                
        Returns:
            List of required field names.
        """
        if not style:
            style = self.config.get('default_style', 'default')
            
        if style not in self.templates:
            style = 'default'
            if style not in self.templates:
                return []
                
        template = self.templates[style]
        
        # Extract field names from template
        # This will match {field_name} and {field_name:option} patterns
        field_pattern = r'\{([a-zA-Z0-9_]+)(?::[a-zA-Z0-9_]+)?\}'
        matches = re.findall(field_pattern, template)
        
        # Get unique field names
        return list(set(matches))
    
    def add_template(self, style_name: str, template: str) -> None:
        """
        Add a new citation template.
        
        Args:
            style_name: Name of the template style.
            template: Template string with field placeholders.
        """
        self.templates[style_name] = template
        
        # Update the config to reflect the new template
        self.config['templates'] = self.templates
    
    def remove_template(self, style_name: str) -> bool:
        """
        Remove a citation template.
        
        Args:
            style_name: Name of the template style to remove.
            
        Returns:
            True if template was removed, False if not found.
        """
        if style_name in self.templates:
            del self.templates[style_name]
            self.config['templates'] = self.templates
            return True
        return False
    
    def add_field_formatter(self, field_name: str, formatter: Callable) -> None:
        """
        Add a custom formatter for a specific field.
        
        Args:
            field_name: Name of the field to format.
            formatter: Function that takes field value and returns formatted string.
        """
        self.field_formatters[field_name] = formatter
    
    def _process_template(self, template: str, source: Dict[str, Any]) -> str:
        """
        Process a template by replacing field placeholders with formatted values.
        
        Args:
            template: Template string with field placeholders.
            source: Source dictionary with field values.
            
        Returns:
            Formatted citation string.
        """
        # Prepare a dictionary of formatted field values
        formatted_fields = {}
        
        # First pass - collect simple field values
        for field in self.get_required_fields():
            if field in source:
                # Check if there's a formatter for this field
                if field in self.field_formatters:
                    formatted_fields[field] = self.field_formatters[field](source[field], source)
                else:
                    formatted_fields[field] = source[field]
            else:
                # For missing fields, use an empty string
                formatted_fields[field] = ""
        
        # Add some common derived fields if they're not explicitly in the source
        if "year" not in formatted_fields and "date" in source:
            # Extract year from date
            date_str = source["date"]
            if date_str:
                try:
                    if isinstance(date_str, str) and "-" in date_str:
                        formatted_fields["year"] = date_str.split("-")[0]
                    else:
                        formatted_fields["year"] = str(date_str)
                except:
                    formatted_fields["year"] = ""
        
        # Second pass - replace placeholders in the template
        result = template
        
        # Replace simple placeholders: {field_name}
        for field, value in formatted_fields.items():
            result = result.replace(f"{{{field}}}", str(value))
        
        # Replace conditional placeholders: {field:?text}
        # This shows 'text' only if 'field' has a value
        conditional_pattern = r'\{([a-zA-Z0-9_]+):\?([^{}]*)\}'
        for match in re.finditer(conditional_pattern, result):
            field = match.group(1)
            text = match.group(2)
            replacement = text if field in formatted_fields and formatted_fields[field] else ""
            result = result.replace(match.group(0), replacement)
        
        # Clean up empty placeholders and handle conditional punctuation
        result = self._clean_citation(result)
        
        return result
    
    def _clean_citation(self, citation: str) -> str:
        """
        Clean up the citation by handling empty fields and fixing punctuation.
        
        Args:
            citation: Raw citation string with possible empty parts.
            
        Returns:
            Cleaned citation string.
        """
        # Remove any remaining unformatted placeholders
        placeholder_pattern = r'\{[a-zA-Z0-9_:?]+\}'
        citation = re.sub(placeholder_pattern, '', citation)
        
        # Fix double spaces
        citation = re.sub(r' +', ' ', citation)
        
        # Fix double punctuation
        citation = re.sub(r'\.\.', '.', citation)
        citation = re.sub(r',,', ',', citation)
        citation = re.sub(r'\.\,', '.', citation)
        citation = re.sub(r'\,\.', '.', citation)
        
        # Fix spaces before punctuation
        citation = re.sub(r' \.', '.', citation)
        citation = re.sub(r' \,', ',', citation)
        
        # Fix missing spaces after punctuation
        citation = re.sub(r'\.([a-zA-Z])', '. \\1', citation)
        citation = re.sub(r'\,([a-zA-Z])', ', \\1', citation)
        
        # Trim whitespace
        citation = citation.strip()
        
        return citation
    
    def _format_authors(self, authors: List[Dict[str, str]], source: Dict[str, Any]) -> str:
        """
        Format author names for citation.
        
        Args:
            authors: List of author dictionaries.
            source: Full source dictionary for context.
            
        Returns:
            Formatted author string.
        """
        # Get the citation style
        style = source.get("_style", self.config.get('default_style', 'default'))
        
        # If authors is just a string, return it as is
        if isinstance(authors, str):
            return authors
            
        # If authors is None or empty, check for group author
        if not authors:
            group_author = source.get("group_author", "")
            return group_author if group_author else ""
            
        # Default to APA-like formatting
        return self.format_author_names(authors, "APA")
    
    def _format_date(self, date_str: str, source: Dict[str, Any]) -> str:
        """
        Format a date for citation.
        
        Args:
            date_str: Date string.
            source: Full source dictionary for context.
            
        Returns:
            Formatted date string.
        """
        date_format = self.config.get('date_format', '%B %d, %Y')
        
        # If date_str is already formatted, return it as is
        if not isinstance(date_str, str) or "-" not in date_str:
            return str(date_str)
            
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            return date_obj.strftime(date_format)
        except ValueError:
            # If date cannot be parsed, return it as is
            return date_str
    
    def _format_access_date(self, date_str: str, source: Dict[str, Any]) -> str:
        """
        Format an access date for citation.
        
        Args:
            date_str: Date string.
            source: Full source dictionary for context.
            
        Returns:
            Formatted access date string.
        """
        if not date_str:
            # Use current date if none provided
            return datetime.now().strftime(self.config.get('date_format', '%B %d, %Y'))
            
        return self._format_date(date_str, source)
    
    def _format_url(self, url: str, source: Dict[str, Any]) -> str:
        """
        Format a URL for citation.
        
        Args:
            url: URL string.
            source: Full source dictionary for context.
            
        Returns:
            Formatted URL string.
        """
        # Return URL as is
        return url
    
    def _format_pages(self, pages: Union[str, int, Dict[str, int]], source: Dict[str, Any]) -> str:
        """
        Format page numbers for citation.
        
        Args:
            pages: Page number information, can be string ("123-145"), int, or dict with 'start' and 'end' keys.
            source: Full source dictionary for context.
            
        Returns:
            Formatted page range string.
        """
        if isinstance(pages, dict):
            start = pages.get('start', '')
            end = pages.get('end', '')
            if start and end:
                return f"{start}-{end}"
            elif start:
                return str(start)
            return ""
        elif isinstance(pages, (int, str)):
            return str(pages)
        return ""
    
    def from_json(self, json_str: str) -> None:
        """
        Load templates and configuration from a JSON string.
        
        Args:
            json_str: JSON string containing templates and configuration.
        """
        try:
            data = json.loads(json_str)
            if "templates" in data:
                self.templates.update(data["templates"])
                self.config["templates"] = self.templates
                
            # Update other config values
            for key, value in data.items():
                if key != "templates" and key != "field_formatters":
                    self.config[key] = value
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON string")
    
    def to_json(self) -> str:
        """
        Export templates and configuration to a JSON string.
        
        Returns:
            JSON string containing templates and configuration.
        """
        # Create a copy of config without the field_formatters which can't be serialized
        export_config = {k: v for k, v in self.config.items() if k != "field_formatters"}
        return json.dumps(export_config, indent=2)
    
    def export_templates(self) -> Dict[str, str]:
        """
        Export all citation templates.
        
        Returns:
            Dictionary of template styles and their template strings.
        """
        return self.templates.copy()
    
    def import_templates(self, templates: Dict[str, str]) -> None:
        """
        Import citation templates.
        
        Args:
            templates: Dictionary mapping style names to template strings.
        """
        self.templates.update(templates)
        self.config["templates"] = self.templates 