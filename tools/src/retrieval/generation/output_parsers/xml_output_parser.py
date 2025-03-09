"""
XML Output Parser

This module defines the XMLOutputParser that converts raw text into XML elements.
"""

import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Union

from .base_output_parser import BaseOutputParser, ParsingError

class XMLOutputParser(BaseOutputParser[ET.Element]):
    """
    Parser for extracting XML data from language model outputs.
    
    This parser handles text that contains XML, even if the XML is
    embedded in other text like markdown code blocks or explanatory text.
    
    Attributes:
        root_tag (str, optional): The expected root tag of the XML.
        required_elements (List[str], optional): A list of element paths that must be present.
        extract_xml (bool): Whether to attempt to extract XML from text that contains non-XML content.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the XML parser.
        
        Args:
            config: Configuration dictionary that may include:
                - root_tag: Optional expected root tag
                - required_elements: Optional list of required element paths
                - extract_xml: Whether to extract XML from markdown blocks (default: True)
        """
        super().__init__(config)
        self.root_tag = self.config.get("root_tag")
        self.required_elements = self.config.get("required_elements", [])
        self.extract_xml = self.config.get("extract_xml", True)
    
    def parse(self, text: str) -> ET.Element:
        """
        Parse text into an XML Element.
        
        Args:
            text: Text that contains XML, possibly embedded in other content.
            
        Returns:
            The parsed XML as an ElementTree Element.
            
        Raises:
            ParsingError: If the text cannot be parsed as valid XML.
        """
        if not text:
            raise ParsingError("Empty text provided", "")
        
        # Try to extract XML if the option is enabled
        xml_str = self._extract_xml(text) if self.extract_xml else text
        
        try:
            # Parse the XML string
            root = ET.fromstring(xml_str)
            
            # Validate the root tag if specified
            if self.root_tag and root.tag != self.root_tag:
                raise ParsingError(
                    f"Expected root tag '{self.root_tag}' but found '{root.tag}'", 
                    text
                )
            
            # Validate required elements if specified
            if not self.validate_output(root):
                raise ParsingError(
                    "XML does not contain all required elements", 
                    text
                )
                
            return root
        except ET.ParseError as e:
            raise ParsingError(
                f"Failed to parse XML: {str(e)}", 
                text, 
                e
            )
    
    def _extract_xml(self, text: str) -> str:
        """
        Extract XML from text that may contain other content.
        
        This method handles common patterns like markdown code blocks and
        extracts only the XML portion of the text.
        
        Args:
            text: Text that may contain XML mixed with other content.
            
        Returns:
            The extracted XML string.
        """
        # Try to extract from markdown code blocks
        code_block_pattern = r"```(?:xml)?(.+?)```"
        code_blocks = re.findall(code_block_pattern, text, re.DOTALL)
        
        if code_blocks:
            # Try each code block until we find valid XML
            for block in code_blocks:
                block = block.strip()
                try:
                    ET.fromstring(block)
                    return block
                except ET.ParseError:
                    continue
        
        # Look for XML-like patterns (content between < and >)
        # This is a simplistic approach and might not work for all XML
        xml_pattern = r"<\?xml.*?>.*?<([a-zA-Z][a-zA-Z0-9]*).*?>.*?</\1>"
        xml_matches = re.findall(xml_pattern, text, re.DOTALL)
        
        if xml_matches:
            # The first match is the root tag name, not the content
            # We need to find the full XML string
            root_tag = xml_matches[0]
            full_xml_pattern = f"<\\?xml.*?>.*?<{root_tag}.*?>.*?</{root_tag}>"
            full_match = re.search(full_xml_pattern, text, re.DOTALL)
            
            if full_match:
                xml_content = full_match.group(0)
                try:
                    ET.fromstring(xml_content)
                    return xml_content
                except ET.ParseError:
                    pass
        
        # If no tag pattern, try to find any XML-like content
        # This might be useful for simpler XML without XML declaration
        xml_element_pattern = r"<([a-zA-Z][a-zA-Z0-9]*).*?>.*?</\1>"
        element_matches = re.findall(xml_element_pattern, text, re.DOTALL)
        
        if element_matches:
            for tag in element_matches:
                pattern = f"<{tag}.*?>.*?</{tag}>"
                full_match = re.search(pattern, text, re.DOTALL)
                
                if full_match:
                    xml_content = full_match.group(0)
                    try:
                        ET.fromstring(xml_content)
                        return xml_content
                    except ET.ParseError:
                        continue
        
        # If no valid XML found in patterns, return the original text
        return text
    
    def validate_output(self, parsed_output: ET.Element) -> bool:
        """
        Validate parsed XML against requirements if provided.
        
        Args:
            parsed_output: The parsed XML Element.
            
        Returns:
            True if valid, False if invalid.
        """
        if not self.required_elements:
            return True
            
        # Check each required element
        for path in self.required_elements:
            # Use ElementTree findall with XPath expressions
            elements = parsed_output.findall(path)
            if not elements:
                return False
        
        return True
    
    def get_format_instructions(self) -> str:
        """
        Get instructions for the language model on how to format its output.
        
        Returns:
            A string containing instructions on the expected XML format.
        """
        instructions = "Respond with a well-formed XML document"
        
        if self.root_tag:
            instructions += f" with root element <{self.root_tag}>"
        
        if self.required_elements:
            instructions += " that includes the following elements:\n"
            for path in self.required_elements:
                instructions += f"- {path}\n"
        
        # Add a basic example
        example_root = self.root_tag if self.root_tag else "response"
        example = f"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<{example_root}>\n"
        
        if self.required_elements:
            # Add example elements based on required paths
            for path in self.required_elements:
                # Simple path parsing logic - doesn't handle complex XPath
                parts = path.split("/")
                element = parts[-1]
                example += f"  <{element}>content</{element}>\n"
        else:
            # Add a generic element if no specific elements required
            example += "  <element>content</element>\n"
        
        example += f"</{example_root}>"
        instructions += f"\n\nExample:\n```xml\n{example}\n```"
        
        return instructions
    
    def __repr__(self) -> str:
        """Return a string representation of the parser."""
        root_str = f"root='{self.root_tag}'" if self.root_tag else "no root constraint"
        elements_str = f", {len(self.required_elements)} required elements" if self.required_elements else ""
        extract_str = ", auto-extract" if self.extract_xml else ""
        return f"XMLOutputParser({root_str}{elements_str}{extract_str})" 