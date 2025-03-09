"""
Markdown Output Parser

This module defines the MarkdownOutputParser that transforms raw text into structured
Markdown components.
"""

import re
from typing import Any, Dict, List, Optional, Union, Tuple

from .base_output_parser import BaseOutputParser, ParsingError

class MarkdownOutputParser(BaseOutputParser[Dict[str, Any]]):
    """
    Parser for extracting structured information from Markdown-formatted text.
    
    This parser can identify and extract headers, lists, code blocks, and other
    Markdown elements, organizing them into a structured dictionary.
    
    Attributes:
        extract_headers (bool): Whether to extract headers.
        extract_code_blocks (bool): Whether to extract code blocks.
        extract_lists (bool): Whether to extract bulleted and numbered lists.
        extract_blockquotes (bool): Whether to extract blockquotes.
        required_sections (List[str], optional): Headers that must be present.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Markdown parser.
        
        Args:
            config: Configuration dictionary that may include:
                - extract_headers: Whether to extract headers (default: True)
                - extract_code_blocks: Whether to extract code blocks (default: True)
                - extract_lists: Whether to extract lists (default: True)
                - extract_blockquotes: Whether to extract blockquotes (default: True)
                - required_sections: List of headers that must be present
        """
        super().__init__(config)
        self.extract_headers = self.config.get("extract_headers", True)
        self.extract_code_blocks = self.config.get("extract_code_blocks", True)
        self.extract_lists = self.config.get("extract_lists", True)
        self.extract_blockquotes = self.config.get("extract_blockquotes", True)
        self.required_sections = self.config.get("required_sections", [])
    
    def parse(self, text: str) -> Dict[str, Any]:
        """
        Parse Markdown text into a structured dictionary.
        
        Args:
            text: Markdown-formatted text.
            
        Returns:
            A dictionary with extracted Markdown components.
            
        Raises:
            ParsingError: If required sections are missing or parsing fails.
        """
        if not text:
            raise ParsingError("Empty text provided", "")
        
        result = {}
        
        # Extract headers and their content
        if self.extract_headers:
            headers = self._extract_headers(text)
            result["headers"] = headers
            
            # Check for required sections
            if self.required_sections:
                header_titles = [h["title"] for h in headers]
                missing = [req for req in self.required_sections if req not in header_titles]
                if missing:
                    raise ParsingError(
                        f"Missing required sections: {', '.join(missing)}", 
                        text
                    )
        
        # Extract code blocks
        if self.extract_code_blocks:
            result["code_blocks"] = self._extract_code_blocks(text)
        
        # Extract lists
        if self.extract_lists:
            result["lists"] = self._extract_lists(text)
        
        # Extract blockquotes
        if self.extract_blockquotes:
            result["blockquotes"] = self._extract_blockquotes(text)
        
        # Validate the parsed output
        if not self.validate_output(result):
            raise ParsingError(
                "Markdown does not meet validation requirements", 
                text
            )
            
        return result
    
    def _extract_headers(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract headers and their content from Markdown text.
        
        Args:
            text: Markdown text.
            
        Returns:
            A list of dictionaries with header information.
        """
        headers = []
        
        # Match both styles of headers (# and ===)
        header_pattern = r"((?:^|\n)#{1,6}\s+.+?(?=\n#{1,6}\s+|\n*$)|(?:^|\n)[^\n]+\n[=-]+\n)"
        header_matches = re.findall(header_pattern, text, re.MULTILINE | re.DOTALL)
        
        for header_match in header_matches:
            # Process # style headers
            if re.match(r"#{1,6}\s+", header_match.strip()):
                # Count the level (number of #)
                level_match = re.match(r"(#+)", header_match.strip())
                level = len(level_match.group(1)) if level_match else 1
                
                # Extract title (text after #s)
                title_match = re.match(r"#+\s+(.+?)(?=\n|$)", header_match.strip())
                title = title_match.group(1) if title_match else ""
                
                # Extract content (everything after the title)
                content = header_match.strip()[len(title_match.group(0)):].strip() if title_match else ""
                
            # Process === or --- style headers
            else:
                lines = header_match.strip().split("\n")
                if len(lines) >= 2:
                    title = lines[0].strip()
                    level = 1 if "=" in lines[1] else 2  # === is h1, --- is h2
                    content = "\n".join(lines[2:]).strip()
                else:
                    title = header_match.strip()
                    level = 1
                    content = ""
            
            headers.append({
                "level": level,
                "title": title,
                "content": content
            })
        
        return headers
    
    def _extract_code_blocks(self, text: str) -> List[Dict[str, str]]:
        """
        Extract code blocks from Markdown text.
        
        Args:
            text: Markdown text.
            
        Returns:
            A list of dictionaries with code block information.
        """
        code_blocks = []
        
        # Match code blocks with or without language specifier
        code_pattern = r"```(?:([\w-]+)\n)?(.*?)```"
        code_matches = re.findall(code_pattern, text, re.DOTALL)
        
        for lang, code in code_matches:
            code_blocks.append({
                "language": lang if lang else "text",
                "code": code.strip()
            })
        
        # Match indented code blocks (4 spaces or 1 tab)
        indented_pattern = r"(?:(?:^|\n)(?:    |\t)[^\n]+)+"
        indented_matches = re.findall(indented_pattern, text)
        
        for indented in indented_matches:
            # Remove the indentation from each line
            code = "\n".join([line[4:] if line.startswith("    ") else line[1:] 
                            for line in indented.split("\n") if line.strip()])
            
            code_blocks.append({
                "language": "text",
                "code": code
            })
        
        return code_blocks
    
    def _extract_lists(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract bulleted and numbered lists from Markdown text.
        
        Args:
            text: Markdown text.
            
        Returns:
            A list of dictionaries with list information.
        """
        lists = []
        
        # Match bulleted lists
        bulleted_pattern = r"(?:(?:^|\n)[ \t]*[-*+][ \t]+[^\n]+)+"
        bulleted_matches = re.findall(bulleted_pattern, text, re.MULTILINE)
        
        for bulleted in bulleted_matches:
            items = []
            for line in bulleted.split("\n"):
                line = line.strip()
                if line and line[0] in "-*+" and line[1:].strip():
                    items.append(line[1:].strip())
            
            if items:  # Only add non-empty lists
                lists.append({
                    "type": "bulleted",
                    "items": items
                })
        
        # Match numbered lists
        numbered_pattern = r"(?:(?:^|\n)[ \t]*\d+\.[ \t]+[^\n]+)+"
        numbered_matches = re.findall(numbered_pattern, text, re.MULTILINE)
        
        for numbered in numbered_matches:
            items = []
            for line in numbered.split("\n"):
                line = line.strip()
                if line and re.match(r"\d+\.", line) and line.split(".", 1)[1].strip():
                    items.append(line.split(".", 1)[1].strip())
            
            if items:  # Only add non-empty lists
                lists.append({
                    "type": "numbered",
                    "items": items
                })
        
        return lists
    
    def _extract_blockquotes(self, text: str) -> List[str]:
        """
        Extract blockquotes from Markdown text.
        
        Args:
            text: Markdown text.
            
        Returns:
            A list of blockquote texts.
        """
        blockquotes = []
        
        # Match blockquotes (> prefixed lines)
        blockquote_pattern = r"(?:(?:^|\n)[ \t]*>[ \t]?.+)+"
        blockquote_matches = re.findall(blockquote_pattern, text, re.MULTILINE)
        
        for blockquote in blockquote_matches:
            # Remove the > prefix from each line
            content = "\n".join([line.split(">", 1)[1].strip() 
                               for line in blockquote.split("\n") 
                               if line.strip() and ">" in line])
            
            if content:  # Only add non-empty blockquotes
                blockquotes.append(content)
        
        return blockquotes
    
    def validate_output(self, parsed_output: Dict[str, Any]) -> bool:
        """
        Validate parsed Markdown against requirements.
        
        Args:
            parsed_output: The parsed Markdown dictionary.
            
        Returns:
            True if valid, False if invalid.
        """
        # For Markdown, the main validation is checking required sections
        # which is already done during parsing
        return True
    
    def get_format_instructions(self) -> str:
        """
        Get instructions for the language model on how to format its output.
        
        Returns:
            A string containing instructions on the expected Markdown format.
        """
        instructions = "Respond using Markdown formatting"
        components = []
        
        if self.extract_headers:
            components.append("headers with appropriate levels (# for main headers, ## for subheaders, etc.)")
        
        if self.required_sections:
            instructions += " that must include the following sections:\n"
            for section in self.required_sections:
                instructions += f"- {section}\n"
        
        if self.extract_code_blocks:
            components.append("code blocks with language specification where appropriate")
        
        if self.extract_lists:
            components.append("bulleted lists (using -, *, or +) and/or numbered lists (using 1., 2., etc.)")
        
        if self.extract_blockquotes:
            components.append("blockquotes using > for quoted content")
        
        if components:
            instructions += "\n\nInclude:\n"
            instructions += "\n".join(f"- {component}" for component in components)
        
        # Add example
        example = ""
        if self.required_sections:
            for section in self.required_sections[:2]:  # Limit to first 2 for brevity
                example += f"# {section}\n\nContent for {section}.\n\n"
        else:
            example += "# Main Section\n\nThis is the main content.\n\n"
        
        if self.extract_code_blocks:
            example += "```python\ndef example_function():\n    return \"This is a code example\"\n```\n\n"
        
        if self.extract_lists:
            example += "## List Examples\n\n"
            example += "Bulleted list:\n- Item 1\n- Item 2\n- Item 3\n\n"
            example += "Numbered list:\n1. First step\n2. Second step\n3. Third step\n\n"
        
        if self.extract_blockquotes:
            example += "## Quote\n\n> This is an important quote that should be highlighted.\n\n"
        
        instructions += f"\n\nExample:\n```markdown\n{example.strip()}\n```"
        
        return instructions
    
    def __repr__(self) -> str:
        """Return a string representation of the parser."""
        components = []
        if self.extract_headers:
            components.append("headers")
        if self.extract_code_blocks:
            components.append("code blocks")
        if self.extract_lists:
            components.append("lists")
        if self.extract_blockquotes:
            components.append("blockquotes")
        
        features = ", ".join(components)
        required = f", {len(self.required_sections)} required sections" if self.required_sections else ""
        
        return f"MarkdownOutputParser(extracts: {features}{required})" 