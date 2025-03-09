"""
Section Splitter for Scientific Papers

This module provides tools for identifying and extracting sections from scientific papers,
with awareness of common section structures across different scientific disciplines.
"""

import re
from typing import List, Dict, Any, Optional, Tuple


class SectionSplitter:
    """
    Specialized tool for identifying and extracting sections from scientific papers.
    Handles different section naming conventions across scientific disciplines.
    """
    
    # Common section patterns by scientific discipline
    DISCIPLINE_PATTERNS = {
        'general': {
            'abstract': r'abstract',
            'introduction': r'introduction|background',
            'methods': r'methods|methodology|materials\s+and\s+methods|experimental',
            'results': r'results',
            'discussion': r'discussion',
            'conclusion': r'conclusion|conclusions|concluding\s+remarks',
            'references': r'references|bibliography|literature\s+cited',
        },
        'computer_science': {
            'abstract': r'abstract',
            'introduction': r'introduction',
            'related_work': r'related\s+work|background|previous\s+work',
            'methods': r'methods|methodology|approach|algorithm|system\s+description',
            'implementation': r'implementation|system|architecture',
            'evaluation': r'evaluation|experiments|experimental\s+results|results',
            'discussion': r'discussion|analysis',
            'conclusion': r'conclusion|future\s+work',
            'references': r'references',
        },
        'biomedical': {
            'abstract': r'abstract',
            'introduction': r'introduction',
            'materials_methods': r'materials\s+and\s+methods|experimental\s+procedures',
            'results': r'results',
            'discussion': r'discussion',
            'conclusion': r'conclusion',
            'acknowledgements': r'acknowledgements?',
            'references': r'references',
            'supplementary': r'supplementary\s+materials?|supporting\s+information',
        },
        'physics': {
            'abstract': r'abstract',
            'introduction': r'introduction',
            'theory': r'theory|theoretical\s+background|model',
            'experimental': r'experimental|experiment|setup|apparatus',
            'results': r'results|observations|measurements',
            'discussion': r'discussion|interpretation',
            'conclusion': r'conclusion|summary',
            'references': r'references',
        }
    }
    
    def __init__(
        self,
        discipline: str = 'general',
        custom_patterns: Optional[Dict[str, str]] = None,
        hierarchical: bool = True,
        numbered_sections: bool = True,
    ):
        """
        Initialize the section splitter with discipline-specific patterns.
        
        Args:
            discipline: Scientific discipline to use patterns for ('general', 'computer_science', etc.)
            custom_patterns: Optional custom patterns to use or override default patterns
            hierarchical: Whether to detect hierarchical section structure (subsections)
            numbered_sections: Whether to handle numbered sections (e.g., "1. Introduction")
        """
        self.discipline = discipline
        self.hierarchical = hierarchical
        self.numbered_sections = numbered_sections
        
        # Use discipline-specific patterns or fall back to general
        self.patterns = self.DISCIPLINE_PATTERNS.get(
            discipline, self.DISCIPLINE_PATTERNS['general']
        )
        
        # Add or override with any custom patterns
        if custom_patterns:
            self.patterns.update(custom_patterns)
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """
        Extract sections from the paper text.
        
        Args:
            text: Full text of the paper
            
        Returns:
            Dictionary mapping section names to section content
        """
        sections = {}
        current_section = 'preamble'
        current_content = []
        
        # Split by lines to process section headers
        lines = text.split('\n')
        
        for line in lines:
            line_strip = line.strip()
            
            # Skip empty lines
            if not line_strip:
                if current_content:  # Only add a blank line if we have content
                    current_content.append('')
                continue
            
            # Check if this is a section header
            section_match = self._match_section_header(line_strip)
            
            if section_match:
                # Save the previous section
                if current_section:
                    sections[current_section] = '\n'.join(current_content)
                
                # Start a new section
                current_section, is_numbered = section_match
                current_content = []
                
                # Add the header without the number if it's numbered
                if is_numbered:
                    header_text = re.sub(r'^\d+(\.\d+)*\s+', '', line_strip)
                    current_content.append(header_text)
                else:
                    current_content.append(line_strip)
            else:
                # Add to current section
                current_content.append(line_strip)
        
        # Add the last section
        if current_section:
            sections[current_section] = '\n'.join(current_content)
        
        return sections
    
    def _match_section_header(self, line: str) -> Optional[Tuple[str, bool]]:
        """
        Check if a line matches any section pattern and return the section name.
        
        Args:
            line: A line of text from the paper
            
        Returns:
            Tuple of (section_name, is_numbered) if matched, None otherwise
        """
        # Check for numbered sections like "1. Introduction" or "2.3 Methods"
        if self.numbered_sections:
            numbered_match = re.match(r'^\d+(\.\d+)*\s+(.*?)$', line)
            if numbered_match:
                section_title = numbered_match.group(2).lower()
                for section_name, pattern in self.patterns.items():
                    if re.search(pattern, section_title):
                        return section_name, True
        
        # Check for non-numbered section headers
        line_lower = line.lower()
        for section_name, pattern in self.patterns.items():
            if re.search(f'^{pattern}$', line_lower) or re.search(f'^{pattern}:', line_lower):
                return section_name, False
        
        return None
    
    def get_section_hierarchy(self, text: str) -> Dict[str, Any]:
        """
        Extract hierarchical section structure (sections and subsections).
        
        Args:
            text: Full text of the paper
            
        Returns:
            Nested dictionary representing the section hierarchy
        """
        if not self.hierarchical:
            # Just return flat sections if hierarchical is disabled
            return self.extract_sections(text)
        
        # This is a placeholder for hierarchical section extraction
        # In a full implementation, we would detect section levels and create a tree
        # structure for sections, subsections, etc.
        
        # For now, just return flat sections
        return self.extract_sections(text)
    
    def get_section_by_name(self, text: str, section_name: str) -> Optional[str]:
        """
        Extract a specific section by name.
        
        Args:
            text: Full text of the paper
            section_name: Name of the section to extract
            
        Returns:
            Section content or None if not found
        """
        sections = self.extract_sections(text)
        return sections.get(section_name) 