"""
Math Formula Handler for Scientific Papers

This module provides tools for detecting, extracting, and processing mathematical
formulas and equations in scientific papers.
"""

import re
import sympy
from typing import List, Dict, Any, Optional, Tuple
import latex2mathml.converter


class MathFormulaHandler:
    """
    Specialized handler for mathematical content in scientific papers.
    Detects, extracts, and processes mathematical formulas and equations.
    """
    
    # Patterns for detecting inline and block math in LaTeX format
    INLINE_MATH_PATTERN = r'\$([^$]+?)\$'
    BLOCK_MATH_PATTERN = r'\$\$(.*?)\$\$|\\\[(.*?)\\\]|\\begin\{equation\}(.*?)\\end\{equation\}'
    
    def __init__(
        self,
        preserve_original: bool = True,
        convert_to_mathml: bool = False,
        convert_to_unicode: bool = True,
        extract_as_tokens: bool = False,
    ):
        """
        Initialize the math formula handler.
        
        Args:
            preserve_original: Whether to keep the original LaTeX in the output
            convert_to_mathml: Whether to convert LaTeX to MathML
            convert_to_unicode: Whether to convert simple expressions to Unicode
            extract_as_tokens: Whether to treat formulas as special tokens
        """
        self.preserve_original = preserve_original
        self.convert_to_mathml = convert_to_mathml
        self.convert_to_unicode = convert_to_unicode
        self.extract_as_tokens = extract_as_tokens
    
    def detect_math(self, text: str) -> List[Tuple[str, bool, int, int]]:
        """
        Detect mathematical content in text.
        
        Args:
            text: Input text that may contain mathematical content
            
        Returns:
            List of tuples containing (formula, is_block, start_pos, end_pos)
        """
        formulas = []
        
        # Find inline math
        for match in re.finditer(self.INLINE_MATH_PATTERN, text):
            formulas.append((
                match.group(1),  # The formula content
                False,           # Not a block formula
                match.start(),   # Start position
                match.end()      # End position
            ))
        
        # Find block math
        for match in re.finditer(self.BLOCK_MATH_PATTERN, text, re.DOTALL):
            # Determine which group matched
            formula = match.group(1) or match.group(2) or match.group(3)
            formulas.append((
                formula,        # The formula content
                True,           # Is a block formula
                match.start(),  # Start position
                match.end()     # End position
            ))
        
        return sorted(formulas, key=lambda x: x[2])  # Sort by start position
    
    def process_text(self, text: str) -> str:
        """
        Process text containing mathematical formulas.
        
        Args:
            text: Input text that may contain mathematical content
            
        Returns:
            Processed text with formulas handled according to configuration
        """
        # Detect math in the text
        formulas = self.detect_math(text)
        
        # If no formulas found, return the original text
        if not formulas:
            return text
        
        # Process the text, replacing formulas as needed
        result = ""
        last_end = 0
        
        for formula, is_block, start, end in formulas:
            # Add text before the formula
            result += text[last_end:start]
            
            # Process the formula based on configuration
            processed_formula = self._process_formula(formula, is_block)
            
            # Add the processed formula
            result += processed_formula
            
            # Update the last end position
            last_end = end
        
        # Add any remaining text
        result += text[last_end:]
        
        return result
    
    def _process_formula(self, formula: str, is_block: bool) -> str:
        """
        Process a single mathematical formula.
        
        Args:
            formula: LaTeX formula content
            is_block: Whether this is a block formula
            
        Returns:
            Processed formula
        """
        result = ""
        
        # Preserve original LaTeX if configured
        if self.preserve_original:
            if is_block:
                result += f"$$${formula}$$$"
            else:
                result += f"${formula}$"
        
        # Add MathML conversion if configured
        if self.convert_to_mathml:
            try:
                mathml = latex2mathml.converter.convert(formula)
                result += f" [MathML: {mathml}]"
            except Exception as e:
                result += f" [MathML conversion failed: {str(e)}]"
        
        # Add Unicode conversion for simple expressions if configured
        if self.convert_to_unicode:
            try:
                # This is a simplified conversion that only works for basic expressions
                # A full implementation would use a more robust LaTeX to Unicode converter
                unicode_formula = self._simple_latex_to_unicode(formula)
                result += f" [Unicode: {unicode_formula}]"
            except Exception as e:
                # If conversion fails, just skip it
                pass
        
        # Add special token if configured
        if self.extract_as_tokens:
            if is_block:
                result = f"[BLOCK_MATH]{result}[/BLOCK_MATH]"
            else:
                result = f"[INLINE_MATH]{result}[/INLINE_MATH]"
        
        return result
    
    def _simple_latex_to_unicode(self, latex: str) -> str:
        """
        Convert simple LaTeX expressions to Unicode.
        This is a placeholder for a more comprehensive conversion.
        
        Args:
            latex: LaTeX math expression
            
        Returns:
            Unicode representation of the expression
        """
        # This is a very simplified conversion for demo purposes
        # A full implementation would use a more comprehensive approach
        
        # Handle subscripts and superscripts
        latex = re.sub(r'_\{([^}]+)\}', r'_\1', latex)
        latex = re.sub(r'\^\{([^}]+)\}', r'^\1', latex)
        
        # Simple replacements
        replacements = {
            r'\alpha': 'α', r'\beta': 'β', r'\gamma': 'γ', r'\delta': 'δ',
            r'\epsilon': 'ε', r'\zeta': 'ζ', r'\eta': 'η', r'\theta': 'θ',
            r'\iota': 'ι', r'\kappa': 'κ', r'\lambda': 'λ', r'\mu': 'μ',
            r'\nu': 'ν', r'\xi': 'ξ', r'\pi': 'π', r'\rho': 'ρ',
            r'\sigma': 'σ', r'\tau': 'τ', r'\upsilon': 'υ', r'\phi': 'φ',
            r'\chi': 'χ', r'\psi': 'ψ', r'\omega': 'ω',
            r'\Gamma': 'Γ', r'\Delta': 'Δ', r'\Theta': 'Θ', r'\Lambda': 'Λ',
            r'\Xi': 'Ξ', r'\Pi': 'Π', r'\Sigma': 'Σ', r'\Upsilon': 'Υ',
            r'\Phi': 'Φ', r'\Psi': 'Ψ', r'\Omega': 'Ω',
            r'\infty': '∞', r'\approx': '≈', r'\ne': '≠', r'\leq': '≤',
            r'\geq': '≥', r'\times': '×', r'\cdot': '·', r'\pm': '±',
            r'\sum': '∑', r'\prod': '∏', r'\int': '∫', r'\partial': '∂',
            r'\nabla': '∇', r'\in': '∈', r'\notin': '∉', r'\subset': '⊂',
            r'\supset': '⊃', r'\cup': '∪', r'\cap': '∩', r'\emptyset': '∅',
            r'\forall': '∀', r'\exists': '∃', r'\rightarrow': '→', r'\Rightarrow': '⇒',
            r'\leftarrow': '←', r'\Leftarrow': '⇐', r'\leftrightarrow': '↔',
            r'\Leftrightarrow': '⇔', r'\sqrt': '√',
        }
        
        for latex_pattern, unicode_char in replacements.items():
            latex = latex.replace(latex_pattern, unicode_char)
        
        return latex
    
    def extract_formulas(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract all mathematical formulas from text with metadata.
        
        Args:
            text: Input text that may contain mathematical content
            
        Returns:
            List of dictionaries with formula information
        """
        formulas = self.detect_math(text)
        return [
            {
                'original': formula,
                'is_block': is_block,
                'position': (start, end),
                'processed': self._process_formula(formula, is_block),
            }
            for formula, is_block, start, end in formulas
        ] 