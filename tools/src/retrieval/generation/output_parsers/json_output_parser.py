"""
JSON Output Parser

This module defines the JSONOutputParser that converts raw text into JSON objects.
"""

import json
import re
from typing import Any, Dict, List, Optional, Type, Union, get_type_hints

from .base_output_parser import BaseOutputParser, ParsingError

class JSONOutputParser(BaseOutputParser[Dict[str, Any]]):
    """
    Parser for extracting JSON data from language model outputs.
    
    This parser handles text that contains a JSON object, even if the JSON is
    embedded in other text like markdown code blocks or explanatory text.
    
    Attributes:
        schema (Dict[str, Any], optional): A schema definition to validate parsed JSON.
        strict (bool): Whether to require exact schema matching.
        extract_json (bool): Whether to attempt to extract JSON from text that contains non-JSON content.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the JSON parser.
        
        Args:
            config: Configuration dictionary that may include:
                - schema: Optional schema dictionary for validation
                - strict: Whether to require strict schema adherence (default: False)
                - extract_json: Whether to extract JSON from markdown blocks (default: True)
        """
        super().__init__(config)
        self.schema = self.config.get("schema")
        self.strict = self.config.get("strict", False)
        self.extract_json = self.config.get("extract_json", True)
    
    def parse(self, text: str) -> Dict[str, Any]:
        """
        Parse text into a JSON object.
        
        Args:
            text: Text that contains JSON, possibly embedded in other content.
            
        Returns:
            The parsed JSON as a Python dictionary.
            
        Raises:
            ParsingError: If the text cannot be parsed as valid JSON.
        """
        if not text:
            raise ParsingError("Empty text provided", "")
        
        # Try to extract JSON if the option is enabled
        json_str = self._extract_json(text) if self.extract_json else text
        
        try:
            parsed_json = json.loads(json_str)
            
            # Validate against schema if provided
            if self.schema and not self.validate_output(parsed_json):
                raise ParsingError(
                    "JSON does not match the required schema", 
                    text
                )
                
            return parsed_json
        except json.JSONDecodeError as e:
            raise ParsingError(
                f"Failed to parse JSON: {str(e)}", 
                text, 
                e
            )
    
    def _extract_json(self, text: str) -> str:
        """
        Extract JSON from text that may contain other content.
        
        This method handles common patterns like markdown code blocks and
        extracts only the JSON portion of the text.
        
        Args:
            text: Text that may contain JSON mixed with other content.
            
        Returns:
            The extracted JSON string.
        """
        # Try to extract from markdown code blocks
        code_block_pattern = r"```(?:json)?(.+?)```"
        code_blocks = re.findall(code_block_pattern, text, re.DOTALL)
        
        if code_blocks:
            # Try each code block until we find valid JSON
            for block in code_blocks:
                try:
                    json.loads(block.strip())
                    return block.strip()
                except json.JSONDecodeError:
                    continue
        
        # Look for JSON-like patterns (objects starting with { and ending with })
        json_pattern = r"\{(?:[^{}]|(?R))*\}"
        json_matches = re.findall(json_pattern, text, re.DOTALL)
        
        if json_matches:
            # Try each match until we find valid JSON
            for match in json_matches:
                try:
                    json.loads(match.strip())
                    return match.strip()
                except json.JSONDecodeError:
                    continue
        
        # If no valid JSON found in patterns, return the original text
        return text
    
    def validate_output(self, parsed_output: Dict[str, Any]) -> bool:
        """
        Validate parsed JSON against a schema if provided.
        
        Args:
            parsed_output: The parsed JSON dictionary.
            
        Returns:
            True if valid, False if invalid.
        """
        if not self.schema:
            return True
            
        # Simple schema validation
        if self.strict:
            # In strict mode, all fields must match exactly
            required_keys = set(self.schema.keys())
            actual_keys = set(parsed_output.keys())
            
            if required_keys != actual_keys:
                return False
        else:
            # In non-strict mode, all required fields must be present
            for key, value_type in self.schema.items():
                if key not in parsed_output:
                    return False
                    
                # Check type if specified
                if value_type and not isinstance(parsed_output[key], value_type):
                    return False
        
        return True
    
    def get_format_instructions(self) -> str:
        """
        Get instructions for the language model on how to format its output.
        
        Returns:
            A string containing instructions on the expected JSON format.
        """
        instructions = "Respond with a JSON object"
        
        if self.schema:
            # Include schema details in the instructions
            schema_desc = []
            for key, value_type in self.schema.items():
                type_name = getattr(value_type, "__name__", str(value_type))
                schema_desc.append(f'- "{key}": {type_name}')
            
            if schema_desc:
                instructions += " with the following structure:\n"
                instructions += "\n".join(schema_desc)
                
                # Add example if helpful
                example = {}
                for key, value_type in self.schema.items():
                    if value_type == str:
                        example[key] = "string value"
                    elif value_type == int:
                        example[key] = 123
                    elif value_type == float:
                        example[key] = 123.45
                    elif value_type == bool:
                        example[key] = True
                    elif value_type == list:
                        example[key] = ["item1", "item2"]
                    elif value_type == dict:
                        example[key] = {"nested_key": "nested_value"}
                    else:
                        example[key] = "value"
                
                if example:
                    example_json = json.dumps(example, indent=2)
                    instructions += f"\n\nExample:\n```json\n{example_json}\n```"
        
        return instructions
    
    def __repr__(self) -> str:
        """Return a string representation of the parser."""
        schema_str = "with schema" if self.schema else "without schema"
        strict_str = ", strict mode" if self.strict else ""
        extract_str = ", auto-extract" if self.extract_json else ""
        return f"JSONOutputParser({schema_str}{strict_str}{extract_str})" 