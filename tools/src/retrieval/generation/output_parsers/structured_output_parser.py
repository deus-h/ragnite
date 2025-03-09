"""
Structured Output Parser

This module defines the StructuredOutputParser that extracts structured data
according to predefined schemas, supporting nested structures and validation.
"""

import json
import re
from typing import Any, Dict, List, Optional, Union, Type, get_type_hints, Callable

from .base_output_parser import BaseOutputParser, ParsingError
from .json_output_parser import JSONOutputParser

class StructuredOutputParser(BaseOutputParser[Dict[str, Any]]):
    """
    Parser for extracting structured data based on predefined schemas.
    
    This parser is built on top of the JSONOutputParser but adds additional
    capabilities like custom extraction patterns, field validation, and
    transformation functions for parsed values.
    
    Attributes:
        schema (Dict[str, Dict[str, Any]]): Schema definition specifying field types and constraints.
        extraction_patterns (Dict[str, str]): Optional regex patterns for extracting fields.
        validators (Dict[str, Callable]): Optional validation functions for fields.
        transformers (Dict[str, Callable]): Optional transformation functions for parsed values.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the structured output parser.
        
        Args:
            config: Configuration dictionary that may include:
                - schema: Required schema dictionary defining the expected output structure
                - extraction_patterns: Optional dictionary of regex patterns for fields
                - validators: Optional dictionary of validation functions
                - transformers: Optional dictionary of transformation functions
                - strict: Whether to require strict schema adherence
        """
        super().__init__(config)
        
        # Schema is required for this parser
        if "schema" not in self.config:
            raise ValueError("Schema is required for StructuredOutputParser")
            
        self.schema = self.config["schema"]
        self.extraction_patterns = self.config.get("extraction_patterns", {})
        self.validators = self.config.get("validators", {})
        self.transformers = self.config.get("transformers", {})
        self.strict = self.config.get("strict", False)
        
        # Initialize the JSON parser for parsing JSON content
        self.json_parser = JSONOutputParser({
            "schema": self.schema,
            "strict": self.strict,
            "extract_json": True
        })
    
    def parse(self, text: str) -> Dict[str, Any]:
        """
        Parse text into a structured dictionary according to the schema.
        
        This method first attempts to parse as JSON, falling back to custom
        extraction patterns if JSON parsing fails.
        
        Args:
            text: Text that contains structured data.
            
        Returns:
            The parsed data as a dictionary matching the schema.
            
        Raises:
            ParsingError: If the text cannot be parsed according to the schema.
        """
        if not text:
            raise ParsingError("Empty text provided", "")
        
        # First try to parse as JSON
        try:
            result = self.json_parser.parse(text)
            # Apply transformations and validate
            return self._process_parsed_data(result)
        except ParsingError:
            # If JSON parsing fails, try custom extraction
            if self.extraction_patterns:
                try:
                    result = self._extract_structured_data(text)
                    # Apply transformations and validate
                    return self._process_parsed_data(result)
                except Exception as e:
                    raise ParsingError(
                        f"Failed to extract structured data: {str(e)}", 
                        text, 
                        e
                    )
            else:
                # Re-raise the JSON parsing error if no extraction patterns are defined
                raise
    
    def _extract_structured_data(self, text: str) -> Dict[str, Any]:
        """
        Extract structured data using custom extraction patterns.
        
        Args:
            text: The text to extract data from.
            
        Returns:
            A dictionary with extracted data.
            
        Raises:
            ParsingError: If extraction fails or required fields are missing.
        """
        result = {}
        
        # Extract each field using its pattern
        for field, pattern in self.extraction_patterns.items():
            if field in self.schema:
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    # Extract the captured group or the entire match
                    value = match.group(1) if match.groups() else match.group(0)
                    
                    # Convert value to the appropriate type
                    field_type = self.schema.get(field)
                    if field_type == int:
                        try:
                            result[field] = int(value)
                        except ValueError:
                            result[field] = 0
                    elif field_type == float:
                        try:
                            result[field] = float(value)
                        except ValueError:
                            result[field] = 0.0
                    elif field_type == bool:
                        result[field] = value.lower() in ("true", "yes", "1", "y")
                    elif field_type == list:
                        # Split by commas, newlines, or semicolons
                        items = re.split(r'[,;\n]+', value)
                        result[field] = [item.strip() for item in items if item.strip()]
                    elif field_type == dict:
                        # Try to parse as JSON
                        try:
                            result[field] = json.loads(value)
                        except json.JSONDecodeError:
                            result[field] = {}
                    else:
                        # Default to string
                        result[field] = value.strip()
        
        # Check for required fields
        missing = [field for field in self.schema if field not in result]
        if missing and self.strict:
            raise ParsingError(
                f"Missing required fields: {', '.join(missing)}", 
                text
            )
        
        return result
    
    def _process_parsed_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply transformations and validation to parsed data.
        
        Args:
            data: The parsed data dictionary.
            
        Returns:
            The processed data dictionary.
            
        Raises:
            ParsingError: If validation fails.
        """
        # Apply transformers
        for field, transformer in self.transformers.items():
            if field in data:
                try:
                    data[field] = transformer(data[field])
                except Exception as e:
                    raise ParsingError(
                        f"Transformation failed for field '{field}': {str(e)}", 
                        json.dumps(data), 
                        e
                    )
        
        # Apply validators
        for field, validator in self.validators.items():
            if field in data:
                try:
                    if not validator(data[field]):
                        raise ParsingError(
                            f"Validation failed for field '{field}'", 
                            json.dumps(data)
                        )
                except Exception as e:
                    if not isinstance(e, ParsingError):
                        raise ParsingError(
                            f"Validator exception for field '{field}': {str(e)}", 
                            json.dumps(data), 
                            e
                        )
                    raise
        
        return data
    
    def validate_output(self, parsed_output: Dict[str, Any]) -> bool:
        """
        Validate parsed output against the schema.
        
        Args:
            parsed_output: The parsed data dictionary.
            
        Returns:
            True if valid, False if invalid.
        """
        # In strict mode, all fields must match exactly
        if self.strict:
            required_keys = set(self.schema.keys())
            actual_keys = set(parsed_output.keys())
            
            if required_keys != actual_keys:
                return False
        
        # Validate each field against expected type
        for field, field_type in self.schema.items():
            if field not in parsed_output:
                if self.strict:
                    return False
                else:
                    continue
            
            # Check type if specified
            if field_type is not None:
                # Handle lists and dicts specially
                if field_type == list and not isinstance(parsed_output[field], (list, tuple)):
                    return False
                elif field_type == dict and not isinstance(parsed_output[field], dict):
                    return False
                # For other types, use isinstance
                elif not isinstance(parsed_output[field], field_type) and field_type not in (list, dict):
                    return False
        
        # Apply custom validators
        for field, validator in self.validators.items():
            if field in parsed_output:
                try:
                    if not validator(parsed_output[field]):
                        return False
                except Exception:
                    return False
        
        return True
    
    def get_format_instructions(self) -> str:
        """
        Get instructions for the language model on how to format its output.
        
        Returns:
            A string containing instructions on the expected output format.
        """
        instructions = "Respond with structured data following this format:\n\n"
        
        # Create schema description
        schema_desc = []
        for field, field_type in self.schema.items():
            type_name = getattr(field_type, "__name__", str(field_type))
            desc = f"- {field} ({type_name})"
            
            # Add validation info if available
            if field in self.validators:
                desc += " [must pass validation]"
            
            schema_desc.append(desc)
        
        if schema_desc:
            instructions += "\n".join(schema_desc) + "\n\n"
        
        # Create an example response
        example = {}
        for field, field_type in self.schema.items():
            if field_type == str:
                example[field] = "sample text"
            elif field_type == int:
                example[field] = 42
            elif field_type == float:
                example[field] = 3.14
            elif field_type == bool:
                example[field] = True
            elif field_type == list:
                example[field] = ["item1", "item2", "item3"]
            elif field_type == dict:
                example[field] = {"key": "value"}
            else:
                example[field] = "value"
        
        # Format as JSON for clarity
        example_json = json.dumps(example, indent=2)
        instructions += f"Example (JSON format preferred):\n```json\n{example_json}\n```\n\n"
        
        # Mention alternative formatting if extraction patterns are defined
        if self.extraction_patterns:
            instructions += "Alternatively, you can format the response with clear field labels, e.g.:\n\n"
            instructions += "\n".join([f"{field}: [value]" for field in self.schema])
        
        return instructions
    
    def __repr__(self) -> str:
        """Return a string representation of the parser."""
        num_fields = len(self.schema) if self.schema else 0
        num_patterns = len(self.extraction_patterns) if self.extraction_patterns else 0
        num_validators = len(self.validators) if self.validators else 0
        num_transformers = len(self.transformers) if self.transformers else 0
        
        return (f"StructuredOutputParser({num_fields} fields, {num_patterns} patterns, "
                f"{num_validators} validators, {num_transformers} transformers, "
                f"strict={self.strict})") 