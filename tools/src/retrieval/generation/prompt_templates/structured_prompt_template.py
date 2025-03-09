"""
Structured Prompt Template

This module provides the StructuredPromptTemplate class for creating prompts
that generate structured outputs like JSON, XML, etc.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Union, Tuple

from .basic_prompt_template import BasicPromptTemplate

# Configure logging
logger = logging.getLogger(__name__)


class StructuredPromptTemplate(BasicPromptTemplate):
    """
    A prompt template for generating structured outputs like JSON or XML.
    
    This template includes the output format specification and schema to help
    language models generate properly structured outputs.
    
    Attributes:
        template: The prompt template string.
        output_format: The desired output format (json, xml, csv, etc.).
        schema: Schema definition for the structured output.
        schema_format: Format of the schema definition (default: 'json_schema').
        config: Configuration options for the prompt template.
    """
    
    SUPPORTED_FORMATS = ["json", "xml", "yaml", "csv", "markdown_table", "custom"]
    SUPPORTED_SCHEMA_FORMATS = ["json_schema", "example", "description", "custom"]
    
    def __init__(self, 
                 template: str,
                 output_format: str = "json",
                 schema: Optional[Union[Dict, str, List[Dict]]] = None,
                 schema_format: str = "json_schema",
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the structured prompt template.
        
        Args:
            template: The prompt template string with {variable} placeholders.
            output_format: The desired output format (json, xml, yaml, csv, markdown_table, custom).
            schema: Schema definition for the structured output. Can be a dict for JSON Schema,
                   a string for description or custom formats, or a list of example objects.
            schema_format: Format of the schema definition (json_schema, example, description, custom).
            config: Configuration options for the prompt template.
                   - include_format_instructions: Whether to include format instructions (default: True)
                   - format_strict: Whether to enforce strict format compliance (default: True)
                   - format_intro: Custom introduction for format instructions
                   - format_outro: Custom conclusion for format instructions
        """
        super().__init__(template, config or {})
        
        if output_format not in self.SUPPORTED_FORMATS:
            logger.warning(f"Unsupported output format: {output_format}. Using 'json' instead.")
            output_format = "json"
        
        if schema_format not in self.SUPPORTED_SCHEMA_FORMATS:
            logger.warning(f"Unsupported schema format: {schema_format}. Using 'json_schema' instead.")
            schema_format = "json_schema"
        
        self.output_format = output_format
        self.schema = schema
        self.schema_format = schema_format
        
        self.config.setdefault("include_format_instructions", True)
        self.config.setdefault("format_strict", True)
        self.config.setdefault("format_intro", "Your response must be in the following format:")
        self.config.setdefault("format_outro", "Ensure your response exactly follows this format.")
    
    def format(self, **kwargs) -> str:
        """
        Format the structured prompt template with the provided variables.
        
        This method formats the main template and appends format instructions
        if configured to do so.
        
        Args:
            **kwargs: Variables to substitute in the template.
            
        Returns:
            str: The formatted prompt with format instructions.
        """
        # Format the main template
        formatted_prompt = super().format(**kwargs)
        
        # Append format instructions if configured
        if self.config.get("include_format_instructions", True):
            format_instructions = self._get_format_instructions()
            formatted_prompt += f"\n\n{format_instructions}"
        
        return formatted_prompt
    
    def _get_format_instructions(self) -> str:
        """
        Generate format instructions based on the output format and schema.
        
        Returns:
            str: Formatted instructions for the specified output format and schema.
        """
        intro = self.config.get("format_intro", "Your response must be in the following format:")
        outro = self.config.get("format_outro", "Ensure your response exactly follows this format.")
        strict_msg = "Strictly adhere to this format without deviation." if self.config.get("format_strict", True) else ""
        
        instructions = [intro]
        
        # Add format-specific instructions
        if self.output_format == "json":
            instructions.append("```json")
            if self.schema_format == "json_schema" and isinstance(self.schema, dict):
                # Format JSON Schema
                instructions.append(self._format_json_schema())
            elif self.schema_format == "example" and (isinstance(self.schema, dict) or isinstance(self.schema, list)):
                # Format JSON example
                instructions.append(json.dumps(self.schema, indent=2))
            elif self.schema_format == "description" and isinstance(self.schema, str):
                # Use schema description directly
                instructions.append(self.schema)
            instructions.append("```")
                
        elif self.output_format == "xml":
            instructions.append("```xml")
            if self.schema_format == "example" and isinstance(self.schema, str):
                instructions.append(self.schema)
            elif self.schema_format == "description" and isinstance(self.schema, str):
                instructions.append(self.schema)
            instructions.append("```")
                
        elif self.output_format == "yaml":
            instructions.append("```yaml")
            if self.schema_format == "example" and isinstance(self.schema, str):
                instructions.append(self.schema)
            elif self.schema_format == "description" and isinstance(self.schema, str):
                instructions.append(self.schema)
            instructions.append("```")
                
        elif self.output_format == "csv":
            instructions.append("```csv")
            if self.schema_format == "example" and isinstance(self.schema, str):
                instructions.append(self.schema)
            elif self.schema_format == "description" and isinstance(self.schema, str):
                instructions.append(self.schema)
            instructions.append("```")
                
        elif self.output_format == "markdown_table":
            instructions.append("```markdown")
            if self.schema_format == "example" and isinstance(self.schema, str):
                instructions.append(self.schema)
            elif self.schema_format == "description" and isinstance(self.schema, str):
                instructions.append(self.schema)
            instructions.append("```")
                
        elif self.output_format == "custom" and isinstance(self.schema, str):
            instructions.append(self.schema)
        
        if strict_msg:
            instructions.append(strict_msg)
        
        instructions.append(outro)
        
        return "\n".join(instructions)
    
    def _format_json_schema(self) -> str:
        """
        Format a JSON schema into a human-readable form.
        
        Returns:
            str: Formatted schema.
        """
        if not isinstance(self.schema, dict):
            return "{}"
            
        schema = self.schema.copy()
        
        # Special handling for simple types to make the output more readable
        if 'type' in schema and schema['type'] == 'object' and 'properties' in schema:
            schema = self._simplify_json_schema(schema)
            
        return json.dumps(schema, indent=2)
    
    def _simplify_json_schema(self, schema: Dict) -> Dict:
        """
        Simplify a JSON schema for better readability.
        
        Args:
            schema: JSON schema.
            
        Returns:
            Dict: Simplified schema.
        """
        # This is a simplified version - in a real implementation you might want
        # to make this more sophisticated
        if 'type' in schema and schema['type'] == 'object' and 'properties' in schema:
            result = {}
            for prop, prop_schema in schema['properties'].items():
                if 'type' in prop_schema:
                    if prop_schema['type'] == 'object' and 'properties' in prop_schema:
                        result[prop] = self._simplify_json_schema(prop_schema)
                    elif prop_schema['type'] == 'array' and 'items' in prop_schema:
                        if 'type' in prop_schema['items'] and prop_schema['items']['type'] == 'object':
                            result[prop] = [self._simplify_json_schema(prop_schema['items'])]
                        else:
                            result[prop] = [prop_schema['items'].get('example', '<item>')]
                    else:
                        result[prop] = prop_schema.get('example', f"<{prop_schema['type']}>")
            return result
        return schema
    
    def set_output_format(self, output_format: str) -> None:
        """
        Set the output format.
        
        Args:
            output_format: The desired output format.
        """
        if output_format not in self.SUPPORTED_FORMATS:
            logger.warning(f"Unsupported output format: {output_format}. Using 'json' instead.")
            output_format = "json"
        self.output_format = output_format
    
    def set_schema(self, schema: Union[Dict, str, List[Dict]], schema_format: str = None) -> None:
        """
        Set the schema definition.
        
        Args:
            schema: Schema definition for the structured output.
            schema_format: Format of the schema definition.
        """
        self.schema = schema
        if schema_format is not None:
            if schema_format not in self.SUPPORTED_SCHEMA_FORMATS:
                logger.warning(f"Unsupported schema format: {schema_format}. Using 'json_schema' instead.")
                schema_format = "json_schema"
            self.schema_format = schema_format
    
    def __repr__(self) -> str:
        """String representation of the StructuredPromptTemplate."""
        return (f"StructuredPromptTemplate(template='{self.template[:30]}...', "
                f"output_format='{self.output_format}', "
                f"schema_format='{self.schema_format}')") 