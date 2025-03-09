"""
Prompt Template Cache Module

This module provides caching for prompt templates and rendered prompts
to reduce overhead when using the same templates repeatedly.
"""

import logging
import hashlib
import json
import time
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field

try:
    from .cache_manager import CacheManager, PromptTemplateCache as BasePromptCache
except ImportError:
    from cache_manager import CacheManager, PromptTemplateCache as BasePromptCache

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """
    Structure to store a prompt template with metadata.
    """
    template_id: str
    template: str
    version: str = "1.0"
    template_type: str = "text"  # text, chat, json
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    variables: List[str] = field(default_factory=list)


@dataclass
class RenderedPrompt:
    """
    Structure to store a rendered prompt with its parameters.
    """
    template_id: str
    rendered: str
    parameters: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    token_count: Optional[int] = None
    format: str = "text"  # text, messages, json


class PromptCache:
    """
    Cache for prompt templates and rendered prompts to reduce
    overhead when using the same templates repeatedly.
    """
    
    def __init__(
        self,
        cache_manager: Optional[CacheManager] = None,
        template_capacity: int = 100,
        rendered_capacity: int = 1000,
        namespace: str = "prompt_cache",
        renderer: Optional[Callable] = None,
        token_counter: Optional[Callable] = None,
        extract_variables: bool = True
    ):
        """
        Initialize the prompt cache.
        
        Args:
            cache_manager: Optional central cache manager to use
            template_capacity: Maximum number of templates to cache
            rendered_capacity: Maximum number of rendered prompts to cache
            namespace: Namespace for cache keys
            renderer: Optional function to render templates
            token_counter: Optional function to count tokens in prompts
            extract_variables: Whether to extract variables from templates
        """
        # External components
        self.cache_manager = cache_manager
        self.using_external_cache = cache_manager is not None
        
        # Internal cache if not using external manager
        if not self.using_external_cache:
            self.template_cache = {}
            self.rendered_cache = {}
            self.template_capacity = template_capacity
            self.rendered_capacity = rendered_capacity
            
            # LRU tracking
            self.template_lru = []
            self.rendered_lru = []
        
        # Configuration
        self.namespace = namespace
        self.renderer = renderer
        self.token_counter = token_counter
        self.extract_variables = extract_variables
        
        # Metrics
        self.metrics = {
            "template_hits": 0,
            "template_misses": 0,
            "rendered_hits": 0,
            "rendered_misses": 0,
            "template_inserts": 0,
            "rendered_inserts": 0,
            "token_savings": 0
        }
    
    def _extract_template_variables(self, template: str) -> List[str]:
        """
        Extract variable names from a template.
        
        Args:
            template: Template string
            
        Returns:
            List of variable names
        """
        if not self.extract_variables:
            return []
        
        import re
        # Match both {variable} and {{variable}} formats
        pattern = r'\{+([a-zA-Z0-9_]+)\}+'
        matches = re.findall(pattern, template)
        
        # Remove duplicates while preserving order
        seen = set()
        variables = [var for var in matches if not (var in seen or seen.add(var))]
        
        return variables
    
    def _create_params_key(self, params: Dict[str, Any]) -> str:
        """
        Create a stable key from template parameters.
        
        Args:
            params: Template parameters
            
        Returns:
            Stable string key
        """
        # Convert parameters to a stable string representation
        param_items = sorted(params.items())
        params_str = json.dumps(param_items, sort_keys=True)
        
        # Use hash for compact representation
        return hashlib.md5(params_str.encode()).hexdigest()
    
    def _count_tokens(self, text: str) -> Optional[int]:
        """
        Count tokens in a text.
        
        Args:
            text: Text to count tokens in
            
        Returns:
            Token count or None if no counter available
        """
        if self.token_counter is not None:
            try:
                return self.token_counter(text)
            except Exception as e:
                logger.error(f"Error counting tokens: {str(e)}")
        
        return None
    
    def _render_template(self, template: str, params: Dict[str, Any]) -> str:
        """
        Render a template with parameters.
        
        Args:
            template: Template string
            params: Parameters to render with
            
        Returns:
            Rendered template
        """
        if self.renderer is not None:
            try:
                return self.renderer(template, params)
            except Exception as e:
                logger.error(f"Error rendering template: {str(e)}")
                # Fall back to simple string formatting
        
        # Simple string formatting
        try:
            return template.format(**params)
        except Exception as e:
            logger.error(f"Error formatting template: {str(e)}")
            return template
    
    def _update_template_lru(self, template_id: str) -> None:
        """
        Update the LRU status of a template.
        
        Args:
            template_id: Template ID to mark as recently used
        """
        if self.using_external_cache:
            return
        
        # Remove if exists
        if template_id in self.template_lru:
            self.template_lru.remove(template_id)
        
        # Add to end (most recently used)
        self.template_lru.append(template_id)
        
        # Evict if over capacity
        while len(self.template_lru) > self.template_capacity:
            evicted = self.template_lru.pop(0)
            if evicted in self.template_cache:
                del self.template_cache[evicted]
    
    def _update_rendered_lru(self, key: str) -> None:
        """
        Update the LRU status of a rendered prompt.
        
        Args:
            key: Cache key to mark as recently used
        """
        if self.using_external_cache:
            return
        
        # Remove if exists
        if key in self.rendered_lru:
            self.rendered_lru.remove(key)
        
        # Add to end (most recently used)
        self.rendered_lru.append(key)
        
        # Evict if over capacity
        while len(self.rendered_lru) > self.rendered_capacity:
            evicted = self.rendered_lru.pop(0)
            if evicted in self.rendered_cache:
                del self.rendered_cache[evicted]
    
    def get_template(self, template_id: str) -> Tuple[bool, Optional[PromptTemplate]]:
        """
        Get a template from the cache.
        
        Args:
            template_id: Template identifier
            
        Returns:
            Tuple of (hit, template) where:
                hit: Whether the template was found in the cache
                template: The cached template if hit is True, otherwise None
        """
        cache_key = f"{self.namespace}:template:{template_id}"
        
        if self.using_external_cache:
            hit, template = self.cache_manager.get_template(cache_key)
            
            if hit:
                self.metrics["template_hits"] += 1
                return True, template
        else:
            if cache_key in self.template_cache:
                template = self.template_cache[cache_key]
                self._update_template_lru(cache_key)
                self.metrics["template_hits"] += 1
                return True, template
        
        self.metrics["template_misses"] += 1
        return False, None
    
    def put_template(self, template_id: str, template: str, template_type: str = "text", version: str = "1.0", metadata: Optional[Dict[str, Any]] = None) -> PromptTemplate:
        """
        Add a template to the cache.
        
        Args:
            template_id: Template identifier
            template: Template string
            template_type: Type of template (text, chat, json)
            version: Version identifier
            metadata: Additional metadata
            
        Returns:
            PromptTemplate object
        """
        cache_key = f"{self.namespace}:template:{template_id}"
        
        # Extract variables if enabled
        variables = self._extract_template_variables(template) if self.extract_variables else []
        
        # Create template object
        template_obj = PromptTemplate(
            template_id=template_id,
            template=template,
            version=version,
            template_type=template_type,
            metadata=metadata or {},
            variables=variables
        )
        
        if self.using_external_cache:
            self.cache_manager.put_template(cache_key, template_obj)
        else:
            self.template_cache[cache_key] = template_obj
            self._update_template_lru(cache_key)
        
        self.metrics["template_inserts"] += 1
        return template_obj
    
    def get_rendered(self, template_id: str, params: Dict[str, Any]) -> Tuple[bool, Optional[RenderedPrompt]]:
        """
        Get a rendered template from the cache.
        
        Args:
            template_id: Template identifier
            params: Parameters used in rendering
            
        Returns:
            Tuple of (hit, rendered) where:
                hit: Whether the rendered template was found in the cache
                rendered: The cached rendered template if hit is True, otherwise None
        """
        params_key = self._create_params_key(params)
        cache_key = f"{self.namespace}:rendered:{template_id}:{params_key}"
        
        if self.using_external_cache and hasattr(self.cache_manager, 'get_rendered_template'):
            hit, rendered = self.cache_manager.get_rendered_template(template_id, params)
            
            if hit:
                self.metrics["rendered_hits"] += 1
                
                # Track token savings
                if rendered.token_count is not None:
                    self.metrics["token_savings"] += rendered.token_count
                
                return True, rendered
        elif not self.using_external_cache:
            if cache_key in self.rendered_cache:
                rendered = self.rendered_cache[cache_key]
                self._update_rendered_lru(cache_key)
                self.metrics["rendered_hits"] += 1
                
                # Track token savings
                if rendered.token_count is not None:
                    self.metrics["token_savings"] += rendered.token_count
                
                return True, rendered
        
        self.metrics["rendered_misses"] += 1
        return False, None
    
    def put_rendered(self, template_id: str, params: Dict[str, Any], rendered: str, format: str = "text") -> RenderedPrompt:
        """
        Add a rendered template to the cache.
        
        Args:
            template_id: Template identifier
            params: Parameters used in rendering
            rendered: Rendered template string
            format: Format of the rendered output (text, messages, json)
            
        Returns:
            RenderedPrompt object
        """
        params_key = self._create_params_key(params)
        cache_key = f"{self.namespace}:rendered:{template_id}:{params_key}"
        
        # Count tokens if counter available
        token_count = self._count_tokens(rendered)
        
        # Create rendered prompt object
        rendered_obj = RenderedPrompt(
            template_id=template_id,
            rendered=rendered,
            parameters=params,
            token_count=token_count,
            format=format
        )
        
        if self.using_external_cache and hasattr(self.cache_manager, 'put_rendered_template'):
            self.cache_manager.put_rendered_template(template_id, params, rendered_obj)
        elif not self.using_external_cache:
            self.rendered_cache[cache_key] = rendered_obj
            self._update_rendered_lru(cache_key)
        
        self.metrics["rendered_inserts"] += 1
        return rendered_obj
    
    def render(self, template_id: str, params: Dict[str, Any], force_render: bool = False, format: str = "text") -> Tuple[str, bool]:
        """
        Render a template with parameters, using cache if available.
        
        Args:
            template_id: Template identifier
            params: Parameters to render with
            force_render: Whether to bypass cache and force rendering
            format: Format of the rendered output
            
        Returns:
            Tuple of (rendered_text, from_cache) where:
                rendered_text: The rendered template
                from_cache: Whether the result came from cache
        """
        # Try to get from rendered cache if not forcing render
        if not force_render:
            hit, rendered = self.get_rendered(template_id, params)
            if hit:
                return rendered.rendered, True
        
        # Get the template
        template_hit, template_obj = self.get_template(template_id)
        
        if not template_hit:
            logger.warning(f"Template not found: {template_id}")
            # Try a very simple rendering of the template ID as a fallback
            if isinstance(template_id, str) and "{" in template_id and "}" in template_id:
                try:
                    result = template_id.format(**params)
                    return result, False
                except:
                    return f"Error: Template {template_id} not found", False
            return f"Error: Template {template_id} not found", False
        
        # Render the template
        rendered_text = self._render_template(template_obj.template, params)
        
        # Cache the rendered result
        self.put_rendered(template_id, params, rendered_text, format)
        
        return rendered_text, False
    
    def add_system_templates(self, templates: Dict[str, str], prefix: str = "system") -> List[str]:
        """
        Add a batch of system templates.
        
        Args:
            templates: Dictionary mapping template IDs to template strings
            prefix: Prefix to add to template IDs
            
        Returns:
            List of registered template IDs
        """
        template_ids = []
        
        for template_id, template in templates.items():
            full_id = f"{prefix}.{template_id}"
            self.put_template(full_id, template, "text", "1.0", {"type": "system"})
            template_ids.append(full_id)
        
        return template_ids
    
    def register_renderer(self, renderer: Callable) -> None:
        """
        Register a custom template renderer.
        
        Args:
            renderer: Function to render templates
        """
        self.renderer = renderer
    
    def register_token_counter(self, counter: Callable) -> None:
        """
        Register a token counting function.
        
        Args:
            counter: Function to count tokens
        """
        self.token_counter = counter
    
    def clear(self, templates_only: bool = False, rendered_only: bool = False) -> None:
        """
        Clear the cache.
        
        Args:
            templates_only: Whether to only clear template cache
            rendered_only: Whether to only clear rendered cache
        """
        if self.using_external_cache:
            if hasattr(self.cache_manager, 'clear'):
                self.cache_manager.clear()
        else:
            if not rendered_only:
                self.template_cache.clear()
                self.template_lru.clear()
            
            if not templates_only:
                self.rendered_cache.clear()
                self.rendered_lru.clear()
        
        # Reset metrics
        if not rendered_only:
            self.metrics["template_hits"] = 0
            self.metrics["template_misses"] = 0
            self.metrics["template_inserts"] = 0
        
        if not templates_only:
            self.metrics["rendered_hits"] = 0
            self.metrics["rendered_misses"] = 0
            self.metrics["rendered_inserts"] = 0
            self.metrics["token_savings"] = 0
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get cache metrics.
        
        Returns:
            Dictionary with cache metrics
        """
        metrics = self.metrics.copy()
        
        # Add cache sizes
        if not self.using_external_cache:
            metrics["template_cache_size"] = len(self.template_cache)
            metrics["rendered_cache_size"] = len(self.rendered_cache)
        
        # Calculate hit rates
        template_total = metrics["template_hits"] + metrics["template_misses"]
        rendered_total = metrics["rendered_hits"] + metrics["rendered_misses"]
        
        metrics["template_hit_rate"] = metrics["template_hits"] / template_total if template_total > 0 else 0.0
        metrics["rendered_hit_rate"] = metrics["rendered_hits"] / rendered_total if rendered_total > 0 else 0.0
        
        return metrics
    
    def get_variable_info(self, template_id: str) -> Tuple[bool, List[str]]:
        """
        Get information about variables in a template.
        
        Args:
            template_id: Template identifier
            
        Returns:
            Tuple of (success, variables) where:
                success: Whether the template was found
                variables: List of variable names in the template
        """
        hit, template = self.get_template(template_id)
        
        if hit:
            return True, template.variables
        
        return False, []
    
    def get_template_ids(self) -> List[str]:
        """
        Get all template IDs in the cache.
        
        Returns:
            List of template IDs
        """
        if self.using_external_cache:
            return []  # Not supported with external cache manager
        
        prefix = f"{self.namespace}:template:"
        return [key[len(prefix):] for key in self.template_cache.keys()]
    
    def get_template_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all templates.
        
        Returns:
            Dictionary mapping template IDs to statistics
        """
        if self.using_external_cache:
            return {}  # Not supported with external cache manager
        
        prefix = f"{self.namespace}:template:"
        rendered_prefix = f"{self.namespace}:rendered:"
        stats = {}
        
        # Get template info
        for key, template in self.template_cache.items():
            if not key.startswith(prefix):
                continue
            
            template_id = key[len(prefix):]
            
            # Count rendered variants
            rendered_variants = sum(1 for k in self.rendered_cache.keys() 
                                   if k.startswith(f"{rendered_prefix}{template_id}:"))
            
            stats[template_id] = {
                "type": template.template_type,
                "version": template.version,
                "variables": len(template.variables),
                "rendered_variants": rendered_variants,
                "created_at": template.created_at
            }
        
        return stats 