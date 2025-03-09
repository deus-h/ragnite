"""
Multi-Modal RAG Module

This module implements retrieval augmented generation for multi-modal content,
supporting both text and image inputs/outputs. It integrates vision-language models
for understanding visual content and specialized retrievers for different content types.
"""

import logging
import re
import json
import base64
import os
from io import BytesIO
from typing import List, Dict, Any, Optional, Union, Callable, Tuple, Set
import time

# Try to import the model provider and related components
try:
    from tools.src.models import (
        LLMProvider, Message, Role, get_model_provider
    )
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

# Try to import image processing libraries
try:
    import PIL
    from PIL import Image
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


class MultiModalRAG:
    """
    Multi-Modal Retrieval Augmented Generation system.
    
    This class implements RAG for multi-modal content, supporting both text
    and image inputs/outputs. It can process diagrams, charts, and other visual
    content to provide comprehensive responses that combine information from
    both textual and visual sources.
    """
    
    def __init__(
        self,
        model_provider: Optional[Union[str, LLMProvider]] = None,
        vision_model_provider: Optional[Union[str, LLMProvider]] = None,
        text_retriever: Optional[Callable] = None,
        image_retriever: Optional[Callable] = None,
        system_prompt: Optional[str] = None,
        image_analysis_prompt: Optional[str] = None,
        chart_analysis_prompt: Optional[str] = None,
        diagram_analysis_prompt: Optional[str] = None,
        cross_modal_reasoning_prompt: Optional[str] = None,
        output_format_prompt: Optional[str] = None,
        max_text_results: int = 5,
        max_image_results: int = 3,
        temperature: float = 0.3,
        image_similarity_threshold: float = 0.75,
        enable_img2text: bool = True,
        enable_text2img: bool = True,
        image_description_detail: str = "medium",  # "low", "medium", "high"
        supported_image_formats: Optional[List[str]] = None,
        cache_processed_images: bool = True,
    ):
        """
        Initialize the Multi-Modal RAG system.
        
        Args:
            model_provider: LLM provider for generating reasoning and answers
            vision_model_provider: Vision-language model provider for image understanding
            text_retriever: Function for retrieving text passages from a knowledge base
            image_retriever: Function for retrieving images from a knowledge base
            system_prompt: Custom system prompt for the LLM
            image_analysis_prompt: Custom prompt for image analysis
            chart_analysis_prompt: Custom prompt for chart/graph analysis
            diagram_analysis_prompt: Custom prompt for diagram analysis
            cross_modal_reasoning_prompt: Custom prompt for cross-modal reasoning
            output_format_prompt: Custom prompt for formatting multi-modal output
            max_text_results: Maximum number of text results to retrieve
            max_image_results: Maximum number of image results to retrieve
            temperature: Temperature for LLM generation
            image_similarity_threshold: Threshold for considering images similar
            enable_img2text: Whether to enable image-to-text capabilities
            enable_text2img: Whether to enable text-to-image capabilities
            image_description_detail: Level of detail for image descriptions
            supported_image_formats: List of supported image formats (default: ["jpg", "jpeg", "png", "gif"])
            cache_processed_images: Whether to cache processed images
        """
        if not MODELS_AVAILABLE:
            raise ImportError(
                "The models module is required. Make sure the 'tools.src.models' module is available."
            )
        
        if not PILLOW_AVAILABLE and (enable_img2text or enable_text2img):
            logger.warning("PIL/Pillow is not available. Image processing capabilities will be limited.")
        
        # Set up the model provider for text generation
        if model_provider is None:
            try:
                # Default to OpenAI GPT-4 if not specified
                self.model_provider = get_model_provider("openai", model="gpt-4o")
                logger.info("Using default OpenAI GPT-4o model provider for Multi-Modal RAG")
            except Exception as e:
                logger.warning(f"Error initializing default model provider: {str(e)}")
                raise
        elif isinstance(model_provider, str):
            try:
                self.model_provider = get_model_provider(model_provider)
                logger.info(f"Using {model_provider} model provider for Multi-Modal RAG")
            except Exception as e:
                logger.warning(f"Error initializing model provider {model_provider}: {str(e)}")
                raise
        else:
            self.model_provider = model_provider
        
        # Set up the vision model provider
        if vision_model_provider is None:
            # Use the same model provider if it supports vision
            if hasattr(self.model_provider, "supports_vision") and self.model_provider.supports_vision:
                self.vision_model_provider = self.model_provider
                logger.info("Using the same model provider for vision tasks")
            else:
                try:
                    # Try to get a vision-capable model
                    self.vision_model_provider = get_model_provider("openai", model="gpt-4o-vision")
                    logger.info("Using GPT-4o Vision for image analysis")
                except Exception as e:
                    logger.warning(f"Error initializing vision model provider: {str(e)}")
                    # Disable image capabilities
                    self.enable_img2text = False
                    self.enable_text2img = False
                    logger.warning("Image processing capabilities disabled due to missing vision model")
        elif isinstance(vision_model_provider, str):
            try:
                self.vision_model_provider = get_model_provider(vision_model_provider)
                logger.info(f"Using {vision_model_provider} for vision tasks")
            except Exception as e:
                logger.warning(f"Error initializing vision model provider {vision_model_provider}: {str(e)}")
                # Disable image capabilities
                self.enable_img2text = False
                self.enable_text2img = False
                logger.warning("Image processing capabilities disabled due to missing vision model")
        else:
            self.vision_model_provider = vision_model_provider
        
        # Set retrievers (can be updated later if None)
        self.text_retriever = text_retriever
        self.image_retriever = image_retriever
        
        # Set multi-modal parameters
        self.max_text_results = max_text_results
        self.max_image_results = max_image_results
        self.temperature = temperature
        self.image_similarity_threshold = image_similarity_threshold
        self.enable_img2text = enable_img2text and PILLOW_AVAILABLE
        self.enable_text2img = enable_text2img and PILLOW_AVAILABLE
        self.image_description_detail = image_description_detail
        self.cache_processed_images = cache_processed_images
        
        # Set supported image formats
        self.supported_image_formats = supported_image_formats or ["jpg", "jpeg", "png", "gif"]
        
        # Initialize default prompts
        self._init_default_prompts()
        
        # Override with custom prompts if provided
        if system_prompt:
            self.system_prompt = system_prompt
        if image_analysis_prompt:
            self.image_analysis_prompt = image_analysis_prompt
        if chart_analysis_prompt:
            self.chart_analysis_prompt = chart_analysis_prompt
        if diagram_analysis_prompt:
            self.diagram_analysis_prompt = diagram_analysis_prompt
        if cross_modal_reasoning_prompt:
            self.cross_modal_reasoning_prompt = cross_modal_reasoning_prompt
        if output_format_prompt:
            self.output_format_prompt = output_format_prompt
        
        # Cache for processed images
        self.image_cache = {}
        
        logger.debug(f"Initialized Multi-Modal RAG with text and image capabilities")
    
    def _init_default_prompts(self):
        """Initialize default prompts for the Multi-Modal RAG system."""
        
        # Main system prompt
        self.system_prompt = """
        You are an expert at answering questions that require understanding both text and visual content.
        Your goal is to provide comprehensive, accurate answers by analyzing both textual information
        and visual elements such as images, charts, diagrams, and other visual content.
        
        When processing a query:
        1. Analyze all provided text and images carefully
        2. Extract relevant information from both modalities
        3. Integrate insights from text and visual sources
        4. Provide clear, thorough explanations that reference both text and visual elements
        5. When appropriate, describe visual content in detail to support your answer
        
        Be precise, thorough, and ensure your reasoning is clear. Explicitly mention when you're
        referencing specific parts of an image or text passage. If information is incomplete
        or uncertain, acknowledge this and explain what additional information would be helpful.
        """
        
        # Prompt for general image analysis
        self.image_analysis_prompt = """
        Please analyze the following image in detail.
        
        Describe:
        1. The main subject or content of the image
        2. Key visual elements, objects, people, or scenes present
        3. Any text visible in the image
        4. The overall composition, setting, or context
        5. Any notable colors, patterns, or visual characteristics
        
        Detail level: {detail_level}
        
        Provide a comprehensive description that captures both the obvious and subtle 
        elements of the image that would be relevant to answering the query: {query}
        """
        
        # Prompt for chart/graph analysis
        self.chart_analysis_prompt = """
        Please analyze the following chart or graph in detail.
        
        Describe:
        1. The type of chart/graph (bar chart, line graph, pie chart, scatter plot, etc.)
        2. The title, axis labels, and legend (if present)
        3. The data being represented and key trends or patterns
        4. The specific values or data points, especially extremes or outliers
        5. The overall conclusion or insight that can be drawn from this visualization
        
        If this is a complex chart with multiple data series or visualization elements,
        break down each component systematically.
        
        Provide a detailed analysis of this chart in relation to the query: {query}
        """
        
        # Prompt for diagram analysis
        self.diagram_analysis_prompt = """
        Please analyze the following diagram in detail.
        
        Describe:
        1. The type of diagram (flowchart, process diagram, architectural diagram, etc.)
        2. The title and any labeled components or sections
        3. The relationships, connections, or flow between elements
        4. Any hierarchies, sequences, or organizational structures depicted
        5. The main concept or system being illustrated by this diagram
        
        Ensure you capture both the overall structure and the specific elements/details
        that would be relevant to answering the query: {query}
        """
        
        # Prompt for cross-modal reasoning
        self.cross_modal_reasoning_prompt = """
        I need to answer a question that requires reasoning across both text and visual information.
        
        Question: {query}
        
        Text Information:
        {text_context}
        
        Visual Information:
        {visual_context}
        
        Based on both the textual and visual information provided, develop a comprehensive answer
        that integrates insights from both sources. When referencing specific parts of the text
        or images, be explicit about which source you're drawing from. Ensure your reasoning
        process is clear, showing how you connected information across different modalities.
        """
        
        # Prompt for formatting multi-modal output
        self.output_format_prompt = """
        Based on your multi-modal reasoning, please provide a final answer to the question.
        
        Your answer should:
        1. Directly address the original question: {query}
        2. Integrate information from both text and visual sources
        3. Include specific references to key text passages and visual elements
        4. Be structured clearly with a logical flow
        5. Include citations to sources when appropriate
        
        {additional_formatting_instructions}
        """
    
    def set_text_retriever(self, retriever: Callable):
        """
        Set or update the text retriever function.
        
        Args:
            retriever: Function for retrieving text passages from a knowledge base
        """
        self.text_retriever = retriever
        logger.info("Text retriever function updated")
    
    def set_image_retriever(self, retriever: Callable):
        """
        Set or update the image retriever function.
        
        Args:
            retriever: Function for retrieving images from a knowledge base
        """
        self.image_retriever = retriever
        logger.info("Image retriever function updated")
    
    def _process_image_input(self, image_data: Union[str, bytes, PIL.Image.Image]) -> Optional[Dict[str, Any]]:
        """
        Process an image input into a standardized format.
        
        Args:
            image_data: The image data (file path, URL, base64, bytes, or PIL Image)
            
        Returns:
            Processed image data dictionary or None if processing failed
        """
        if not self.enable_img2text:
            logger.warning("Image-to-text capabilities are disabled")
            return None
        
        # Check cache if enabled and input is hashable
        if self.cache_processed_images and isinstance(image_data, (str, bytes)):
            cache_key = str(hash(image_data))
            if cache_key in self.image_cache:
                return self.image_cache[cache_key]
        else:
            cache_key = None
        
        try:
            pil_image = None
            image_source = "unknown"
            
            # Handle different input types
            if isinstance(image_data, PIL.Image.Image):
                pil_image = image_data
                image_source = "pil_image"
            
            elif isinstance(image_data, bytes):
                # Bytes data
                pil_image = Image.open(BytesIO(image_data))
                image_source = "bytes"
            
            elif isinstance(image_data, str):
                if image_data.startswith(('http://', 'https://')):
                    # URL - we'd need to add a dependency like requests to handle this properly
                    # For now, just note that we need external libraries
                    logger.warning("Processing image URLs requires the 'requests' library")
                    import requests
                    response = requests.get(image_data, stream=True)
                    response.raise_for_status()
                    pil_image = Image.open(BytesIO(response.content))
                    image_source = "url"
                    
                elif image_data.startswith('data:image/'):
                    # Base64 encoded image
                    # Extract the actual base64 content
                    base64_data = image_data.split(',')[1] if ',' in image_data else image_data
                    image_bytes = base64.b64decode(base64_data)
                    pil_image = Image.open(BytesIO(image_bytes))
                    image_source = "base64"
                    
                elif os.path.isfile(image_data):
                    # Local file path
                    pil_image = Image.open(image_data)
                    image_source = "file"
                    
                else:
                    # Try as direct base64
                    try:
                        image_bytes = base64.b64decode(image_data)
                        pil_image = Image.open(BytesIO(image_bytes))
                        image_source = "direct_base64"
                    except Exception as e:
                        logger.error(f"Could not process image data as base64: {str(e)}")
                        return None
            
            else:
                logger.error(f"Unsupported image data type: {type(image_data)}")
                return None
            
            # If we successfully loaded the image, process it
            if pil_image:
                # Get basic metadata
                width, height = pil_image.size
                format_name = pil_image.format or "UNKNOWN"
                mode = pil_image.mode
                
                # Convert to RGB if needed
                if mode != "RGB":
                    pil_image = pil_image.convert("RGB")
                
                # Create base64 representation for the vision model
                buffered = BytesIO()
                pil_image.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                # Create result dictionary
                result = {
                    "image": pil_image,
                    "width": width,
                    "height": height,
                    "format": format_name,
                    "mode": mode,
                    "source": image_source,
                    "base64": img_str,
                    "mime_type": f"image/jpeg"  # We converted to JPEG
                }
                
                # Cache result if enabled
                if self.cache_processed_images and cache_key:
                    self.image_cache[cache_key] = result
                
                return result
                
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return None
    
    def _analyze_image(self, 
                     image_data: Dict[str, Any], 
                     query: str,
                     content_type: str = "general") -> Dict[str, Any]:
        """
        Analyze an image using the vision model.
        
        Args:
            image_data: Processed image data dictionary
            query: The user's query to provide context for analysis
            content_type: Type of content ("general", "chart", "diagram")
            
        Returns:
            Analysis results dictionary
        """
        if not self.enable_img2text:
            logger.warning("Image-to-text capabilities are disabled")
            return {"description": "Image analysis not available", "error": "Image analysis disabled"}
        
        # Select the appropriate prompt based on content type
        if content_type == "chart":
            prompt = self.chart_analysis_prompt.format(query=query)
        elif content_type == "diagram":
            prompt = self.diagram_analysis_prompt.format(query=query)
        else:
            # General image analysis with detail level
            prompt = self.image_analysis_prompt.format(
                query=query,
                detail_level=self.image_description_detail
            )
        
        # Create messages for the vision model
        messages = [
            Message(role=Role.SYSTEM, content=self.system_prompt),
            Message(
                role=Role.USER,
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image", 
                        "image_url": {
                            "url": f"data:{image_data['mime_type']};base64,{image_data['base64']}",
                            "detail": self.image_description_detail
                        }
                    }
                ]
            )
        ]
        
        # Generate analysis with vision model
        try:
            response = self.vision_model_provider.generate(
                messages=messages,
                temperature=self.temperature
            )
            
            analysis = response.get("content", "")
            
            # Create result dictionary
            result = {
                "description": analysis,
                "content_type": content_type,
                "width": image_data.get("width"),
                "height": image_data.get("height"),
                "format": image_data.get("format"),
                "analyzed_for_query": query
            }
            
            logger.info(f"Generated image analysis of {len(analysis.split())} words")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            return {
                "description": f"Error analyzing image: {str(e)}",
                "content_type": content_type,
                "error": str(e)
            }
    
    def _detect_content_type(self, image_data: Dict[str, Any], query: str) -> str:
        """
        Detect the type of visual content (general image, chart, diagram).
        
        Args:
            image_data: Processed image data dictionary
            query: The user's query to provide context
            
        Returns:
            Content type ("general", "chart", "diagram")
        """
        # Use vision model to classify the image
        # We'll use a simplified prompt for this
        classify_prompt = """
        Classify this image into one of the following categories:
        1. "chart" - If it's a chart, graph, plot, or data visualization
        2. "diagram" - If it's a diagram, flowchart, architecture, process flow, or schematic
        3. "general" - If it's a general photograph, screenshot, or other image
        
        Reply with ONLY one word: "chart", "diagram", or "general".
        The query related to this image is: {query}
        """
        
        # Create messages for the vision model
        messages = [
            Message(
                role=Role.USER,
                content=[
                    {"type": "text", "text": classify_prompt.format(query=query)},
                    {
                        "type": "image", 
                        "image_url": {
                            "url": f"data:{image_data['mime_type']};base64,{image_data['base64']}",
                            "detail": "low"  # Low detail is sufficient for classification
                        }
                    }
                ]
            )
        ]
        
        # Generate classification with vision model
        try:
            response = self.vision_model_provider.generate(
                messages=messages,
                temperature=0.1  # Low temperature for more consistent classification
            )
            
            classification = response.get("content", "").strip().lower()
            
            # Extract just the classification word
            if "chart" in classification:
                return "chart"
            elif "diagram" in classification:
                return "diagram"
            else:
                return "general"
            
        except Exception as e:
            logger.error(f"Error classifying image: {str(e)}")
            return "general"  # Default to general on error
    
    def _retrieve_text(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve text passages relevant to the query.
        
        Args:
            query: The user's query
            
        Returns:
            List of retrieved text passages
        """
        if not self.text_retriever:
            logger.warning("Text retriever not configured")
            return []
        
        try:
            # Retrieve passages
            logger.info(f"Retrieving text passages for query: {query}")
            passages = self.text_retriever(
                query=query,
                limit=self.max_text_results
            )
            
            logger.info(f"Retrieved {len(passages)} text passages")
            return passages
            
        except Exception as e:
            logger.error(f"Error retrieving text passages: {str(e)}")
            return []
    
    def _retrieve_images(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve images relevant to the query.
        
        Args:
            query: The user's query
            
        Returns:
            List of retrieved and processed images
        """
        if not self.image_retriever or not self.enable_img2text:
            logger.warning("Image retrieval not available")
            return []
        
        try:
            # Retrieve images
            logger.info(f"Retrieving images for query: {query}")
            images = self.image_retriever(
                query=query,
                limit=self.max_image_results
            )
            
            # Process each image
            processed_images = []
            for img in images:
                # Extract image data from the retrieval result
                image_data = img.get("image", img.get("content", img.get("data")))
                
                if not image_data:
                    logger.warning(f"No image data found in retrieval result: {img}")
                    continue
                
                # Process the image
                processed = self._process_image_input(image_data)
                if processed:
                    # Add metadata from the original retrieval result
                    processed.update({
                        "id": img.get("id", str(hash(image_data))[:10]),
                        "score": img.get("score", 1.0),
                        "title": img.get("title", "Untitled Image"),
                        "source": img.get("source", processed.get("source", "unknown")),
                        "metadata": img.get("metadata", {})
                    })
                    processed_images.append(processed)
            
            logger.info(f"Retrieved and processed {len(processed_images)} images")
            return processed_images
            
        except Exception as e:
            logger.error(f"Error retrieving images: {str(e)}")
            return []
    
    def _analyze_retrieved_images(self, 
                               images: List[Dict[str, Any]], 
                               query: str) -> List[Dict[str, Any]]:
        """
        Analyze retrieved images for their content and relevance.
        
        Args:
            images: List of retrieved and processed images
            query: The user's query
            
        Returns:
            List of images with analysis results
        """
        if not images:
            return []
        
        analyzed_images = []
        
        for img in images:
            # Detect content type
            content_type = self._detect_content_type(img, query)
            
            # Analyze the image
            analysis = self._analyze_image(img, query, content_type)
            
            # Combine original image data with analysis
            img_with_analysis = {**img, **analysis}
            analyzed_images.append(img_with_analysis)
        
        return analyzed_images
    
    def _format_text_context(self, passages: List[Dict[str, Any]]) -> str:
        """
        Format text passages for inclusion in reasoning prompts.
        
        Args:
            passages: List of retrieved text passages
            
        Returns:
            Formatted text context string
        """
        if not passages:
            return "No relevant text passages found."
        
        formatted = ""
        
        for i, passage in enumerate(passages):
            content = passage.get("content", "")
            source = passage.get("source", "Unknown")
            title = passage.get("title", "Untitled")
            
            formatted += f"\nPASSAGE {i+1} [{source}: {title}]:\n{content}\n"
        
        return formatted
    
    def _format_visual_context(self, images: List[Dict[str, Any]]) -> str:
        """
        Format analyzed images for inclusion in reasoning prompts.
        
        Args:
            images: List of analyzed images
            
        Returns:
            Formatted visual context string
        """
        if not images:
            return "No relevant images found."
        
        formatted = ""
        
        for i, img in enumerate(images):
            description = img.get("description", "No description available")
            content_type = img.get("content_type", "general")
            title = img.get("title", "Untitled Image")
            source = img.get("source", "Unknown")
            
            formatted += f"\nIMAGE {i+1} [{content_type.upper()}: {title} from {source}]:\n{description}\n"
        
        return formatted
    
    def _reason_across_modalities(self, 
                               query: str, 
                               text_context: str, 
                               visual_context: str) -> str:
        """
        Perform cross-modal reasoning to answer the query.
        
        Args:
            query: The user's query
            text_context: Formatted text context
            visual_context: Formatted visual context
            
        Returns:
            Cross-modal reasoning result
        """
        # Format the cross-modal reasoning prompt
        formatted_prompt = self.cross_modal_reasoning_prompt.format(
            query=query,
            text_context=text_context,
            visual_context=visual_context
        )
        
        # Create messages for reasoning
        messages = [
            Message(role=Role.SYSTEM, content=self.system_prompt),
            Message(role=Role.USER, content=formatted_prompt)
        ]
        
        # Generate reasoning
        try:
            response = self.model_provider.generate(
                messages=messages,
                temperature=self.temperature
            )
            
            reasoning = response.get("content", "")
            
            logger.info(f"Generated cross-modal reasoning of {len(reasoning.split())} words")
            return reasoning
            
        except Exception as e:
            logger.error(f"Error in cross-modal reasoning: {str(e)}")
            return f"Error generating cross-modal reasoning: {str(e)}"
    
    def _format_final_output(self, 
                          query: str, 
                          reasoning: str,
                          include_sources: bool = True) -> str:
        """
        Format the final output response.
        
        Args:
            query: The user's query
            reasoning: Cross-modal reasoning text
            include_sources: Whether to include source information
            
        Returns:
            Formatted final output
        """
        # Additional formatting instructions
        if include_sources:
            additional_instructions = "Include citations to source materials where appropriate."
        else:
            additional_instructions = "Focus on a clear, direct answer without explicit citations."
        
        # Format the output formatting prompt
        formatted_prompt = self.output_format_prompt.format(
            query=query,
            additional_formatting_instructions=additional_instructions
        )
        
        # Create messages for output formatting
        messages = [
            Message(role=Role.SYSTEM, content=self.system_prompt),
            Message(role=Role.USER, content=formatted_prompt),
            Message(role=Role.ASSISTANT, content=reasoning)
        ]
        
        # Generate formatted output
        try:
            response = self.model_provider.generate(
                messages=messages,
                temperature=self.temperature
            )
            
            output = response.get("content", "")
            
            logger.info(f"Generated final output of {len(output.split())} words")
            return output
            
        except Exception as e:
            logger.error(f"Error formatting final output: {str(e)}")
            return reasoning  # Fall back to the reasoning text if formatting fails
    
    def process_query(self, 
                    query: str, 
                    input_images: Optional[List[Union[str, bytes, PIL.Image.Image]]] = None,
                    include_source_images: bool = True,
                    include_source_texts: bool = True) -> Dict[str, Any]:
        """
        Process a multi-modal query with optional input images.
        
        Args:
            query: The user's query
            input_images: Optional list of input images
            include_source_images: Whether to include source images in the response
            include_source_texts: Whether to include source texts in the response
            
        Returns:
            Dictionary with query results
        """
        start_time = time.time()
        logger.info(f"Processing multi-modal query: {query}")
        
        # Initialize result structure
        result = {
            "query": query,
            "answer": "",
            "reasoning": "",
            "retrieved_texts": [],
            "retrieved_images": [],
            "input_images_analysis": [],
            "metrics": {
                "text_count": 0,
                "image_count": 0,
                "input_image_count": 0,
                "processing_time": 0
            }
        }
        
        try:
            # Process input images if provided
            input_image_analyses = []
            if input_images and self.enable_img2text:
                for img_data in input_images:
                    # Process the image
                    processed = self._process_image_input(img_data)
                    if processed:
                        # Detect content type
                        content_type = self._detect_content_type(processed, query)
                        
                        # Analyze the image
                        analysis = self._analyze_image(processed, query, content_type)
                        
                        # Combine with original data
                        img_with_analysis = {**processed, **analysis}
                        input_image_analyses.append(img_with_analysis)
                
                result["metrics"]["input_image_count"] = len(input_image_analyses)
                result["input_images_analysis"] = input_image_analyses
            
            # Retrieve relevant text passages
            text_passages = self._retrieve_text(query)
            result["metrics"]["text_count"] = len(text_passages)
            
            if include_source_texts:
                result["retrieved_texts"] = text_passages
            
            # Retrieve relevant images
            retrieved_images = self._retrieve_images(query)
            
            # Analyze retrieved images
            analyzed_images = self._analyze_retrieved_images(retrieved_images, query)
            result["metrics"]["image_count"] = len(analyzed_images)
            
            if include_source_images:
                result["retrieved_images"] = analyzed_images
            
            # Format contexts for reasoning
            text_context = self._format_text_context(text_passages)
            
            # Combine retrieved and input images for visual context
            all_analyzed_images = input_image_analyses + analyzed_images
            visual_context = self._format_visual_context(all_analyzed_images)
            
            # Perform cross-modal reasoning
            reasoning = self._reason_across_modalities(query, text_context, visual_context)
            result["reasoning"] = reasoning
            
            # Format final output
            answer = self._format_final_output(
                query=query,
                reasoning=reasoning,
                include_sources=include_source_texts or include_source_images
            )
            result["answer"] = answer
            
            # Calculate processing time
            end_time = time.time()
            processing_time = end_time - start_time
            result["metrics"]["processing_time"] = processing_time
            
            logger.info(f"Multi-modal processing completed in {processing_time:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Error in multi-modal RAG: {str(e)}")
            
            # Add error information to results
            result["error"] = str(e)
            
            # Try to generate a partial answer if possible
            if not result["answer"] and result["reasoning"]:
                result["answer"] = result["reasoning"]
            elif not result["answer"]:
                result["answer"] = f"Unable to generate an answer due to an error: {str(e)}"
            
            # Calculate processing time even in case of error
            end_time = time.time()
            result["metrics"]["processing_time"] = end_time - start_time
            
            return result
    
    def process_image_query(self, 
                          image: Union[str, bytes, PIL.Image.Image], 
                          query: str) -> Dict[str, Any]:
        """
        Process a query about a specific image.
        
        Args:
            image: The image to analyze
            query: The user's query about the image
            
        Returns:
            Dictionary with query results
        """
        # Simplify by using the main process_query method with just the input image
        return self.process_query(
            query=query,
            input_images=[image],
            include_source_images=False,  # No need for additional retrieved images
            include_source_texts=True     # Still include relevant text if available
        ) 