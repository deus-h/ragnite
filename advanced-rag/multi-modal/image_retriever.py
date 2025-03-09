"""
Multi-Modal RAG Image Retriever Module

This module provides image retrieval capabilities for the multi-modal RAG system,
including specialized retrievers for different content types and image-to-text/text-to-image
search functionality.
"""

import logging
import os
import base64
import time
from io import BytesIO
from typing import List, Dict, Any, Optional, Union, Callable, Tuple

# Try to import the model provider and related components
try:
    from tools.src.models import LLMProvider, Message, Role, get_model_provider
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


class ImageRetriever:
    """
    Image retriever for multi-modal RAG system.
    
    This class provides methods for retrieving images based on text queries,
    including specialized retrievers for charts, diagrams, and other visual content.
    It also supports image-to-image similarity search and hybrid retrieval.
    """
    
    def __init__(
        self,
        vector_db_client=None,
        image_embedder: Optional[Callable] = None,
        text_embedder: Optional[Callable] = None,
        image_collection: str = "images",
        image_directory: Optional[str] = None,
        support_content_types: bool = True,
        similarity_threshold: float = 0.75,
        max_results: int = 10,
        enable_hybrid_search: bool = True,
        enable_img2img: bool = True,
        cache_embeddings: bool = True,
    ):
        """
        Initialize the image retriever.
        
        Args:
            vector_db_client: Client for vector database storage
            image_embedder: Function to generate embeddings for images
            text_embedder: Function to generate embeddings for text
            image_collection: Name of the collection for image storage
            image_directory: Directory containing local images (optional)
            support_content_types: Whether to support specialized content types
            similarity_threshold: Threshold for considering images similar
            max_results: Maximum number of results to return
            enable_hybrid_search: Whether to enable hybrid search
            enable_img2img: Whether to enable image-to-image search
            cache_embeddings: Whether to cache embeddings
        """
        # Check for image processing capabilities
        if not PILLOW_AVAILABLE and (image_embedder or enable_img2img):
            logger.warning("PIL/Pillow is not available. Image processing capabilities will be limited.")
        
        # Set up the vector database client (if provided)
        self.vector_db_client = vector_db_client
        
        # Set up embedders
        self.image_embedder = image_embedder
        self.text_embedder = text_embedder
        
        # Set up image storage
        self.image_collection = image_collection
        self.image_directory = image_directory
        
        # Set parameters
        self.support_content_types = support_content_types
        self.similarity_threshold = similarity_threshold
        self.max_results = max_results
        self.enable_hybrid_search = enable_hybrid_search
        self.enable_img2img = enable_img2img and PILLOW_AVAILABLE
        self.cache_embeddings = cache_embeddings
        
        # Cache storage
        self.embedding_cache = {}
        
        # Register content type handlers
        self.content_type_handlers = {
            "chart": self._retrieve_charts,
            "diagram": self._retrieve_diagrams,
            "photo": self._retrieve_photos,
            "general": self._retrieve_general,
        }
        
        # Initialize if we have a vector DB client
        if self.vector_db_client and hasattr(self.vector_db_client, "get_collection"):
            self._ensure_collection_exists()
        
        logger.debug(f"Initialized ImageRetriever with collection '{image_collection}'")
    
    def _ensure_collection_exists(self):
        """Ensure the image collection exists in the vector database."""
        try:
            # Check if collection exists
            if hasattr(self.vector_db_client, "list_collections"):
                collections = self.vector_db_client.list_collections()
                if self.image_collection not in collections:
                    # Create collection if it doesn't exist
                    if hasattr(self.vector_db_client, "create_collection"):
                        logger.info(f"Creating image collection '{self.image_collection}'")
                        self.vector_db_client.create_collection(
                            name=self.image_collection,
                            dimension=512  # Default dimension, may need adjustment
                        )
                    else:
                        logger.warning(f"Cannot create collection '{self.image_collection}' automatically")
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {str(e)}")
    
    def set_image_embedder(self, embedder: Callable):
        """
        Set or update the image embedding function.
        
        Args:
            embedder: Function to generate embeddings for images
        """
        self.image_embedder = embedder
        # Clear embedding cache if we have a new embedder
        if self.cache_embeddings:
            self.embedding_cache = {}
        logger.info("Image embedder updated")
    
    def set_text_embedder(self, embedder: Callable):
        """
        Set or update the text embedding function.
        
        Args:
            embedder: Function to generate embeddings for text
        """
        self.text_embedder = embedder
        logger.info("Text embedder updated")
    
    def _process_image(self, image_data: Union[str, bytes, PIL.Image.Image]) -> Optional[Dict[str, Any]]:
        """
        Process an image into a standardized format.
        
        Args:
            image_data: Image data (path, URL, base64, bytes, or PIL Image)
            
        Returns:
            Processed image data or None if processing failed
        """
        if not PILLOW_AVAILABLE:
            logger.warning("PIL/Pillow is not available. Cannot process image.")
            return None
        
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
                    # URL - would need requests library
                    logger.warning("Processing image URLs requires the 'requests' library")
                    try:
                        import requests
                        response = requests.get(image_data, stream=True)
                        response.raise_for_status()
                        pil_image = Image.open(BytesIO(response.content))
                        image_source = "url"
                    except ImportError:
                        logger.error("Requests library not available, cannot process URLs")
                        return None
                    
                elif image_data.startswith('data:image/'):
                    # Base64 encoded image
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
                    except Exception:
                        logger.error("Could not process image data as base64")
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
                
                # Create base64 representation
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
                
                return result
                
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return None
    
    def _get_content_type(self, image_data: Dict[str, Any], query: str = "") -> str:
        """
        Determine the content type of an image.
        
        Args:
            image_data: Processed image data
            query: Optional query to provide context
            
        Returns:
            Content type ("chart", "diagram", "photo", or "general")
        """
        # This would ideally use a vision model to classify the image
        # For simplicity, we'll use heuristics based on image properties
        
        # If we have metadata that specifies the content type, use that
        if "metadata" in image_data and "content_type" in image_data["metadata"]:
            return image_data["metadata"]["content_type"]
        
        # Check for content type in query
        if query:
            query_lower = query.lower()
            if any(term in query_lower for term in ["chart", "graph", "plot", "visualization"]):
                return "chart"
            elif any(term in query_lower for term in ["diagram", "flowchart", "architecture", "process flow"]):
                return "diagram"
            elif any(term in query_lower for term in ["photo", "picture", "image", "photograph"]):
                return "photo"
        
        # Use simple heuristics based on image properties
        img = image_data.get("image")
        width = image_data.get("width", 0)
        height = image_data.get("height", 0)
        
        if img and hasattr(img, "histogram"):
            # Check if the image looks like a chart/diagram (fewer colors, more white space)
            histogram = img.histogram()
            
            # Check if image has mostly a few dominant colors (common in charts/diagrams)
            dominant_colors = sum(1 for count in histogram if count > (width * height * 0.05))
            if dominant_colors < 10:
                # Further check for charts vs diagrams
                if width > height * 1.5:  # Charts often have wider aspect ratios
                    return "chart"
                else:
                    return "diagram"
        
        # Default to general
        return "general"
    
    def _embed_image(self, image_data: Dict[str, Any]) -> List[float]:
        """
        Generate an embedding vector for an image.
        
        Args:
            image_data: Processed image data
            
        Returns:
            Embedding vector for the image
        """
        if not self.image_embedder:
            logger.warning("Image embedder not configured")
            return []
        
        try:
            # Check cache first if enabled
            if self.cache_embeddings:
                cache_key = image_data.get("base64", "")[:100]  # Use prefix of base64 as cache key
                if cache_key in self.embedding_cache:
                    return self.embedding_cache[cache_key]
            
            # Generate embedding
            img = image_data.get("image")
            if img:
                embedding = self.image_embedder(img)
                
                # Cache the result if enabled
                if self.cache_embeddings and cache_key:
                    self.embedding_cache[cache_key] = embedding
                
                return embedding
            else:
                logger.error("No image found in image data")
                return []
                
        except Exception as e:
            logger.error(f"Error embedding image: {str(e)}")
            return []
    
    def _embed_text(self, text: str) -> List[float]:
        """
        Generate an embedding vector for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector for the text
        """
        if not self.text_embedder:
            logger.warning("Text embedder not configured")
            return []
        
        try:
            # Check cache first if enabled
            if self.cache_embeddings:
                cache_key = f"text_{hash(text)}"
                if cache_key in self.embedding_cache:
                    return self.embedding_cache[cache_key]
            
            # Generate embedding
            embedding = self.text_embedder(text)
            
            # Cache the result if enabled
            if self.cache_embeddings:
                self.embedding_cache[cache_key] = embedding
            
            return embedding
                
        except Exception as e:
            logger.error(f"Error embedding text: {str(e)}")
            return []
    
    def _retrieve_by_embedding(self, 
                             embedding: List[float], 
                             limit: int = None, 
                             content_type: Optional[str] = None,
                             **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve images by embedding similarity.
        
        Args:
            embedding: Query embedding vector
            limit: Maximum number of results to return
            content_type: Optional content type filter
            **kwargs: Additional parameters for the query
            
        Returns:
            List of image results
        """
        if not self.vector_db_client:
            logger.warning("Vector database client not configured")
            return []
        
        if not embedding:
            logger.warning("Empty embedding provided")
            return []
        
        limit = limit or self.max_results
        
        try:
            # Prepare filters
            filters = {}
            if content_type:
                filters["content_type"] = content_type
            
            # Add any additional filters from kwargs
            metadata_filters = kwargs.get("metadata_filters", {})
            if metadata_filters:
                filters.update(metadata_filters)
            
            # Perform the query
            collection = self.vector_db_client.get_collection(self.image_collection)
            results = collection.query(
                query_embeddings=[embedding],
                n_results=limit,
                filter=filters
            )
            
            # Process results
            processed_results = []
            for idx, (id, distance) in enumerate(zip(results.ids[0], results.distances[0])):
                # Convert distance to similarity score (1.0 is most similar)
                similarity = 1.0 - min(1.0, float(distance))
                
                # Skip results below threshold
                if similarity < self.similarity_threshold:
                    continue
                
                # Get metadata
                metadata = results.metadatas[0][idx] if results.metadatas else {}
                
                # Construct result
                result = {
                    "id": id,
                    "score": similarity,
                    "content_type": metadata.get("content_type", "general"),
                    "title": metadata.get("title", f"Image {id}"),
                    "source": metadata.get("source", "unknown"),
                    "metadata": metadata,
                }
                
                # Add image data if available
                if hasattr(results, "documents") and results.documents:
                    result["data"] = results.documents[0][idx]
                
                processed_results.append(result)
            
            logger.info(f"Retrieved {len(processed_results)} images by embedding similarity")
            return processed_results
            
        except Exception as e:
            logger.error(f"Error retrieving images by embedding: {str(e)}")
            return []
    
    def _retrieve_charts(self, query: str, limit: int = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve chart images relevant to the query.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            **kwargs: Additional parameters for the query
            
        Returns:
            List of chart image results
        """
        # Enhance the query to be more chart-specific
        chart_query = f"chart visualization showing {query}"
        
        # Generate embedding for the chart-specific query
        embedding = self._embed_text(chart_query)
        
        # Retrieve with content type filter
        return self._retrieve_by_embedding(
            embedding=embedding,
            limit=limit,
            content_type="chart",
            **kwargs
        )
    
    def _retrieve_diagrams(self, query: str, limit: int = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve diagram images relevant to the query.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            **kwargs: Additional parameters for the query
            
        Returns:
            List of diagram image results
        """
        # Enhance the query to be more diagram-specific
        diagram_query = f"diagram showing {query}"
        
        # Generate embedding for the diagram-specific query
        embedding = self._embed_text(diagram_query)
        
        # Retrieve with content type filter
        return self._retrieve_by_embedding(
            embedding=embedding,
            limit=limit,
            content_type="diagram",
            **kwargs
        )
    
    def _retrieve_photos(self, query: str, limit: int = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve photo images relevant to the query.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            **kwargs: Additional parameters for the query
            
        Returns:
            List of photo image results
        """
        # Generate embedding for the query
        embedding = self._embed_text(query)
        
        # Retrieve with content type filter
        return self._retrieve_by_embedding(
            embedding=embedding,
            limit=limit,
            content_type="photo",
            **kwargs
        )
    
    def _retrieve_general(self, query: str, limit: int = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve general images relevant to the query.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            **kwargs: Additional parameters for the query
            
        Returns:
            List of general image results
        """
        # Generate embedding for the query
        embedding = self._embed_text(query)
        
        # Retrieve without content type filter
        return self._retrieve_by_embedding(
            embedding=embedding,
            limit=limit,
            **kwargs
        )
    
    def retrieve(self, 
               query: str = None, 
               image: Union[str, bytes, PIL.Image.Image, Dict[str, Any]] = None,
               content_type: str = None,
               limit: int = None,
               hybrid_weight: float = 0.5,
               **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve images relevant to a text query or similar to an input image.
        
        This method supports:
        1. Text-to-image retrieval (with query)
        2. Image-to-image retrieval (with image)
        3. Hybrid retrieval (with both query and image)
        
        Args:
            query: Text query to search for (optional if image provided)
            image: Input image to find similar images (optional if query provided)
            content_type: Specific content type to search for
            limit: Maximum number of results to return
            hybrid_weight: Weight for hybrid search (0.0-1.0, higher values favor image similarity)
            **kwargs: Additional parameters
            
        Returns:
            List of image results
        """
        limit = limit or self.max_results
        
        # Validate inputs
        if not query and not image:
            logger.error("Either query or image must be provided")
            return []
        
        results = []
        
        # Process image if provided
        processed_image = None
        image_embedding = None
        if image:
            if isinstance(image, dict) and "image" in image:
                # Already processed
                processed_image = image
            else:
                # Process the image
                processed_image = self._process_image(image)
            
            if processed_image and self.image_embedder:
                # Generate embedding for the image
                image_embedding = self._embed_image(processed_image)
        
        # For text-to-image retrieval (when only query is provided)
        if query and not image_embedding:
            # Determine if we should use a specialized content type handler
            if content_type and content_type in self.content_type_handlers:
                handler = self.content_type_handlers[content_type]
                results = handler(query, limit, **kwargs)
            elif self.support_content_types:
                # Try to infer content type from query
                chart_results = self._retrieve_charts(query, limit // 2, **kwargs)
                diagram_results = self._retrieve_diagrams(query, limit // 2, **kwargs)
                general_results = self._retrieve_general(query, limit, **kwargs)
                
                # Combine results
                all_results = chart_results + diagram_results + general_results
                
                # Sort by score and de-duplicate
                seen_ids = set()
                sorted_results = []
                for result in sorted(all_results, key=lambda x: x["score"], reverse=True):
                    if result["id"] not in seen_ids:
                        sorted_results.append(result)
                        seen_ids.add(result["id"])
                        if len(sorted_results) >= limit:
                            break
                
                results = sorted_results
            else:
                # Just use general retrieval
                results = self._retrieve_general(query, limit, **kwargs)
        
        # For image-to-image retrieval (when only image is provided)
        elif image_embedding and not query:
            results = self._retrieve_by_embedding(
                embedding=image_embedding,
                limit=limit,
                content_type=content_type,
                **kwargs
            )
        
        # For hybrid retrieval (when both query and image are provided)
        elif query and image_embedding and self.enable_hybrid_search:
            # Get text-based results
            text_results = self._retrieve_general(query, limit * 2, **kwargs)
            
            # Get image-based results
            image_results = self._retrieve_by_embedding(
                embedding=image_embedding,
                limit=limit * 2,
                content_type=content_type,
                **kwargs
            )
            
            # Combine results with hybrid scoring
            combined_results = {}
            
            # Process text results
            for result in text_results:
                result_id = result["id"]
                combined_results[result_id] = {
                    **result,
                    "text_score": result["score"],
                    "image_score": 0.0,
                    "hybrid_score": result["score"] * (1.0 - hybrid_weight)
                }
            
            # Process image results and merge with text results
            for result in image_results:
                result_id = result["id"]
                if result_id in combined_results:
                    # Update existing entry
                    combined_results[result_id]["image_score"] = result["score"]
                    combined_results[result_id]["hybrid_score"] += result["score"] * hybrid_weight
                else:
                    # Add new entry
                    combined_results[result_id] = {
                        **result,
                        "text_score": 0.0,
                        "image_score": result["score"],
                        "hybrid_score": result["score"] * hybrid_weight
                    }
            
            # Convert to list and sort by hybrid score
            results = list(combined_results.values())
            results.sort(key=lambda x: x["score"], reverse=True)
            
            # Update the main score to be the hybrid score
            for result in results:
                result["score"] = result["hybrid_score"]
            
            # Limit results
            results = results[:limit]
        
        logger.info(f"Retrieved {len(results)} image results")
        return results
    
    def add_image(self, 
                image_data: Union[str, bytes, PIL.Image.Image, Dict[str, Any]],
                metadata: Optional[Dict[str, Any]] = None,
                content_type: Optional[str] = None,
                generate_embedding: bool = True) -> Optional[str]:
        """
        Add an image to the vector database.
        
        Args:
            image_data: Image to add (file path, URL, bytes, PIL Image, or processed data)
            metadata: Additional metadata for the image
            content_type: Content type of the image
            generate_embedding: Whether to generate an embedding for the image
            
        Returns:
            ID of the added image or None if failed
        """
        if not self.vector_db_client:
            logger.warning("Vector database client not configured")
            return None
        
        try:
            # Process image if needed
            if isinstance(image_data, dict) and "image" in image_data:
                # Already processed
                processed_image = image_data
            else:
                # Process the image
                processed_image = self._process_image(image_data)
            
            if not processed_image:
                logger.error("Failed to process image")
                return None
            
            # Determine content type if not provided
            if not content_type:
                content_type = self._get_content_type(processed_image)
            
            # Prepare metadata
            image_metadata = metadata or {}
            image_metadata["content_type"] = content_type
            image_metadata["width"] = processed_image.get("width")
            image_metadata["height"] = processed_image.get("height")
            image_metadata["format"] = processed_image.get("format")
            image_metadata["source"] = processed_image.get("source")
            
            # Generate embedding if requested
            embedding = None
            if generate_embedding and self.image_embedder:
                embedding = self._embed_image(processed_image)
            
            if not embedding:
                logger.warning("No embedding generated for image")
                return None
            
            # Store in vector database
            collection = self.vector_db_client.get_collection(self.image_collection)
            
            # Create document (using base64 representation)
            document = processed_image.get("base64", "")
            
            # Add to collection
            ids = collection.add(
                embeddings=[embedding],
                metadatas=[image_metadata],
                documents=[document],
                ids=[f"img_{int(time.time())}_{os.urandom(4).hex()}"]
            )
            
            image_id = ids[0]
            logger.info(f"Added image to vector database with ID: {image_id}")
            return image_id
            
        except Exception as e:
            logger.error(f"Error adding image to vector database: {str(e)}")
            return None 