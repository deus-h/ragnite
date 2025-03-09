"""
Streaming RAG Client Library

This module provides client libraries for the streaming RAG system,
supporting various output formats and user interfaces including
CLI, web, and API integration.
"""

import logging
import json
import sys
import time
import asyncio
import threading
from typing import List, Dict, Any, Optional, Union, Callable, AsyncGenerator, Generator, TextIO, Tuple

# Configure logging
logger = logging.getLogger(__name__)


class StreamingRAGClient:
    """
    Client for the Streaming RAG system.
    
    This class provides methods for interacting with the streaming RAG system,
    supporting various output formats and user interfaces.
    """
    
    def __init__(
        self,
        streaming_rag=None,
        output_format: str = "text",  # "text", "json", "markdown", "html"
        show_thinking: bool = False,
        show_citations: bool = True,
        show_retrieval_info: bool = False,
        color_output: bool = True,
        output_stream: TextIO = sys.stdout,
    ):
        """
        Initialize the Streaming RAG client.
        
        Args:
            streaming_rag: StreamingRAG instance
            output_format: Format for output
            show_thinking: Whether to show thinking process
            show_citations: Whether to show citations
            show_retrieval_info: Whether to show retrieval information
            color_output: Whether to colorize output (for CLI)
            output_stream: Stream to write output to
        """
        self.streaming_rag = streaming_rag
        self.output_format = output_format
        self.show_thinking = show_thinking
        self.show_citations = show_citations
        self.show_retrieval_info = show_retrieval_info
        self.color_output = color_output
        self.output_stream = output_stream
        
        # Set up color codes if enabled
        if self.color_output:
            self.colors = {
                "reset": "\033[0m",
                "bold": "\033[1m",
                "thinking": "\033[36m",  # Cyan
                "answer": "\033[32m",    # Green
                "citation": "\033[33m",  # Yellow
                "error": "\033[31m",     # Red
                "info": "\033[94m",      # Bright Blue
                "prompt": "\033[35m",    # Magenta
            }
        else:
            self.colors = {k: "" for k in ["reset", "bold", "thinking", "answer", "citation", "error", "info", "prompt"]}
    
    def set_streaming_rag(self, streaming_rag):
        """
        Set or update the StreamingRAG instance.
        
        Args:
            streaming_rag: StreamingRAG instance
        """
        self.streaming_rag = streaming_rag
        logger.info("StreamingRAG instance updated")
    
    def _colorize(self, text: str, color: str) -> str:
        """
        Add color codes to text.
        
        Args:
            text: Text to colorize
            color: Color name
            
        Returns:
            Colorized text
        """
        if not self.color_output:
            return text
        
        color_code = self.colors.get(color, self.colors["reset"])
        return f"{color_code}{text}{self.colors['reset']}"
    
    def _format_chunk_text(self, chunk: Dict[str, Any], is_thinking: bool = False) -> str:
        """
        Format a chunk as text.
        
        Args:
            chunk: Chunk to format
            is_thinking: Whether this is a thinking chunk
            
        Returns:
            Formatted text
        """
        content = chunk.get("content", "")
        
        if not content:
            return ""
        
        if is_thinking:
            return self._colorize(content, "thinking")
        else:
            return self._colorize(content, "answer")
    
    def _format_chunk_json(self, chunk: Dict[str, Any]) -> str:
        """
        Format a chunk as JSON.
        
        Args:
            chunk: Chunk to format
            
        Returns:
            Formatted JSON string
        """
        # Create a clean version of the chunk for JSON output
        clean_chunk = {
            "content": chunk.get("content", ""),
            "is_complete": chunk.get("is_complete", False),
            "is_thinking": chunk.get("content", "").strip().startswith("Thinking:"),
            "timestamp": time.time()
        }
        
        # Add retrieval info if requested
        if self.show_retrieval_info and "retrieval_info" in chunk:
            clean_chunk["retrieval_info"] = chunk["retrieval_info"]
        
        # Add progressive retrieval info if available
        if "progressive_retrieval" in chunk:
            clean_chunk["progressive_retrieval"] = {
                "query": chunk["progressive_retrieval"]["query"]
            }
        
        return json.dumps(clean_chunk)
    
    def _format_chunk_markdown(self, chunk: Dict[str, Any]) -> str:
        """
        Format a chunk as Markdown.
        
        Args:
            chunk: Chunk to format
            
        Returns:
            Formatted Markdown string
        """
        content = chunk.get("content", "")
        
        if not content:
            return ""
        
        # Check if this is a thinking chunk
        is_thinking = content.strip().startswith("Thinking:")
        
        if is_thinking:
            # Format thinking as blockquote
            return f"> {content}"
        else:
            return content
    
    def _format_chunk_html(self, chunk: Dict[str, Any]) -> str:
        """
        Format a chunk as HTML.
        
        Args:
            chunk: Chunk to format
            
        Returns:
            Formatted HTML string
        """
        content = chunk.get("content", "")
        
        if not content:
            return ""
        
        # Check if this is a thinking chunk
        is_thinking = content.strip().startswith("Thinking:")
        
        if is_thinking:
            # Format thinking as a different color
            return f'<span class="thinking">{content}</span>'
        else:
            return f'<span class="answer">{content}</span>'
    
    def _format_retrieval_info(self, retrieval_info: Dict[str, Any]) -> str:
        """
        Format retrieval information based on output format.
        
        Args:
            retrieval_info: Retrieval information
            
        Returns:
            Formatted retrieval info
        """
        if not retrieval_info:
            return ""
        
        # Extract relevant information
        progressive_retrievals = retrieval_info.get("progressive_retrievals", [])
        new_passages_count = retrieval_info.get("new_passages_count", 0)
        total_tokens = retrieval_info.get("total_tokens", 0)
        
        if self.output_format == "json":
            # For JSON, we just return the data, it will be serialized later
            return retrieval_info
        
        elif self.output_format == "markdown":
            # Format as Markdown
            output = "\n\n### Retrieval Information\n\n"
            output += f"* **Progressive retrievals:** {len(progressive_retrievals)}\n"
            output += f"* **New passages:** {new_passages_count}\n"
            output += f"* **Total tokens:** {total_tokens}\n\n"
            
            if progressive_retrievals:
                output += "#### Progressive Retrieval Queries\n\n"
                for i, query in enumerate(progressive_retrievals):
                    output += f"{i+1}. {query}\n"
            
            return output
            
        elif self.output_format == "html":
            # Format as HTML
            output = '<div class="retrieval-info">\n'
            output += '<h3>Retrieval Information</h3>\n'
            output += '<ul>\n'
            output += f'<li><strong>Progressive retrievals:</strong> {len(progressive_retrievals)}</li>\n'
            output += f'<li><strong>New passages:</strong> {new_passages_count}</li>\n'
            output += f'<li><strong>Total tokens:</strong> {total_tokens}</li>\n'
            output += '</ul>\n'
            
            if progressive_retrievals:
                output += '<h4>Progressive Retrieval Queries</h4>\n'
                output += '<ol>\n'
                for query in progressive_retrievals:
                    output += f'<li>{query}</li>\n'
                output += '</ol>\n'
            
            output += '</div>\n'
            return output
            
        else:  # text format
            # Format as plain text
            output = "\n\nRetrieval Information:\n"
            output += f"- Progressive retrievals: {len(progressive_retrievals)}\n"
            output += f"- New passages: {new_passages_count}\n"
            output += f"- Total tokens: {total_tokens}\n"
            
            if progressive_retrievals:
                output += "\nProgressive Retrieval Queries:\n"
                for i, query in enumerate(progressive_retrievals):
                    output += f"{i+1}. {query}\n"
            
            return self._colorize(output, "info")
    
    def _write_to_output(self, text: str, flush: bool = True):
        """
        Write text to the output stream.
        
        Args:
            text: Text to write
            flush: Whether to flush the output stream
        """
        if not text:
            return
        
        self.output_stream.write(text)
        if flush:
            self.output_stream.flush()
    
    def _process_response_chunk(self, chunk: Dict[str, Any]) -> str:
        """
        Process a response chunk based on output format.
        
        Args:
            chunk: Response chunk
            
        Returns:
            Formatted output string
        """
        # Check if this is a thinking chunk
        content = chunk.get("content", "")
        is_thinking = content.strip().startswith("Thinking:")
        
        # Skip thinking chunks if not showing thinking
        if is_thinking and not self.show_thinking:
            return ""
        
        # Format chunk based on output format
        if self.output_format == "json":
            return self._format_chunk_json(chunk) + "\n"
        elif self.output_format == "markdown":
            return self._format_chunk_markdown(chunk)
        elif self.output_format == "html":
            return self._format_chunk_html(chunk)
        else:  # text format
            return self._format_chunk_text(chunk, is_thinking)
    
    def _process_final_response(self, 
                              is_stream: bool, 
                              response: Union[Dict[str, Any], str],
                              full_response: Optional[str] = None) -> str:
        """
        Process the final response.
        
        Args:
            is_stream: Whether this was a streaming response
            response: Final response or chunk
            full_response: Full response for streaming (optional)
            
        Returns:
            Formatted final response
        """
        # For streaming, response is the final chunk
        if is_stream:
            # Add retrieval info if requested
            if self.show_retrieval_info and "retrieval_info" in response:
                retrieval_info = self._format_retrieval_info(response["retrieval_info"])
                
                if self.output_format == "json":
                    # For JSON, we just return a JSON object with both the response and retrieval info
                    return json.dumps({
                        "response": full_response,
                        "retrieval_info": retrieval_info
                    }) + "\n"
                else:
                    return retrieval_info
            
            return ""  # No final processing needed
        
        # For non-streaming, response is the full response object
        else:
            answer = response.get("answer", "")
            
            # Format based on output format
            if self.output_format == "json":
                # Create clean response object
                clean_response = {
                    "answer": answer,
                    "query": response.get("query", ""),
                    "timestamp": time.time()
                }
                
                # Add retrieval info if requested
                if self.show_retrieval_info:
                    clean_response["retrieval_info"] = response.get("retrieval_info", {})
                
                return json.dumps(clean_response, indent=2) + "\n"
                
            elif self.output_format == "markdown":
                output = f"## Answer\n\n{answer}\n\n"
                
                # Add retrieval info if requested
                if self.show_retrieval_info:
                    retrieval_info = self._format_retrieval_info(response.get("retrieval_info", {}))
                    output += retrieval_info
                
                return output
                
            elif self.output_format == "html":
                output = f'<div class="rag-response">\n<div class="answer">{answer}</div>\n'
                
                # Add retrieval info if requested
                if self.show_retrieval_info:
                    retrieval_info = self._format_retrieval_info(response.get("retrieval_info", {}))
                    output += retrieval_info
                
                output += '</div>\n'
                return output
                
            else:  # text format
                output = self._colorize(answer, "answer")
                
                # Add retrieval info if requested
                if self.show_retrieval_info:
                    retrieval_info = self._format_retrieval_info(response.get("retrieval_info", {}))
                    output += retrieval_info
                
                return output
    
    async def stream_response(self, 
                            query: str,
                            context: Optional[List[Dict[str, Any]]] = None,
                            **kwargs) -> str:
        """
        Stream a response to a query, writing to the output stream.
        
        Args:
            query: The user's query
            context: Initial context (optional)
            **kwargs: Additional parameters
            
        Returns:
            Full response as a string
        """
        if not self.streaming_rag:
            error_msg = "StreamingRAG instance not set"
            logger.error(error_msg)
            self._write_to_output(self._colorize(f"Error: {error_msg}\n", "error"))
            return ""
        
        # Log the query
        logger.info(f"Streaming response for query: {query}")
        
        # Print the query if this is a CLI-style interaction
        if self.output_format == "text":
            self._write_to_output(self._colorize(f"\nQuery: {query}\n\n", "prompt"))
        
        full_response = ""
        final_chunk = None
        
        try:
            # Stream the response
            async for chunk in self.streaming_rag.generate_stream(
                query=query,
                context=context,
                enable_thought_streaming=self.show_thinking,
                **kwargs
            ):
                # Process the chunk
                output = self._process_response_chunk(chunk)
                self._write_to_output(output)
                
                # Update full response
                content = chunk.get("content", "")
                if content and not (not self.show_thinking and content.strip().startswith("Thinking:")):
                    full_response += content
                
                # Save final chunk for additional processing
                if chunk.get("is_complete", False):
                    final_chunk = chunk
        
        except Exception as e:
            logger.error(f"Error streaming response: {str(e)}")
            self._write_to_output(self._colorize(f"\nError: {str(e)}\n", "error"))
            return full_response
        
        # Process final information if available
        if final_chunk:
            final_output = self._process_final_response(True, final_chunk, full_response)
            if final_output:
                self._write_to_output(final_output)
        
        return full_response
    
    async def get_response(self, 
                         query: str,
                         context: Optional[List[Dict[str, Any]]] = None,
                         stream_output: bool = True,
                         **kwargs) -> Dict[str, Any]:
        """
        Get a response to a query, optionally streaming to the output stream.
        
        Args:
            query: The user's query
            context: Initial context (optional)
            stream_output: Whether to stream output
            **kwargs: Additional parameters
            
        Returns:
            Response dictionary
        """
        if not self.streaming_rag:
            error_msg = "StreamingRAG instance not set"
            logger.error(error_msg)
            if stream_output:
                self._write_to_output(self._colorize(f"Error: {error_msg}\n", "error"))
            return {"error": error_msg}
        
        # Log the query
        logger.info(f"Getting response for query: {query}")
        
        if stream_output:
            # Use streaming method
            full_response = await self.stream_response(query, context, **kwargs)
            
            # Return a response-like dictionary
            return {
                "query": query,
                "answer": full_response
            }
        
        else:
            # Print the query if this is a CLI-style interaction and we're not streaming
            if self.output_format == "text":
                self._write_to_output(self._colorize(f"\nQuery: {query}\n\n", "prompt"))
            
            try:
                # Get the full response
                response = await self.streaming_rag.generate(
                    query=query,
                    context=context,
                    enable_thought_streaming=self.show_thinking,
                    **kwargs
                )
                
                # Process and write the response
                output = self._process_final_response(False, response)
                self._write_to_output(output)
                
                return response
                
            except Exception as e:
                logger.error(f"Error getting response: {str(e)}")
                if stream_output:
                    self._write_to_output(self._colorize(f"\nError: {str(e)}\n", "error"))
                return {"error": str(e)}
    
    # Synchronous versions of the async methods
    
    def stream_response_sync(self, 
                           query: str,
                           context: Optional[List[Dict[str, Any]]] = None,
                           **kwargs) -> str:
        """
        Synchronous version of stream_response.
        
        Args:
            query: The user's query
            context: Initial context (optional)
            **kwargs: Additional parameters
            
        Returns:
            Full response as a string
        """
        import asyncio
        
        # Create a new event loop if there isn't one
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.stream_response(query, context, **kwargs)
        )
    
    def get_response_sync(self, 
                        query: str,
                        context: Optional[List[Dict[str, Any]]] = None,
                        stream_output: bool = True,
                        **kwargs) -> Dict[str, Any]:
        """
        Synchronous version of get_response.
        
        Args:
            query: The user's query
            context: Initial context (optional)
            stream_output: Whether to stream output
            **kwargs: Additional parameters
            
        Returns:
            Response dictionary
        """
        import asyncio
        
        # Create a new event loop if there isn't one
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.get_response(query, context, stream_output, **kwargs)
        )
    
    def run_cli_session(self):
        """
        Run an interactive CLI session.
        """
        if not self.streaming_rag:
            self._write_to_output(self._colorize("Error: StreamingRAG instance not set\n", "error"))
            return
        
        # Set output format to text for CLI
        original_format = self.output_format
        self.output_format = "text"
        
        self._write_to_output(self._colorize("Streaming RAG CLI Session\n", "bold"))
        self._write_to_output(self._colorize("Type 'exit' or 'quit' to end the session\n\n", "info"))
        
        try:
            while True:
                # Get user input
                user_input = input(self._colorize("Query: ", "prompt"))
                
                # Check for exit command
                if user_input.lower() in ["exit", "quit", "q"]:
                    self._write_to_output(self._colorize("\nSession ended\n", "info"))
                    break
                
                # Skip empty input
                if not user_input.strip():
                    continue
                
                # Process the query
                self.stream_response_sync(user_input)
                
                # Add a newline after the response
                self._write_to_output("\n")
                
        except KeyboardInterrupt:
            self._write_to_output(self._colorize("\nSession interrupted\n", "info"))
        except Exception as e:
            self._write_to_output(self._colorize(f"\nError: {str(e)}\n", "error"))
        finally:
            # Restore original output format
            self.output_format = original_format
    
    def create_webapp(self, host: str = "localhost", port: int = 8000):
        """
        Create and run a simple web application for the streaming RAG system.
        
        Args:
            host: Host to run the web app on
            port: Port to run the web app on
        """
        try:
            from flask import Flask, request, jsonify, Response, stream_with_context
            
            app = Flask("StreamingRAG")
            
            @app.route("/api/query", methods=["POST"])
            def query_endpoint():
                # Get query from request
                data = request.json
                query = data.get("query", "")
                stream = data.get("stream", True)
                
                if not query:
                    return jsonify({"error": "No query provided"}), 400
                
                # For streaming responses
                if stream:
                    def generate():
                        # Create a copy of this client with JSON output format
                        client_copy = StreamingRAGClient(
                            streaming_rag=self.streaming_rag,
                            output_format="json",
                            show_thinking=data.get("show_thinking", self.show_thinking),
                            show_citations=data.get("show_citations", self.show_citations),
                            show_retrieval_info=data.get("show_retrieval_info", self.show_retrieval_info),
                            color_output=False
                        )
                        
                        # Get a synchronous generator
                        for chunk in self.streaming_rag.generate_stream_sync(
                            query=query,
                            enable_thought_streaming=client_copy.show_thinking,
                            **data.get("params", {})
                        ):
                            # Format as JSON
                            yield client_copy._format_chunk_json(chunk) + "\n"
                    
                    return Response(stream_with_context(generate()), mimetype="application/json")
                
                # For non-streaming responses
                else:
                    # Get response
                    response = self.streaming_rag.generate_sync(
                        query=query,
                        enable_thought_streaming=data.get("show_thinking", self.show_thinking),
                        **data.get("params", {})
                    )
                    
                    # Create a clean response object
                    clean_response = {
                        "answer": response.get("answer", ""),
                        "query": query
                    }
                    
                    # Add retrieval info if requested
                    if data.get("show_retrieval_info", self.show_retrieval_info):
                        clean_response["retrieval_info"] = response.get("retrieval_info", {})
                    
                    return jsonify(clean_response)
            
            # Add a simple HTML interface
            @app.route("/")
            def index():
                return """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Streaming RAG Interface</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 20px; }
                        #input { width: 100%; padding: 10px; margin-bottom: 10px; }
                        #output { width: 100%; height: 400px; padding: 10px; border: 1px solid #ccc; overflow-y: auto; white-space: pre-wrap; }
                        .thinking { color: #0077cc; }
                        .answer { color: #333; }
                        .controls { margin-bottom: 10px; }
                        button { padding: 8px 15px; }
                    </style>
                </head>
                <body>
                    <h1>Streaming RAG Interface</h1>
                    <div class="controls">
                        <label><input type="checkbox" id="showThinking"> Show thinking process</label>
                        <label><input type="checkbox" id="showRetrievalInfo"> Show retrieval info</label>
                    </div>
                    <textarea id="input" placeholder="Enter your query here..."></textarea>
                    <button id="submitBtn">Submit</button>
                    <button id="clearBtn">Clear</button>
                    <div id="output"></div>
                    
                    <script>
                        const input = document.getElementById('input');
                        const output = document.getElementById('output');
                        const submitBtn = document.getElementById('submitBtn');
                        const clearBtn = document.getElementById('clearBtn');
                        const showThinking = document.getElementById('showThinking');
                        const showRetrievalInfo = document.getElementById('showRetrievalInfo');
                        
                        clearBtn.addEventListener('click', () => {
                            output.innerHTML = '';
                        });
                        
                        submitBtn.addEventListener('click', async () => {
                            const query = input.value.trim();
                            if (!query) return;
                            
                            output.innerHTML += `<div><strong>Query:</strong> ${query}</div>`;
                            
                            try {
                                const response = await fetch('/api/query', {
                                    method: 'POST',
                                    headers: {
                                        'Content-Type': 'application/json'
                                    },
                                    body: JSON.stringify({
                                        query,
                                        stream: true,
                                        show_thinking: showThinking.checked,
                                        show_retrieval_info: showRetrievalInfo.checked
                                    })
                                });
                                
                                const reader = response.body.getReader();
                                const decoder = new TextDecoder();
                                
                                while (true) {
                                    const {value, done} = await reader.read();
                                    if (done) break;
                                    
                                    const text = decoder.decode(value);
                                    const lines = text.split('\\n');
                                    
                                    for (const line of lines) {
                                        if (!line.trim()) continue;
                                        
                                        try {
                                            const chunk = JSON.parse(line);
                                            const content = chunk.content || '';
                                            
                                            if (content) {
                                                const isThinking = chunk.is_thinking;
                                                const className = isThinking ? 'thinking' : 'answer';
                                                output.innerHTML += `<span class="${className}">${content}</span>`;
                                                output.scrollTop = output.scrollHeight;
                                            }
                                        } catch (e) {
                                            console.error('Error parsing chunk:', e);
                                        }
                                    }
                                }
                                
                                output.innerHTML += '<hr>';
                                
                            } catch (e) {
                                output.innerHTML += `<div style="color: red">Error: ${e.message}</div>`;
                            }
                        });
                        
                        // Submit on Enter (but allow multiline with Shift+Enter)
                        input.addEventListener('keydown', (e) => {
                            if (e.key === 'Enter' && !e.shiftKey) {
                                e.preventDefault();
                                submitBtn.click();
                            }
                        });
                    </script>
                </body>
                </html>
                """
            
            # Start the web app
            app.run(host=host, port=port)
            
        except ImportError:
            self._write_to_output(self._colorize("Error: Flask not installed. Install it with 'pip install flask'.\n", "error"))
            return 