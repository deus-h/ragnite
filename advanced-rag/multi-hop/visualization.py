"""
Multi-Hop RAG Visualization Module

This module provides tools for visualizing multi-hop reasoning processes
and knowledge graphs generated during multi-hop retrieval.
"""

import logging
from typing import Dict, Any, List, Optional
import json
import uuid
import base64
import networkx as nx
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Configure logging
logger = logging.getLogger(__name__)

class MultiHopVisualizer:
    """
    Visualizer for Multi-Hop RAG processes and knowledge graphs.
    
    This class provides methods for visualizing the multi-hop reasoning
    process and knowledge graphs in different formats for better
    understanding complex RAG queries.
    """
    
    def __init__(self, color_scheme: Optional[Dict[str, str]] = None):
        """
        Initialize the Multi-Hop Visualizer.
        
        Args:
            color_scheme: Custom color scheme for different node types
        """
        # Default color scheme
        self.default_colors = {
            "query": "#4285F4",  # Google Blue
            "sub_question": "#EA4335",  # Google Red
            "follow_up": "#FBBC05",  # Google Yellow
            "context": "#34A853",  # Google Green
            "answer": "#9C27B0",  # Purple
            "unknown": "#9E9E9E"   # Gray
        }
        
        # Use custom color scheme if provided
        self.colors = color_scheme if color_scheme else self.default_colors
    
    def generate_html_visualization(self, data: Dict[str, Any]) -> str:
        """
        Generate an HTML visualization of the knowledge graph.
        
        Args:
            data: Visualization data from MultiHopRAG
            
        Returns:
            HTML string with interactive visualization
        """
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        
        # Generate unique ID for the visualization
        vis_id = f"multihop_vis_{uuid.uuid4().hex[:8]}"
        
        # Create styles based on node types
        node_styles = ""
        for node_type, color in self.colors.items():
            node_styles += f"""
            .{node_type}-node {{
                background-color: {color};
                border: 2px solid {color};
                color: white;
                text-align: center;
                padding: 5px;
                border-radius: 5px;
                width: fit-content;
                max-width: 250px;
                font-weight: bold;
            }}
            """
        
        # Create HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Multi-Hop RAG Visualization</title>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.23.0/cytoscape.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                }}
                #cy-container {{
                    display: flex;
                    flex-direction: row;
                }}
                #{vis_id} {{
                    width: 100%;
                    height: 800px;
                    border: 1px solid #ccc;
                }}
                .controls {{
                    margin-bottom: 20px;
                }}
                .legend {{
                    display: flex;
                    flex-wrap: wrap;
                    margin-bottom: 20px;
                }}
                .legend-item {{
                    display: flex;
                    align-items: center;
                    margin-right: 20px;
                    margin-bottom: 10px;
                }}
                .legend-color {{
                    width: 20px;
                    height: 20px;
                    margin-right: 5px;
                    border-radius: 3px;
                }}
                .node-details {{
                    position: fixed;
                    right: 20px;
                    top: 20px;
                    width: 300px;
                    padding: 15px;
                    background-color: white;
                    border: 1px solid #ccc;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    display: none;
                    max-height: 80vh;
                    overflow-y: auto;
                }}
                {node_styles}
            </style>
        </head>
        <body>
            <h1>Multi-Hop RAG Knowledge Graph</h1>
            
            <div class="controls">
                <button id="zoomFit">Zoom to Fit</button>
                <button id="toggleLabels">Toggle Labels</button>
                <button id="toggleLayout">Change Layout</button>
            </div>
            
            <div class="legend">
        """
        
        # Add legend items
        for node_type, color in self.colors.items():
            html += f"""
                <div class="legend-item">
                    <div class="legend-color" style="background-color: {color};"></div>
                    <div>{node_type.replace('_', ' ').title()}</div>
                </div>
            """
        
        html += """
            </div>
            
            <div id="cy-container">
                <div id="{vis_id}"></div>
                <div class="node-details" id="node-details">
                    <h3 id="details-title">Node Details</h3>
                    <div id="details-content"></div>
                </div>
            </div>
            
            <script>
        """
        
        # Add JavaScript for Cytoscape visualization
        html += f"""
                // Initialize Cytoscape
                var cy = cytoscape({{
                    container: document.getElementById('{vis_id}'),
                    style: [
                        {{
                            selector: 'node',
                            style: {{
                                'label': 'data(label)',
                                'color': '#fff',
                                'text-valign': 'center',
                                'text-halign': 'center',
                                'background-color': 'data(color)',
                                'text-wrap': 'wrap',
                                'text-max-width': '100px',
                                'font-size': '12px',
                                'width': 'label',
                                'height': 'label',
                                'padding': '10px',
                                'shape': 'round-rectangle'
                            }}
                        }},
                        {{
                            selector: 'edge',
                            style: {{
                                'width': 2,
                                'line-color': '#ccc',
                                'target-arrow-color': '#ccc',
                                'target-arrow-shape': 'triangle',
                                'curve-style': 'bezier',
                                'label': 'data(type)',
                                'font-size': '10px',
                                'text-rotation': 'autorotate',
                                'text-background-color': 'white',
                                'text-background-opacity': 1,
                                'text-background-padding': '2px'
                            }}
                        }}
                    ],
                    layout: {{
                        name: 'dagre',
                        rankDir: 'LR',
                        padding: 50,
                        animate: true
                    }},
                    wheelSensitivity: 0.2
                }});
                
                // Add nodes
                var nodes = {json.dumps(nodes)};
                nodes.forEach(function(node) {{
                    var type = node.type || 'unknown';
                    var color = {json.dumps(self.colors)};
                    cy.add({{
                        group: 'nodes',
                        data: {{
                            id: node.id,
                            label: node.label,
                            type: type,
                            color: color[type] || color['unknown'],
                            fullData: node
                        }}
                    }});
                }});
                
                // Add edges
                var edges = {json.dumps(edges)};
                edges.forEach(function(edge, i) {{
                    cy.add({{
                        group: 'edges',
                        data: {{
                            id: 'e' + i,
                            source: edge.source,
                            target: edge.target,
                            type: edge.type || 'connects'
                        }}
                    }});
                }});
                
                // Apply layout
                var layout = cy.layout({{ name: 'dagre', rankDir: 'LR', padding: 50 }});
                layout.run();
                
                // Add event handlers
                document.getElementById('zoomFit').addEventListener('click', function() {{
                    cy.fit();
                }});
                
                document.getElementById('toggleLabels').addEventListener('click', function() {{
                    var currentLabels = cy.style().selector('node').style('label');
                    if (currentLabels) {{
                        cy.style().selector('node').style('label', '').update();
                    }} else {{
                        cy.style().selector('node').style('label', 'data(label)').update();
                    }}
                }});
                
                var layouts = ['dagre', 'breadthfirst', 'circle', 'concentric', 'grid'];
                var currentLayout = 0;
                document.getElementById('toggleLayout').addEventListener('click', function() {{
                    currentLayout = (currentLayout + 1) % layouts.length;
                    var options = {{ name: layouts[currentLayout], animate: true }};
                    
                    if (layouts[currentLayout] === 'dagre') {{
                        options.rankDir = 'LR';
                        options.padding = 50;
                    }}
                    
                    cy.layout(options).run();
                }});
                
                // Node click handler for details
                cy.on('tap', 'node', function(evt) {{
                    var node = evt.target;
                    var data = node.data('fullData');
                    var detailsDiv = document.getElementById('node-details');
                    var detailsTitle = document.getElementById('details-title');
                    var detailsContent = document.getElementById('details-content');
                    
                    detailsTitle.innerText = data.type.replace('_', ' ').toUpperCase() + ': ' + node.id();
                    
                    var content = '<dl>';
                    for (var key in data) {{
                        if (key !== 'id' && key !== 'type' && key !== 'fullData') {{
                            content += '<dt><strong>' + key + ':</strong></dt>';
                            content += '<dd>' + data[key] + '</dd>';
                        }}
                    }}
                    content += '</dl>';
                    
                    detailsContent.innerHTML = content;
                    detailsDiv.style.display = 'block';
                }});
                
                cy.on('tap', function(evt) {{
                    if (evt.target === cy) {{
                        document.getElementById('node-details').style.display = 'none';
                    }}
                }});
                
                // Initial fit
                cy.fit();
            </script>
        </body>
        </html>
        """
        
        return html
    
    def generate_mermaid_diagram(self, data: Dict[str, Any]) -> str:
        """
        Generate a Mermaid.js diagram of the knowledge graph.
        
        Args:
            data: Visualization data from MultiHopRAG
            
        Returns:
            Mermaid.js diagram code
        """
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        
        # Start with flowchart definition
        mermaid = "flowchart LR\n"
        
        # Define nodes with styles based on type
        for node in nodes:
            node_id = node.get("id", "").replace(" ", "_").replace("-", "_")
            node_label = node.get("label", "")[:30]  # Truncate long labels
            node_type = node.get("type", "unknown")
            
            # Use different styles based on node type
            if node_type == "query":
                mermaid += f'    {node_id}["{node_label}"]:::query\n'
            elif node_type == "sub_question":
                mermaid += f'    {node_id}["{node_label}"]:::subQuestion\n'
            elif node_type == "follow_up":
                mermaid += f'    {node_id}["{node_label}"]:::followUp\n'
            elif node_type == "context":
                mermaid += f'    {node_id}["{node_label}"]:::context\n'
            elif node_type == "answer":
                mermaid += f'    {node_id}["{node_label}"]:::answer\n'
            else:
                mermaid += f'    {node_id}["{node_label}"]\n'
        
        # Define edges
        for edge in edges:
            source = edge.get("source", "").replace(" ", "_").replace("-", "_")
            target = edge.get("target", "").replace(" ", "_").replace("-", "_")
            edge_type = edge.get("type", "")
            
            # Add edge with label
            mermaid += f'    {source} -->|{edge_type}| {target}\n'
        
        # Define class styles
        mermaid += """
    classDef query fill:#4285F4,stroke:#4285F4,color:white
    classDef subQuestion fill:#EA4335,stroke:#EA4335,color:white
    classDef followUp fill:#FBBC05,stroke:#FBBC05,color:white
    classDef context fill:#34A853,stroke:#34A853,color:white
    classDef answer fill:#9C27B0,stroke:#9C27B0,color:white
        """
        
        return mermaid
    
    def generate_matplotlib_visualization(self, data: Dict[str, Any]) -> BytesIO:
        """
        Generate a static image visualization using matplotlib.
        
        Args:
            data: Visualization data from MultiHopRAG
            
        Returns:
            BytesIO containing the PNG image
        """
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes
        node_colors = []
        node_labels = {}
        
        for node in data.get("nodes", []):
            node_id = node.get("id", "")
            G.add_node(node_id)
            
            # Add node label
            node_labels[node_id] = node.get("label", "")[:20]  # Truncate long labels
            
            # Determine node color
            node_type = node.get("type", "unknown")
            hex_color = self.colors.get(node_type, self.colors["unknown"])
            
            # Convert hex to RGB and append
            rgb_color = mcolors.hex2color(hex_color)
            node_colors.append(rgb_color)
        
        # Add edges
        for edge in data.get("edges", []):
            source = edge.get("source", "")
            target = edge.get("target", "")
            edge_type = edge.get("type", "")
            
            G.add_edge(source, target, type=edge_type)
        
        # Create a position layout
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        except ImportError:
            # Fall back to networkx layout if graphviz is not available
            pos = nx.spring_layout(G, seed=42)
        
        # Create a figure
        plt.figure(figsize=(16, 12))
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_size=1500, node_color=node_colors, alpha=0.9)
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_color='white')
        
        # Draw edges with labels
        nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=15, alpha=0.7)
        edge_labels = {(u, v): d['type'] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        # Add a legend
        legend_elements = []
        for node_type, color in self.colors.items():
            from matplotlib.patches import Patch
            legend_elements.append(Patch(facecolor=color, label=node_type.replace('_', ' ').title()))
        
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.axis('off')
        plt.tight_layout()
        
        # Save the figure to BytesIO
        image_stream = BytesIO()
        plt.savefig(image_stream, format='png', dpi=150)
        plt.close()
        
        # Reset stream position
        image_stream.seek(0)
        
        return image_stream
    
    def get_base64_image(self, data: Dict[str, Any]) -> str:
        """
        Generate a base64-encoded PNG image for embedding in HTML/Markdown.
        
        Args:
            data: Visualization data from MultiHopRAG
            
        Returns:
            Base64-encoded PNG image
        """
        image_stream = self.generate_matplotlib_visualization(data)
        encoded = base64.b64encode(image_stream.read()).decode('utf-8')
        return f"data:image/png;base64,{encoded}"
    
    def save_visualization(self, 
                          data: Dict[str, Any], 
                          output_path: str, 
                          format: str = "html") -> str:
        """
        Save visualization to a file.
        
        Args:
            data: Visualization data from MultiHopRAG
            output_path: Path to save the visualization
            format: Format to save as ("html", "mermaid", "png")
            
        Returns:
            Path to the saved file
        """
        try:
            if format == "html":
                # Generate HTML visualization
                html = self.generate_html_visualization(data)
                
                # Save to file
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(html)
                
            elif format == "mermaid":
                # Generate Mermaid diagram
                mermaid = self.generate_mermaid_diagram(data)
                
                # Save to file
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(mermaid)
                
            elif format == "png":
                # Generate matplotlib visualization
                image_stream = self.generate_matplotlib_visualization(data)
                
                # Save to file
                with open(output_path, 'wb') as f:
                    f.write(image_stream.read())
            
            logger.info(f"Visualization saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving visualization: {str(e)}")
            return "" 