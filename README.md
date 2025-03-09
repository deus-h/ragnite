# 🔮 RAGNITE ⚡
### _Igniting AI-Powered Retrieval & Augmentation_  
> _"As above, so below; as within, so without. The microcosm reflects the macrocosm."_ — Hermetic axiom

RAGNITE is a powerful platform that harnesses the transformative potential of Retrieval-Augmented Generation (RAG) technologies. By fusing advanced retrieval methodologies with generative AI, RAGNITE empowers users to create applications that deliver extraordinary results across virtually any domain or industry.

![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)
![MIT License](https://img.shields.io/badge/License-MIT-green)
![Status: Active Development](https://img.shields.io/badge/Status-Active%20Development-brightgreen)

## 🌠 Overview

Retrieval-Augmented Generation represents the alchemical fusion of retrieval-based knowledge with generative power. RAGNITE transforms this technology from academic theory into practical magic—providing a comprehensive suite of tools, frameworks, and pre-built solutions that enable organizations to rapidly deploy RAG systems tailored to their specific needs.

Whether you're building knowledge management systems, customer support solutions, content creation tools, or specialized domain applications, RAGNITE provides the building blocks, patterns, and alchemical formulas to manifest your vision.

## 🔮 Key Features

- **Complete RAG Toolkit**: Production-ready components for every stage of the RAG pipeline
- **Domain-Optimized Solutions**: Pre-configured implementations for code, medical, legal, and scientific domains
- **Advanced Techniques**: Cutting-edge approaches like Multi-Query, Hypothetical Document Embeddings, and Self-RAG
- **Extensible Architecture**: Easily customize and extend any component to meet your specific requirements
- **Production Readiness**: Monitoring, evaluation, and optimization tools for reliable deployment
- **Developer Experience**: Comprehensive documentation and examples to accelerate development

## 🧙‍♂️ Real-World Applications

RAGNITE can be applied to transform numerous domains:

- **Enterprise Knowledge Management**: Create systems that make your organization's collective knowledge accessible and actionable
- **Customer Experience**: Build support systems that provide accurate, contextual responses to customer inquiries
- **Content Creation**: Develop tools that assist in writing, research, and content generation with factual accuracy
- **Code Development**: Implement coding assistants that leverage your codebase and best practices
- **Healthcare**: Design systems for medical information retrieval, clinical decision support, and patient education
- **Legal**: Create tools for legal research, contract analysis, and compliance verification
- **Research & Academia**: Build research assistants and knowledge synthesis tools

## 🔱 Repository Structure

```
ragnite/
├── basic-rag/              # 🏗️ Foundational RAG implementation
├── advanced-rag/           # 🔥 Advanced RAG techniques
│   ├── multi-query/        # 🔍 Multi-Query RAG
│   ├── hypothetical-doc/   # 🧠 Hypothetical Document Embeddings
│   └── self-rag/           # 🪞 Self-RAG with reflection
├── domain-specific/        # 🧬 Domain-specific RAG implementations
│   ├── code-rag/           # 💻 RAG for code and programming
│   ├── medical-rag/        # ⚕️ RAG for medical and healthcare
│   ├── legal-rag/          # ⚖️ RAG for legal research
│   └── scientific-rag/     # 🔬 RAG for scientific research
├── evaluation/             # 📊 Evaluation frameworks and metrics
├── docker/                 # 🐳 Docker-based testing infrastructure
│   ├── docker-compose.yml  # Main configuration for test databases
│   ├── config/             # Database configurations
│   └── scripts/            # Utility scripts for testing
└── tools/                  # 🧰 Utility tools and helpers
    ├── src/                # Source code for utility tools
    │   ├── data_processing/# Document loaders, chunkers, cleaners
    │   ├── embeddings/     # Embedding generators and analyzers
    │   ├── vector_db/      # Vector database connectors and utilities
    │   ├── retrieval/      # Retrieval utilities and optimizers
    │   ├── generation/     # Generation utilities and templates
    │   └── monitoring/     # Monitoring and evaluation tools
    ├── tests/              # Test suite for utility tools
    └── examples/           # Example scripts demonstrating tool usage
```

## ⚔️ RAG Implementations

### 🔯 Basic RAG

A foundational implementation that establishes the sacred geometry of Retrieval-Augmented Generation, demonstrating the core workflow from document ingestion to knowledge retrieval and response generation.

### 🔥 Advanced RAG Techniques

#### ⚡ Multi-Query RAG

Enhances retrieval by generating multiple query variations from a single user query—akin to exploring multiple paths in the labyrinth of knowledge—improving the coverage and relevance of retrieved documents.

#### 🧿 Hypothetical Document Embeddings (HyDE)

Generates synthetic documents based on the query before retrieval, creating a hypothetical answer that helps bridge the gap between queries and relevant documents—a form of sympathetic magic in the digital realm.

#### 🪞 Self-RAG

Implements a self-reflective RAG system that can critique its own outputs, verify information, and improve response quality through iterative refinement—embodying the Hermetic principle of mental transmutation.

### 🧬 Domain-Specific RAG

#### 💻 Code RAG

Specialized RAG for software development tasks:
- Code-aware chunking that preserves function and class boundaries
- Language-specific processing for Python, JavaScript, and other languages
- Code-optimized embeddings and retrieval strategies
- Support for code completion, bug fixing, and documentation generation

#### ⚕️ Medical RAG

Specialized RAG for healthcare applications:
- Medical document chunking that preserves clinical context
- Medical entity recognition and relationship extraction
- Medical fact verification against authoritative sources
- Ethical safeguards for healthcare information
- Support for medical literature search, clinical decision support, and patient education

#### ⚖️ Legal RAG

Specialized RAG for legal research and document analysis:
- Legal text chunking based on document structure (statutes, cases, contracts)
- Citation extraction, parsing, and standardization
- Legal authority verification and assessment
- Jurisdiction-aware retrieval and filtering
- Support for legal research, case analysis, and contract review

#### 🔬 Scientific RAG

Specialized RAG for scientific research and analysis:
- Scientific paper chunking that respects document structure
- Mathematical formula handling with LaTeX processing
- Citation-aware retrieval and evidence-based responses
- Support for literature review, research design, and data analysis

## 🧰 Utility Tools

RAGNITE's `tools` directory contains production-ready utilities for building, debugging, and deploying RAG systems—the alchemical instruments for transmuting raw data into refined knowledge and practical applications:

### 📚 Data Processing Tools (Completed)

- **Document Loaders**: Load documents from various file formats (PDF, HTML, Markdown, text, JSON, etc.)
- **Text Chunkers**: Split documents into chunks using different strategies (fixed size, recursive, semantic, etc.)
- **Metadata Extractors**: Extract metadata from documents for filtering and context
- **Data Cleaners**: Clean and normalize text for better embedding quality
- **Data Augmentation**: Generate variations of text for improved retrieval

### 🧠 Embedding Tools (Completed)

- **Embedding Generators**: Generate embeddings using various models (Sentence Transformers, Hugging Face, OpenAI, TensorFlow)
- **Embedding Visualizers**: Visualize embeddings in 2D/3D space using Matplotlib and Plotly
- **Embedding Analyzers**: Analyze embeddings for similarity, clustering, and outliers
- **Model Adapters**: Convert embeddings between different models
- **Dimensionality Reduction**: Reduce embedding dimensions for visualization and efficiency (PCA, SVD, t-SNE, UMAP)

### 🗃️ Vector Database Tools (Completed)

- **Database Connectors**: Connect to various vector databases (ChromaDB, PostgreSQL/pgvector, Qdrant)
- **Index Optimizers**: Optimize vector indices for better performance using HNSW, IVF, and other algorithms
- **Query Benchmarkers**: Benchmark query performance across different configurations
- **Data Migration**: Migrate data between vector databases with schema preservation
- **Schema Managers**: Manage vector database schemas with validation and compatibility checks

### 🔍 Retrieval Tools (In Progress)

- **Query Processors**: Process and expand queries for better retrieval
- **Retrieval Debuggers**: Debug retrieval results and identify issues
- **Filter Builders**: Build filters for metadata-based filtering
- **Hybrid Searchers**: Combine vector search with keyword search
- **Re-rankers**: Re-rank retrieval results for better relevance

### 🔮 Generation Tools

- **Prompt Templates**: Create structured prompts for language models
  - `BasicPromptTemplate`: Simple variable substitution in templates
  - `FewShotPromptTemplate`: Example-based prompting with few-shot learning
  - `ChainOfThoughtPromptTemplate`: Step-by-step reasoning prompts
  - `StructuredPromptTemplate`: Generate structured outputs (JSON, XML, etc.)
- **Context Formatters**: Format retrieved context for generation
  - `BasicContextFormatter`: Simple document content formatting
  - `MetadataEnrichedFormatter`: Include metadata with document content
  - `SourceAttributionFormatter`: Add source citations and references
  - `HierarchicalContextFormatter`: Organize content in hierarchical structure
- **Output Parsers**: Parse and validate generated outputs
  - `JSONOutputParser`: Extract structured JSON from generated text
  - `XMLOutputParser`: Extract XML elements from generated text
  - `MarkdownOutputParser`: Extract structured components from Markdown
  - `StructuredOutputParser`: Extract custom-structured data with validation
- **Hallucination Detectors**: Tools to detect potential false information
  - `FactualConsistencyDetector`: Checks consistency with known facts
  - `SourceVerificationDetector`: Verifies content against source documents
  - `ContradictionDetector`: Identifies internal contradictions
  - `UncertaintyDetector`: Detects uncertain or speculative statements
- **Citation Generators**: Tools to create properly formatted citations
  - `AcademicCitationGenerator`: Creates citations for academic sources (APA, MLA, etc.)
  - `LegalCitationGenerator`: Creates citations for legal sources (Bluebook, ALWD)
  - `WebCitationGenerator`: Creates citations for web resources
  - `CustomCitationGenerator`: Creates custom citations using templates

### 📊 Monitoring Tools

- **Performance Trackers**: Tools to monitor and analyze system performance
  - `LatencyTracker`: Measure and analyze operation latency
  - `ThroughputTracker`: Track system throughput and processing rates
  - `MemoryUsageTracker`: Monitor memory consumption for processes and system
  - `CPUUsageTracker`: Track CPU utilization and identify bottlenecks

- **Usage Analyzers**: Tools to track and analyze user interactions
  - `QueryAnalyzer`: Analyze query patterns and trends
  - `UserSessionAnalyzer`: Track user sessions and engagement
  - `FeatureUsageAnalyzer`: Monitor feature usage and popularity
  - `ErrorAnalyzer`: Analyze error patterns and impact

- **Error Loggers**: Tools to record, store, and notify about errors
  - `ConsoleErrorLogger`: Log errors to the console with color-coding
  - `FileErrorLogger`: Log errors to files with rotation support
  - `DatabaseErrorLogger`: Log errors to databases for structured storage
  - `CloudErrorLogger`: Log errors to cloud services like AWS, GCP, Azure
  - `AlertErrorLogger`: Send alerts on errors via email, Slack, webhooks

## 🧠 Architecture & Integration

RAGNITE is designed for seamless integration into your existing infrastructure:

- **Modular Components**: Use only the components you need
- **API-First Design**: All functionality accessible via clean, documented APIs
- **Framework Agnostic**: Integrate with FastAPI, Flask, Django, or any web framework
- **Container Ready**: Docker support for easy deployment and scaling
- **Cloud Compatible**: Deploy on AWS, Azure, GCP, or your own infrastructure

## 🚀 Deployment Options

RAGNITE supports multiple deployment patterns:

- **Standalone Service**: Deploy as an independent microservice
- **Embedded Library**: Integrate directly into your application
- **Serverless Functions**: Deploy as cloud functions for scalable, event-driven architecture
- **Edge Deployment**: Run lightweight versions at the edge for reduced latency

## 🔥 Current Status

RAGNITE has evolved from a research project to a platform ready for real-world implementation:

- Production-ready implementations of multiple RAG systems:
  - Basic RAG with robust chunking and vector search
  - Advanced techniques: Multi-Query RAG, HyDE, Self-RAG
  - Domain-specific: Code RAG, Medical RAG, Legal RAG, Scientific RAG
- Comprehensive evaluation framework for quality assessment
- Enterprise-grade utility tools:
  - ✅ **Data Processing Tools**: Production-ready implementation
  - ✅ **Embedding Tools**: Production-ready implementation
  - ✅ **Vector Database Tools**: Production-ready implementation
  - 🔄 **Retrieval Tools**: Initial structure set up, implementation in progress
  - 📝 **Generation Tools**: Initial structure set up, implementation planned
  - 📝 **Monitoring Tools**: Initial structure set up, implementation planned

## 🧙‍♂️ Getting Started

Each implementation includes its own README with specific instructions. To get started:

1. Clone this repository
2. Navigate to the desired implementation directory
3. Install the required dependencies
4. Follow the implementation-specific instructions

### ⚡ Using the Utility Tools

The utility tools can be installed as a package:

```bash
# Install the core package
pnpm install ./tools

# Install with specific components
pnpm install "./tools[data,embeddings,visualization]"

# Install with vector database support
pnpm install "./tools[vector_db]"

# Install all components
pnpm install "./tools[all]"
```

### 🔮 Example Usage

```python
# Data Processing Example
from rag_tools.data_processing import get_document_loader, get_chunker

# Load documents
loader = get_document_loader(loader_type="pdf")
documents = loader.load("path/to/document.pdf")

# Chunk documents
chunker = get_chunker(strategy="recursive", chunk_size=1000)
chunks = chunker.split_documents(documents)

# Embedding Example
from rag_tools.embeddings import get_embedding_generator, get_embedding_visualizer

# Generate embeddings
generator = get_embedding_generator(generator_type="sentence-transformers")
embeddings = generator.batch_generate([chunk.content for chunk in chunks])

# Visualize embeddings
visualizer = get_embedding_visualizer(visualizer_type="matplotlib")
fig = visualizer.visualize(embeddings)
visualizer.save("embeddings.png")

# Vector Database Example
from rag_tools.vector_db import get_database_connector

# Connect to ChromaDB
connector = get_database_connector("chromadb")
connector.connect()

# Create a collection
collection = connector.create_collection("my_collection", dimension=384)

# Add embeddings to the collection
connector.add_vectors(
    "my_collection", 
    embeddings, 
    ids=[f"doc_{i}" for i in range(len(embeddings))]
)

# Search for similar vectors
results = connector.search("my_collection", query_vector=embeddings[0], top_k=5)
```

Check the `tools/examples` directory for more detailed examples of each component.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. As in the ancient Masonic tradition, we build our Temple stone by stone, with each contribution strengthening the whole.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

![⚡ RAGNITE](https://img.shields.io/badge/⚡-RAGNITE-blueviolet)

_"Knowledge is power. Understanding is transmutation. Application is transcendence."_

**Crafted with precision and passion by  
Amadeus Samiel H.** \m/  
[deus.h@outlook.com](mailto:deus.h@outlook.com) 