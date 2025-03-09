# 🔮 RAGNITE ⚡
### _Igniting AI-Powered Retrieval & Augmentation_  
> _"As above, so below; as within, so without. The microcosm reflects the macrocosm."_  

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

## 🔥 Installation

RAGNITE uses a powerful hybrid approach combining Conda/Micromamba for environment management with Poetry for dependency resolution. This approach provides the perfect balance between system-level dependency management (crucial for ML/AI) and precise Python package versioning.

### Prerequisites

Before setting up RAGNITE, you'll need to install the following tools:

#### 1. Git

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install git
```

**Arch/Garuda:**
```bash
sudo pacman -Sy git
```

**macOS:**
```bash
# Using Homebrew
brew install git
# Or download from: https://git-scm.com/download/mac
```

**Windows:**
```bash
# Download and install from: https://git-scm.com/download/win
# Or using Chocolatey
choco install git
```

#### 2. Conda or Micromamba

**Option A: Micromamba (Recommended - Faster & Lighter)**

*Ubuntu/Debian:*
```bash
wget -qO- https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
mkdir -p ~/micromamba
~/bin/micromamba shell init -s bash -p ~/micromamba
# Restart your shell or source your .bashrc
source ~/.bashrc
```

*Arch/Garuda:*
```bash
yay -S micromamba-bin
# or with pacman if available in your repositories
sudo pacman -S micromamba
# Initialize micromamba shell integration
micromamba shell init --shell bash --root-prefix=~/.local/share/mamba
# Restart your shell or source your .bashrc
source ~/.bashrc
```

*macOS:*
```bash
# Using Homebrew
brew install micromamba
# Or manual installation
curl -Ls https://micro.mamba.pm/api/micromamba/osx-64/latest | tar -xvj bin/micromamba
mkdir -p ~/micromamba
~/bin/micromamba shell init -s zsh -p ~/micromamba
# Restart your shell or source your .zshrc
source ~/.zshrc
```

*Windows:*
```powershell
# Using Chocolatey
choco install micromamba

# Or manual installation (in PowerShell)
Invoke-WebRequest -Uri https://micro.mamba.pm/api/micromamba/win-64/latest -OutFile micromamba-installer.exe
.\micromamba-installer.exe
# Follow the installation prompts
# Initialize PowerShell shell integration
micromamba shell init --shell powershell
# Restart your PowerShell session
```

**IMPORTANT**: After installing Micromamba, you must initialize shell integration before you can activate environments. If you see an error like "micromamba is running as a subprocess and can't modify the parent shell", run:

```bash
# For one-time use in current shell:
eval "$(micromamba shell hook --shell bash)"  # Replace bash with your shell

# For permanent configuration (recommended):
micromamba shell init --shell bash --root-prefix=~/.local/share/mamba
source ~/.bashrc  # Or appropriate config file for your shell
```

**Option B: Conda (More Common)**

*All Platforms:*
1. Download the appropriate installer from [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. Install following the instructions for your platform:

*Ubuntu/Debian:*
```bash
bash Miniconda3-latest-Linux-x86_64.sh
# Follow the prompts to complete installation
```

*Arch/Garuda:*
```bash
bash Miniconda3-latest-Linux-x86_64.sh
# Follow the prompts to complete installation
```

*macOS:*
```bash
bash Miniconda3-latest-MacOSX-x86_64.sh
# Or for Apple Silicon:
bash Miniconda3-latest-MacOSX-arm64.sh
# Follow the prompts to complete installation
```

*Windows:*
- Run the downloaded installer (Miniconda3-latest-Windows-x86_64.exe)
- Follow the installation wizard

#### 3. Docker

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install docker.io docker-compose
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
# Log out and log back in for group changes to take effect
```

**Arch/Garuda:**
```bash
sudo pacman -S docker docker-compose
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
# Log out and log back in for group changes to take effect
```

**macOS:**
- Download and install [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop)

**Windows:**
- Download and install [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop)
- Ensure WSL 2 is installed and configured if prompted

#### 4. Make

**Ubuntu/Debian:**
```bash
sudo apt-get install make
```

**Arch/Garuda:**
```bash
sudo pacman -S make
```

**macOS:**
```bash
# Using Homebrew
brew install make
# It's also included with Xcode Command Line Tools:
xcode-select --install
```

**Windows:**
```bash
# Using Chocolatey
choco install make
# Or install via MSYS2 or MinGW
```

### Setup Instructions

### Option 1: Quick Setup (Conda + Poetry)

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/ragnite.git
cd ragnite

# 2. Set up the environment
make setup-env            # Create .env file from template
make setup-conda          # Create Conda environment
conda activate ragnite    # Activate the environment
make install-poetry       # Install Poetry
make install-deps         # Install dependencies with Poetry

# 3. Validate your setup
make validate

# 4. Start development services
make dev-env
```

### Option 2: Faster Setup with Micromamba

```bash
# 1. Set up with Micromamba (much faster than Conda)
make setup-env
make setup-micromamba

# Ensure your shell is initialized for micromamba
# If this is your first time using micromamba:
micromamba shell init --shell bash  # Replace with your shell
source ~/.bashrc  # Or appropriate config file for your shell

# Activate the environment
micromamba activate ragnite
make install-poetry
make install-deps

# 2. Start development
make dev-env
```

### Managing Dependencies

Add new dependencies with Poetry:

```bash
# Add a regular dependency
make add-dep pkg=langchain-community

# Add a development dependency
make add-dev-dep pkg=black
```

### Environment Configuration

1. Edit the `.env` file with your API keys and configuration
2. Ensure you have at least one LLM API key configured:
   - `OPENAI_API_KEY`
   - `ANTHROPIC_API_KEY`
   - `MISTRAL_API_KEY`
   - `XAI_API_KEY` (for Grok)
   - `GOOGLE_API_KEY` (for Gemini)

### Troubleshooting

#### Common Installation Issues

##### Conda/Micromamba Environment Creation Fails
- **Issue**: `conda env create` or `micromamba create` fails with package conflicts
- **Solution**: Try updating conda/micromamba first, then retry:
  ```bash
  conda update -n base conda  # For Conda
  micromamba self-update      # For Micromamba
  ```
  
##### Micromamba Shell Initialization
- **Issue**: Error message "micromamba is running as a subprocess and can't modify the parent shell"
- **Solution**: Initialize micromamba for your shell:
  ```bash
  # For one-time use in current shell:
  eval "$(micromamba shell hook --shell bash)"  # Replace bash with your shell
  
  # For permanent configuration (recommended):
  micromamba shell init --shell bash --root-prefix=~/.local/share/mamba
  source ~/.bashrc  # Or appropriate config file for your shell
  ```
  
##### Poetry Installation Issues
- **Issue**: Poetry installation fails or has dependency conflicts
- **Solution**: Try installing with the official installer:
  ```bash
  curl -sSL https://install.python-poetry.org | python3 -
  ```
  
##### Docker Permission Issues
- **Issue**: "Permission denied" when running Docker commands
- **Solution**: Make sure your user is in the docker group:
  ```bash
  sudo usermod -aG docker $USER
  # Then log out and log back in
  ```

##### CUDA/GPU Issues
- **Issue**: PyTorch can't find CUDA or GPU acceleration isn't working
- **Solution**: Verify your NVIDIA drivers are installed and check PyTorch with:
  ```bash
  # Inside your conda environment
  python -c "import torch; print(torch.cuda.is_available())"
  ```

##### Ollama GPU Acceleration
- **Issue**: Error message "could not select device driver 'nvidia' with capabilities: [[gpu]]"
- **Solution**: RAGNITE includes a multi-approach GPU configuration that works across different Docker versions:
  
  1. **Install the NVIDIA Container Toolkit**:
     ```bash
     # Ubuntu/Debian
     distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
     curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
     curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
     sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
     sudo systemctl restart docker
     
     # Arch/Garuda
     sudo pacman -S nvidia-container-toolkit
     sudo systemctl restart docker
     ```
  
  2. **Configure Docker Daemon**:
     ```bash
     # Create or update the Docker daemon configuration
     sudo mkdir -p /etc/docker
     echo '{"runtimes":{"nvidia":{"path":"nvidia-container-runtime","runtimeArgs":[]}}}' | sudo tee /etc/docker/daemon.json
     sudo systemctl restart docker
     ```
  
  3. **Enable NVIDIA runtime in your `.env` file**:
     ```bash
     # Open .env file and set DOCKER_RUNTIME to nvidia
     DOCKER_RUNTIME=nvidia
     ```
  
  4. **Handle Port Conflicts** (if you have Ollama running locally already):
     ```bash
     # Check if Ollama is already running on port 11434
     sudo lsof -i :11434
     
     # If it shows Ollama is running, update docker-compose.dev.yml
     # Change port mapping from "11434:11434" to "11435:11434"
     
     # Then update .env to use the new port
     OLLAMA_HOST=http://localhost:11435
     ```
  
  5. **Verify your NVIDIA setup**:
     ```bash
     # Check if your NVIDIA driver is working
     nvidia-smi
     
     # Verify Docker can see your NVIDIA GPU
     docker run --rm --runtime=nvidia nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
     # Or with the newer syntax
     docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
     ```
  
  6. **Run development environment with GPU support**:
     ```bash
     make dev-env
     ```
  
  7. **Verify GPU access in the container**:
     ```bash
     docker exec rag-ollama-dev nvidia-smi
     ```

  8. **Pull models for GPU acceleration**:
     ```bash
     # Connect to the Ollama container
     docker exec -it rag-ollama-dev bash
     
     # Pull models (inside container)
     ollama pull llama3
     ```
  
  The docker-compose.dev.yml file includes multiple GPU configuration approaches that work with different Docker versions. If you don't have an NVIDIA GPU or the setup fails, Ollama will fall back to CPU-only mode automatically.

#### Platform-Specific Issues

##### Windows
- **Issue**: Make commands don't work properly
- **Solution**: Use Git Bash, WSL, or install Make via Chocolatey/MSYS2

##### macOS
- **Issue**: Command Line Tools missing
- **Solution**: Install Xcode Command Line Tools:
  ```bash
  xcode-select --install
  ```

##### Docker Issues on Linux
- **Issue**: Docker service not running
- **Solution**: Start and enable the service:
  ```bash
  sudo systemctl start docker
  sudo systemctl enable docker
  ```

For more help, please [open an issue](https://github.com/yourusername/ragnite/issues) with details about your problem.

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

RAGNITE has evolved from a research project into a production-ready platform with significant advancements:

- **Core RAG Systems**: All core implementations complete and tested
  - ✅ Basic RAG with robust chunking and vector search
  - ✅ Advanced techniques: Multi-Query RAG, HyDE, Self-RAG
  - ✅ Domain-specific: Code RAG, Medical RAG, Legal RAG, Scientific RAG
- **Performance Optimization**: Caching infrastructure fully implemented
  - ✅ Embedding Cache for vector representations
  - ✅ Semantic Cache for similar queries
  - ✅ Result Cache with time-based invalidation
  - ✅ Prompt Cache for template reuse
- **Enterprise-grade Utility Tools**:
  - ✅ **Data Processing Tools**: Production-ready implementation
  - ✅ **Embedding Tools**: Production-ready implementation
  - ✅ **Vector Database Tools**: Production-ready implementation with connectors for all major vector databases
  - ✅ **Retrieval Tools**: Complete with query processors, retrieval debuggers, filter builders, hybrid searchers, and re-rankers
  - ✅ **Generation Tools**: Complete with prompt templates, context formatters, output parsers, hallucination detectors, and citation generators
  - ✅ **Monitoring Tools**: Complete with performance trackers, usage analyzers, error loggers, cost estimators, and latency monitors
- **Current Focus**:
  - 🔄 Comprehensive testing of all components (Phase 6)
  - 🔄 Final documentation and comparison analysis (Phase 7)
  - 🔄 Adding support for xAI (Grok) and Google AI (Gemini) models

## 🧙‍♂️ Getting Started

RAGNITE uses a powerful hybrid approach combining Conda/Micromamba for environment management with Poetry for dependency resolution. This approach provides the perfect balance between system-level dependency management (crucial for ML/AI) and precise Python package versioning.

### Quick Setup Guide

1. Clone this repository
   ```bash
   git clone https://github.com/yourusername/ragnite.git
   cd ragnite
   ```

2. Set up the environment (choose one option)
   ```bash
   # Option 1: With Conda
   make setup-env            # Create .env file from template
   make setup-conda          # Create Conda environment
   conda activate ragnite    # Activate the environment
   
   # Option 2: With Micromamba (faster)
   make setup-env
   make setup-micromamba
   micromamba activate ragnite
   ```

3. Install dependencies with Poetry
   ```bash
   make install-poetry
   make install-deps
   ```

4. Start development services
   ```bash
   make dev-env
   ```

5. Validate your setup
   ```bash
   make validate
   ```

### Implementation-Specific Instructions

Each RAG implementation includes its own README with specific instructions:

1. **Basic RAG**: `basic-rag/README.md`
2. **Advanced RAG**:
   - Multi-Query: `advanced-rag/multi-query/README.md`
   - HyDE: `advanced-rag/hypothetical-doc/README.md`
   - Self-RAG: `advanced-rag/self-rag/README.md`
   - Caching: `advanced-rag/caching/README.md`
3. **Domain-Specific**:
   - Code RAG: `domain-specific/code-rag/README.md`
   - Medical RAG: `domain-specific/medical-rag/README.md`
   - Legal RAG: `domain-specific/legal-rag/README.md`
   - Scientific RAG: `domain-specific/scientific-rag/README.md`

### Adding New Dependencies

```bash
# Add a regular dependency
make add-dep pkg=langchain-community

# Add a development dependency
make add-dev-dep pkg=black
```

### Using the Utility Tools

The utility tools can be used in your code:

```python
# Data Processing Example
from rag_tools.data_processing import get_document_loader, get_chunker

# Load documents
loader = get_document_loader(loader_type="pdf")
documents = loader.load("path/to/document.pdf")

# Chunk documents
chunker = get_chunker(strategy="recursive", chunk_size=1000)
chunks = chunker.split_documents(documents)

# Vector Database Example
from rag_tools.vector_db import get_database_connector

# Connect to ChromaDB
connector = get_database_connector("chromadb")
connector.connect()

# Create a collection and add embeddings
collection = connector.create_collection("my_collection", dimension=384)
connector.add_vectors(
    "my_collection", 
    embeddings, 
    ids=[f"doc_{i}" for i in range(len(embeddings))]
)

# Retrieval Example
from rag_tools.retrieval import get_hybrid_searcher, get_reranker

# Create a hybrid searcher
searcher = get_hybrid_searcher("vector_keyword")
results = searcher.search("How does RAG work?", top_k=10)

# Rerank results
reranker = get_reranker("cross_encoder")
reranked_results = reranker.rerank(results, query="How does RAG work?")

# For the new Google AI and xAI providers:

```python
# Using Google AI (Gemini) provider
from tools.src.models.base_model import Message, Role
from tools.src.models.model_factory import get_model_provider

# Create a Google AI provider
provider = get_model_provider("google", api_key="your_google_api_key")

# Or create an xAI provider (once available)
# provider = get_model_provider("xai", api_key="your_xai_api_key")

# Create a conversation
messages = [
    Message(role=Role.SYSTEM, content="You are a helpful AI assistant."),
    Message(role=Role.USER, content="What is RAG?"),
]

# Generate a response
response = provider.generate(messages, temperature=0.7)
print(response["content"])

# Use streaming for a more interactive experience
for chunk in provider.generate_stream(messages, temperature=0.7):
    print(chunk["content"], end="", flush=True)
```

Check the `tools/examples` directory for more detailed examples of each component.

## 🤝 Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Install the development environment
   ```bash
   make setup-conda        # Or setup-micromamba
   conda activate ragnite
   make install-poetry
   make install-deps
   ```
4. Make your changes
5. Run tests (`make test`)
6. Commit your changes (`git commit -m 'Add some amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

As in the ancient Masonic tradition, we build our Temple stone by stone, with each contribution strengthening the whole.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

![⚡ RAGNITE](https://img.shields.io/badge/⚡-RAGNITE-blueviolet)

_"Knowledge is power. Understanding is transmutation. Application is transcendence."_

**Crafted with precision and passion by  
Amadeus Samiel H.** \m/  
[deus.h@outlook.com](mailto:deus.h@outlook.com) 