# RAGNITE Hybrid Dependency Management

This document explains the hybrid Conda/Micromamba + Poetry approach used in RAGNITE for managing dependencies.

## Why Hybrid?

RAGNITE uses a hybrid approach to dependency management for several reasons:

1. **Complex System Dependencies**: AI and RAG systems require complex non-Python dependencies (CUDA, C++ libraries) that Poetry alone can't handle
2. **Precise Python Dependency Resolution**: Poetry provides deterministic dependency resolution via lockfiles
3. **Performance**: Tools like UV can accelerate package installation

## Setup

### 1. Environment Management with Conda/Micromamba

We use Conda or Micromamba to manage the system-level environment and complex dependencies:

```bash
# Create environment from environment.yml
conda env create -f environment.yml
conda activate ragnite

# OR with Micromamba (much faster)
micromamba create -n ragnite -f environment.yml
micromamba activate ragnite
```

### 2. Python Dependency Management with Poetry

Within the Conda/Micromamba environment, we use Poetry for precise Python dependency management:

```bash
# Install Poetry
pip install poetry

# Install dependencies from pyproject.toml
poetry install

# Add a new dependency
poetry add some-package
```

### 3. Fast Installation with UV (Optional)

For faster installation, we can use UV as a drop-in replacement for pip:

```bash
# Install UV
pip install uv

# Use UV to install from Poetry's lock file
uv pip install --requirement <(poetry export)
```

## Best Practices

1. **Use Conda/Micromamba for**:
   - Creating and managing isolated environments
   - Installing GPU-related packages (PyTorch, TensorFlow)
   - Managing system dependencies (CUDA, database drivers)

2. **Use Poetry for**:
   - Managing Python package dependencies
   - Generating lockfiles for deterministic builds
   - Version management

3. **Use UV for**:
   - Accelerating package installation
   - Fast dependency resolution

## Integration with RAGNITE Caching

The caching system has been designed to work seamlessly with this hybrid approach:

- **Cache Manager**: Works with any Python environment
- **Embedding Cache**: Compatible with embedding models installed via Conda or Poetry
- **Result Cache**: System-agnostic caching of query results

## Recommended Workflow

1. Create and activate the Conda/Micromamba environment
2. Install dependencies with Poetry
3. Use the provided Makefile commands for common tasks
4. Keep dependencies updated with `poetry update` 