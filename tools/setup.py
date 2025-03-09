#!/usr/bin/env python3
"""
Setup script for RAG Utility Tools.
"""

from setuptools import setup, find_packages

# Read the long description from README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read version from src/__init__.py
with open("src/__init__.py", "r", encoding="utf-8") as fh:
    for line in fh:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"\'')
            break

setup(
    name="rag-tools",
    version=version,
    author="RAG Research Team",
    author_email="info@rag-research.org",
    description="Utility tools for building, debugging, and optimizing RAG systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rag-research/rag-tools",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "data": [
            "pypdf>=3.0.0",
            "beautifulsoup4>=4.9.0",
            "markdown>=3.3.0",
            "python-docx>=0.8.10",
        ],
        "embeddings": [
            "sentence-transformers>=2.2.0",
            "transformers>=4.20.0",
            "torch>=1.10.0",
        ],
        "visualization": [
            "matplotlib>=3.5.0",
            "plotly>=5.5.0",
        ],
        "analysis": [
            "scikit-learn>=1.0.0",
            "umap-learn>=0.5.0",
        ],
        "vector_db": [
            "chromadb>=0.4.0",
            "qdrant-client>=1.0.0",
            "pinecone-client>=2.0.0",
        ],
        "generation": [
            "openai>=1.0.0",
            "langchain>=0.0.200",
        ],
        "monitoring": [
            "prometheus-client>=0.14.0",
            "psutil>=5.9.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "mypy>=0.950",
            "flake8>=4.0.0",
        ],
        "all": [
            "pypdf>=3.0.0",
            "beautifulsoup4>=4.9.0",
            "markdown>=3.3.0",
            "python-docx>=0.8.10",
            "sentence-transformers>=2.2.0",
            "transformers>=4.20.0",
            "torch>=1.10.0",
            "matplotlib>=3.5.0",
            "plotly>=5.5.0",
            "scikit-learn>=1.0.0",
            "umap-learn>=0.5.0",
            "chromadb>=0.4.0",
            "qdrant-client>=1.0.0",
            "pinecone-client>=2.0.0",
            "openai>=1.0.0",
            "langchain>=0.0.200",
            "prometheus-client>=0.14.0",
            "psutil>=5.9.0",
        ],
    },
) 