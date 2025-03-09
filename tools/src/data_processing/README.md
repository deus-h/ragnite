# Data Processing Tools

This module provides a comprehensive suite of tools for processing, cleaning, and augmenting text data for RAG (Retrieval-Augmented Generation) systems.

## Overview

The data processing module includes the following components:

1. **Document Loaders**: Tools for loading documents from various file formats.
2. **Text Chunkers**: Tools for splitting documents into smaller pieces for embedding and retrieval.
3. **Metadata Extractors**: Tools for extracting metadata from documents.
4. **Data Cleaners**: Tools for cleaning and normalizing text data.
5. **Data Augmentation**: Tools for augmenting text data to improve diversity and robustness.

## Installation

To use all features of the data processing tools, install the required dependencies:

```bash
pip install nltk spacy beautifulsoup4 python-docx PyPDF2 langdetect transformers sentencepiece
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt stopwords wordnet
```

## Usage Examples

### Document Loaders

```python
from rag_research.tools.data_processing.document_loaders import TextLoader, PDFLoader, DirectoryLoader

# Load a text file
text_loader = TextLoader()
documents = text_loader.load("path/to/document.txt")

# Load a PDF file
pdf_loader = PDFLoader()
documents = pdf_loader.load("path/to/document.pdf")

# Load all documents in a directory
dir_loader = DirectoryLoader()
documents = dir_loader.load("path/to/directory")
```

### Text Chunkers

```python
from rag_research.tools.data_processing.text_chunkers import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    SentenceTextSplitter,
    SemanticTextSplitter,
    get_chunker
)

# Using a specific chunker
chunker = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " "]
)
chunks = chunker.split_text("Your long text content here...")

# Using the factory function
chunker = get_chunker(
    strategy="recursive",
    chunk_size=1000,
    chunk_overlap=200
)
chunks = chunker.split_text("Your long text content here...")

# Split documents
documents = [...] # Your list of Document objects
chunked_documents = chunker.split_documents(documents)
```

### Metadata Extractors

```python
from rag_research.tools.data_processing.metadata_extractors import (
    BasicMetadataExtractor,
    TitleExtractor,
    AuthorExtractor,
    DateExtractor,
    EntityExtractor,
    KeywordExtractor,
    CompositeMetadataExtractor,
    create_comprehensive_extractor
)

# Using a specific extractor
title_extractor = TitleExtractor()
metadata = title_extractor.extract(document)

# Using the composite extractor
comprehensive_extractor = create_comprehensive_extractor()
documents_with_metadata = comprehensive_extractor.extract_from_documents(documents)

# Extract entities
entity_extractor = EntityExtractor(entity_types=["PERSON", "ORG", "GPE"])
metadata = entity_extractor.extract(document)
```

### Data Cleaners

```python
from rag_research.tools.data_processing.data_cleaners import (
    WhitespaceNormalizer,
    HTMLCleaner,
    UnicodeNormalizer,
    TextNormalizer,
    RegexCleaner,
    StopwordRemover,
    DuplicateRemover,
    NoiseRemover,
    CompositeCleaner,
    create_standard_cleaner
)

# Using a specific cleaner
html_cleaner = HTMLCleaner()
cleaned_text = html_cleaner.clean(html_content)

# Using the standard cleaner
standard_cleaner = create_standard_cleaner()
cleaned_document = standard_cleaner.clean_document(document)

# Clean multiple documents
cleaned_documents = standard_cleaner.clean_documents(documents)

# Create a custom composite cleaner
custom_cleaner = CompositeCleaner([
    HTMLCleaner(),
    UnicodeNormalizer(),
    WhitespaceNormalizer(),
    NoiseRemover(remove_urls=True, remove_emails=True)
])
cleaned_text = custom_cleaner.clean(text)
```

### Data Augmentation

```python
from rag_research.tools.data_processing.data_augmentation import (
    WordReplacementAugmenter,
    RandomInsertionAugmenter,
    RandomDeletionAugmenter,
    RandomSwapAugmenter,
    CompositeAugmenter,
    create_standard_augmentation_pipeline
)

# Using a specific augmenter
synonym_augmenter = WordReplacementAugmenter(
    replace_fraction=0.2,
    num_variations=3
)
augmented_texts = synonym_augmenter.augment("Your text content here...")

# Augment a document
augmented_documents = synonym_augmenter.augment_document(document, max_variations=3)

# Using the standard augmentation pipeline
standard_augmenter = create_standard_augmentation_pipeline(num_variations=5)
augmented_texts = standard_augmenter.augment("Your text content here...")
```

## Complete Pipeline Example

Here's an example of a complete data processing pipeline:

```python
from rag_research.tools.data_processing.document_loaders import DirectoryLoader
from rag_research.tools.data_processing.metadata_extractors import create_comprehensive_extractor
from rag_research.tools.data_processing.data_cleaners import create_standard_cleaner
from rag_research.tools.data_processing.text_chunkers import get_chunker
from rag_research.tools.data_processing.data_augmentation import create_standard_augmentation_pipeline

# 1. Load documents
loader = DirectoryLoader()
documents = loader.load("path/to/docs")

# 2. Extract metadata
metadata_extractor = create_comprehensive_extractor()
documents_with_metadata = metadata_extractor.extract_from_documents(documents)

# 3. Clean documents
cleaner = create_standard_cleaner()
cleaned_documents = cleaner.clean_documents(documents_with_metadata)

# 4. Chunk documents
chunker = get_chunker(strategy="recursive", chunk_size=1000, chunk_overlap=200)
chunked_documents = chunker.split_documents(cleaned_documents)

# 5. Augment documents (optional)
augmenter = create_standard_augmentation_pipeline(num_variations=2)
augmented_documents = augmenter.augment_documents(
    chunked_documents,
    max_variations_per_doc=2,
    max_total_variations=100
)

# Final processed documents ready for embedding
processed_documents = chunked_documents + augmented_documents
```

## Customization

Each component is designed to be customizable through various parameters. See the docstrings and method signatures for detailed information on the available options.

## Dependencies

Different features have different dependencies:

- Basic functionality: No external dependencies
- Text chunking: NLTK for tokenization, spaCy for semantic chunking
- Metadata extraction: NLTK, spaCy, langdetect, BeautifulSoup4, PyPDF2
- Data cleaning: NLTK for stopword removal and lemmatization
- Data augmentation: NLTK for synonym replacement, spaCy for contextual word embeddings, transformers for back-translation

## Error Handling

Most components will gracefully degrade if optional dependencies are not available. Check the logs for warnings about missing dependencies and install them as needed. 