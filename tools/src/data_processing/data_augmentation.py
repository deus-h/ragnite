"""
Data Augmentation

This module provides tools for augmenting text data to improve the diversity
and robustness of training datasets for RAG systems.
"""

import re
import random
import string
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple, Set, Callable
import logging
from collections import defaultdict

try:
    import nltk
    from nltk.corpus import wordnet
    NLTK_AVAILABLE = True
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
except ImportError:
    NLTK_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from transformers import MarianMTModel, MarianTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)

# Import Document class from document_loaders
from .document_loaders import Document


class BaseAugmenter(ABC):
    """
    Abstract base class for text augmenters.
    """
    
    @abstractmethod
    def augment(self, text: str) -> List[str]:
        """
        Augment text to create variations.
        
        Args:
            text: Text to augment
            
        Returns:
            List of augmented text variations
        """
        pass
    
    def augment_document(self, document: Document, max_variations: int = 3) -> List[Document]:
        """
        Augment a document to create variations.
        
        Args:
            document: Document to augment
            max_variations: Maximum number of variations to create
            
        Returns:
            List of augmented document variations
        """
        # Generate augmented text variations
        augmented_texts = self.augment(document.content)
        
        # Limit the number of variations
        if len(augmented_texts) > max_variations:
            augmented_texts = random.sample(augmented_texts, max_variations)
        
        # Create augmented documents
        augmented_docs = []
        
        for i, aug_text in enumerate(augmented_texts):
            # Create metadata for augmented document
            aug_metadata = document.metadata.copy()
            aug_metadata["augmented"] = True
            aug_metadata["augmenter"] = self.__class__.__name__
            aug_metadata["augmentation_index"] = i
            aug_metadata["original_id"] = aug_metadata.get("id", "")
            aug_metadata["id"] = f"{aug_metadata.get('id', '')}_{self.__class__.__name__}_{i}"
            
            # Create augmented document
            aug_doc = Document(content=aug_text, metadata=aug_metadata)
            augmented_docs.append(aug_doc)
        
        return augmented_docs
    
    def augment_documents(
        self, 
        documents: List[Document], 
        max_variations_per_doc: int = 3,
        max_total_variations: Optional[int] = None
    ) -> List[Document]:
        """
        Augment multiple documents.
        
        Args:
            documents: List of documents to augment
            max_variations_per_doc: Maximum number of variations per document
            max_total_variations: Maximum total number of augmented documents to return
            
        Returns:
            List of augmented documents
        """
        all_augmented_docs = []
        
        for doc in documents:
            augmented_docs = self.augment_document(doc, max_variations_per_doc)
            all_augmented_docs.extend(augmented_docs)
        
        # Limit the total number of variations if specified
        if max_total_variations is not None and len(all_augmented_docs) > max_total_variations:
            all_augmented_docs = random.sample(all_augmented_docs, max_total_variations)
        
        return all_augmented_docs


class WordReplacementAugmenter(BaseAugmenter):
    """
    Augment text by replacing words with their synonyms.
    """
    
    def __init__(
        self,
        replace_fraction: float = 0.2,
        num_variations: int = 3,
        preserve_stopwords: bool = True,
        min_word_length: int = 3,
        language: str = 'english'
    ):
        """
        Initialize a word replacement augmenter.
        
        Args:
            replace_fraction: Fraction of words to replace with synonyms
            num_variations: Number of variations to generate
            preserve_stopwords: Whether to avoid replacing stopwords
            min_word_length: Minimum length of words to consider for replacement
            language: Language for stopwords and synonym lookup
        """
        self.replace_fraction = min(1.0, max(0.0, replace_fraction))
        self.num_variations = num_variations
        self.preserve_stopwords = preserve_stopwords
        self.min_word_length = min_word_length
        self.language = language
        
        if not NLTK_AVAILABLE:
            raise ImportError(
                "NLTK is required for synonym replacement. "
                "Install it with `pip install nltk`."
            )
        
        # Load stopwords if needed
        if preserve_stopwords:
            try:
                from nltk.corpus import stopwords
                nltk.data.find('corpora/stopwords')
                self.stopwords = set(stopwords.words(language))
            except (ImportError, LookupError):
                nltk.download('stopwords')
                from nltk.corpus import stopwords
                self.stopwords = set(stopwords.words(language))
        else:
            self.stopwords = set()
    
    def augment(self, text: str) -> List[str]:
        """
        Augment text by replacing words with their synonyms.
        
        Args:
            text: Text to augment
            
        Returns:
            List of augmented text variations
        """
        if not text:
            return []
        
        augmented_texts = []
        
        # Tokenize text into words
        words = text.split()
        
        # Find candidate words for replacement (excluding stopwords and short words)
        candidate_indices = [
            i for i, word in enumerate(words)
            if (word.lower() not in self.stopwords) and (len(word) >= self.min_word_length)
        ]
        
        # Generate variations
        for _ in range(self.num_variations):
            if not candidate_indices:
                continue
                
            # Create a copy of the words list
            new_words = words.copy()
            
            # Calculate number of words to replace
            num_to_replace = max(1, int(len(candidate_indices) * self.replace_fraction))
            
            # Select random words to replace
            indices_to_replace = random.sample(candidate_indices, min(num_to_replace, len(candidate_indices)))
            
            # Replace words with synonyms
            for idx in indices_to_replace:
                word = words[idx]
                synonyms = self._get_synonyms(word)
                
                if synonyms:
                    new_words[idx] = random.choice(synonyms)
            
            # Join words to create augmented text
            augmented_text = ' '.join(new_words)
            
            # Add augmented text to the list if it's different from the original
            if augmented_text != text and augmented_text not in augmented_texts:
                augmented_texts.append(augmented_text)
        
        return augmented_texts
    
    def _get_synonyms(self, word: str) -> List[str]:
        """
        Get synonyms for a word using WordNet.
        
        Args:
            word: Word to find synonyms for
            
        Returns:
            List of synonyms
        """
        # Convert word to lowercase for lookup
        word_lower = word.lower()
        
        # Check if the word is capitalized
        is_capitalized = word[0].isupper() if word else False
        
        # Get synsets from WordNet
        synsets = wordnet.synsets(word_lower)
        
        # Extract synonyms from synsets
        synonyms = []
        for synset in synsets:
            for lemma in synset.lemmas():
                synonym = lemma.name().replace('_', ' ')
                
                # Skip if the synonym is the same as the original word
                if synonym.lower() == word_lower:
                    continue
                
                # Match capitalization of the original word
                if is_capitalized:
                    synonym = synonym[0].upper() + synonym[1:] if synonym else ''
                
                # Add synonym to the list
                synonyms.append(synonym)
        
        # Remove duplicates
        return list(set(synonyms))


class BackTranslationAugmenter(BaseAugmenter):
    """
    Augment text using back-translation through intermediate languages.
    
    This augmenter requires the transformers library with MarianMT models.
    """
    
    def __init__(
        self,
        intermediate_languages: List[str] = None,
        num_variations: int = 2,
        max_length: int = 512
    ):
        """
        Initialize a back-translation augmenter.
        
        Args:
            intermediate_languages: List of intermediate language codes (default: ['fr', 'de', 'es'])
            num_variations: Number of variations to generate
            max_length: Maximum sequence length for translation
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "The transformers library is required for back-translation. "
                "Install it with `pip install transformers sentencepiece`."
            )
        
        self.intermediate_languages = intermediate_languages or ['fr', 'de', 'es']
        self.num_variations = min(num_variations, len(self.intermediate_languages))
        self.max_length = max_length
        
        # Load translation models
        self.models = {}
        self.tokenizers = {}
        
        logger.info("Loading translation models. This may take some time...")
        
        for lang in self.intermediate_languages:
            # Load models for translating to intermediate language
            model_name = f"Helsinki-NLP/opus-mt-en-{lang}"
            tokenizer_name = f"Helsinki-NLP/opus-mt-en-{lang}"
            
            try:
                self.tokenizers[f"en-{lang}"] = MarianTokenizer.from_pretrained(tokenizer_name)
                self.models[f"en-{lang}"] = MarianMTModel.from_pretrained(model_name)
                
                # Load models for translating back to English
                model_name = f"Helsinki-NLP/opus-mt-{lang}-en"
                tokenizer_name = f"Helsinki-NLP/opus-mt-{lang}-en"
                
                self.tokenizers[f"{lang}-en"] = MarianTokenizer.from_pretrained(tokenizer_name)
                self.models[f"{lang}-en"] = MarianMTModel.from_pretrained(model_name)
                
                logger.info(f"Loaded translation models for language: {lang}")
            except Exception as e:
                logger.warning(f"Failed to load translation models for language {lang}: {e}")
                # Remove language from the list if model loading failed
                self.intermediate_languages.remove(lang)
        
        if not self.intermediate_languages:
            raise ValueError(
                "No translation models could be loaded. "
                "Check your internet connection and available disk space."
            )
        
        self.num_variations = min(self.num_variations, len(self.intermediate_languages))
    
    def augment(self, text: str) -> List[str]:
        """
        Augment text using back-translation.
        
        Args:
            text: Text to augment
            
        Returns:
            List of augmented text variations
        """
        if not text:
            return []
        
        augmented_texts = []
        
        # Select random intermediate languages
        selected_languages = random.sample(
            self.intermediate_languages, 
            min(self.num_variations, len(self.intermediate_languages))
        )
        
        for lang in selected_languages:
            try:
                # Translate to intermediate language
                intermediate_text = self._translate(text, f"en-{lang}")
                
                # Translate back to English
                back_translated_text = self._translate(intermediate_text, f"{lang}-en")
                
                # Add to list if different from the original
                if back_translated_text != text and back_translated_text not in augmented_texts:
                    augmented_texts.append(back_translated_text)
            except Exception as e:
                logger.warning(f"Translation failed for language {lang}: {e}")
        
        return augmented_texts
    
    def _translate(self, text: str, model_key: str) -> str:
        """
        Translate text using the specified model.
        
        Args:
            text: Text to translate
            model_key: Key for the translation model and tokenizer
            
        Returns:
            Translated text
        """
        # Ensure text is not too long by splitting into sentences
        sentences = nltk.sent_tokenize(text)
        
        translated_sentences = []
        
        for sentence in sentences:
            # Skip empty sentences
            if not sentence.strip():
                continue
                
            # Encode the text
            tokenizer = self.tokenizers[model_key]
            model = self.models[model_key]
            
            encoded = tokenizer.encode(sentence, return_tensors="pt", max_length=self.max_length, truncation=True)
            
            # Generate translation
            translated = model.generate(encoded, max_length=self.max_length)
            
            # Decode the translation
            translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
            
            translated_sentences.append(translated_text)
        
        # Join the translated sentences
        return " ".join(translated_sentences)


class RandomInsertionAugmenter(BaseAugmenter):
    """
    Augment text by randomly inserting words.
    """
    
    def __init__(
        self,
        insert_fraction: float = 0.1,
        num_variations: int = 3,
        insertion_candidates: Optional[List[str]] = None,
        use_synonyms: bool = True,
        language: str = 'english'
    ):
        """
        Initialize a random insertion augmenter.
        
        Args:
            insert_fraction: Fraction of text length to insert
            num_variations: Number of variations to generate
            insertion_candidates: List of words to insert (if None, use synonyms)
            use_synonyms: Whether to use synonyms of existing words for insertion
            language: Language for synonym lookup
        """
        self.insert_fraction = min(0.5, max(0.0, insert_fraction))  # Cap at 50%
        self.num_variations = num_variations
        self.insertion_candidates = insertion_candidates
        self.use_synonyms = use_synonyms
        self.language = language
        
        if use_synonyms and not NLTK_AVAILABLE:
            raise ImportError(
                "NLTK is required for synonym-based insertion. "
                "Install it with `pip install nltk`."
            )
    
    def augment(self, text: str) -> List[str]:
        """
        Augment text by randomly inserting words.
        
        Args:
            text: Text to augment
            
        Returns:
            List of augmented text variations
        """
        if not text:
            return []
        
        augmented_texts = []
        
        # Tokenize text into words
        words = text.split()
        
        # Generate variations
        for _ in range(self.num_variations):
            # Create a copy of the words list
            new_words = words.copy()
            
            # Calculate number of words to insert
            num_to_insert = max(1, int(len(words) * self.insert_fraction))
            
            # Get words to insert
            if self.insertion_candidates:
                # Use provided candidates
                words_to_insert = random.choices(self.insertion_candidates, k=num_to_insert)
            elif self.use_synonyms and words:
                # Use synonyms of random words from the text
                words_to_insert = []
                for _ in range(num_to_insert):
                    # Select a random word from the text
                    random_word = random.choice(words)
                    
                    # Get synonyms
                    synonyms = self._get_synonyms(random_word)
                    
                    if synonyms:
                        words_to_insert.append(random.choice(synonyms))
                    else:
                        # Fall back to the original word if no synonyms found
                        words_to_insert.append(random_word)
            else:
                # Skip if no insertion candidates are available
                continue
            
            # Insert words at random positions
            for word_to_insert in words_to_insert:
                # Choose a random position
                insert_pos = random.randint(0, len(new_words))
                
                # Insert the word
                new_words.insert(insert_pos, word_to_insert)
            
            # Join words to create augmented text
            augmented_text = ' '.join(new_words)
            
            # Add augmented text to the list if it's different from the original
            if augmented_text != text and augmented_text not in augmented_texts:
                augmented_texts.append(augmented_text)
        
        return augmented_texts
    
    def _get_synonyms(self, word: str) -> List[str]:
        """
        Get synonyms for a word using WordNet.
        
        Args:
            word: Word to find synonyms for
            
        Returns:
            List of synonyms
        """
        # Convert word to lowercase for lookup
        word_lower = word.lower()
        
        # Check if the word is capitalized
        is_capitalized = word[0].isupper() if word else False
        
        # Get synsets from WordNet
        synsets = wordnet.synsets(word_lower)
        
        # Extract synonyms from synsets
        synonyms = []
        for synset in synsets:
            for lemma in synset.lemmas():
                synonym = lemma.name().replace('_', ' ')
                
                # Skip if the synonym is the same as the original word
                if synonym.lower() == word_lower:
                    continue
                
                # Match capitalization of the original word
                if is_capitalized:
                    synonym = synonym[0].upper() + synonym[1:] if synonym else ''
                
                # Add synonym to the list
                synonyms.append(synonym)
        
        # Remove duplicates
        return list(set(synonyms))


class RandomDeletionAugmenter(BaseAugmenter):
    """
    Augment text by randomly deleting words.
    """
    
    def __init__(
        self,
        delete_fraction: float = 0.1,
        num_variations: int = 3,
        preserve_stopwords: bool = True,
        min_length: int = 3,
        language: str = 'english'
    ):
        """
        Initialize a random deletion augmenter.
        
        Args:
            delete_fraction: Fraction of words to delete
            num_variations: Number of variations to generate
            preserve_stopwords: Whether to avoid deleting stopwords
            min_length: Minimum length of the text after deletion (in words)
            language: Language for stopwords
        """
        self.delete_fraction = min(0.5, max(0.0, delete_fraction))  # Cap at 50%
        self.num_variations = num_variations
        self.preserve_stopwords = preserve_stopwords
        self.min_length = min_length
        self.language = language
        
        # Load stopwords if needed
        if preserve_stopwords and NLTK_AVAILABLE:
            try:
                from nltk.corpus import stopwords
                nltk.data.find('corpora/stopwords')
                self.stopwords = set(stopwords.words(language))
            except (ImportError, LookupError):
                nltk.download('stopwords')
                from nltk.corpus import stopwords
                self.stopwords = set(stopwords.words(language))
        else:
            self.stopwords = set()
    
    def augment(self, text: str) -> List[str]:
        """
        Augment text by randomly deleting words.
        
        Args:
            text: Text to augment
            
        Returns:
            List of augmented text variations
        """
        if not text:
            return []
        
        augmented_texts = []
        
        # Tokenize text into words
        words = text.split()
        
        # Skip if the text is too short
        if len(words) <= self.min_length:
            return []
        
        # Find candidate words for deletion (excluding stopwords if requested)
        candidate_indices = [
            i for i, word in enumerate(words)
            if not (self.preserve_stopwords and word.lower() in self.stopwords)
        ]
        
        # Generate variations
        for _ in range(self.num_variations):
            if not candidate_indices:
                continue
                
            # Create a copy of the words list
            new_words = words.copy()
            
            # Calculate number of words to delete
            num_to_delete = min(
                max(1, int(len(words) * self.delete_fraction)),  # At least 1, but follow fraction
                len(words) - self.min_length,  # Ensure we maintain minimum length
                len(candidate_indices)  # Cannot delete more than available candidates
            )
            
            # Select random words to delete
            indices_to_delete = sorted(
                random.sample(candidate_indices, num_to_delete),
                reverse=True  # Delete from end to start to avoid index issues
            )
            
            # Delete words
            for idx in indices_to_delete:
                if idx < len(new_words):  # Check if the index is valid
                    del new_words[idx]
            
            # Join words to create augmented text
            augmented_text = ' '.join(new_words)
            
            # Add augmented text to the list if it's different from the original
            if augmented_text != text and augmented_text not in augmented_texts:
                augmented_texts.append(augmented_text)
        
        return augmented_texts


class RandomSwapAugmenter(BaseAugmenter):
    """
    Augment text by randomly swapping adjacent words.
    """
    
    def __init__(
        self,
        swap_fraction: float = 0.1,
        num_variations: int = 3,
        preserve_stopwords: bool = False,
        max_swaps: int = 10,
        language: str = 'english'
    ):
        """
        Initialize a random swap augmenter.
        
        Args:
            swap_fraction: Fraction of words to consider for swapping
            num_variations: Number of variations to generate
            preserve_stopwords: Whether to avoid swapping stopwords
            max_swaps: Maximum number of swaps per variation
            language: Language for stopwords
        """
        self.swap_fraction = min(0.5, max(0.0, swap_fraction))  # Cap at 50%
        self.num_variations = num_variations
        self.preserve_stopwords = preserve_stopwords
        self.max_swaps = max_swaps
        self.language = language
        
        # Load stopwords if needed
        if preserve_stopwords and NLTK_AVAILABLE:
            try:
                from nltk.corpus import stopwords
                nltk.data.find('corpora/stopwords')
                self.stopwords = set(stopwords.words(language))
            except (ImportError, LookupError):
                nltk.download('stopwords')
                from nltk.corpus import stopwords
                self.stopwords = set(stopwords.words(language))
        else:
            self.stopwords = set()
    
    def augment(self, text: str) -> List[str]:
        """
        Augment text by randomly swapping adjacent words.
        
        Args:
            text: Text to augment
            
        Returns:
            List of augmented text variations
        """
        if not text:
            return []
        
        augmented_texts = []
        
        # Tokenize text into words
        words = text.split()
        
        # Skip if the text is too short
        if len(words) < 2:
            return []
        
        # Find candidate positions for swapping (excluding stopwords if requested)
        candidate_positions = []
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            if self.preserve_stopwords and (w1.lower() in self.stopwords or w2.lower() in self.stopwords):
                continue
            candidate_positions.append(i)
        
        # Generate variations
        for _ in range(self.num_variations):
            if not candidate_positions:
                continue
                
            # Create a copy of the words list
            new_words = words.copy()
            
            # Calculate number of swaps to perform
            num_swaps = min(
                max(1, int(len(candidate_positions) * self.swap_fraction)),  # At least 1
                self.max_swaps,  # Cap at max_swaps
                len(candidate_positions)  # Cannot swap more than available positions
            )
            
            # Select random positions for swapping
            positions_to_swap = random.sample(candidate_positions, num_swaps)
            
            # Perform swaps
            for pos in positions_to_swap:
                if pos + 1 < len(new_words):  # Check if the position is valid
                    new_words[pos], new_words[pos + 1] = new_words[pos + 1], new_words[pos]
            
            # Join words to create augmented text
            augmented_text = ' '.join(new_words)
            
            # Add augmented text to the list if it's different from the original
            if augmented_text != text and augmented_text not in augmented_texts:
                augmented_texts.append(augmented_text)
        
        return augmented_texts


class ContextualWordEmbeddingsAugmenter(BaseAugmenter):
    """
    Augment text by replacing words with contextually similar ones using word embeddings.
    Requires spaCy with word vectors or a similar library.
    """
    
    def __init__(
        self,
        replace_fraction: float = 0.1,
        num_variations: int = 3,
        model_name: str = 'en_core_web_md',
        similarity_threshold: float = 0.7,
        preserve_stopwords: bool = True
    ):
        """
        Initialize a contextual word embeddings augmenter.
        
        Args:
            replace_fraction: Fraction of words to replace
            num_variations: Number of variations to generate
            model_name: spaCy model name with word vectors
            similarity_threshold: Minimum similarity threshold for replacements
            preserve_stopwords: Whether to avoid replacing stopwords
        """
        self.replace_fraction = min(0.5, max(0.0, replace_fraction))  # Cap at 50%
        self.num_variations = num_variations
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.preserve_stopwords = preserve_stopwords
        
        if not SPACY_AVAILABLE:
            raise ImportError(
                "spaCy is required for contextual word embeddings augmentation. "
                "Install it with `pip install spacy` and download a model "
                "with word vectors using `python -m spacy download en_core_web_md`."
            )
        
        # Load spaCy model
        try:
            self.nlp = spacy.load(model_name)
            if not self.nlp.has_pipe("tok2vec") and not self.nlp.has_pipe("transformer"):
                logger.warning(
                    f"The spaCy model '{model_name}' may not have word vectors. "
                    f"Consider using a model like 'en_core_web_md' or 'en_core_web_lg'."
                )
        except OSError:
            logger.warning(f"spaCy model '{model_name}' not found. Downloading...")
            try:
                spacy.cli.download(model_name)
                self.nlp = spacy.load(model_name)
            except Exception as e:
                raise ValueError(f"Failed to download spaCy model: {e}")
    
    def augment(self, text: str) -> List[str]:
        """
        Augment text using contextually similar word replacements.
        
        Args:
            text: Text to augment
            
        Returns:
            List of augmented text variations
        """
        if not text:
            return []
        
        augmented_texts = []
        
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Generate variations
        for _ in range(self.num_variations):
            # Get a copy of the tokens
            tokens = [token.text for token in doc]
            
            # Find candidate tokens for replacement
            candidates = []
            for i, token in enumerate(doc):
                if (not token.is_stop or not self.preserve_stopwords) and \
                   not token.is_punct and not token.is_space and \
                   token.has_vector and token.vector_norm > 0:
                    candidates.append(i)
            
            if not candidates:
                continue
            
            # Calculate number of tokens to replace
            num_to_replace = max(1, int(len(candidates) * self.replace_fraction))
            num_to_replace = min(num_to_replace, len(candidates))
            
            # Select random tokens to replace
            indices_to_replace = random.sample(candidates, num_to_replace)
            
            # Replace tokens with similar ones
            for idx in indices_to_replace:
                original_token = doc[idx]
                
                # Find similar tokens
                similar_tokens = self._find_similar_tokens(original_token)
                
                if similar_tokens:
                    # Replace with a random similar token
                    tokens[idx] = random.choice(similar_tokens)
            
            # Join tokens to create augmented text
            augmented_text = ' '.join(tokens)
            
            # Add augmented text to the list if it's different from the original
            if augmented_text != text and augmented_text not in augmented_texts:
                augmented_texts.append(augmented_text)
        
        return augmented_texts
    
    def _find_similar_tokens(self, token) -> List[str]:
        """
        Find tokens similar to the given one using word vectors.
        
        Args:
            token: spaCy token
            
        Returns:
            List of similar tokens
        """
        # Skip if token has no vector or is not a content word
        if not token.has_vector or token.is_stop or token.is_punct:
            return []
        
        # Try to find similar words using the vocabulary
        similar_tokens = []
        
        # Get most similar words from the vocabulary
        queries = [w for w in self.nlp.vocab if w.has_vector and w.is_lower == token.is_lower]
        
        by_similarity = sorted(queries, key=lambda w: token.similarity(w), reverse=True)
        
        # Take the top similar words
        for similar_token in by_similarity[1:20]:  # Skip the first one (it's the same word)
            similarity = token.similarity(similar_token)
            if similarity >= self.similarity_threshold and similar_token.text.lower() != token.text.lower():
                similar_tokens.append(similar_token.text)
                if len(similar_tokens) >= 5:  # Limit to 5 similar tokens
                    break
        
        return similar_tokens


class CompositeAugmenter(BaseAugmenter):
    """
    Apply multiple augmenters in sequence or combination.
    """
    
    def __init__(
        self,
        augmenters: List[BaseAugmenter],
        max_variations: int = 5,
        combine_augmenters: bool = False
    ):
        """
        Initialize a composite augmenter.
        
        Args:
            augmenters: List of augmenters to apply
            max_variations: Maximum number of variations to generate
            combine_augmenters: Whether to apply augmenters in combination (True) or sequence (False)
        """
        self.augmenters = augmenters
        self.max_variations = max_variations
        self.combine_augmenters = combine_augmenters
    
    def augment(self, text: str) -> List[str]:
        """
        Apply multiple augmenters to generate text variations.
        
        Args:
            text: Text to augment
            
        Returns:
            List of augmented text variations
        """
        if not text:
            return []
        
        if not self.augmenters:
            return []
        
        if self.combine_augmenters:
            # Apply augmenters in combination
            all_variations = []
            
            # Get variations from each augmenter
            for augmenter in self.augmenters:
                variations = augmenter.augment(text)
                all_variations.extend(variations)
            
            # Remove duplicates
            unique_variations = list(set(all_variations))
            
            # Limit to max_variations
            if len(unique_variations) > self.max_variations:
                unique_variations = random.sample(unique_variations, self.max_variations)
            
            return unique_variations
        else:
            # Apply augmenters in sequence
            variations = [text]
            result_variations = []
            
            for augmenter in self.augmenters:
                new_variations = []
                
                for variation in variations:
                    # Apply the current augmenter to each variation
                    augmented = augmenter.augment(variation)
                    new_variations.extend(augmented)
                
                # Update variations for the next augmenter
                variations = new_variations
                
                # Add to results
                result_variations.extend(variations)
            
            # Remove the original text and duplicates
            unique_variations = list(set(result_variations) - {text})
            
            # Limit to max_variations
            if len(unique_variations) > self.max_variations:
                unique_variations = random.sample(unique_variations, self.max_variations)
            
            return unique_variations


# Factory function to get an augmenter
def get_augmenter(
    augmenter_type: str = "synonym",
    **kwargs
) -> BaseAugmenter:
    """
    Get an augmenter based on the specified type.
    
    Args:
        augmenter_type: Type of augmenter to use
        **kwargs: Additional arguments for the specific augmenter
        
    Returns:
        Augmenter instance
        
    Raises:
        ValueError: If an unsupported augmenter type is specified
    """
    if augmenter_type == "synonym":
        return WordReplacementAugmenter(**kwargs)
    elif augmenter_type == "backtranslation":
        return BackTranslationAugmenter(**kwargs)
    elif augmenter_type == "insertion":
        return RandomInsertionAugmenter(**kwargs)
    elif augmenter_type == "deletion":
        return RandomDeletionAugmenter(**kwargs)
    elif augmenter_type == "swap":
        return RandomSwapAugmenter(**kwargs)
    elif augmenter_type == "contextual":
        return ContextualWordEmbeddingsAugmenter(**kwargs)
    else:
        raise ValueError(
            f"Unsupported augmenter type: {augmenter_type}. "
            "Supported types are: synonym, backtranslation, insertion, deletion, swap, contextual."
        )


# Create a standard augmentation pipeline for general use
def create_standard_augmentation_pipeline(num_variations: int = 5) -> CompositeAugmenter:
    """
    Create a standard augmentation pipeline with reasonable defaults.
    
    Args:
        num_variations: Maximum number of variations to generate
    
    Returns:
        Composite augmenter with standard configuration
    """
    augmenters = []
    
    # Add word replacement augmenter if NLTK is available
    if NLTK_AVAILABLE:
        augmenters.append(WordReplacementAugmenter(
            replace_fraction=0.15,
            num_variations=2
        ))
    
    # Add random swap augmenter
    augmenters.append(RandomSwapAugmenter(
        swap_fraction=0.1,
        num_variations=2
    ))
    
    # Add random deletion augmenter
    augmenters.append(RandomDeletionAugmenter(
        delete_fraction=0.1,
        num_variations=2,
        preserve_stopwords=True
    ))
    
    return CompositeAugmenter(
        augmenters=augmenters,
        max_variations=num_variations,
        combine_augmenters=True
    ) 