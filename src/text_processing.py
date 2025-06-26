"""
Text processing utilities for news headline analysis.
"""
import os
import nltk
from typing import List, Dict
from collections import Counter

def setup_nltk(download_dir: str = None) -> None:
    """
    Set up NLTK by downloading required resources.

    Args:
        download_dir: Optional custom directory for NLTK data
    """
    if download_dir:
        os.makedirs(download_dir, exist_ok=True)
        nltk.data.path.insert(0, download_dir)

    # Download required NLTK data
    resources = ['punkt', 'averaged_perceptron_tagger', 'stopwords']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True, download_dir=download_dir)
        except Exception as e:
            print(f"Error downloading {resource}: {str(e)}")

def process_text(text: str, remove_stopwords: bool = True) -> List[str]:
    """
    Process text by tokenizing and optionally removing stopwords.

    Args:
        text: Input text to process
        remove_stopwords: Whether to remove stopwords

    Returns:
        List of processed tokens
    """
    # Convert to string if not already
    text = str(text).lower()

    # Tokenize
    tokens = nltk.word_tokenize(text)

    if remove_stopwords:
        # Remove stopwords and non-alphanumeric tokens
        stop_words = set(nltk.corpus.stopwords.words('english'))
        tokens = [token for token in tokens
                 if token.isalnum() and token not in stop_words]

    return tokens

def analyze_text(text: str) -> Dict:
    """
    Perform basic text analysis.

    Args:
        text: Input text to analyze

    Returns:
        Dictionary containing analysis results
    """
    # Basic stats
    char_count = len(text)
    word_count = len(text.split())

    # Tokenize and tag parts of speech
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)

    # Count POS tags
    pos_counts = Counter(tag for word, tag in pos_tags)

    return {
        'char_count': char_count,
        'word_count': word_count,
        'pos_counts': dict(pos_counts),
        'tokens': tokens,
        'pos_tags': pos_tags
    }
