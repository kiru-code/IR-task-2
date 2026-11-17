"""
Text Preprocessing Module for Product Review Sentiment Classification
Author: Kirubaanithi R (Reg No: RA2411026050203)
Course: 21CSE377T - Information Retrieval
SRM Institute of Science and Technology, Tiruchirappalli
"""

import os
import re
import string
import nltk

# Ensure local nltk_data directory is registered in nltk data path
current_dir = os.path.dirname(os.path.abspath(__file__))
local_nltk = os.path.join(current_dir, 'nltk_data')
if os.path.exists(local_nltk):
    nltk.data.path.insert(0, local_nltk)

# Common contractions dictionary for normalisation
CONTRACTIONS = {
    "won't": "will not",
    "can't": "cannot",
    "can not": "cannot",
    "ain't": "is not",
    "aren't": "are not",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "isn't": "is not",
    "mustn't": "must not",
    "shan't": "shall not",
    "shouldn't": "should not",
    "wasn't": "was not",
    "weren't": "were not",
    "wouldn't": "would not",
    "it's": "it is",
    "i'm": "i am",
    "they're": "they are",
    "we're": "we are",
    "you're": "you are"
}

# Negation words that MUST be preserved for sentiment classification
# Stripping these words destroys polarity (e.g. 'not good' -> 'good')
NEGATION_WORDS = {
    "not", "no", "nor", "neither", "never", "none", "nobody", 
    "nowhere", "nothing", "barely", "hardly", "scarcely", 
    "rarely", "seldom", "without", "cannot", "do not", "does not",
    "did not", "is not", "was not", "were not", "have not", "has not"
}

# Try loading NLTK stop words and lemmatizer, with robust fallbacks
try:
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    raw_stopwords = set(stopwords.words('english'))
    # Remove negation words from standard stop words list
    STOP_WORDS = raw_stopwords - NEGATION_WORDS
    LEMMATIZER = WordNetLemmatizer()
except Exception:
    # Minimal fallback stop words if NLTK data is unavailable
    FALLBACK_STOPWORDS = {
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", 
        "your", "yours", "yourself", "yourselves", "he", "him", "his", 
        "himself", "she", "her", "hers", "herself", "it", "its", "itself", 
        "they", "them", "their", "theirs", "themselves", "what", "which", 
        "who", "whom", "this", "that", "these", "those", "am", "is", "are", 
        "was", "were", "be", "been", "being", "have", "has", "had", "having", 
        "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", 
        "or", "because", "as", "until", "while", "of", "at", "by", "for", 
        "with", "about", "against", "between", "into", "through", "during", 
        "before", "after", "above", "below", "to", "from", "up", "down", 
        "in", "out", "on", "off", "over", "under", "again", "further", 
        "then", "once", "here", "there", "when", "where", "why", "how", 
        "all", "any", "both", "each", "few", "more", "most", "other", 
        "some", "such", "only", "own", "same", "so", "than", "too", "very", 
        "s", "t", "can", "will", "just", "don", "should", "now"
    }
    STOP_WORDS = FALLBACK_STOPWORDS - NEGATION_WORDS
    LEMMATIZER = None


def expand_contractions(text: str) -> str:
    """Expand short-form contractions (e.g. didn't -> did not)."""
    text_lower = text.lower()
    for contraction, expansion in CONTRACTIONS.items():
        text_lower = re.sub(r'\b' + re.escape(contraction) + r'\b', expansion, text_lower)
    return text_lower


def clean_html_and_special(text: str) -> str:
    """Strip HTML markup, URLs, and non-alphanumeric noise."""
    # Remove HTML tags (e.g., <br />, <p>)
    text = re.sub(r'<.*?>', ' ', text)
    # Remove web URLs
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text, flags=re.MULTILINE)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)
    # Remove digits and punctuation, keeping words and spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Normalize extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize(text: str) -> list:
    """Split text into lowercase token words."""
    return text.lower().split()


def remove_stopwords(tokens: list, preserve_negations: bool = True) -> list:
    """
    Remove standard stop words while preserving critical negation words
    to retain sentiment polarity.
    """
    active_stopwords = STOP_WORDS if preserve_negations else (STOP_WORDS | NEGATION_WORDS)
    return [token for token in tokens if token not in active_stopwords and len(token) > 1]


def lemmatize_tokens(tokens: list) -> list:
    """Apply WordNet lemmatization to reduce words to dictionary root forms."""
    if LEMMATIZER:
        try:
            return [LEMMATIZER.lemmatize(token) for token in tokens]
        except Exception:
            return tokens
    return tokens


def preprocess_review(text: str, preserve_negations: bool = True) -> str:
    """
    Complete end-to-end preprocessing pipeline:
    1. Contraction expansion
    2. HTML tag & noise removal
    3. Lowercase conversion
    4. Tokenization
    5. Sentiment-aware stop-word removal (preserving negations)
    6. Lemmatization
    """
    if not isinstance(text, str):
        return ""
    
    expanded = expand_contractions(text)
    cleaned = clean_html_and_special(expanded)
    tokens = tokenize(cleaned)
    filtered = remove_stopwords(tokens, preserve_negations=preserve_negations)
    lemmatized = lemmatize_tokens(filtered)
    return " ".join(lemmatized)


def get_preprocessing_steps(raw_text: str) -> dict:
    """
    Deconstructs preprocessing into explicit stages for demonstration,
    UI visualization, and academic defense.
    """
    step1_raw = raw_text
    step2_contractions = expand_contractions(raw_text)
    step3_html_clean = clean_html_and_special(step2_contractions)
    step4_tokens = tokenize(step3_html_clean)
    step5_filtered = remove_stopwords(step4_tokens, preserve_negations=True)
    step6_lemmatized = lemmatize_tokens(step5_filtered)
    step7_final = " ".join(step6_lemmatized)

    # Calculate token counts
    raw_word_count = len(step1_raw.split())
    cleaned_word_count = len(step6_lemmatized)
    reduction_pct = ((raw_word_count - cleaned_word_count) / max(raw_word_count, 1)) * 100

    return {
        "raw_text": step1_raw,
        "contractions_expanded": step2_contractions,
        "html_and_noise_removed": step3_html_clean,
        "tokens": step4_tokens,
        "stopwords_removed": step5_filtered,
        "lemmatized_tokens": step6_lemmatized,
        "final_preprocessed_text": step7_final,
        "raw_word_count": raw_word_count,
        "cleaned_word_count": cleaned_word_count,
        "reduction_percentage": round(reduction_pct, 2)
    }


if __name__ == "__main__":
    sample = (
        "I didn't like this movie at all! The acting was terrible and the plot "
        "made no sense.<br /><br />However, the visual effects weren't bad."
    )
    print("--- SAMPLE PREPROCESSING DEMO ---")
    steps = get_preprocessing_steps(sample)
    for k, v in steps.items():
        print(f"\n[{k}]:\n{v}")

