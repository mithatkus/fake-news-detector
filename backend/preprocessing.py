"""
Text preprocessing pipeline.

- preprocess_tfidf: strips noise, lowercases, removes stopwords, stems (Porter).
  Used for the TF-IDF Logistic Regression model.
- preprocess_cnn: same pipeline but WITHOUT stemming.
  Used for the CNN model (the Keras tokenizer maps un-stemmed tokens).

NOTE: The saved keras_tokenizer.pkl was fit on the `clean_text` column of
cleaned_isot.csv.  If that column was produced with stemming, swap
preprocess_cnn → preprocess_tfidf inside model_manager.py.
"""

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# ── Ensure NLTK resources are available ───────────────────────────────────────
# In Lambda, /opt/nltk_data is baked into the image (NLTK_DATA env var points there).
# We use os.path checks to avoid nltk.data.find() which can raise OSError on
# partial/legacy punkt data.  Downloads go to /tmp (only writable dir in Lambda).
import os as _os

_NLTK_FALLBACK = "/tmp/nltk_data"

def _ensure_nltk(resource: str, rel_path: str) -> None:
    for search_dir in nltk.data.path:
        if _os.path.exists(_os.path.join(search_dir, rel_path)):
            return
    _os.makedirs(_NLTK_FALLBACK, exist_ok=True)
    nltk.download(resource, quiet=True, download_dir=_NLTK_FALLBACK)
    if _NLTK_FALLBACK not in nltk.data.path:
        nltk.data.path.insert(0, _NLTK_FALLBACK)

_ensure_nltk("punkt_tab", "tokenizers/punkt_tab")
_ensure_nltk("stopwords", "corpora/stopwords")

STOP_WORDS: set[str] = set(stopwords.words("english"))
_stemmer = PorterStemmer()


# ── Shared noise-removal ──────────────────────────────────────────────────────
def _clean(text: str) -> str:
    text = re.sub(r"[A-Z\s]+\(Reuters\)\s*-\s*", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"[@#]\w+", "", text)
    text = re.sub(r"featured image via.*", "", text, flags=re.IGNORECASE)
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


# ── Public API ────────────────────────────────────────────────────────────────
def preprocess_tfidf(text: str) -> str:
    """Stemmed preprocessing for the TF-IDF model."""
    text = _clean(text)
    tokens = word_tokenize(text)
    tokens = [
        _stemmer.stem(t)
        for t in tokens
        if t not in STOP_WORDS and len(t) > 2
    ]
    return " ".join(tokens)


def preprocess_cnn(text: str) -> str:
    """Un-stemmed preprocessing for the CNN model."""
    text = _clean(text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)
