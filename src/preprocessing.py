"""
preprocessing.py
-----------------
Implements the classical NLP text-preprocessing pipeline:
    1. Lowercasing
    2. Noise removal (URLs, digits, punctuation, extra whitespace)
    3. Tokenization
    4. Stopword removal
    5. Lemmatization (falls back to a light suffix-stripping stemmer
       if NLTK / WordNet data is not available offline)

NLTK is used if it is installed and its data has been downloaded;
otherwise a small built-in stopword list + rule-based lemmatizer keeps
the pipeline fully offline-runnable.
"""

import re
import string

# ---------------------------------------------------------------------------
# Try to use NLTK (better linguistic quality). Fall back gracefully if the
# library or its corpora are not available (e.g. no internet access).
# ---------------------------------------------------------------------------
_USE_NLTK = False
try:
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    try:
        _STOPWORDS = set(nltk_stopwords.words("english"))
        _lemmatizer = WordNetLemmatizer()
        _lemmatizer.lemmatize("test")  # triggers WordNet load -> confirms data present
        _USE_NLTK = True
    except LookupError:
        _USE_NLTK = False
except ImportError:
    _USE_NLTK = False

# ---------------------------------------------------------------------------
# Offline fallback resources
# ---------------------------------------------------------------------------
_BASIC_STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be
because been before being below between both but by can't cannot could
couldn't did didn't do does doesn't doing don't down during each few for
from further had hadn't has hasn't have haven't having he he'd he'll he's
her here here's hers herself him himself his how how's i i'd i'll i'm i've
if in into is isn't it it's its itself let's me more most mustn't my myself
no nor not of off on once only or other ought our ours ourselves out over
own same shan't she she'd she'll she's should shouldn't so some such than
that that's the their theirs them themselves then there there's these they
they'd they'll they're they've this those through to too under until up
very was wasn't we we'd we'll we're we've were weren't what what's when
when's where where's which while who who's whom why why's with won't would
wouldn't you you'd you'll you're you've your yours yourself yourselves
""".split())

_SUFFIXES = ["ational", "tional", "alize", "icate", "iciti", "ative",
             "ical", "ness", "ful", "ing", "edly", "ies", "ied", "ed",
             "es", "s", "ly"]


def _simple_lemmatize(token: str) -> str:
    """Very light rule-based suffix stripper used only if NLTK/WordNet
    is unavailable. Not linguistically perfect but keeps the pipeline
    self-contained and dependency-free."""
    for suf in _SUFFIXES:
        if token.endswith(suf) and len(token) - len(suf) >= 3:
            return token[: -len(suf)]
    return token


def clean_text(text: str) -> str:
    """Lowercase + remove URLs, digits, punctuation and extra whitespace."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # URLs
    text = re.sub(r"\d+", " ", text)                         # digits
    text = text.translate(str.maketrans("", "", string.punctuation))  # punctuation
    text = re.sub(r"\s+", " ", text).strip()                 # extra whitespace
    return text


def tokenize(text: str):
    if _USE_NLTK:
        try:
            return word_tokenize(text)
        except LookupError:
            pass
    return text.split()


def remove_stopwords(tokens):
    stop_set = _STOPWORDS if _USE_NLTK else _BASIC_STOPWORDS
    return [t for t in tokens if t not in stop_set and len(t) > 1]


def lemmatize(tokens):
    if _USE_NLTK:
        return [_lemmatizer.lemmatize(t) for t in tokens]
    return [_simple_lemmatize(t) for t in tokens]


def preprocess_text(text: str) -> str:
    """Full pipeline: clean -> tokenize -> remove stopwords -> lemmatize.
    Returns a single cleaned string (space-joined tokens), suitable for
    both TF-IDF/BoW vectorizers and for the Keras Tokenizer used by the
    self-attention model."""
    text = clean_text(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return " ".join(tokens)


def preprocess_series(series):
    """Apply preprocess_text to a pandas Series of raw abstracts."""
    return series.astype(str).apply(preprocess_text)


if __name__ == "__main__":
    sample = ("We propose a novel Self-Attention Transformer for long-context "
               "representation learning! Visit https://arxiv.org for 2024 details.")
    print("RAW  :", sample)
    print("CLEAN:", preprocess_text(sample))
    print("Using NLTK:", _USE_NLTK)
