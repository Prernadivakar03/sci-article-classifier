"""
feature_engineering.py
------------------------
Implements the feature-engineering techniques required by the project:

    1. Bag of Words (CountVectorizer)
    2. TF-IDF (TfidfVectorizer)               -> used by the baseline ML models
    3. Keras Tokenizer + padded sequences      -> used by the Self-Attention /
                                                   Transformer deep-learning model
    4. (Optional) Word2Vec / GloVe embedding matrix builder

These are kept as separate, reusable functions so the same preprocessed
text column can be fed into either the classical ML baselines or the
deep self-attention model.
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def build_bow_features(train_texts, test_texts, max_features=5000):
    """Bag-of-Words representation using CountVectorizer."""
    vectorizer = CountVectorizer(max_features=max_features, ngram_range=(1, 1))
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_test, vectorizer


def build_tfidf_features(train_texts, test_texts, max_features=8000, ngram_range=(1, 2)):
    """TF-IDF representation (unigrams + bigrams) - main feature set for baselines."""
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range,
                                  sublinear_tf=True)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_test, vectorizer


def build_sequence_features(train_texts, test_texts, vocab_size=10000, max_len=200):
    """
    Tokenizer + padded integer sequences for the deep self-attention model.
    Requires TensorFlow/Keras (see requirements.txt). Long max_len (default
    200 tokens) is chosen deliberately since the whole point of this project
    is LONG-CONTEXT representation learning over full scientific abstracts.
    """
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_texts)

    train_seq = tokenizer.texts_to_sequences(train_texts)
    test_seq = tokenizer.texts_to_sequences(test_texts)

    X_train = pad_sequences(train_seq, maxlen=max_len, padding="post", truncating="post")
    X_test = pad_sequences(test_seq, maxlen=max_len, padding="post", truncating="post")

    return X_train, X_test, tokenizer


def build_embedding_matrix(tokenizer, glove_path=None, embedding_dim=100, vocab_size=10000):
    """
    Optional: build a pretrained GloVe embedding matrix aligned with the
    Keras tokenizer's word index. If glove_path is None (default, since it
    requires downloading ~800MB glove.6B.100d.txt from
    https://nlp.stanford.edu/projects/glove/), a random-uniform matrix is
    returned instead and the embedding layer is trained from scratch.
    """
    word_index = tokenizer.word_index
    num_words = min(vocab_size, len(word_index) + 1)
    embedding_matrix = np.random.uniform(-0.05, 0.05, (num_words, embedding_dim)).astype("float32")

    if glove_path is not None:
        embeddings_index = {}
        with open(glove_path, encoding="utf8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vec = np.asarray(values[1:], dtype="float32")
                embeddings_index[word] = vec

        for word, i in word_index.items():
            if i >= num_words:
                continue
            vec = embeddings_index.get(word)
            if vec is not None:
                embedding_matrix[i] = vec

    return embedding_matrix
