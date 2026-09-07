"""
document_analysis.py
----------------------
Powers the "Analyze Article" WOW page:
    - document statistics (words, sentences, pages, reading time)
    - TF-IDF keyword extraction
    - chunk-level "importance" scoring — an always-available proxy for
      self-attention, built from the trained Logistic Regression weights,
      answering "which part of the article pushed the prediction?"
    - (optional/advanced) REAL self-attention weight extraction from the
      trained Transformer model, if you have TensorFlow + the saved
      self_attention_model.keras available
    - short-context vs long-context prediction comparison
"""

import re
import numpy as np


# ---------------------------------------------------------------------------
# 1. Basic document statistics
# ---------------------------------------------------------------------------
def compute_document_stats(text: str, num_pages=None) -> dict:
    words = text.split()
    sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    n_words = len(words)
    n_sentences = max(len(sentences), 1)
    reading_time_min = round(n_words / 200, 1)  # ~200 words/min average

    return {
        "pages": num_pages if num_pages is not None else "-",
        "words": n_words,
        "sentences": n_sentences,
        "reading_time_min": reading_time_min,
    }


# ---------------------------------------------------------------------------
# 2. Chunking — used for both keyword analysis and the "attention" viz
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size_words: int = 120) -> list:
    """Splits text into roughly equal word-count chunks — a simple stand-in
    for sections (Abstract / Introduction / Methodology / ...) when the
    PDF has no clean, machine-readable section headers."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size_words):
        chunk_words = words[i:i + chunk_size_words]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
    return chunks if chunks else [text]


# ---------------------------------------------------------------------------
# 3. TF-IDF based keyword extraction
# ---------------------------------------------------------------------------
def extract_keywords(clean_text: str, tfidf_vectorizer, top_n: int = 10) -> list:
    """Top-N highest TF-IDF-weighted terms for this single document, using
    the SAME vectorizer fit during training (models/tfidf_vectorizer.joblib)."""
    vec = tfidf_vectorizer.transform([clean_text])
    scores = vec.toarray().flatten()
    feature_names = np.array(tfidf_vectorizer.get_feature_names_out())

    top_idx = scores.argsort()[::-1][:top_n]
    keywords = [(feature_names[i], float(scores[i])) for i in top_idx if scores[i] > 0]
    return keywords


# ---------------------------------------------------------------------------
# 4. Chunk importance — "Why did the model choose this?" (always available,
#    based on trained Logistic Regression coefficients)
# ---------------------------------------------------------------------------
def chunk_importance_lr(chunks, clean_chunks, tfidf_vectorizer, lr_model, predicted_class_idx) -> list:
    """
    Importance(chunk) = sum over the chunk's TF-IDF-weighted terms of the
    Logistic Regression coefficient for the PREDICTED class. Chunks whose
    vocabulary pushed the model most strongly towards the winning class
    get the highest score. Returned scores are min-max normalized to [0, 1].
    """
    coef = lr_model.coef_[predicted_class_idx]
    raw_scores = []
    for clean_chunk in clean_chunks:
        vec = tfidf_vectorizer.transform([clean_chunk]).toarray().flatten()
        raw_scores.append(float(np.dot(vec, coef)))

    raw_scores = np.array(raw_scores)
    if raw_scores.max() - raw_scores.min() > 1e-9:
        norm = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
    else:
        norm = np.zeros_like(raw_scores)
    return norm.tolist()


# ---------------------------------------------------------------------------
# 5. OPTIONAL / ADVANCED: real self-attention weights from the trained
#    Transformer (only if TensorFlow + self_attention_model.keras exist).
#    Not called by default in app.py — wire it in yourself if you finish
#    training the deep model and want the "real" version instead of the
#    Logistic-Regression proxy above.
# ---------------------------------------------------------------------------
def chunk_importance_self_attention(chunks, tokenizer, model, max_len=200):
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    transformer_layers = [l for l in model.layers if l.__class__.__name__ == "TransformerBlock"]
    if not transformer_layers:
        raise ValueError("No TransformerBlock layer found in the model.")
    last_block = transformer_layers[-1]
    idx = model.layers.index(last_block)
    sub_model = tf.keras.Model(inputs=model.input, outputs=model.layers[idx - 1].output)

    scores = []
    for chunk in chunks:
        seq = tokenizer.texts_to_sequences([chunk])
        padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
        block_input = sub_model.predict(padded, verbose=0)
        _, attn_scores = last_block.att(block_input, block_input, return_attention_scores=True)
        scores.append(float(tf.reduce_mean(attn_scores).numpy()))

    scores = np.array(scores)
    if scores.max() - scores.min() > 1e-9:
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    return scores.tolist()


# ---------------------------------------------------------------------------
# 6. Short-context vs Long-context comparison
# ---------------------------------------------------------------------------
def short_vs_long_context_predict(full_clean_text, tfidf_vectorizer, lr_model, label_encoder,
                                   short_word_limit=80):
    """Compares prediction when the model only sees the first
    `short_word_limit` words vs the ENTIRE cleaned article — this is the
    demo that directly justifies why long-context modeling matters."""
    words = full_clean_text.split()
    short_text = " ".join(words[:short_word_limit])

    def _predict(text):
        vec = tfidf_vectorizer.transform([text])
        probs = lr_model.predict_proba(vec)[0]
        pred_idx = int(np.argmax(probs))
        label = label_encoder.inverse_transform([pred_idx])[0]
        return label, float(probs[pred_idx]), probs

    s_label, s_conf, s_probs = _predict(short_text)
    l_label, l_conf, l_probs = _predict(full_clean_text)

    return {
        "short": {"label": s_label, "confidence": s_conf, "probs": s_probs, "n_words": len(words[:short_word_limit])},
        "long": {"label": l_label, "confidence": l_conf, "probs": l_probs, "n_words": len(words)},
    }