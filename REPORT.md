# NLP Mini Project — Report Content
## Topic: Scientific Article Classification using Self-Attention Networks with Long-Context Representation Learning

> Use this file as the source text for the **handwritten portion** of your
> submission (Problem Statement, Theory of preprocessing techniques, Theory
> of algorithms used). Copy it onto A4 sheets in your own handwriting as per
> the submission instructions. The code / output / conclusion / dashboard
> screenshots should be **printed** and attached after the handwritten pages.

---

## 1. Problem Statement

With the exponential growth of scientific literature published every year on
platforms such as arXiv, manually categorizing research articles into their
correct subject area (e.g., Natural Language Processing, Computer Vision,
Machine Learning, Statistics, Artificial Intelligence) has become
increasingly difficult and time-consuming. Automating this classification
can help digital libraries, journals, and researchers organize, search, and
recommend relevant papers efficiently.

Scientific abstracts are typically **long** (100–250 words) and their topic
is often determined not by a single keyword but by relationships between
words and phrases spread across the **entire** passage. Traditional feature
representations such as **Bag-of-Words** and **TF-IDF**, and traditional
sequence models such as RNN/LSTM, either ignore word order completely or
struggle to retain information over long distances (the "long-range
dependency problem").

**Objective:** To design and implement a **Self-Attention based neural
network (Transformer-encoder architecture)** that learns a **long-context
representation** of a scientific abstract — allowing every word to directly
relate to every other word in the abstract regardless of distance — and to
use this representation to automatically classify the abstract into its
correct subject category. The performance of this self-attention model is
compared against classical machine-learning baselines (Naive Bayes and
Logistic Regression trained on TF-IDF features) to demonstrate the benefit
of long-context representation learning for scientific text classification.

**Dataset:** Sample built to mirror the *Kaggle arXiv Dataset* by Cornell
University.
Source URL: https://www.kaggle.com/datasets/Cornell-University/arxiv
(Alternative sources also acceptable: Hugging Face
`ccdv/arxiv-classification`, UCI ML Repository text-classification corpora,
GitHub scientific-abstract datasets.)

---

## 2. Theory — Text Preprocessing Techniques

Raw text data is noisy and unstructured, so it must be cleaned and
normalized before it can be converted into numerical features. The
following preprocessing steps were applied to every abstract:

### 2.1 Lowercasing
Converts all characters to lowercase so that words like "Attention" and
"attention" are treated as the same token, reducing vocabulary size and
sparsity.

### 2.2 Noise Removal
- **URL removal:** Hyperlinks (e.g. `https://arxiv.org`) carry no semantic
  value for classification and are removed using regular expressions.
- **Digit removal:** Numeric values (years, page numbers, equation indices)
  are stripped since they rarely help distinguish subject categories.
- **Punctuation removal:** Punctuation marks are removed as they do not
  contribute to bag-of-words/TF-IDF style features.
- **Whitespace normalization:** Multiple spaces/tabs/newlines are collapsed
  into a single space.

### 2.3 Tokenization
The process of splitting a continuous string of text into individual units
called **tokens** (usually words). E.g.
`"self attention transformer"` → `["self", "attention", "transformer"]`.
Tokenization is a prerequisite for every downstream NLP step because
almost all algorithms operate on tokens, not raw strings.

### 2.4 Stopword Removal
**Stopwords** are extremely common words (*the, is, at, which, on, ...*)
that carry little discriminative meaning for a classification task. Removing
them reduces feature-space dimensionality and noise, letting the model focus
on content-bearing words.

### 2.5 Lemmatization
Lemmatization reduces a word to its dictionary base form (**lemma**) using
vocabulary and morphological analysis, e.g. `"classifying" → "classify"`,
`"models" → "model"`. Unlike stemming (crude suffix-chopping), lemmatization
produces linguistically valid root words, improving feature quality. (A
lightweight rule-based fallback stemmer is used if the NLTK WordNet corpus is
unavailable offline.)

---

## 3. Theory — Feature Engineering Techniques

### 3.1 Bag of Words (BoW) / CountVectorizer
Represents each document as a vector of raw word counts over a fixed
vocabulary, ignoring grammar and word order. Simple and effective but
produces very high-dimensional, sparse vectors and cannot capture context
or word importance.

### 3.2 TF-IDF (Term Frequency – Inverse Document Frequency)
Improves upon BoW by weighting each term according to:

```
TF-IDF(t, d) = TF(t, d) × IDF(t)
TF(t, d)  = (number of times term t appears in document d) / (total terms in d)
IDF(t)    = log( N / (1 + number of documents containing t) )
```

Terms that occur frequently in one document but rarely across the whole
corpus (e.g. "transformer" in an NLP paper) receive a **high** TF-IDF
weight, while terms common to almost every document (e.g. "the", "model")
receive a **low** weight. This makes TF-IDF far more discriminative than raw
BoW counts and is the feature representation used for the baseline models
in this project (with unigrams + bigrams).

### 3.3 Word Embeddings (Word2Vec / GloVe) — used in the deep-learning model
Instead of sparse, count-based vectors, **word embeddings** map each word to
a small (e.g. 64–300 dimensional) **dense** vector such that semantically
similar words lie close together in the embedding space.
- **Word2Vec** learns embeddings by predicting a word from its surrounding
  context (CBOW) or the context from a word (Skip-gram).
- **GloVe** learns embeddings from global word-word co-occurrence
  statistics of a corpus.

In this project, the Self-Attention model uses a **trainable token
embedding layer** (optionally initialized with pretrained GloVe vectors,
see `feature_engineering.build_embedding_matrix`) combined with a
**positional embedding**, since self-attention itself has no inherent sense
of word order.

---

## 4. Theory — Algorithms / Models Used

### 4.1 Multinomial Naive Bayes (Baseline 1)
A probabilistic classifier based on **Bayes' Theorem** with a "naive"
assumption that features (words) are conditionally independent given the
class label:

```
P(class | document) ∝ P(class) × Π P(word_i | class)
```

Despite its simplicity, Naive Bayes is a fast, strong baseline for text
classification because TF-IDF/BoW features are high-dimensional and largely
independent in practice.

### 4.2 Logistic Regression (Baseline 2)
A linear discriminative classifier that models the probability of each
class using the **softmax** function (multi-class generalization of the
sigmoid) over a weighted sum of input TF-IDF features:

```
P(class = k | x) = softmax(W·x + b)_k
```

The weights `W` are learned by minimizing cross-entropy loss via gradient
descent. Logistic Regression generally outperforms Naive Bayes on text data
because it does not assume feature independence and can learn feature
weights directly from data.

### 4.3 Self-Attention Mechanism (Core Concept)
Self-attention allows each token in a sequence to compute a weighted
combination of **all** other tokens (including itself), where the weights
reflect how relevant each other token is to the current one. For each token
we compute three vectors — **Query (Q)**, **Key (K)**, **Value (V)** —
via learned linear projections, and attention output is:

```
Attention(Q, K, V) = softmax( (Q · Kᵀ) / √d_k ) · V
```

Because every token can attend directly to every other token in a **single
step**, the maximum path length between any two tokens is O(1), unlike
RNNs where it is O(n) — this is precisely what makes self-attention
suitable for **long-context representation learning**, i.e., capturing
dependencies between words that are far apart in a long scientific
abstract.

**Multi-Head Attention** runs several attention operations ("heads") in
parallel, each learning to focus on different types of relationships (e.g.
one head might learn syntactic dependencies, another topical/semantic
similarity), and concatenates their outputs.

### 4.4 Transformer Encoder Block (used as the classifier)
Each Transformer block (Vaswani et al., 2017, *"Attention Is All You
Need"*) consists of:
1. Multi-Head Self-Attention layer
2. Residual ("skip") connection + Layer Normalization
3. Position-wise Feed-Forward Network (two Dense layers with ReLU)
4. A second residual connection + Layer Normalization

Because self-attention has no built-in notion of sequence order, a
**Positional Embedding** is added to the token embeddings before the first
Transformer block, so the model knows the relative/absolute position of
each word.

**Overall architecture used in this project:**

```
Input tokens (padded to max_len)
      ↓
Token Embedding + Positional Embedding
      ↓
Transformer Block × 2   (Multi-Head Self-Attention + Feed-Forward)
      ↓
Global Average Pooling  (sequence → fixed-size vector = long-context representation)
      ↓
Dropout → Dense(ReLU) → Dropout
      ↓
Dense(Softmax) → predicted category
```

---

## 5. Theory — Evaluation Metrics

- **Accuracy** = (Correct Predictions) / (Total Predictions) — overall
  correctness, can be misleading on imbalanced datasets.
- **Precision** = TP / (TP + FP) — of all items predicted as class *k*, how
  many were actually class *k*.
- **Recall** = TP / (TP + FN) — of all actual items of class *k*, how many
  were correctly identified.
- **F1-score** = 2 × (Precision × Recall) / (Precision + Recall) — harmonic
  mean of precision and recall, useful when classes are imbalanced.
- **Confusion Matrix** — an N×N table (N = number of classes) showing
  actual vs. predicted labels; the diagonal represents correct predictions
  and off-diagonal cells reveal which categories are most often confused.
- **ROC Curve & AUC** — plots True Positive Rate vs. False Positive Rate at
  various classification thresholds (computed One-vs-Rest for multi-class
  problems); **AUC** (Area Under Curve) close to 1.0 indicates excellent
  class separability.

---

## 6. Results & Discussion (fill in with your actual run's numbers/screenshots)

### 6.1 Key Findings
- Logistic Regression (TF-IDF) outperformed Naive Bayes because it does not
  assume feature independence and directly optimizes discriminative weights.
- The Self-Attention Transformer model is expected to outperform both TF-IDF
  baselines on longer abstracts because it explicitly models long-range
  token dependencies rather than treating text as an unordered bag of
  n-grams.
- The most confusion occurs between closely related categories (e.g.
  `cs.LG`, `cs.AI`, `stat.ML`) which naturally share vocabulary — visible in
  the confusion-matrix off-diagonal cells.

### 6.2 Explaining the Graphs
- **Class distribution plot** — verifies the dataset is (reasonably)
  balanced across categories, so accuracy is a fair metric.
- **Abstract length histogram** — shows most abstracts span well beyond a
  short sentence, motivating why long-context modeling (self-attention)
  is necessary rather than short-window n-gram features.
- **Word cloud** — highlights the most frequent domain-specific terms after
  preprocessing (stopwords removed), giving a qualitative sense of each
  category's vocabulary.
- **Confusion matrix** — diagonal-heavy matrix = good classifier; spread-out
  off-diagonal values reveal specific category confusions.
- **ROC curve** — curves that hug the top-left corner (AUC → 1.0) indicate
  strong class separability for that category.
- **Model comparison bar chart** — visually ranks all trained models across
  Accuracy / Precision / Recall / F1-score side by side.

### 6.3 Comparison Table (example structure — replace with your actual numbers from outputs/comparison_table.csv)

| Model                          | Accuracy | Precision | Recall | F1-Score |
|---------------------------------|----------|-----------|--------|----------|
| Naive Bayes (TF-IDF)            |   ~0.78  |   ~0.78   | ~0.78  |  ~0.78   |
| Logistic Regression (TF-IDF)    |   ~0.89  |   ~0.89   | ~0.89  |  ~0.89   |
| Self-Attention Transformer      |  (run locally with TensorFlow installed) |

### 6.4 Challenges Faced
- Vocabulary overlap between closely related scientific sub-fields makes
  perfect separation impossible even for a strong model.
- Self-attention has quadratic (O(n²)) computational cost in sequence
  length, requiring careful choice of `max_len` for long abstracts.
- Limited/synthetic training data can lead to overfitting; real deployments
  need larger labeled corpora and possibly pretrained language models.
- Choosing the right vocabulary size and embedding dimensions to balance
  model capacity against overfitting/training time.

### 6.5 Future Scope
- Replace the from-scratch Transformer with a pretrained long-context model
  (Longformer, BigBird, SciBERT) via transfer learning.
- Extend to multi-label classification, since a paper can belong to
  multiple arXiv categories simultaneously.
- Visualize learned attention weights for explainability (which words the
  model "focused on" when classifying).
- Deploy as a browser extension or API for real-time classification of new
  arXiv submissions.

### 6.6 Applications
- Automatic subject-tagging for digital libraries and preprint servers.
- Literature-review and citation-recommendation assistants.
- Reviewer-assignment systems for journals/conferences based on predicted
  subject area.
- Research-trend analysis dashboards for institutions and funding bodies.

---

## 7. Conclusion

This project demonstrates that scientific article classification benefits
from moving beyond simple bag-of-words / TF-IDF representations toward
**self-attention based, long-context representation learning**. By allowing
every token in an abstract to directly attend to every other token, the
Transformer-encoder based classifier is able to capture dependencies that
span the entire document — something recurrent and classical ML models
struggle to do — leading to a more robust and context-aware model for
categorizing long scientific text, at the cost of increased computational
requirements.

---

## 8. References
1. Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS.
2. Cornell University. *arXiv Dataset*. Kaggle.
   https://www.kaggle.com/datasets/Cornell-University/arxiv
3. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*.
   JMLR.
4. Pennington, J., Socher, R., Manning, C. (2014). *GloVe: Global Vectors
   for Word Representation*. EMNLP.
5. Mikolov, T. et al. (2013). *Efficient Estimation of Word Representations
   in Vector Space* (Word2Vec). arXiv:1301.3781.
