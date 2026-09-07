"""
generate_dataset.py
--------------------
Dataset source (REAL DATA, recommended for final submission):
    Kaggle - arXiv Dataset (Cornell University)
    URL: https://www.kaggle.com/datasets/Cornell-University/arxiv

    Alternative sources (any one may be cited in the report):
    - Hugging Face: https://huggingface.co/datasets/ccdv/arxiv-classification
    - UCI ML Repository (Reuters-21578 / 20 Newsgroups style corpora)
    - GitHub: https://github.com/qingyunzou/arxiv-abstract-classification

This script builds a LOCAL, OFFLINE, reproducible sample of the same shape
(title, abstract, category) so that the full pipeline (preprocessing ->
feature engineering -> self-attention model -> evaluation -> dashboard)
can be developed, tested and demoed even without an internet connection.

For the actual submission:
    1. Download "arxiv-metadata-oai-snapshot.json" from the Kaggle link above.
    2. Filter it down to ~5-8 categories of your choice (e.g. cs.CL, cs.CV,
       cs.LG, stat.ML, cs.AI) and 500-1000 rows per category.
    3. Save as data/arxiv_abstracts.csv with columns: title, abstract, category
    4. Re-run the pipeline -- no other code changes are required, the loader
       in this file (`load_real_kaggle_json`) already handles the raw file.

Running this script directly creates data/arxiv_abstracts.csv (synthetic,
offline sample used for the demo run of this project).
"""

import json
import random
import pandas as pd
import os

random.seed(42)

CATEGORIES = ["cs.CL", "cs.CV", "cs.LG", "stat.ML", "cs.AI"]

CATEGORY_LABELS = {
    "cs.CL": "Computation and Language (NLP)",
    "cs.CV": "Computer Vision",
    "cs.LG": "Machine Learning",
    "stat.ML": "Statistics - Machine Learning",
    "cs.AI": "Artificial Intelligence",
}

# ---- Vocabulary banks used to synthesize long, topically-coherent abstracts ----
VOCAB = {
    "cs.CL": {
        "subjects": ["language model", "transformer encoder", "sequence-to-sequence model",
                     "attention-based parser", "neural machine translation system",
                     "text classification pipeline", "named entity recognizer",
                     "question answering system", "dialogue agent", "summarization model"],
        "tasks": ["sentiment analysis", "machine translation", "text summarization",
                  "named entity recognition", "part-of-speech tagging", "question answering",
                  "coreference resolution", "document classification", "language modeling"],
        "methods": ["self-attention", "recurrent neural networks", "long short-term memory networks",
                    "byte-pair encoding", "word embeddings", "contextual embeddings",
                    "pretraining and fine-tuning", "beam search decoding"],
        "datasets": ["Penn Treebank", "SQuAD", "GLUE benchmark", "IMDB reviews", "WMT14 corpus",
                     "CoNLL-2003", "20 Newsgroups"],
    },
    "cs.CV": {
        "subjects": ["convolutional neural network", "vision transformer", "object detector",
                     "image segmentation network", "generative adversarial network",
                     "video understanding model", "pose estimation framework",
                     "image captioning model", "face recognition system", "depth estimation network"],
        "tasks": ["image classification", "object detection", "semantic segmentation",
                  "instance segmentation", "image captioning", "action recognition",
                  "optical flow estimation", "3D reconstruction"],
        "methods": ["convolution operations", "residual connections", "region proposal networks",
                    "self-attention over patches", "data augmentation", "transfer learning",
                    "anchor-free detection", "feature pyramid networks"],
        "datasets": ["ImageNet", "COCO", "Pascal VOC", "CIFAR-10", "Cityscapes", "Kinetics-400"],
    },
    "cs.LG": {
        "subjects": ["deep neural network", "gradient boosting ensemble", "reinforcement learning agent",
                     "graph neural network", "meta-learning framework", "self-supervised model",
                     "generative model", "representation learning framework", "optimization algorithm"],
        "tasks": ["classification", "regression", "clustering", "anomaly detection",
                  "recommendation", "time-series forecasting", "reinforcement learning control"],
        "methods": ["stochastic gradient descent", "dropout regularization", "batch normalization",
                    "attention mechanisms", "contrastive learning", "curriculum learning",
                    "hyperparameter optimization", "ensemble learning"],
        "datasets": ["UCI Machine Learning Repository", "MNIST", "OpenAI Gym", "Criteo dataset",
                     "Amazon reviews", "KDD Cup dataset"],
    },
    "stat.ML": {
        "subjects": ["Bayesian model", "Gaussian process", "probabilistic graphical model",
                     "variational autoencoder", "kernel method", "mixture model",
                     "statistical estimator", "causal inference framework"],
        "tasks": ["density estimation", "uncertainty quantification", "causal effect estimation",
                  "model selection", "hypothesis testing", "dimensionality reduction"],
        "methods": ["variational inference", "Markov chain Monte Carlo sampling",
                    "maximum likelihood estimation", "regularization", "cross-validation",
                    "bootstrap resampling", "expectation-maximization"],
        "datasets": ["UCI Machine Learning Repository", "synthetic simulation data",
                     "Boston Housing dataset", "clinical trial data"],
    },
    "cs.AI": {
        "subjects": ["knowledge graph reasoning system", "planning agent", "multi-agent system",
                     "expert system", "automated reasoning engine", "recommendation engine",
                     "hybrid symbolic-neural model", "conversational agent"],
        "tasks": ["automated planning", "knowledge graph completion", "game playing",
                  "multi-agent coordination", "commonsense reasoning", "decision making"],
        "methods": ["search algorithms", "logic-based reasoning", "attention-guided reasoning",
                    "reinforcement learning", "constraint satisfaction", "heuristic search"],
        "datasets": ["FreeBase", "ConceptNet", "OpenAI Gym", "StarCraft II Learning Environment"],
    },
}

TEMPLATES = [
    "In this paper, we propose a novel {subject} for the task of {task}. "
    "Traditional approaches to {task} often struggle to capture long-range dependencies "
    "present in real-world data, which limits their performance on complex benchmarks. "
    "To address this limitation, our approach leverages {method1} combined with {method2} "
    "to build a long-context representation of the input. We evaluate our model on the "
    "{dataset} dataset and show that it consistently outperforms strong baselines in terms "
    "of accuracy and generalization. Extensive ablation studies further demonstrate that "
    "the proposed {method1} module is critical for capturing dependencies across distant "
    "tokens, leading to more robust and interpretable predictions. Our results suggest that "
    "combining {subject} architectures with {method2} is a promising direction for {task} "
    "and related problems that require reasoning over long sequences.",

    "{task} remains a challenging problem due to the difficulty of modeling long-range "
    "contextual information. In this work, we introduce a {subject} that uses {method1} to "
    "learn a rich, long-context representation of the input data. Unlike prior methods that "
    "rely primarily on {method2}, our architecture is explicitly designed to scale to longer "
    "sequences without a significant loss in performance. We benchmark our system against "
    "several state-of-the-art baselines on the {dataset} dataset, achieving competitive or "
    "superior results across multiple evaluation metrics. We also present a detailed "
    "analysis of attention weights, revealing that the model learns to focus on semantically "
    "relevant, long-distance context when performing {task}. These findings highlight the "
    "importance of long-context representation learning for building robust {subject} systems.",

    "We present a new approach for {task} based on a {subject} enhanced with {method1}. "
    "Motivated by the observation that important contextual cues in scientific and natural "
    "language text are often spread across long spans, we design our model to explicitly "
    "capture such dependencies using stacked {method1} layers combined with {method2}. "
    "Experiments on the widely used {dataset} dataset demonstrate that our proposed method "
    "achieves substantial improvements over conventional baselines, particularly on long "
    "documents where context spans hundreds of tokens. We further discuss the computational "
    "trade-offs of long-context modeling and propose efficient training strategies that make "
    "the {subject} practical for real-world {task} applications.",

    "Scientific literature classification and related {task} problems require models that "
    "can effectively summarize information distributed across long passages of text. This "
    "paper proposes a {subject} that integrates {method1} with {method2} for improved "
    "long-context representation learning. We conduct extensive experiments on the {dataset} "
    "dataset and report significant gains in precision, recall and F1-score compared to "
    "classical machine learning baselines such as bag-of-words and TF-IDF based classifiers. "
    "Our qualitative analysis shows that the self-attention weights align well with human "
    "intuition about which parts of an abstract are most indicative of its subject area, "
    "confirming the model's ability to reason over long, information-dense text.",
]


def _pick(category: str, key: str, cross_prob: float = 0.35):
    """Pick a term for `key` mostly from `category`'s own vocab bank, but
    sometimes (cross_prob) borrow from a *related* category to reflect the
    genuine vocabulary overlap between neighbouring sub-fields (e.g. cs.LG
    / cs.AI / stat.ML all talk about 'attention mechanisms', 'optimization',
    etc.). This keeps the classification task realistically non-trivial."""
    if random.random() < cross_prob:
        other_cat = random.choice([c for c in CATEGORIES if c != category])
        return random.choice(VOCAB[other_cat][key])
    return random.choice(VOCAB[category][key])


def _make_abstract(category: str) -> str:
    template = random.choice(TEMPLATES)
    text = template.format(
        subject=_pick(category, "subjects", cross_prob=0.15),
        task=_pick(category, "tasks", cross_prob=0.30),
        method1=_pick(category, "methods", cross_prob=0.35),
        method2=_pick(category, "methods", cross_prob=0.35),
        dataset=_pick(category, "datasets", cross_prob=0.30),
    )
    return text


def _make_title(category: str) -> str:
    v = VOCAB[category]
    return f"A {random.choice(v['subjects']).title()} for {random.choice(v['tasks']).title()}"


def _inject_cross_category_noise(text: str, noise_ratio: float = 0.12) -> str:
    """Randomly swaps a fraction of words with terms from OTHER categories'
    vocabularies. Real scientific abstracts across related sub-fields
    (e.g. cs.CL / cs.LG / cs.AI) genuinely share a lot of vocabulary, so
    this keeps the synthetic dataset from being trivially/perfectly
    separable and gives a more realistic, non-saturated evaluation."""
    all_terms = []
    for v in VOCAB.values():
        for key in ("subjects", "tasks", "methods"):
            all_terms.extend(v[key])
    words = text.split()
    n_noise = int(len(words) * noise_ratio)
    idxs = random.sample(range(len(words)), min(n_noise, len(words)))
    for i in idxs:
        words[i] = random.choice(all_terms).split()[0]
    return " ".join(words)


def build_synthetic_dataset(n_per_class: int = 400) -> pd.DataFrame:
    rows = []
    for cat in CATEGORIES:
        for _ in range(n_per_class):
            abstract = _make_abstract(cat)
            abstract = _inject_cross_category_noise(abstract, noise_ratio=0.20)
            rows.append({
                "title": _make_title(cat),
                "abstract": abstract,
                "category": cat,
            })
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df


def load_real_kaggle_json(path_to_json: str, categories=CATEGORIES, per_class: int = 1000) -> pd.DataFrame:
    """
    Loader for the REAL Kaggle arXiv dataset
    (arxiv-metadata-oai-snapshot.json, one JSON object per line).
    Use this instead of build_synthetic_dataset() for the final submission.
    """
    rows, counts = [], {c: 0 for c in categories}
    with open(path_to_json, "r") as f:
        for line in f:
            if all(v >= per_class for v in counts.values()):
                break
            rec = json.loads(line)
            cats = rec.get("categories", "").split()
            primary = next((c for c in cats if c in categories), None)
            if primary and counts[primary] < per_class:
                rows.append({
                    "title": rec.get("title", "").strip().replace("\n", " "),
                    "abstract": rec.get("abstract", "").strip().replace("\n", " "),
                    "category": primary,
                })
                counts[primary] += 1
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    df = build_synthetic_dataset(n_per_class=400)
    out_path = os.path.join(out_dir, "arxiv_abstracts.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print(df["category"].value_counts())
