"""
train_evaluate.py
--------------------
End-to-end pipeline:
    1. Load dataset (data/arxiv_abstracts.csv)
    2. Text preprocessing
    3. Feature engineering (TF-IDF for baselines, tokenizer+padding for the
       self-attention model)
    4. Train baseline models (Naive Bayes, Logistic Regression)
    5. Train the Self-Attention (Transformer) model  [requires tensorflow;
       the script degrades gracefully and still produces the baseline
       results + plots if tensorflow is not installed]
    6. Evaluate all models: accuracy, precision, recall, F1, confusion
       matrix, ROC curve (one-vs-rest, multi-class)
    7. Save all plots to ../outputs/, a metrics.json and
       comparison_table.csv that are consumed by the Streamlit dashboard
       (dashboard/app.py)

Run with:  python3 train_evaluate.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_curve, auc,
)

from preprocessing import preprocess_series
from feature_engineering import build_tfidf_features, build_bow_features
from baseline_model import train_naive_bayes, train_logistic_regression

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "arxiv_abstracts.csv")
OUT_DIR = os.path.join(BASE_DIR, "..", "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42
sns.set_style("whitegrid")


# ---------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["abstract", "category"]).reset_index(drop=True)
    return df


def plot_class_distribution(df):
    plt.figure(figsize=(7, 5))
    order = df["category"].value_counts().index
    sns.countplot(data=df, y="category", order=order, palette="viridis")
    plt.title("Class Distribution of Scientific Article Categories")
    plt.xlabel("Number of Articles")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "class_distribution.png"), dpi=150)
    plt.close()


def plot_abstract_length_distribution(df):
    lengths = df["abstract"].astype(str).apply(lambda t: len(t.split()))
    plt.figure(figsize=(7, 5))
    sns.histplot(lengths, bins=30, kde=True, color="steelblue")
    plt.title("Abstract Length Distribution (word count) — motivates\nthe need for LONG-CONTEXT representation learning")
    plt.xlabel("Number of words per abstract")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "abstract_length_distribution.png"), dpi=150)
    plt.close()


def plot_wordcloud(df):
    text = " ".join(df["clean"].tolist())
    try:
        from wordcloud import WordCloud
        wc = WordCloud(width=1000, height=500, background_color="white",
                        colormap="viridis", max_words=150).generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title("Word Cloud of Preprocessed Abstracts")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "wordcloud.png"), dpi=150)
        plt.close()
    except ImportError:
        # Offline fallback: frequency bar chart instead of a word cloud image.
        from collections import Counter
        counts = Counter(text.split()).most_common(25)
        words, freqs = zip(*counts)
        plt.figure(figsize=(9, 6))
        sns.barplot(x=list(freqs), y=list(words), palette="mako")
        plt.title("Top 25 Most Frequent Terms (wordcloud package not installed —\n"
                   "install `wordcloud` to render an actual word-cloud image)")
        plt.xlabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "wordcloud.png"), dpi=150)
        plt.close()


def evaluate_model(name, y_true, y_pred, y_score, class_names, results):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    results[name] = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
    }
    print(f"\n=== {name} ===")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix — {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    plt.savefig(os.path.join(OUT_DIR, f"confusion_matrix_{safe_name}.png"), dpi=150)
    plt.close()

    # ROC curve (one-vs-rest, multi-class)
    if y_score is not None:
        y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
        plt.figure(figsize=(7, 6))
        for i, cname in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{cname} (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--", linewidth=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve (One-vs-Rest) — {name}")
        plt.legend(loc="lower right", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"roc_curve_{safe_name}.png"), dpi=150)
        plt.close()

    return cm


def main():
    print("Loading dataset ...")
    df = load_data()

    print("Preprocessing text ...")
    df["clean"] = preprocess_series(df["abstract"])
    df.to_csv(os.path.join(OUT_DIR, "preprocessed_dataset.csv"), index=False)

    plot_class_distribution(df)
    plot_abstract_length_distribution(df)
    plot_wordcloud(df)

    le = LabelEncoder()
    y = le.fit_transform(df["category"])
    class_names = list(le.classes_)

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["clean"], y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    results = {}

    # ---------------- Baseline 1: TF-IDF + Naive Bayes ----------------
    Xtr_tfidf, Xte_tfidf, tfidf_vec = build_tfidf_features(X_train_text, X_test_text)
    nb = train_naive_bayes(Xtr_tfidf, y_train)
    nb_pred = nb.predict(Xte_tfidf)
    nb_score = nb.predict_proba(Xte_tfidf)
    evaluate_model("Naive Bayes (TF-IDF)", y_test, nb_pred, nb_score, class_names, results)

    # ---------------- Baseline 2: TF-IDF + Logistic Regression ----------------
    lr = train_logistic_regression(Xtr_tfidf, y_train)
    lr_pred = lr.predict(Xte_tfidf)
    lr_score = lr.predict_proba(Xte_tfidf)
    evaluate_model("Logistic Regression (TF-IDF)", y_test, lr_pred, lr_score, class_names, results)

    import joblib
    joblib.dump(tfidf_vec, os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
    joblib.dump(nb, os.path.join(MODEL_DIR, "naive_bayes.joblib"))
    joblib.dump(lr, os.path.join(MODEL_DIR, "logistic_regression.joblib"))
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.joblib"))

    # ---------------- Main model: Self-Attention (Transformer) ----------------
    try:
        import tensorflow as tf
        from feature_engineering import build_sequence_features
        from self_attention_model import build_self_attention_classifier

        MAX_LEN = 200
        VOCAB_SIZE = 10000

        Xtr_seq, Xte_seq, tokenizer = build_sequence_features(
            X_train_text, X_test_text, vocab_size=VOCAB_SIZE, max_len=MAX_LEN
        )

        model = build_self_attention_classifier(
            vocab_size=VOCAB_SIZE, max_len=MAX_LEN, embed_dim=64,
            num_heads=4, ff_dim=128, num_transformer_blocks=2,
            num_classes=len(class_names),
        )

        history = model.fit(
            Xtr_seq, y_train,
            validation_split=0.15,
            epochs=10,
            batch_size=32,
            verbose=2,
        )

        # Training curves
        plt.figure(figsize=(7, 5))
        plt.plot(history.history["accuracy"], label="Train Accuracy")
        plt.plot(history.history["val_accuracy"], label="Val Accuracy")
        plt.title("Self-Attention Model — Training vs Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "self_attention_training_curve.png"), dpi=150)
        plt.close()

        plt.figure(figsize=(7, 5))
        plt.plot(history.history["loss"], label="Train Loss")
        plt.plot(history.history["val_loss"], label="Val Loss")
        plt.title("Self-Attention Model — Training vs Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "self_attention_loss_curve.png"), dpi=150)
        plt.close()

        sa_score = model.predict(Xte_seq)
        sa_pred = np.argmax(sa_score, axis=1)
        evaluate_model("Self-Attention Transformer", y_test, sa_pred, sa_score, class_names, results)

        model.save(os.path.join(MODEL_DIR, "self_attention_model.keras"))
        with open(os.path.join(MODEL_DIR, "tokenizer.json"), "w") as f:
            f.write(tokenizer.to_json())

    except ImportError as e:
        print("\n[INFO] TensorFlow is not installed in this environment, so the "
              "Self-Attention Transformer model was skipped.\n"
              "Run `pip install tensorflow` and re-run this script to train and "
              "evaluate the main deep-learning model.\n", e)

    # ---------------- Comparison table ----------------
    comp_df = pd.DataFrame(results).T
    comp_df.index.name = "Model"
    comp_df.to_csv(os.path.join(OUT_DIR, "comparison_table.csv"))
    print("\nComparison table:\n", comp_df)

    plt.figure(figsize=(8, 5))
    comp_df[["accuracy", "precision", "recall", "f1_score"]].plot(kind="bar", figsize=(9, 5))
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "model_comparison.png"), dpi=150)
    plt.close()

    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump({
            "results": results,
            "class_names": class_names,
            "n_samples": int(len(df)),
            "n_classes": len(class_names),
        }, f, indent=2)

    print("\nAll outputs saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
