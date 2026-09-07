"""
baseline_model.py
-------------------
Classical ML baselines trained on TF-IDF features, used purely as a
COMPARISON POINT against the Self-Attention / Transformer model that is
the actual topic of this mini project.

Models:
    - Multinomial Naive Bayes
    - Logistic Regression (One-vs-Rest, multi-class)
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression


def train_naive_bayes(X_train, y_train):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(X_train, y_train, max_iter=1000):
    model = LogisticRegression(max_iter=max_iter, n_jobs=-1)
    model.fit(X_train, y_train)
    return model
