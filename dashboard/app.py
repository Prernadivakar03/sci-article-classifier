"""
app.py — Streamlit Analytical Dashboard
==========================================
Scientific Article Classification using Self-Attention Networks
with Long-Context Representation Learning

Run with:
    streamlit run app.py

Expects that ../src/train_evaluate.py has already been run once so that
../outputs/ (metrics.json, comparison_table.csv, *.png, preprocessed_dataset.csv)
and ../models/ (saved vectorizer + models) are populated.

Theming: this dashboard intentionally uses Streamlit's OWN native
light/dark theme switcher (hamburger menu, top-right -> Settings -> Theme).
All custom CSS below reads Streamlit's built-in CSS variables
(--text-color, --background-color, --secondary-background-color) so it
automatically matches whichever native theme the user has picked, instead
of shipping a second, competing theme system.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

try:
    from streamlit_option_menu import option_menu
    HAS_OPTION_MENU = True
except ImportError:
    HAS_OPTION_MENU = False

# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "..", "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
SRC_DIR = os.path.join(BASE_DIR, "..", "src")
DATA_PATH = os.path.join(OUT_DIR, "preprocessed_dataset.csv")
METRICS_PATH = os.path.join(OUT_DIR, "metrics.json")
COMPARISON_PATH = os.path.join(OUT_DIR, "comparison_table.csv")

sys.path.append(SRC_DIR)

st.set_page_config(
    page_title="Scientific Article Classification | NLP Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT = "#6C5CE7"
ACCENT_SOFT = "#8B7CFF"
ACCENT2 = "#12B8A0"
CHART_SEQUENCE = ["#6C5CE7", "#12B8A0", "#F2994A", "#EE5D8F", "#2F80ED", "#9B59B6"]

# ---------------------------------------------------------------------------
# Fonts + icon set (kept in its own tiny call, separate from the style block)
# ---------------------------------------------------------------------------
st.markdown(
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">'
    '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css">',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Global CSS — small, native-theme-aware, no competing dark/light system.
# Built with plain string concatenation (no f-string brace escaping) to
# keep this block simple to audit.
# ---------------------------------------------------------------------------
_CSS = """
<style>
html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

.block-container {
    padding-top: 1.6rem;
    padding-bottom: 2rem;
    max-width: 100%;
}

.hero-banner {
    background: linear-gradient(135deg, ACCENT_C 0%, ACCENT_SOFT_C 55%, ACCENT2_C 130%);
    padding: 26px 30px;
    border-radius: 16px;
    color: #ffffff;
    margin-bottom: 18px;
    box-shadow: 0 10px 26px rgba(108,92,231,0.28);
}
.hero-banner .eyebrow {
    font-size: 11.5px; font-weight: 700; letter-spacing: .12em; text-transform: uppercase;
    opacity: 0.85; margin-bottom: 6px;
}
.hero-banner h1 { font-size: 27px; font-weight: 800; margin: 0 0 6px 0; color: #ffffff !important; }
.hero-banner p  { font-size: 14.5px; opacity: 0.95; margin: 0; max-width: 760px; line-height: 1.5; }

.section-header {
    display: flex; align-items: center; gap: 9px;
    font-size: 17.5px; font-weight: 700; color: var(--text-color);
    margin-top: 6px; margin-bottom: 10px;
}
.section-header i { color: ACCENT_C; font-size: 16px; }
.section-header .bar { width: 4px; height: 16px; background: ACCENT_C; border-radius: 3px; }

.sub-note { font-size: 13px; color: var(--text-color); opacity: 0.65; margin-top: -6px; margin-bottom: 12px; }

.wow-card {
    background: var(--secondary-background-color);
    color: var(--text-color);
    padding: 18px 20px;
    border-radius: 14px;
    border: 1px solid rgba(128,128,128,0.2);
    margin-bottom: 14px;
}
.wow-card h4 { margin-top: 0; font-size: 13.5px; opacity: 0.7; font-weight: 600; }

.pipeline-row { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin-top: 4px; }
.pipe-chip {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.25);
    color: var(--text-color);
    padding: 7px 13px; border-radius: 9px; font-size: 12.5px; font-weight: 600;
}
.pipe-arrow { color: ACCENT_C; font-size: 15px; }

[data-testid="stMetric"] {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.2);
    padding: 12px 14px;
    border-radius: 12px;
}
[data-testid="stMetricLabel"] { font-weight: 600; opacity: 0.75; }

.stButton>button {
    background: linear-gradient(135deg, ACCENT_C, ACCENT_SOFT_C);
    color: white !important;
    border: none;
    border-radius: 9px;
    padding: 9px 22px;
    font-weight: 600;
}
.stButton>button:hover { filter: brightness(1.06); }

.wow-tag {
    background: rgba(108, 92, 231, 0.13);
    color: ACCENT_SOFT_C;
    border-radius: 8px;
    padding: 5px 12px 5px 10px;
    margin: 3px 4px 3px 0;
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 12.5px; font-weight: 600;
    border: 1px solid rgba(108, 92, 231, 0.28);
}
.wow-tag i { font-size: 11px; }

[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

.badge-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(18,184,160,0.13); color: ACCENT2_C;
    border: 1px solid rgba(18,184,160,0.3);
    padding: 4px 11px; border-radius: 20px; font-size: 12px; font-weight: 700;
}
</style>
"""
_CSS = (
    _CSS.replace("ACCENT_SOFT_C", ACCENT_SOFT)
        .replace("ACCENT2_C", ACCENT2)
        .replace("ACCENT_C", ACCENT)
)
st.markdown(_CSS, unsafe_allow_html=True)


def themed_fig(fig, height=340, legend=True):
    """Consistent, compact plotly styling that follows Streamlit's own
    light/dark setting via a transparent background (so the app background
    always shows through correctly, in either theme)."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12.5),
        margin=dict(l=10, r=10, t=48, b=10),
        height=height,
        title_font=dict(size=14.5, family="Inter, sans-serif"),
        showlegend=legend,
        colorway=CHART_SEQUENCE,
        hoverlabel=dict(font_family="Inter, sans-serif"),
    )
    return fig


def section_header(icon, text):
    st.markdown(
        '<div class="section-header"><span class="bar"></span>'
        '<i class="bi bi-' + icon + '"></i><span>' + text + '</span></div>',
        unsafe_allow_html=True,
    )


def page_title(icon, text):
    st.markdown(
        '<div style="display:flex;align-items:center;gap:11px;margin-bottom:14px;">'
        '<div style="width:38px;height:38px;border-radius:10px;'
        'background:linear-gradient(135deg,' + ACCENT + ',' + ACCENT_SOFT + ');'
        'display:flex;align-items:center;justify-content:center;">'
        '<i class="bi bi-' + icon + '" style="color:#fff;font-size:18px;"></i></div>'
        '<span style="font-size:23px;font-weight:800;">' + text + '</span></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------
@st.cache_data
def load_dataset():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None


@st.cache_data
def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return None


@st.cache_data
def load_comparison_table():
    if os.path.exists(COMPARISON_PATH):
        return pd.read_csv(COMPARISON_PATH, index_col=0)
    return None


@st.cache_resource
def load_baseline_artifacts():
    try:
        vec = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
        lr = joblib.load(os.path.join(MODEL_DIR, "logistic_regression.joblib"))
        le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
        return vec, lr, le
    except FileNotFoundError:
        return None, None, None


def img_path(name):
    p = os.path.join(OUT_DIR, name)
    return p if os.path.exists(p) else None


# ---------------------------------------------------------------------------
# Sidebar — brand + navigation (no theme toggle; use Streamlit's native
# Settings -> Theme switcher in the top-right hamburger menu instead)
# ---------------------------------------------------------------------------
NAV_OPTIONS = ["Overview", "Dataset Insights", "Model Performance",
               "Try a Prediction", "Analyze Article", "Conclusion"]
NAV_ICONS = ["speedometer2", "bar-chart-line", "cpu",
             "search", "file-earmark-richtext", "journal-check"]

with st.sidebar:
    st.markdown(
        '<div style="display:flex;align-items:center;gap:10px;padding-bottom:14px;'
        'margin-bottom:6px;border-bottom:1px solid rgba(128,128,128,0.25);">'
        '<div style="width:34px;height:34px;border-radius:9px;flex-shrink:0;'
        'background:linear-gradient(135deg,' + ACCENT + ',' + ACCENT_SOFT + ');'
        'display:flex;align-items:center;justify-content:center;color:#fff;font-size:16px;">'
        '<i class="bi bi-diagram-3-fill"></i></div>'
        '<div style="line-height:1.15;">'
        '<div style="font-weight:800;font-size:14.5px;">ArticleClassifier</div>'
        '<div style="font-size:11px;opacity:0.7;">Self-Attention NLP Suite</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    if HAS_OPTION_MENU:
        page = option_menu(
            menu_title=None,
            options=NAV_OPTIONS,
            icons=NAV_ICONS,
            default_index=0,
            styles={
                "container": {"padding": "2px 0", "background-color": "transparent"},
                "icon": {"font-size": "15px"},
                "nav-link": {
                    "font-size": "13.5px", "font-weight": "600", "border-radius": "9px",
                    "padding": "9px 12px", "margin": "2px 0",
                },
                "nav-link-selected": {"background-color": ACCENT, "color": "white"},
            },
        )
    else:
        page = st.radio("Navigate", NAV_OPTIONS, label_visibility="collapsed")

    st.markdown(
        '<div style="font-size:11.5px;opacity:0.65;line-height:1.6;'
        'border-top:1px solid rgba(128,128,128,0.25);padding-top:10px;margin-top:10px;">'
        '<b>Project</b><br>Scientific Article Classification using '
        'Self-Attention Networks with Long-Context Representation Learning'
        '<br><br><b>NLP Mini Project</b> &middot; 2026'
        '</div>',
        unsafe_allow_html=True,
    )

df = load_dataset()
metrics = load_metrics()
comparison_df = load_comparison_table()


# ===========================================================================
# PAGE 1 — OVERVIEW
# ===========================================================================
if page == "Overview":
    st.markdown("""
    <div class="hero-banner">
        <div class="eyebrow">NLP Mini Project &middot; 2026</div>
        <h1>Scientific Article Classification</h1>
        <p>Self-attention networks with long-context representation learning, benchmarked
        against classical TF-IDF baselines for arXiv-style abstract categorization.</p>
    </div>
    """, unsafe_allow_html=True)

    section_header("bullseye", "Problem Statement")
    st.markdown(
        """
        <div class="wow-card">
        Scientific abstracts are long, information-dense documents whose subject area often
        depends on cues scattered across the <b>entire</b> passage rather than a few keywords
        near the beginning. Classical bag-of-words / TF-IDF classifiers treat text as an
        unordered set of tokens and struggle to model such long-range dependencies. This
        project builds a <b>self-attention (Transformer-encoder) classifier</b> that learns a
        <b>long-context representation</b> of each abstract &mdash; letting every token attend
        directly to every other token regardless of distance &mdash; and compares it against
        classical ML baselines (Naive Bayes, Logistic Regression) trained on TF-IDF features.
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_header("collection", "Dataset Overview")
    if df is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Articles", f"{len(df):,}")
        c2.metric("Number of Classes", df["category"].nunique())
        avg_len = int(df["abstract"].astype(str).apply(lambda t: len(t.split())).mean())
        c3.metric("Avg. Abstract Length", f"{avg_len} words")
        c4.metric("Feature Technique", "TF-IDF / Self-Attention")
    else:
        st.warning("Run `python src/train_evaluate.py` first to generate the dataset & outputs.")

    st.markdown(
        """
        <div class="sub-note">Dataset source: simulated sample built to mirror the structure
        of the <a href="https://www.kaggle.com/datasets/Cornell-University/arxiv" target="_blank">
        Kaggle arXiv Dataset (Cornell University)</a>. For the final submission, replace
        <code>data/arxiv_abstracts.csv</code> with real abstracts downloaded from the link above
        (see <code>data/generate_dataset.py</code> &rarr; <code>load_real_kaggle_json()</code>).</div>
        """,
        unsafe_allow_html=True,
    )

    section_header("diagram-3", "Pipeline")
    st.markdown(
        """
        <div class="pipeline-row">
            <span class="pipe-chip">Raw Abstracts</span><span class="pipe-arrow">&rarr;</span>
            <span class="pipe-chip">Text Preprocessing</span><span class="pipe-arrow">&rarr;</span>
            <span class="pipe-chip">Feature Engineering</span><span class="pipe-arrow">&rarr;</span>
            <span class="pipe-chip">Self-Attention Transformer</span><span class="pipe-arrow">&rarr;</span>
            <span class="pipe-chip">Baseline Comparison</span><span class="pipe-arrow">&rarr;</span>
            <span class="pipe-chip">Evaluation</span><span class="pipe-arrow">&rarr;</span>
            <span class="pipe-chip">This Dashboard</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ===========================================================================
# PAGE 2 — DATASET INSIGHTS
# ===========================================================================
elif page == "Dataset Insights":
    page_title("bar-chart-line", "Dataset Insights")

    if df is None:
        st.error("Dataset not found. Run `python src/train_evaluate.py` first.")
    else:
        section_header("table", "Sample Records")
        st.dataframe(df[["title", "abstract", "category"]].sample(min(5, len(df))),
                     use_container_width=True, height=210)

        counts = df["category"].value_counts().reset_index()
        counts.columns = ["category", "count"]

        col1, col2 = st.columns(2)
        with col1:
            section_header("bar-chart-steps", "Category Distribution")
            fig = px.bar(counts, x="category", y="count", color="category", text="count")
            st.plotly_chart(themed_fig(fig, legend=False), use_container_width=True)

        with col2:
            section_header("pie-chart", "Category Share")
            fig2 = px.pie(counts, names="category", values="count", hole=0.55)
            st.plotly_chart(themed_fig(fig2), use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            section_header("diagram-2", "Category Weight (Treemap)")
            fig_tree = px.treemap(counts, path=["category"], values="count",
                                   color="count", color_continuous_scale="Purples")
            st.plotly_chart(themed_fig(fig_tree, legend=False), use_container_width=True)

        with col4:
            section_header("rulers", "Average Abstract Length by Category")
            df["_len"] = df["abstract"].astype(str).apply(lambda t: len(t.split()))
            avg_len_cat = df.groupby("category")["_len"].mean().reset_index().sort_values("_len")
            fig_avg = px.bar(avg_len_cat, x="_len", y="category", orientation="h",
                              color="_len", color_continuous_scale="Teal",
                              labels={"_len": "Avg. words", "category": ""})
            st.plotly_chart(themed_fig(fig_avg, legend=False), use_container_width=True)

        section_header("distribute-horizontal", "Abstract Length Distribution")
        fig3 = px.histogram(df, x="_len", nbins=30, color="category", marginal="box",
                             labels={"_len": "Number of words"})
        st.plotly_chart(themed_fig(fig3, height=380), use_container_width=True)

        section_header("cloud", "Word Cloud of Preprocessed Abstracts")
        wc_img = img_path("wordcloud.png")
        if wc_img:
            st.image(wc_img, use_container_width=True)
        else:
            st.info("Word cloud image not found — run the training pipeline first.")

        section_header("type", "Most Frequent Terms per Category")
        cat_choice = st.selectbox("Choose a category", sorted(df["category"].unique()))
        subset_text = " ".join(df.loc[df["category"] == cat_choice, "clean"].astype(str))
        top_words = Counter(subset_text.split()).most_common(15)
        if top_words:
            words, freqs = zip(*top_words)
            fig4 = px.bar(x=list(freqs), y=list(words), orientation="h",
                          labels={"x": "Frequency", "y": "Term"},
                          color=list(freqs), color_continuous_scale="Purples")
            fig4.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(themed_fig(fig4, height=420, legend=False), use_container_width=True)

# ===========================================================================
# PAGE 3 — MODEL PERFORMANCE
# ===========================================================================
elif page == "Model Performance":
    page_title("cpu", "Model Performance & Comparison")

    if comparison_df is None:
        st.error("Metrics not found. Run `python src/train_evaluate.py` first.")
    else:
        section_header("card-checklist", "Model Comparison Table")
        st.dataframe(comparison_df.style.highlight_max(axis=0, color="#12B8A0"),
                     use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            section_header("bar-chart", "Metric Comparison")
            comp_long = comparison_df.reset_index().melt(
                id_vars="Model", var_name="Metric", value_name="Score")
            fig = px.bar(comp_long, x="Model", y="Score", color="Metric", barmode="group")
            st.plotly_chart(themed_fig(fig), use_container_width=True)

        with col2:
            section_header("hexagon", "Model Profile (Radar)")
            fig_radar = go.Figure()
            categories = list(comparison_df.columns)
            for model_name in comparison_df.index:
                values = comparison_df.loc[model_name, categories].tolist()
                fig_radar.add_trace(go.Scatterpolar(
                    r=values + values[:1],
                    theta=categories + categories[:1],
                    fill="toself",
                    name=model_name,
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            )
            st.plotly_chart(themed_fig(fig_radar), use_container_width=True)

        if "Self-Attention Transformer" not in comparison_df.index:
            st.info(
                "The Self-Attention Transformer row will appear here once TensorFlow is "
                "installed (`pip install tensorflow`) and `src/train_evaluate.py` is re-run "
                "— it is skipped automatically when TensorFlow is unavailable so the baseline "
                "results above are still produced."
            )

        st.markdown("---")
        section_header("grid-3x3", "Confusion Matrices & ROC Curves")
        model_pick = st.selectbox("Select model", list(comparison_df.index))
        cm_file = f"confusion_matrix_{model_pick.lower().replace(' ', '_')}.png"
        cm_img = img_path(cm_file)
        roc_file = f"roc_curve_{model_pick.lower().replace(' ', '_')}.png"
        roc_img = img_path(roc_file)

        c1, c2 = st.columns(2)
        with c1:
            if cm_img:
                st.image(cm_img, caption=f"Confusion Matrix — {model_pick}", use_container_width=True)
            else:
                st.info("Confusion matrix image not found for this model.")
        with c2:
            if roc_img:
                st.image(roc_img, caption=f"ROC Curve — {model_pick}", use_container_width=True)
            else:
                st.info("ROC curve image not found for this model.")

        if img_path("self_attention_training_curve.png"):
            st.markdown("---")
            section_header("graph-up", "Self-Attention Model — Training Curves")
            c3, c4 = st.columns(2)
            with c3:
                st.image(img_path("self_attention_training_curve.png"), use_container_width=True)
            with c4:
                st.image(img_path("self_attention_loss_curve.png"), use_container_width=True)

# ===========================================================================
# PAGE 4 — TRY A PREDICTION
# ===========================================================================
elif page == "Try a Prediction":
    page_title("search", "Try a Live Prediction")
    st.markdown(
        '<div class="sub-note">Paste a scientific abstract below. The TF-IDF + Logistic '
        'Regression baseline model will predict its category. To use the trained '
        'self-attention model instead, load <code>models/self_attention_model.keras</code> '
        'with TensorFlow — see README.</div>',
        unsafe_allow_html=True,
    )

    vec, lr, le = load_baseline_artifacts()
    default_text = (
        "We propose a novel transformer encoder that leverages self-attention "
        "to capture long-context dependencies in scientific abstracts, "
        "improving classification accuracy over TF-IDF baselines."
    )
    user_text = st.text_area("Enter an abstract:", value=default_text, height=170)

    if st.button("Predict Category", type="primary"):
        if vec is None or lr is None:
            st.error("Trained baseline model not found. Run `python src/train_evaluate.py` first.")
        else:
            from preprocessing import preprocess_text

            cleaned = preprocess_text(user_text)
            X = vec.transform([cleaned])
            probs = lr.predict_proba(X)[0]
            pred_idx = int(np.argmax(probs))
            pred_label = le.inverse_transform([pred_idx])[0]

            st.markdown(
                '<span class="badge-pill"><i class="bi bi-check-circle-fill"></i> '
                'Predicted category: ' + str(pred_label) + '</span>',
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)

            prob_df = pd.DataFrame({"Category": le.classes_, "Probability": probs}).sort_values(
                "Probability", ascending=False
            )
            fig = px.bar(prob_df, x="Category", y="Probability", color="Category")
            st.plotly_chart(themed_fig(fig, legend=False), use_container_width=True)


# ===========================================================================
# PAGE — ANALYZE ARTICLE (the "WOW" page)
# ===========================================================================
elif page == "Analyze Article":
    page_title("file-earmark-richtext", "Analyze Article — Full NLP Breakdown")
    st.markdown(
        '<div class="sub-note">Upload a PDF, or paste text, to get a complete breakdown: '
        'prediction with confidence, document statistics, a "why this prediction" view, '
        'keyword extraction, and a short-vs-long-context demo.</div>',
        unsafe_allow_html=True,
    )

    from preprocessing import preprocess_text
    from document_analysis import (
        compute_document_stats, chunk_text, extract_keywords,
        chunk_importance_lr, short_vs_long_context_predict,
    )

    vec, lr, le = load_baseline_artifacts()

    tab1, tab2 = st.tabs(["Upload PDF", "Paste Text"])
    raw_text, num_pages = None, None

    with tab1:
        uploaded_pdf = st.file_uploader("Upload a scientific article (PDF)", type=["pdf"])
        if uploaded_pdf is not None:
            from pdf_utils import extract_text_from_pdf
            raw_text, num_pages = extract_text_from_pdf(uploaded_pdf)
            st.success(f"Extracted {len(raw_text.split())} words from {num_pages} page(s).")

    with tab2:
        pasted = st.text_area("Or paste abstract / article text here:", height=190)
        if pasted.strip():
            raw_text = pasted

    if raw_text and vec is not None and lr is not None:
        if st.button("Analyze Article", type="primary"):
            clean = preprocess_text(raw_text)

            # ---- 1. Prediction + confidence ----
            X = vec.transform([clean])
            probs = lr.predict_proba(X)[0]
            pred_idx = int(np.argmax(probs))
            pred_label = le.inverse_transform([pred_idx])[0]

            section_header("bullseye", "Prediction")
            st.markdown(
                '<span class="badge-pill"><i class="bi bi-check-circle-fill"></i> '
                + str(pred_label) + ' &middot; ' + f"{probs[pred_idx]*100:.1f}"
                + '% confidence</span>',
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)
            prob_df = pd.DataFrame({"Category": le.classes_, "Confidence": probs}).sort_values(
                "Confidence", ascending=False
            )
            fig = px.bar(prob_df, x="Confidence", y="Category", orientation="h",
                         color="Confidence", color_continuous_scale="Purples")
            fig.update_layout(xaxis_tickformat=".0%")
            st.plotly_chart(themed_fig(fig, legend=False), use_container_width=True)

            # ---- 2. Document insights ----
            section_header("clipboard-data", "Document Insights")
            stats = compute_document_stats(raw_text, num_pages)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Pages", stats["pages"])
            c2.metric("Words", f"{stats['words']:,}")
            c3.metric("Sentences", stats["sentences"])
            c4.metric("Est. Reading Time", f"{stats['reading_time_min']} min")

            # ---- 3. Why this prediction? ----
            section_header("lightbulb", "Why This Prediction?")
            chunks = chunk_text(raw_text, chunk_size_words=120)
            clean_chunks = [preprocess_text(c) for c in chunks]
            importance = chunk_importance_lr(chunks, clean_chunks, vec, lr, pred_idx)

            imp_df = pd.DataFrame({
                "Chunk": [f"Chunk {i+1}" for i in range(len(chunks))],
                "Importance": importance,
                "Preview": [c[:150] + "..." if len(c) > 150 else c for c in chunks],
            }).sort_values("Importance", ascending=False)

            fig2 = px.bar(imp_df, x="Importance", y="Chunk", orientation="h",
                          hover_data=["Preview"], color="Importance",
                          color_continuous_scale="Teal")
            fig2.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(themed_fig(fig2, height=380, legend=False), use_container_width=True)

            with st.expander("View most influential chunk's text"):
                top_chunk_idx = int(np.argmax(importance))
                st.write(chunks[top_chunk_idx])

            # ---- 4. Keyword extraction ----
            section_header("key", "Key Concepts Detected")
            keywords = extract_keywords(clean, vec, top_n=12)
            if keywords:
                tags_html = " ".join(
                    "<span class='wow-tag'><i class='bi bi-hash'></i>" + str(kw) + "</span>"
                    for kw, _ in keywords
                )
                st.markdown(tags_html, unsafe_allow_html=True)
            else:
                st.info("No distinctive keywords found (try a longer article).")

            # ---- 5. Short vs Long context ----
            section_header("arrow-left-right", "Short Context vs Long Context")
            comparison = short_vs_long_context_predict(clean, vec, lr, le, short_word_limit=80)
            colA, colB = st.columns(2)
            with colA:
                st.markdown(
                    '<div class="wow-card"><h4>SHORT CONTEXT — first '
                    + str(comparison["short"]["n_words"]) + ' words</h4>'
                    + '<div style="font-size:15px;font-weight:700;margin-bottom:4px;">'
                    + str(comparison["short"]["label"]) + '</div>'
                    + '<div style="opacity:0.65;font-size:13px;">Confidence: '
                    + f"{comparison['short']['confidence']*100:.1f}" + '%</div></div>',
                    unsafe_allow_html=True,
                )
            with colB:
                st.markdown(
                    '<div class="wow-card"><h4>LONG CONTEXT — all '
                    + str(comparison["long"]["n_words"]) + ' words</h4>'
                    + '<div style="font-size:15px;font-weight:700;margin-bottom:4px;">'
                    + str(comparison["long"]["label"]) + '</div>'
                    + '<div style="opacity:0.65;font-size:13px;">Confidence: '
                    + f"{comparison['long']['confidence']*100:.1f}" + '%</div></div>',
                    unsafe_allow_html=True,
                )
            if (comparison["short"]["label"] != comparison["long"]["label"]
                    or abs(comparison["long"]["confidence"] - comparison["short"]["confidence"]) > 0.05):
                st.info(
                    "Notice how the prediction/confidence shifts once the model sees the "
                    "FULL article instead of just the first few sentences — this is exactly "
                    "why long-context representation learning (self-attention over the whole "
                    "sequence) matters for scientific text."
                )
    elif raw_text is None:
        st.info("Upload a PDF or paste some text above, then click Analyze.")
    else:
        st.error("Trained baseline model not found. Run `python src/train_evaluate.py` first.")


# ===========================================================================
# PAGE 5 — CONCLUSION
# ===========================================================================
elif page == "Conclusion":
    page_title("journal-check", "Conclusion & Discussion")

    section_header("check2-circle", "Key Findings")
    st.markdown(
        """
        <div class="wow-card">
        <ul style="margin:0; padding-left:20px; line-height:1.75;">
        <li>TF-IDF based <b>Logistic Regression</b> clearly outperforms <b>Naive Bayes</b> as a
        classical baseline, since it can weigh discriminative n-grams rather than assuming
        feature independence.</li>
        <li>The <b>Self-Attention Transformer</b> model is expected to further improve
        performance on longer abstracts because self-attention lets every token attend to
        every other token, directly modeling long-range dependencies that TF-IDF and
        bag-of-words features cannot capture.</li>
        <li>Categories with highly overlapping vocabulary (e.g. <code>cs.LG</code>,
        <code>cs.AI</code>, <code>stat.ML</code>) are the hardest to separate — visible as the
        largest off-diagonal blocks in their confusion matrices.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_header("exclamation-triangle", "Challenges Faced")
    st.markdown(
        """
        <div class="wow-card">
        <ul style="margin:0; padding-left:20px; line-height:1.75;">
        <li>Real scientific abstracts from closely related sub-fields share a lot of technical
        vocabulary, making the classification boundary inherently noisy.</li>
        <li>Long sequences increase the computational cost of self-attention (quadratic in
        sequence length), requiring careful tuning of <code>max_len</code>, embedding size and
        number of attention heads.</li>
        <li>Class imbalance and out-of-vocabulary technical terms need careful
        tokenizer/vocabulary-size tuning.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_header("rocket-takeoff", "Future Scope")
    st.markdown(
        """
        <div class="wow-card">
        <ul style="margin:0; padding-left:20px; line-height:1.75;">
        <li>Fine-tune a pretrained long-context transformer (e.g. Longformer, BigBird, or
        SciBERT) instead of training self-attention from scratch.</li>
        <li>Extend to multi-label classification, since a single article can genuinely belong
        to more than one arXiv category.</li>
        <li>Add explainability (attention-weight visualization) to show which parts of the
        abstract most influenced the prediction.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_header("globe2", "Applications")
    st.markdown(
        """
        <div class="wow-card">
        <ul style="margin:0; padding-left:20px; line-height:1.75;">
        <li>Automatic tagging/routing of papers on preprint servers (arXiv, bioRxiv) and
        digital libraries.</li>
        <li>Literature-review assistants that group related papers by topic.</li>
        <li>Journal/conference submission systems that suggest reviewers based on predicted
        subject area.</li>
        <li>Research trend analysis and citation-recommendation systems.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if metrics:
        section_header("bar-chart-fill", "Final Summary")
        st.json(metrics["results"])

    st.markdown(
        """
        <div class="wow-card" style="border-left:4px solid #6C5CE7;">
        <b>Overall Conclusion:</b> Self-attention based, long-context representation learning
        provides a principled way to classify long, technical documents like scientific
        abstracts by directly modeling dependencies across the entire sequence, offering a
        meaningful upgrade over classical bag-of-words / TF-IDF pipelines — at the cost of
        higher computational and data requirements.
        </div>
        """,
        unsafe_allow_html=True,
    )
