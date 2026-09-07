# Scientific Article Classification using Self-Attention Networks with Long-Context Representation Learning

## 📁 Folder Structure
```
nlp_mini_project/
├── data/
│   ├── generate_dataset.py     # builds the offline demo dataset / loads real Kaggle data
│   └── arxiv_abstracts.csv     # generated dataset (title, abstract, category)
├── src/
│   ├── preprocessing.py        # text cleaning, tokenization, stopwords, lemmatization
│   ├── feature_engineering.py  # BoW, TF-IDF, tokenizer+padding, GloVe embedding matrix
│   ├── baseline_model.py       # Naive Bayes & Logistic Regression
│   ├── self_attention_model.py # Transformer-encoder self-attention classifier (Keras)
│   └── train_evaluate.py       # MAIN SCRIPT: runs the whole pipeline end-to-end
├── outputs/                    # generated plots, metrics.json, comparison_table.csv
├── models/                     # saved trained models + vectorizer + tokenizer
├── dashboard/
│   └── app.py                  # Streamlit analytical dashboard
├── requirements.txt
├── REPORT.md                   # theory + report content for the handwritten submission
├── pink_cover_page.html        # printable pink front/back cover page template
└── README.md                   # this file
```

## ⚙️ Setup
```bash
cd nlp_mini_project
python -m venv venv
source venv/bin/activate        # (Windows: venv\Scripts\activate)
pip install -r requirements.txt

# (first time only, if you want NLTK's higher-quality stopwords/lemmatizer)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## ▶️ Run the pipeline
```bash
# 1. Generate the dataset (or replace with the real Kaggle arXiv CSV, see below)
python data/generate_dataset.py

# 2. Run preprocessing + feature engineering + train + evaluate ALL models
#    (produces every plot / metric / model file needed for the report & dashboard)
python src/train_evaluate.py
```
This creates:
- `outputs/*.png` — class distribution, abstract-length histogram, word cloud,
  confusion matrices, ROC curves, training curves, model comparison chart
- `outputs/metrics.json`, `outputs/comparison_table.csv`
- `models/*.joblib`, `models/self_attention_model.keras` (if TensorFlow is installed)

> **Note:** if TensorFlow isn't installed, the script automatically skips the
> Self-Attention model and still produces the two baseline models + all
> plots, so you always get a complete run. Install `tensorflow` and re-run
> to include the main deep-learning model in your results/screenshots.

## 📊 Launch the dashboard
```bash
cd dashboard
streamlit run app.py
```
Then open the local URL Streamlit prints (usually http://localhost:8501).

The dashboard has 5 pages (sidebar):
1. **Overview** — project title, problem statement, dataset overview, key stats
2. **Dataset Insights** — class distribution, abstract-length histogram, word cloud, top terms per category
3. **Model Performance** — comparison table, interactive metric chart, confusion matrices, ROC curves, training curves
4. **Try a Prediction** — paste a new abstract and get a live predicted category
5. **Conclusion** — key findings, challenges, future scope, applications, final conclusion

## 🗃️ Using the REAL Kaggle dataset (recommended before final submission)
1. Download `arxiv-metadata-oai-snapshot.json` from
   https://www.kaggle.com/datasets/Cornell-University/arxiv
2. In `data/generate_dataset.py`, call:
   ```python
   from generate_dataset import load_real_kaggle_json
   df = load_real_kaggle_json("path/to/arxiv-metadata-oai-snapshot.json", per_class=800)
   df.to_csv("arxiv_abstracts.csv", index=False)
   ```
3. Re-run `python src/train_evaluate.py` — no other code changes required.

## ✅ Submission Checklist (as per instructions given)
**Handwritten (on plain A4 sheets):**
- [ ] Problem Statement (REPORT.md §1)
- [ ] Theory of all preprocessing techniques (REPORT.md §2)
- [ ] Theory of all ML/other algorithms used — Naive Bayes, Logistic
      Regression, Self-Attention/Transformer (REPORT.md §3–4)
- [ ] Theory of evaluation metrics (REPORT.md §5)

**Printed (attach after handwritten pages):**
- [ ] Code printouts (`src/*.py`, `data/generate_dataset.py`, `dashboard/app.py`)
- [ ] Output printouts (screenshots of terminal output + all PNGs in `outputs/`)
- [ ] Conclusion (REPORT.md §7, plus §6 results discussion)
- [ ] Dashboard screenshots (each of the 5 Streamlit pages)

**Cover pages (light pink paper, front & back):**
- [ ] Use `pink_cover_page.html` — open in a browser and print on **light
      pink** paper (not dark/fluorescent pink) — fields in the top-right
      corner: Name, Roll No, Div, Class, Topic
- [ ] Bind all sheets with a **blue ribbon** (no stapler pins)

## 👥 Choosing your project partner
This is a two-member submission. I can't pick a real classmate for you —
that's a decision to make with someone in your class (check the roll-number
→ topic mapping in your Excel sheet to see who else was assigned this exact
topic, since pairs are often formed with a topic-mate, or ask a friend and
confirm with your instructor whether partner choice is free or fixed by
roll number). Once decided, just fill both names into the pink cover page.
