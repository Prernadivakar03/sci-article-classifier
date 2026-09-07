# Scientific Article Classification using Self-Attention Networks with Long-Context Representation Learning

## The APP is LIVE : https://sci-article-classifier-3.streamlit.app/

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
└── README.md                   # this file
```

## ⚙️ Setup
```bash
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


## 🚀 Deployment

The Streamlit dashboard is deployed and publicly accessible online.

## 🌐 Live Demo

### Scientific Article Classification App:
### https://sci-article-classifier-3.streamlit.app/

The deployed application allows users to:

Explore the scientific article dataset
View model performance and evaluation results
Enter a scientific abstract for live classification
Analyze articles using the NLP pipeline
View prediction confidence and other analytical insights

☁️ Deployment Platform

The application is deployed using Streamlit Community Cloud.
The deployment is connected to this GitHub project, so the Streamlit dashboard can be accessed directly through the live URL above.

Note: The deployed version uses the trained model artifacts and project files available in the repository. If the project is updated, the deployed application can be redeployed to reflect the latest changes.
