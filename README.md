# Fake News Detector

A beginner-friendly machine learning project that reads a news article's text and predicts whether it is **REAL** or **FAKE**, using TF-IDF text vectorization and a Passive Aggressive Classifier. Includes a Streamlit web UI so you can paste an article and get an instant verdict.

## How it works

1. **Text → numbers:** each article's text is converted into a TF-IDF vector, which weights words by how important they are relative to the whole dataset.
2. **Learn:** a **Passive Aggressive Classifier** is trained on ~7,800 labeled news articles (80% train / 20% test).
3. **Predict:** for new text, the same vectorizer turns it into a vector and the classifier returns `REAL` or `FAKE`, along with a confidence score.

## What's inside

```
├── PROJECT_PLAN.md      # in-depth step-by-step build plan & pitfalls
├── requirements.txt     # Python dependencies
├── data/
│   └── news.csv         # the dataset (see below)
├── models/
│   ├── tfidf.pkl        # saved TF-IDF vectorizer
│   └── pac_model.pkl    # saved classifier
├── train.py             # load data → train → save model artifacts
├── predict.py           # load artifacts → classify a single input
└── app.py               # Streamlit web UI
```

> Note: some of these files (`train.py`, `predict.py`, `app.py`, the `data/` and `models/` folders) are created as you follow the plan in [PROJECT_PLAN.md](PROJECT_PLAN.md).

## Getting started

### 1. Get the dataset

Download `news.csv` (~7,796 rows, columns: `id`, `title`, `text`, `label`) and save it to `data/news.csv`. Good sources:

- **Kaggle** — search "fake or real news" (`fake_or_real_news.csv`)
- **GitHub** — many repos mirror the same dataset
- The **Google Drive** link in the [original DataFlair tutorial](https://data-flair.training/blogs/advanced-python-project-detecting-fake-news/)

### 2. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> `scikit-learn` is the correct package name. The older `sklearn` name is deprecated.

### 3. Train the model

```bash
python train.py
```

This prints the accuracy (~92–93%) and saves both model files to `models/`.

### 4. Classify a single input

```bash
python predict.py
```

Edit the `sample` string in the script to test your own headline or article.

### 5. Run the web app

```bash
streamlit run app.py
```

Opens a browser tab at `http://localhost:8501` where you can paste an article and get a verdict.

## Example

```
Input:  "Scientists confirm a new cure for all diseases overnight."
Output: FAKE   (confidence: -0.842)
```

## Limitations

- The dataset is an **older, US-politics-heavy snapshot**, so the model does **not** generalize well to arbitrary modern articles — real BBC/NYT pieces may get flagged `FAKE`. This is a data limitation, not a code bug.
- The vectorizer only knows words it saw during training, so unfamiliar phrasing gets weighted differently.

## Stretch goals

- Surface the confidence score as a human-readable percentage.
- Try a more modern/balanced dataset, or add a third class (satire).
- Compare against Logistic Regression or an SVM.
- Use cross-validation for a more reliable accuracy estimate.
- Deploy via Streamlit Community Cloud / Hugging Face Spaces.

## Credits

Built as a learning exercise based on the [DataFlair — Detecting Fake News with Python](https://data-flair.training/blogs/advanced-python-project-detecting-fake-news/) tutorial. See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the full build plan.
