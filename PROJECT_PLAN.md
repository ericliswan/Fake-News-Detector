# Detecting Fake News — Project Plan

**Project type:** Beginner-friendly ML classifier (TF-IDF + Passive Aggressive Classifier)
**Language/stack:** Python 3.11, scikit-learn, pandas, numpy
**Inspiration:** [DataFlair — Detecting Fake News with Python](https://data-flair.training/blogs/advanced-python-project-detecting-fake-news/)
**Outcome:** A classifier that reads article text and predicts `REAL` or `FAKE`, with a simple UI to test it live.

---

## 1. Overview / What we're building

- **Input:** a news article's text.
- **Output:** a prediction of `REAL` or `FAKE`, plus a confidence score.
- **How:** convert text into TF-IDF numeric vectors, train a **Passive Aggressive Classifier** on labeled examples, then wrap it in a small **Streamlit** web app so you can paste a headline/article and get a verdict.

### Key concepts (know these before you start)
| Term | Meaning |
|------|---------|
| **TF-IDF** | Text → numbers. Weights words by how important they are in a document *relative to the whole corpus* (rare-but-significant words get higher weight; "the", "and" get ~0). |
| **Passive Aggressive Classifier** | An online learning algorithm — if a prediction is correct it stays "passive" (does little); on a mistake it "aggressively" updates its weights. Fast and works well for text. |
| **Train/test split** | Train on 80% of data, evaluate on the untouched 20% to see how well it generalizes. |
| **Confusion matrix** | Table of correct/incorrect predictions per class (TP/FP/TN/FN). |

---

## 2. Environment Setup

**Goal:** a reproducible Python environment with dependencies installed.

```bash
# 1. (Recommended) Create + activate a virtual environment so deps don't pollute your system
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# Windows: .venv\Scripts\activate

# 2. Install dependencies
# NOTE: the article says `sklearn` — that package name is deprecated.
# Use `scikit-learn` (same library; code still does `import sklearn`).
pip install numpy pandas scikit-learn jupyterlab streamlit joblib

# 3. (Optional) Freeze deps for reproducibility
pip freeze > requirements.txt
```

**Check:** `python -c "import sklearn, pandas, numpy; print('ok')"` prints `ok`.

> **Project layout** (we'll create this as we go):
> ```
> Fake-News-Detector/
> ├── PROJECT_PLAN.md        # this file
> ├── requirements.txt
> ├── data/
> │   └── news.csv           # dataset
> ├── notebooks/
> │   └── explore.ipynb      # (optional) exploratory notebook
> ├── train.py               # loads data, trains model, saves artifacts
> ├── predict.py             # loads model, classifies a single input
> ├── app.py                 # Streamlit web UI
> └── models/
>     ├── tfidf.pkl          # saved vectorizer
>     └── pac_model.pkl      # saved classifier
> ```

---

## 3. Get the Dataset

**What you need:** `news.csv` (~7,796 rows) with columns: `id`, `title`, `text`, `label` (`REAL`/`FAKE`).

Options, in order of preference:
1. **Kaggle** — search "fake or real news" (`fake_or_real_news.csv`). Cleanest to download via the Kaggle API or browser.
2. **GitHub** — many repos mirror the same dataset.
3. **Google Drive link** from the DataFlair article (may require login / may be stale).

Save it as `data/news.csv` inside the project.

**Verify it loaded correctly** before building anything:
```python
import pandas as pd
df = pd.read_csv("data/news.csv")
print(df.shape)        # (~7796, 4)
print(df.head())       # inspect columns + a few rows
print(df.label.value_counts())  # roughly balanced between REAL/FAKE
```
Check for `NaN` / missing values in `text` and handle them (drop rows).

> **Gotcha:** this dataset is an older, US-politics-heavy snapshot. It does **not** generalize to arbitrary modern articles — real BBC/NYT articles may get flagged `FAKE`. Keep that in mind; it's a limitation of the data, not your code.

---

## 4. Training Script (`train.py`)

**Goal:** one runnable script that turns `news.csv` into two saved model files.

### Step 4a — Imports
```python
import numpy as np
import pandas as pd
import itertools               # NOTE: "itertools", not "intertools" (common typo)
import joblib                  # for saving/loading models
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
```

### Step 4b — Load & prep data
```python
df = pd.read_csv("data/news.csv")
df = df.dropna(subset=["text"])       # drop rows with missing text
labels = df["label"]                  # the target
```

### Step 4c — Train/test split
```python
x_train, x_test, y_train, y_test = train_test_split(
    df["text"], labels, test_size=0.2, random_state=7
)
```

### Step 4d — Vectorize (TF-IDF)
```python
tfidf = TfidfVectorizer(stop_words="english", max_df=0.7)
tfidf_train = tfidf.fit_transform(x_train)   # fit ONLY on training
tfidf_test  = tfidf.transform(x_test)        # transform (no fit) on test
```
> **THE #1 BUG:** if you call `fit_transform` on the test set too, train and test get different vocabularies and the model throws *"feature mismatch"* errors. **Only** `fit_transform` on train; **only** `transform` on test. This trips up nearly everyone in the article's comments.

### Step 4e — Train the classifier
```python
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)
```

### Step 4f — Evaluate
```python
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f"Accuracy: {round(score * 100, 2)}%")   # expect ~92–93%

cm = confusion_matrix(y_test, y_pred, labels=["FAKE", "REAL"])
print(cm)
```
> **Confusion-matrix gotcha:** read the matrix in the order *you* pass labels — `[[TN, FP], [FN, TP]]`. Don't trust a blog caption; verify which cell is which. (This is literally the source of one of the most-upvoted comment corrections on the DataFlair page.)

### Step 4g — Save the model + vectorizer
```python
import os
os.makedirs("models", exist_ok=True)
joblib.dump(pac, "models/pac_model.pkl")
joblib.dump(tfidf, "models/tfidf.pkl")
```
Now you don't retrain every time you want to make a prediction.

---

## 5. Prediction Script (`predict.py`)

**Goal:** load the saved artifacts and classify any new text.

```python
import joblib

pac  = joblib.load("models/pac_model.pkl")
tfidf = joblib.load("models/tfidf.pkl")

def predict(text: str) -> str:
    vec = tfidf.transform([text])        # NOTE: must be a LIST, and use transform
    return pac.predict(vec)[0]

if __name__ == "__main__":
    sample = "Scientists confirm a new cure for all diseases overnight."
    print(predict(sample))
```
> **Gotchas:**
> - Pass text as a **list** (`[text]`), not a bare string.
> - The vectorizer's vocabulary was built only from training data, so unseen wording won't be weighted the same way — part of why new articles can mispredict.
> - Optionally add probabilities: `pac.decision_function(vec)` gives a signed distance → map to a confidence %.

---

## 6. Web UI (`app.py` — Streamlit)

**Goal:** a textbox → verdict interface you can demo.

```python
import streamlit as st
import joblib

pac   = joblib.load("models/pac_model.pkl")
tfidf = joblib.load("models/tfidf.pkl")

st.title("📰 Fake News Detector")
text = st.text_area("Paste an article or headline:", height=200)

if st.button("Predict"):
    if text.strip():
        vec = tfidf.transform([text])
        pred = pac.predict(vec)[0]
        conf = pac.decision_function(vec)[0]   # higher magnitude = more confident
        verdict = "🟢 REAL" if pred == "REAL" else "🔴 FAKE"
        st.write(f"**Verdict:** {verdict}")
        st.write(f"**Confidence score:** {conf:.3f}")
    else:
        st.warning("Enter some text first.")
```

**Run it:**
```bash
streamlit run app.py
```
Opens a browser tab at `http://localhost:8501`.

---

## 7. Suggested Build Order (Milestones)

Do these in order — each is independently runnable/testable:

| # | Milestone | Definition of done | Est. effort |
|---|-----------|--------------------|-------------|
| 1 | Env + dataset | venv active, `news.csv` loads with `df.shape == (~7796, 4)` | 30 min |
| 2 | Train script | `train.py` prints ~92–93% accuracy + saves both `.pkl` files | 1–2 hrs |
| 3 | Predict script | `predict.py` classifies a sample headline correctly | 30 min |
| 4 | Streamlit UI | `app.py` runs and returns verdicts in browser | 1 hr |
| 5 | (Stretch) Evaluation | print confusion matrix + a short analysis of failure cases | 1 hr |

---

## 8. Stretch Goals / Where to take it next

- **Confidence scoring** — surface `decision_function` as a human-readable %.
- **Better data** — a more modern/balanced dataset, or more classes (real / fake / satirical).
- **Better model** — try Logistic Regression or an SVM; compare accuracy against the Passive Aggressive baseline.
- **Cross-validation** — `cross_val_score` to reduce variance in the accuracy estimate.
- **Feature understanding** — `tfidf.get_feature_names_out()` to see which words drive FAKE vs REAL.
- **Deployment** — wrap the Streamlit app in a container (Docker) or host on Streamlit Community Cloud / Hugging Face Spaces.

---

## 9. Known Pitfalls Checklist (refer back when stuck)

- [ ] Used `pip install scikit-learn`, not `sklearn`.
- [ ] Spelled `itertools`, not `intertools`.
- [ ] `fit_transform` only on train; `transform` only on test.
- [ ] Handled `NaN` / missing `text` rows before training.
- [ ] `predict()` input wrapped in a list: `tfidf.transform([text])`.
- [ ] Confusion matrix order checked against the `labels=` argument.
- [ ] `stop_words="english"` and `max_df=0.7` tuned to reduce noise.
- [ ] Model artifacts saved with `joblib` (works across scripts / the UI).
