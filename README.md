# 📰 Fake News Detector

A machine learning project that reads a news article's text and predicts whether it is **REAL** or **FAKE** — powered by TF-IDF text vectorization and a Passive Aggressive Classifier, with a Streamlit web interface for live testing.

## About

The goal is simple: paste in an article or headline and get back a verdict — real or fake. Under the hood it's a supervised text-classification model trained on roughly 7,800 hand-labeled news articles.

## How it works

The pipeline is three steps:

1. **Text → numbers (TF-IDF).** Each article's raw text is converted into a TF-IDF (Term Frequency–Inverse Document Frequency) vector. TF-IDF weights words by how distinctive they are: common words like "the" or "and" carry almost no signal, while rare, meaningful words carry more. This turns free-form text into a numeric representation a model can learn from.

2. **Learn (Passive Aggressive Classifier).** The vectors are fed into a Passive Aggressive Classifier, an online learning algorithm that fits well to text data. It stays "passive" when predictions are correct and "aggressively" updates its weights when it makes a mistake. The model is trained on 80% of the data and evaluated on the held-out 20%, achieving roughly 92–93% accuracy.

3. **Predict.** For new text, the same vectorizer turns the input into a vector and the classifier returns a `REAL` or `FAKE` verdict along with a confidence score.

## Tools used

- **Python** — the core language
- **scikit-learn** — `TfidfVectorizer` for text vectorization and `PassiveAggressiveClassifier` for classification, plus train/test splitting and evaluation metrics (accuracy, confusion matrix)
- **pandas** / **numpy** — data loading and manipulation
- **Streamlit** — the web UI for entering text and viewing predictions
- **joblib** — serializing the trained model and vectorizer so they can be reused without retraining

## Limitations

The model is only as good as its training data. The dataset is an older, US-politics-heavy snapshot, so the classifier doesn't generalize well to arbitrary modern articles — real BBC/NYT pieces may occasionally be flagged `FAKE`. This is a data limitation, not a code bug. Additionally, the vectorizer only understands words it encountered during training, so unfamiliar phrasing can be weighted unpredictably.

## Credits

Built as a learning exercise based on the [DataFlair — Detecting Fake News with Python](https://data-flair.training/blogs/advanced-python-project-detecting-fake-news/) tutorial. A full step-by-step build plan lives in [PROJECT_PLAN.md](PROJECT_PLAN.md).
