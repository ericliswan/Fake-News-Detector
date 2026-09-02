"""Load the trained classifier and vectorizer, then classify a single input.

Usage:
    python predict.py                 # runs the built-in sample
    python predict.py "your text"     # classify a custom article/headline
"""

import sys

import joblib


def load_models():
    """Load the saved classifier and vectorizer from disk."""
    pac = joblib.load("models/pac_model.pkl")
    tfidf = joblib.load("models/tfidf.pkl")
    return pac, tfidf


def predict(text: str, pac, tfidf) -> str:
    """Classify a single text and return the prediction."""
    vec = tfidf.transform([text])  # NOTE: must be a LIST; use transform (not fit_transform)
    return pac.predict(vec)[0]


def confidence(text: str, pac, tfidf) -> float:
    """Return a signed confidence score (positive = REAL, negative = FAKE)."""
    vec = tfidf.transform([text])
    return float(pac.decision_function(vec)[0])


if __name__ == "__main__":
    pac, tfidf = load_models()

    if len(sys.argv) > 1:
        sample = " ".join(sys.argv[1:])
    else:
        sample = "Scientists confirm a new cure for all diseases overnight."

    prediction = predict(sample, pac, tfidf)
    score = confidence(sample, pac, tfidf)

    verdict = "🟢 REAL" if prediction == "REAL" else "🔴 FAKE"
    print(f"Input:   {sample!r}")
    print(f"Verdict: {verdict}")
    print(f"Confidence score: {score:.3f}  (positive = REAL, negative = FAKE)")
