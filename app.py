"""Streamlit web UI for the Fake News Detector.

Run with:  streamlit run app.py
Opens a browser at http://localhost:8501
"""

import joblib
import streamlit as st

# Load the saved artifacts once at startup
pac = joblib.load("models/pac_model.pkl")
tfidf = joblib.load("models/tfidf.pkl")


def predict(text: str):
    vec = tfidf.transform([text])
    pred = pac.predict(vec)[0]
    conf = float(pac.decision_function(vec)[0])
    return pred, conf


st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Fake News Detector")
st.markdown(
    "Paste an article or headline below and get an instant **REAL** / **FAKE** verdict. "
    "Powered by TF-IDF vectorization + a Passive-Aggressive classifier."
)

text = st.text_area("Article or headline:", height=200, placeholder="Paste your text here...")

if st.button("Predict", type="primary"):
    if text.strip():
        pred, conf = predict(text)
        verdict = "🟢 REAL" if pred == "REAL" else "🔴 FAKE"
        st.write(f"### {verdict}")
        st.metric("Confidence score", f"{conf:.3f}", help="Positive = REAL, negative = FAKE")
    else:
        st.warning("Enter some text first.")

st.caption("Note: the model is trained on an older, US-politics-heavy dataset and may not generalize to modern articles.")
