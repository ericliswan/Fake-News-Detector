#!/usr/bin/env python3
"""
Training script for the Fake News Detector.

Loads news.csv, trains an SGDClassifier (Passive-Aggressive mode) on
TF-IDF vectors, and saves the trained model and vectorizer for later use.
"""

# Step 4a — Imports
import numpy as np
import pandas as pd
import itertools
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 4b — Load & prep data
df = pd.read_csv("data/news.csv")
df = df.dropna(subset=["text"])  # drop rows with missing text
labels = df["label"]

print(f"Dataset shape: {df.shape}")
print(f"Label distribution:\n{labels.value_counts()}\n")

# Step 4c — Train/test split
x_train, x_test, y_train, y_test = train_test_split(
    df["text"], labels, test_size=0.2, random_state=7
)

print(f"Training set size: {len(x_train)}")
print(f"Test set size: {len(x_test)}\n")

# Step 4d — Vectorize (TF-IDF)
tfidf = TfidfVectorizer(stop_words="english", max_df=0.7)
tfidf_train = tfidf.fit_transform(x_train)  # fit ONLY on training
tfidf_test = tfidf.transform(x_test)  # transform (no fit) on test

print(f"TF-IDF vectorizer fit with vocabulary size: {len(tfidf.vocabulary_)}\n")

# Step 4e — Train the classifier
# SGDClassifier with 'pa1' replicates the Passive Aggressive algorithm
# (PassiveAggressiveClassifier is deprecated in sklearn 1.8+).
pac = SGDClassifier(
    loss="hinge",
    penalty=None,
    learning_rate="pa1",
    eta0=1.0,
    max_iter=50,
)
pac.fit(tfidf_train, y_train)

print("Classifier (SGD/Passive-Aggressive) trained.\n")

# Step 4f — Evaluate
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f"Accuracy: {round(score * 100, 2)}%\n")

cm = confusion_matrix(y_test, y_pred, labels=["FAKE", "REAL"])
print("Confusion Matrix (labels order: FAKE, REAL):")
print(cm)
print()

# Step 4g — Save the model + vectorizer
import os
os.makedirs("models", exist_ok=True)
joblib.dump(pac, "models/pac_model.pkl")
joblib.dump(tfidf, "models/tfidf.pkl")

print("Model and vectorizer saved to models/ directory.")
print("Training complete")
