import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Base directory (Classify/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "yogesh.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")

# Create model directory
os.makedirs(MODEL_DIR, exist_ok=True)

# Load CSV
df = pd.read_csv(DATA_PATH, encoding="latin1")

# 🔴 CRITICAL FIXES
df = df.dropna(subset=["Text", "Category"])   # remove empty rows
df["Text"] = df["Text"].astype(str)            # force text type

# Features and labels
X_text = df["Text"]
y = df["Category"]

# Vectorization
vectorizer = TfidfVectorizer(
    stop_words="english",
    lowercase=True
)

X = vectorizer.fit_transform(X_text)

# Train classifier
classifier = MultinomialNB()
classifier.fit(X, y)

# Save model
with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)

with open(os.path.join(MODEL_DIR, "classifier.pkl"), "wb") as f:
    pickle.dump(classifier, f)

print("✅ Model trained successfully and saved in model/")
