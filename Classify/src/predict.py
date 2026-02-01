import pickle
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

vectorizer_path = os.path.join(BASE_DIR, "model", "vectorizer.pkl")
classifier_path = os.path.join(BASE_DIR, "model", "classifier.pkl")

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

with open(classifier_path, "rb") as f:
    classifier = pickle.load(f)

def predict_category(text):
    text_vector = vectorizer.transform([text])
    prediction = classifier.predict(text_vector)
    return prediction[0]
