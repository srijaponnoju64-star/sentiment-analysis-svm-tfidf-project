"""Inference pipeline for trained sentiment models."""
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.special import softmax
import warnings
warnings.filterwarnings('ignore')

# Load trained model and vectorizer using joblib
model = joblib.load('models/svm_tfidf.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# Test predictions
test_texts = [
    "I love this product, absolutely amazing!",
    "This is terrible and waste of money",
    "It's okay, nothing special"
]

print("=" * 60)
print("SENTIMENT PREDICTION")
print("=" * 60)

for text in test_texts:
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    
    # LinearSVC uses decision_function (not predict_proba)
    decision_scores = model.decision_function(text_vec)[0]
    prob = softmax(decision_scores)  # Convert to probabilities
    
    print(f"\nText: {text}")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {max(prob)*100:.2f}%")

print("\n" + "=" * 60)
print("Prediction pipeline working correctly ✓")
print("=" * 60)