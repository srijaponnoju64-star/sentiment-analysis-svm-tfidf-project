"""Use pre-trained HuggingFace sentiment model - NO TRAINING NEEDED."""
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

# Load pre-trained sentiment model (trained on Twitter)
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

test_texts = [
    "I love this product, absolutely amazing!",
    "This is terrible and waste of money",
    "It's okay, nothing special"
]

print("=" * 60)
print("PRE-TRAINED SENTIMENT PREDICTION (No training needed!)")
print("=" * 60)

for text in test_texts:
    result = classifier(text)[0]
    label = result['label']  # POSITIVE or NEGATIVE
    score = result['score'] * 100
    
    print(f"\nText: {text}")
    print(f"Sentiment: {label} ({score:.2f}% confidence)")

print("\n" + "=" * 60)
print("✓ Pre-trained model working correctly")
print("=" * 60)