"""
Evaluates a pretrained, already fine-tuned transformer
(cardiffnlp/twitter-roberta-base-sentiment-latest) on our held-out test set.
No training needed — this model was fine-tuned by CardiffNLP (the same lab
behind the TweetEval benchmark) specifically for 3-class tweet sentiment.

Run:
    python evaluate_pretrained_transformer.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

import config
from src.data.loader import load_real_dataset
from src.data.splits import make_splits
from src.evaluation.metrics import evaluate_model, plot_confusion_matrix, save_results

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
# This model's own label order (check its config): negative, neutral, positive
MODEL_LABEL_ORDER = ["Negative", "Neutral", "Positive"]


class PretrainedTransformerWrapper:
    """Wraps a HF pipeline so it plays nicely with evaluate_model()'s
    model.predict(X) interface used by our other scripts."""

    def __init__(self, model_name):
        print(f"Loading {model_name} (first run downloads ~500MB)...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def predict(self, texts, batch_size=32):
        all_preds = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = list(texts[i:i + batch_size])
                enc = self.tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=128, return_tensors="pt",
                )
                logits = self.model(**enc).logits
                pred_ids = torch.argmax(logits, dim=1).tolist()
                all_preds.extend([MODEL_LABEL_ORDER[i] for i in pred_ids])
                if i % (batch_size * 20) == 0:
                    print(f"  predicted {i}/{len(texts)}")
        return np.array(all_preds)


def main():
    df = load_real_dataset()
    _, _, X_test, _, _, y_test = make_splits(df)
    print(f"Evaluating on {len(X_test)} test examples...")

    model = PretrainedTransformerWrapper(MODEL_NAME)
    results, y_pred = evaluate_model(model, X_test, y_test, "pretrained_twitter_roberta")

    print(f"\nAccuracy: {results['accuracy']:.4f}")
    print(f"Macro F1: {results['macro_f1']:.4f}")

    plot_confusion_matrix(y_test, y_pred, "pretrained_twitter_roberta")
    save_results([results], path=os.path.join(config.MODELS_DIR, "evaluation_results_pretrained_transformer.json"))


if __name__ == "__main__":
    main()