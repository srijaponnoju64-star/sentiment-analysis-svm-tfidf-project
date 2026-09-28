"""Evaluation metrics: classification report, confusion matrix, comparison table."""
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
)
import config


def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", labels=config.CLASSES, zero_division=0
    )
    report = classification_report(
        y_test, y_pred, labels=config.CLASSES, output_dict=True, zero_division=0
    )
    results = {
        "model": model_name,
        "accuracy": round(acc, 4),
        "macro_precision": round(precision, 4),
        "macro_recall": round(recall, 4),
        "macro_f1": round(f1, 4),
        "per_class_report": report,
    }
    return results, y_pred


def plot_confusion_matrix(y_test, y_pred, model_name, save_dir=config.MODELS_DIR):
    cm = confusion_matrix(y_test, y_pred, labels=config.CLASSES)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=config.CLASSES, yticklabels=config.CLASSES)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    path = os.path.join(save_dir, f"confusion_matrix_{model_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix -> {path}")


def save_results(all_results, path=None):
    path = path or os.path.join(config.MODELS_DIR, "evaluation_results.json")
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved evaluation results -> {path}")


def plot_model_comparison(all_results, save_dir=config.MODELS_DIR):
    names = [r["model"] for r in all_results]
    accs = [r["accuracy"] for r in all_results]
    f1s = [r["macro_f1"] for r in all_results]

    x = range(len(names))
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar([i - width / 2 for i in x], accs, width, label="Accuracy")
    plt.bar([i + width / 2 for i in x], f1s, width, label="Macro F1")
    plt.xticks(list(x), names, rotation=15)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Model Comparison")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, "metrics_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved comparison chart -> {path}")