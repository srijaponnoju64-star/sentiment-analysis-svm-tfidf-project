"""
Master training script: traditional ML models (Naive Bayes, Logistic
Regression, SVM) x two feature sets (BoW, TF-IDF).

Run:
    python train_all.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.data.loader import load_real_dataset, generate_synthetic_dataset, save_dataset
from src.data.splits import make_splits
from src.preprocessing.cleaner import preprocess_batch
from src.features.bow import build_bow_vectorizer, save_vectorizer as save_bow
from src.features.tfidf import build_tfidf_vectorizer, save_vectorizer as save_tfidf
from src.models.traditional import train_model, save_model
from src.evaluation.metrics import evaluate_model, plot_confusion_matrix, save_results, plot_model_comparison


def load_or_create_dataset():
    if os.path.exists(config.RAW_DATASET_PATH):
        print(f"Loading dataset from {config.RAW_DATASET_PATH}")
        return load_real_dataset()
    print("No dataset found — generating a synthetic one so you can run the "
          "pipeline today. Swap in a real CSV at data/raw/sentiment_dataset.csv "
          "before reporting final results.")
    df = generate_synthetic_dataset(n_per_class=1500)
    save_dataset(df)
    return df


def main():
    df = load_or_create_dataset()
    print(f"\nDataset size: {len(df)}")
    print(df[config.LABEL_COLUMN].value_counts(), "\n")

    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(df)

    print("Preprocessing text...")
    X_train_clean = preprocess_batch(X_train)
    X_test_clean = preprocess_batch(X_test)

    feature_sets = {}

    print("Building BoW features...")
    bow_vec = build_bow_vectorizer()
    X_train_bow = bow_vec.fit_transform(X_train_clean)
    X_test_bow = bow_vec.transform(X_test_clean)
    save_bow(bow_vec, os.path.join(config.MODELS_DIR, "bow_vectorizer.pkl"))
    feature_sets["bow"] = (X_train_bow, X_test_bow)

    print("Building TF-IDF features...")
    tfidf_vec = build_tfidf_vectorizer()
    X_train_tfidf = tfidf_vec.fit_transform(X_train_clean)
    X_test_tfidf = tfidf_vec.transform(X_test_clean)
    save_tfidf(tfidf_vec, os.path.join(config.MODELS_DIR, "tfidf_vectorizer.pkl"))
    feature_sets["tfidf"] = (X_train_tfidf, X_test_tfidf)

    model_names = ["naive_bayes", "logistic_regression", "svm"]
    all_results = []

    for feature_name, (X_tr, X_te) in feature_sets.items():
        for model_name in model_names:
            full_name = f"{model_name}_{feature_name}"
            print(f"\nTraining {full_name}...")
            model = train_model(model_name, X_tr, y_train)
            save_model(model, os.path.join(config.MODELS_DIR, f"{full_name}.pkl"))

            results, y_pred = evaluate_model(model, X_te, y_test, full_name)
            print(f"  Accuracy: {results['accuracy']:.4f} | Macro F1: {results['macro_f1']:.4f}")
            all_results.append(results)
            plot_confusion_matrix(y_test, y_pred, full_name)

    save_results(all_results)
    plot_model_comparison(all_results)

    print("\n=== SUMMARY ===")
    for r in sorted(all_results, key=lambda x: -x["accuracy"]):
        print(f"{r['model']:30s}  acc={r['accuracy']:.4f}  f1={r['macro_f1']:.4f}")


if __name__ == "__main__":
    main()