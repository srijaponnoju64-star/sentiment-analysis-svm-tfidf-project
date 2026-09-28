"""
Hyperparameter tuning with GridSearchCV, using class-balanced weighting,
on TF-IDF features (the best-performing feature set from train_all.py).

Run AFTER train_all.py has already generated the TF-IDF vectorizer + dataset:
    python tune_models.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from src.data.loader import load_real_dataset
from src.data.splits import make_splits
from src.preprocessing.cleaner import preprocess_batch
from src.features.tfidf import build_tfidf_vectorizer, save_vectorizer
from src.models.traditional import save_model
from src.evaluation.metrics import evaluate_model, plot_confusion_matrix, save_results, plot_model_comparison


def tune_and_train(name, base_estimator, param_grid, X_train, y_train):
    print(f"\nTuning {name} (grid: {param_grid})...")
    grid = GridSearchCV(
        base_estimator, param_grid,
        cv=config.CV_FOLDS, scoring="f1_macro", n_jobs=-1, verbose=1,
    )
    grid.fit(X_train, y_train)
    print(f"  Best params: {grid.best_params_}  |  Best CV macro-F1: {grid.best_score_:.4f}")
    return grid.best_estimator_


def main():
    df = load_real_dataset()
    print(f"Dataset size: {len(df)}")

    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(df)

    print("Preprocessing text...")
    X_train_clean = preprocess_batch(X_train)
    X_test_clean = preprocess_batch(X_test)

    print("Building TF-IDF features...")
    tfidf_vec = build_tfidf_vectorizer()
    X_train_tfidf = tfidf_vec.fit_transform(X_train_clean)
    X_test_tfidf = tfidf_vec.transform(X_test_clean)
    save_vectorizer(tfidf_vec, os.path.join(config.MODELS_DIR, "tfidf_vectorizer_tuned.pkl"))

    all_results = []

    # Logistic Regression
    best_lr = tune_and_train(
        "logistic_regression",
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=config.RANDOM_STATE),
        config.LOGREG_GRID,
        X_train_tfidf, y_train,
    )
    save_model(best_lr, os.path.join(config.MODELS_DIR, "logistic_regression_tfidf_tuned.pkl"))
    results, y_pred = evaluate_model(best_lr, X_test_tfidf, y_test, "logistic_regression_tfidf_tuned")
    print(f"  Test accuracy: {results['accuracy']:.4f} | Test macro F1: {results['macro_f1']:.4f}")
    all_results.append(results)
    plot_confusion_matrix(y_test, y_pred, "logistic_regression_tfidf_tuned")

    # Linear SVM
    best_svm = tune_and_train(
        "svm",
        LinearSVC(class_weight="balanced", random_state=config.RANDOM_STATE, max_iter=5000),
        config.SVM_GRID,
        X_train_tfidf, y_train,
    )
    save_model(best_svm, os.path.join(config.MODELS_DIR, "svm_tfidf_tuned.pkl"))
    results, y_pred = evaluate_model(best_svm, X_test_tfidf, y_test, "svm_tfidf_tuned")
    print(f"  Test accuracy: {results['accuracy']:.4f} | Test macro F1: {results['macro_f1']:.4f}")
    all_results.append(results)
    plot_confusion_matrix(y_test, y_pred, "svm_tfidf_tuned")

    # Naive Bayes (no class_weight support, but alpha tuning still helps)
    best_nb = tune_and_train(
        "naive_bayes",
        MultinomialNB(),
        config.NB_GRID,
        X_train_tfidf, y_train,
    )
    save_model(best_nb, os.path.join(config.MODELS_DIR, "naive_bayes_tfidf_tuned.pkl"))
    results, y_pred = evaluate_model(best_nb, X_test_tfidf, y_test, "naive_bayes_tfidf_tuned")
    print(f"  Test accuracy: {results['accuracy']:.4f} | Test macro F1: {results['macro_f1']:.4f}")
    all_results.append(results)
    plot_confusion_matrix(y_test, y_pred, "naive_bayes_tfidf_tuned")

    save_results(all_results, path=os.path.join(config.MODELS_DIR, "evaluation_results_tuned.json"))
    plot_model_comparison(all_results)

    print("\n=== TUNED SUMMARY ===")
    for r in sorted(all_results, key=lambda x: -x["macro_f1"]):
        print(f"{r['model']:35s}  acc={r['accuracy']:.4f}  f1={r['macro_f1']:.4f}")


if __name__ == "__main__":
    main()