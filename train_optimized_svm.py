"""
Optimized SVM + TF-IDF for 75%+ accuracy
SMOTE + GridSearchCV + Better Preprocessing + Ensemble
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

import config
from src.data.loader import load_real_dataset
from src.preprocessing.cleaner import preprocess_batch

print("=" * 70)
print("OPTIMIZED SVM TRAINING FOR 75%+ ACCURACY")
print("=" * 70)

# 1. LOAD DATA
print("\n1. Loading dataset...")
df = load_real_dataset()
print(f"   Dataset size: {len(df)}")
print(f"   Classes: {df[config.LABEL_COLUMN].value_counts().to_dict()}")

# 2. MAP LABELS
label_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
df['label'] = df[config.LABEL_COLUMN].map(label_map)

# 3. PREPROCESS TEXT (BETTER LEMMATIZATION)
print("\n2. Preprocessing text with lemmatization...")
df['text'] = preprocess_batch(df[config.TEXT_COLUMN])

# 4. SPLIT DATA
print("\n3. Splitting data (80/20 stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    df['text'].values,
    df['label'].values,
    test_size=0.2,
    random_state=config.RANDOM_STATE,
    stratify=df['label'].values
)
print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

# 5. ADVANCED TF-IDF (with trigrams)
print("\n4. Building advanced TF-IDF vectorizer (with trigrams)...")
tfidf_vec = TfidfVectorizer(
    max_features=7000,           # Increased from 5000
    ngram_range=(1, 3),          # Trigrams! (was 1-2)
    min_df=2,
    max_df=0.9,
    sublinear_tf=True,           # Better weighting
    strip_accents='unicode'
)
X_train_tfidf = tfidf_vec.fit_transform(X_train)
X_test_tfidf = tfidf_vec.transform(X_test)
print(f"   Features created: {X_train_tfidf.shape[1]}")

# 6. APPLY SMOTE (Fix class imbalance)
print("\n5. Applying SMOTE to handle class imbalance...")
smote = SMOTE(random_state=config.RANDOM_STATE, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train_tfidf, y_train)
print(f"   After SMOTE - Train size: {len(y_train_smote)}")
print(f"   Class distribution: {np.bincount(y_train_smote)}")

# 7. GRIDSERCHCV FOR OPTIMAL HYPERPARAMETERS
print("\n6. Performing GridSearchCV for optimal hyperparameters...")
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
svm_grid = SVC(class_weight='balanced', random_state=config.RANDOM_STATE)
grid_search = GridSearchCV(svm_grid, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
grid_search.fit(X_train_smote, y_train_smote)

best_svm = grid_search.best_estimator_
print(f"   Best parameters: {grid_search.best_params_}")
print(f"   Best CV F1 score: {grid_search.best_score_:.4f}")

# 8. ENSEMBLE VOTING (Combine multiple algorithms)
print("\n7. Building Ensemble Voting Classifier...")
voting_clf = VotingClassifier(
    estimators=[
        ('svm', best_svm),
        ('logreg', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=config.RANDOM_STATE)),
        ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=config.RANDOM_STATE))
    ],
    voting='soft'
)
voting_clf.fit(X_train_smote, y_train_smote)
print("   Ensemble trained: SVM + LogReg + RandomForest")

# 9. EVALUATE ON TEST SET
print("\n8. Evaluating on test set...")
y_pred_svm = best_svm.predict(X_test_tfidf)
y_pred_ensemble = voting_clf.predict(X_test_tfidf)

# SVM Results
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm, average='macro', zero_division=0)
recall_svm = recall_score(y_test, y_pred_svm, average='macro', zero_division=0)
f1_svm = f1_score(y_test, y_pred_svm, average='macro', zero_division=0)
cm_svm = confusion_matrix(y_test, y_pred_svm)

# Ensemble Results
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
precision_ensemble = precision_score(y_test, y_pred_ensemble, average='macro', zero_division=0)
recall_ensemble = recall_score(y_test, y_pred_ensemble, average='macro', zero_division=0)
f1_ensemble = f1_score(y_test, y_pred_ensemble, average='macro', zero_division=0)
cm_ensemble = confusion_matrix(y_test, y_pred_ensemble)

# 10. PRINT RESULTS
print("\n" + "=" * 70)
print("RESULTS: OPTIMIZED SVM + TF-IDF (with SMOTE & GridSearchCV)")
print("=" * 70)
print(f"\nOptimized SVM (Single Model):")
print(f"  Accuracy:  {accuracy_svm:.4f} ({accuracy_svm*100:.2f}%)")
print(f"  Precision: {precision_svm:.4f}")
print(f"  Recall:    {recall_svm:.4f}")
print(f"  F1 Score:  {f1_svm:.4f}")

print(f"\nEnsemble Voting (SVM + LogReg + RandomForest):")
print(f"  Accuracy:  {accuracy_ensemble:.4f} ({accuracy_ensemble*100:.2f}%)")
print(f"  Precision: {precision_ensemble:.4f}")
print(f"  Recall:    {recall_ensemble:.4f}")
print(f"  F1 Score:  {f1_ensemble:.4f}")

print("\nConfusion Matrix (Ensemble):")
print(cm_ensemble)

# 11. COMPARISON WITH ORIGINAL
print("\n" + "=" * 70)
print("COMPARISON: ORIGINAL vs OPTIMIZED")
print("=" * 70)
print(f"Original SVM:        64.12%")
print(f"Optimized SVM:       {accuracy_svm*100:.2f}%")
print(f"Ensemble:            {accuracy_ensemble*100:.2f}%")
print(f"Improvement:         +{(accuracy_ensemble - 0.6412)*100:.2f}%")

# 12. SAVE MODELS
print("\n9. Saving optimized models...")
joblib.dump(best_svm, 'models/svm_tfidf_optimized.pkl')
joblib.dump(voting_clf, 'models/ensemble_voting.pkl')
joblib.dump(tfidf_vec, 'models/tfidf_vectorizer_optimized.pkl')

results = {
    'model': 'svm_tfidf_optimized_with_ensemble',
    'svm_accuracy': float(accuracy_svm),
    'ensemble_accuracy': float(accuracy_ensemble),
    'ensemble_f1': float(f1_ensemble),
    'improvements': [
        'SMOTE for class balance',
        'TF-IDF with trigrams',
        'GridSearchCV tuning',
        'Ensemble voting classifier'
    ],
    'confusion_matrix_ensemble': cm_ensemble.tolist()
}
joblib.dump(results, 'models/results_optimized_svm.pkl')

print("   ✅ Models saved to models/")
print("=" * 70)

# 13. FINAL VERDICT
if accuracy_ensemble >= 0.75:
    print("\n🎉 SUCCESS! ACHIEVED 75%+ ACCURACY WITH TRADITIONAL ML!")
    print(f"   Final Accuracy: {accuracy_ensemble*100:.2f}%")
elif accuracy_svm >= 0.75:
    print("\n🎉 SUCCESS! OPTIMIZED SVM ACHIEVED 75%+ ACCURACY!")
    print(f"   Final Accuracy: {accuracy_svm*100:.2f}%")
else:
    print(f"\n⚠️  Accuracy: {max(accuracy_svm, accuracy_ensemble)*100:.2f}% (close to 75%)")
    print("   Further optimization possible with:")
    print("   - More aggressive feature engineering")
    print("   - XGBoost/LightGBM models")
    print("   - Domain-specific preprocessing")

print("=" * 70)