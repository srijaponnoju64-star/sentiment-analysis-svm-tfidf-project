"""
Central configuration for the Advanced Sentiment Analysis project.
Edit values here rather than hardcoding them across modules.
"""
import os

# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")

RAW_DATASET_PATH = os.path.join(DATA_RAW_DIR, "sentiment_dataset.csv")

# ---------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------
TEXT_COLUMN = "text"
LABEL_COLUMN = "sentiment"
CLASSES = ["Negative", "Neutral", "Positive"]
LABEL2ID = {c: i for i, c in enumerate(CLASSES)}
ID2LABEL = {i: c for c, i in LABEL2ID.items()}

TEST_SIZE = 0.20
VAL_SIZE = 0.10          # fraction of the remaining train split, used by DL models later
RANDOM_STATE = 42

# ---------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------
MAX_FEATURES_BOW = 5000
MAX_FEATURES_TFIDF = 5000
NGRAM_RANGE = (1, 2)          # unigrams + bigrams
MIN_DF = 2

# ---------------------------------------------------------------
# Traditional ML models
# ---------------------------------------------------------------
LOGREG_PARAMS = dict(max_iter=2000, solver="lbfgs", random_state=RANDOM_STATE)
SVM_PARAMS = dict(kernel="linear", probability=True, random_state=RANDOM_STATE)
NB_PARAMS = dict(alpha=1.0)

# ---------------------------------------------------------------
# Misc
# ---------------------------------------------------------------
os.makedirs(DATA_RAW_DIR, exist_ok=True)
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
# ---------------------------------------------------------------
# Hyperparameter tuning (GridSearchCV)
# ---------------------------------------------------------------
CV_FOLDS = 5

LOGREG_GRID = {
    "C": [0.1, 1.0, 3.0, 10.0],
}
SVM_GRID = {
    "C": [0.1, 1.0, 3.0, 10.0],
}
NB_GRID = {
    "alpha": [0.1, 0.5, 1.0, 2.0],
}