"""TF-IDF feature extraction."""
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import config


def build_tfidf_vectorizer():
    return TfidfVectorizer(
        max_features=config.MAX_FEATURES_TFIDF,
        ngram_range=config.NGRAM_RANGE,
        min_df=config.MIN_DF,
        sublinear_tf=True,
    )


def fit_transform_tfidf(vectorizer, texts):
    return vectorizer.fit_transform(texts)


def transform_tfidf(vectorizer, texts):
    return vectorizer.transform(texts)


def save_vectorizer(vectorizer, path):
    joblib.dump(vectorizer, path)


def load_vectorizer(path):
    return joblib.load(path)