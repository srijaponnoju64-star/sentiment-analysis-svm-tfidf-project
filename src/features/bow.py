"""Bag-of-Words feature extraction."""
import joblib
from sklearn.feature_extraction.text import CountVectorizer
import config


def build_bow_vectorizer():
    return CountVectorizer(
        max_features=config.MAX_FEATURES_BOW,
        ngram_range=config.NGRAM_RANGE,
        min_df=config.MIN_DF,
    )


def fit_transform_bow(vectorizer, texts):
    return vectorizer.fit_transform(texts)


def transform_bow(vectorizer, texts):
    return vectorizer.transform(texts)


def save_vectorizer(vectorizer, path):
    joblib.dump(vectorizer, path)


def load_vectorizer(path):
    return joblib.load(path)