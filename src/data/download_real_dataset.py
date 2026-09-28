"""
Downloads the TweetEval 'sentiment' dataset directly from its Hugging Face
repo (cardiffnlp/tweet_eval) via direct parquet file URLs — a real, labeled,
3-class (negative/neutral/positive) Twitter sentiment benchmark with
~60,000 examples.

Run:
    python src/data/download_real_dataset.py
"""
import io
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import requests
import pandas as pd
import config

LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}

BASE_URL = "https://huggingface.co/datasets/cardiffnlp/tweet_eval/resolve/main/sentiment"
SPLIT_FILES = {
    "train": f"{BASE_URL}/train-00000-of-00001.parquet",
    "validation": f"{BASE_URL}/validation-00000-of-00001.parquet",
    "test": f"{BASE_URL}/test-00000-of-00001.parquet",
}


def download_and_convert():
    print("Downloading TweetEval sentiment dataset (cardiffnlp/tweet_eval)...")
    frames = []
    for split, url in SPLIT_FILES.items():
        print(f"  downloading {split} from {url}")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        df_part = pd.read_parquet(io.BytesIO(resp.content))
        frames.append(df_part)
        print(f"    -> {len(df_part)} rows")

    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"text": config.TEXT_COLUMN})
    df[config.LABEL_COLUMN] = df["label"].map(LABEL_MAP)
    df = df[[config.TEXT_COLUMN, config.LABEL_COLUMN]].dropna()
    df = df.drop_duplicates(subset=[config.TEXT_COLUMN]).reset_index(drop=True)

    print(f"\nTotal after dedup: {len(df)} rows")
    print(df[config.LABEL_COLUMN].value_counts())

    df.to_csv(config.RAW_DATASET_PATH, index=False)
    print(f"\nSaved -> {config.RAW_DATASET_PATH}")


if __name__ == "__main__":
    download_and_convert()