"""
Dataset loading for the sentiment analysis project.

Two modes:
1. load_real_dataset(path)   -> loads a CSV you've downloaded (e.g. from Kaggle)
2. generate_synthetic_dataset(n_per_class) -> builds a labeled dataset from
   templated sentence patterns, useful for developing/testing the pipeline
   before you plug in a real dataset.

Real dataset CSV must have two columns: 'text' and 'sentiment'
(sentiment values: Negative / Neutral / Positive).
"""
import os
import random
import pandas as pd

import config


def load_real_dataset(path: str = config.RAW_DATASET_PATH) -> pd.DataFrame:
    """Load a real, user-provided sentiment dataset from CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No dataset found at {path}. Download one (e.g. IMDB reviews or "
            f"Sentiment140 from Kaggle), rename the text/label columns to "
            f"'{config.TEXT_COLUMN}'/'{config.LABEL_COLUMN}', and place it there."
        )
    df = pd.read_csv(path)
    df = df[[config.TEXT_COLUMN, config.LABEL_COLUMN]].dropna()
    df = df[df[config.LABEL_COLUMN].isin(config.CLASSES)]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------
# Synthetic dataset generator
# ---------------------------------------------------------------
_POSITIVE_TEMPLATES = [
    "I absolutely love {obj}, it's {pos_adj}!",
    "This {obj} is {pos_adj}, highly recommended.",
    "Best {obj} I've ever used, truly {pos_adj}.",
    "What a {pos_adj} experience with {obj}, made my day!",
    "{obj} exceeded my expectations, {pos_adj} quality.",
    "So happy with this {obj}, {pos_adj} in every way.",
    "The {obj} works perfectly and looks {pos_adj}.",
    "Fantastic {obj}! {pos_adj} service and support too.",
]
_NEGATIVE_TEMPLATES = [
    "I hate this {obj}, it's {neg_adj}.",
    "This {obj} is {neg_adj}, total waste of money.",
    "Worst {obj} I've ever used, completely {neg_adj}.",
    "Very disappointed with {obj}, {neg_adj} experience overall.",
    "{obj} broke after a day, {neg_adj} build quality.",
    "Terrible {obj}, {neg_adj} customer service.",
    "The {obj} stopped working, {neg_adj} product.",
    "Regret buying this {obj}, {neg_adj} in every way.",
]
_NEUTRAL_TEMPLATES = [
    "The {obj} comes in three sizes.",
    "{obj} was delivered on Tuesday afternoon.",
    "This {obj} weighs about two kilograms.",
    "The {obj} is available in five colors.",
    "Instructions for the {obj} are in the manual.",
    "The {obj} costs around fifty dollars.",
    "The meeting about {obj} is scheduled for 3 PM.",
    "{obj} specifications are listed on the website.",
]
_OBJECTS = [
    "product", "phone", "laptop", "movie", "restaurant", "service", "app",
    "book", "hotel", "course", "game", "headphones", "car", "software",
    "delivery", "camera", "watch", "shoes", "flight", "subscription",
]
_POS_ADJ = ["amazing", "excellent", "wonderful", "outstanding", "fantastic", "brilliant", "superb"]
_NEG_ADJ = ["awful", "horrible", "disappointing", "useless", "frustrating", "poor", "unacceptable"]


def generate_synthetic_dataset(n_per_class: int = 1000, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """
    Build a synthetic 3-class dataset from templated sentences.
    NOT a substitute for a real dataset in a final report — use this to get
    the pipeline running end-to-end, then swap in a real CSV via
    load_real_dataset() before you report final numbers.
    """
    rng = random.Random(seed)
    rows = []

    def fill(template, is_pos=False, is_neg=False):
        obj = rng.choice(_OBJECTS)
        if is_pos:
            return template.format(obj=obj, pos_adj=rng.choice(_POS_ADJ))
        if is_neg:
            return template.format(obj=obj, neg_adj=rng.choice(_NEG_ADJ))
        return template.format(obj=obj)

    for _ in range(n_per_class):
        rows.append((fill(rng.choice(_POSITIVE_TEMPLATES), is_pos=True), "Positive"))
        rows.append((fill(rng.choice(_NEGATIVE_TEMPLATES), is_neg=True), "Negative"))
        rows.append((fill(rng.choice(_NEUTRAL_TEMPLATES)), "Neutral"))

    df = pd.DataFrame(rows, columns=[config.TEXT_COLUMN, config.LABEL_COLUMN])
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # shuffle
    return df


def save_dataset(df: pd.DataFrame, path: str = config.RAW_DATASET_PATH):
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows to {path}")


if __name__ == "__main__":
    df = generate_synthetic_dataset(n_per_class=1000)
    save_dataset(df)
    print(df[config.LABEL_COLUMN].value_counts())