"""Stratified train/val/test splitting."""
from sklearn.model_selection import train_test_split
import config


def make_splits(df, text_col=config.TEXT_COLUMN, label_col=config.LABEL_COLUMN):
    """
    Returns (X_train, X_val, X_test, y_train, y_val, y_test).
    Stratified so class distribution is preserved in every split.
    """
    X = df[text_col].values
    y = df[label_col].values

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )

    # Carve a validation set out of the remaining training data
    val_fraction_of_train = config.VAL_SIZE / (1 - config.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=val_fraction_of_train,
        stratify=y_train_full,
        random_state=config.RANDOM_STATE,
    )

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test