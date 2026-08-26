import pandas as pd

from src.config import DATA_PATH, ID_COLUMN, NUMERIC_FEATURES, TARGET_COLUMN


def load_data(path=DATA_PATH) -> pd.DataFrame:
    """Load the raw Telco customer churn CSV."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the customer identifier and fix the numeric-as-string TotalCharges column.

    Eleven rows ship with a blank TotalCharges (new customers with zero tenure);
    those become NaN on coercion and are median-imputed rather than dropped.
    """
    df = df.copy()
    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned dataframe into features X and a binary target y (Yes -> 1)."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].map({"No": 0, "Yes": 1})
    return X, y


def get_feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric_features, categorical_features) present in a cleaned dataframe."""
    categorical_features = [
        col for col in df.columns if col not in NUMERIC_FEATURES + [TARGET_COLUMN]
    ]
    return NUMERIC_FEATURES, categorical_features
