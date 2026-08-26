"""Load the trained pipeline and score new customer records.

Usage:
    python -m src.predict input.csv output.csv
"""
import argparse
from pathlib import Path

import pandas as pd
from joblib import load

from src.config import MODEL_PATH


def load_model(path: Path = MODEL_PATH):
    return load(path)


def predict(model, records: pd.DataFrame | list[dict]) -> tuple[list[int], list[float]]:
    """Score raw customer records (a DataFrame or list of dicts).

    The model's ColumnTransformer selects its expected columns by name, so
    extra columns (e.g. customerID) in `records` are ignored automatically.
    """
    if isinstance(records, list):
        records = pd.DataFrame(records)
    proba = model.predict_proba(records)[:, 1]
    labels = (proba >= 0.5).astype(int)
    return labels.tolist(), proba.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch churn predictions on a CSV file.")
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    model = load_model(args.model_path)
    df = pd.read_csv(args.input_csv)
    labels, proba = predict(model, df)

    result = df.copy()
    result["churn_prediction"] = labels
    result["churn_probability"] = proba
    result.to_csv(args.output_csv, index=False)
    print(f"Wrote predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
