import numpy as np
import pandas as pd

from src.pipeline import build_model_pipeline


def _sample_data(n: int = 40, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "tenure": rng.integers(1, 72, size=n),
            "MonthlyCharges": rng.uniform(18, 120, size=n),
            "TotalCharges": rng.uniform(20, 8000, size=n),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], size=n),
        }
    )
    # p=0.3 minority share keeps SMOTE's default k_neighbors=5 satisfiable at this sample size.
    y = pd.Series(rng.choice([0, 1], size=n, p=[0.7, 0.3]))
    return X, y


def test_pipeline_fits_and_predicts():
    X, y = _sample_data()
    pipeline = build_model_pipeline(["tenure", "MonthlyCharges", "TotalCharges"], ["Contract"])
    pipeline.fit(X, y)

    preds = pipeline.predict(X)
    proba = pipeline.predict_proba(X)

    assert len(preds) == len(X)
    assert proba.shape == (len(X), 2)
    assert set(np.unique(preds)) <= {0, 1}


def test_pipeline_handles_unseen_categories_at_predict_time():
    X, y = _sample_data()
    pipeline = build_model_pipeline(["tenure", "MonthlyCharges", "TotalCharges"], ["Contract"])
    pipeline.fit(X, y)

    unseen = X.iloc[:2].copy()
    unseen["Contract"] = "Some New Contract Type"

    preds = pipeline.predict(unseen)
    assert len(preds) == 2
