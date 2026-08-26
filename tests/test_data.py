import pandas as pd

from src.data import clean_data, get_feature_lists, split_features_target


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customerID": ["a", "b", "c"],
            "gender": ["Female", "Male", "Male"],
            "SeniorCitizen": [0, 1, 0],
            "tenure": [1, 34, 2],
            "MonthlyCharges": [29.85, 56.95, 53.85],
            "TotalCharges": ["29.85", " ", "108.15"],
            "Contract": ["Month-to-month", "One year", "Month-to-month"],
            "Churn": ["No", "No", "Yes"],
        }
    )


def test_clean_data_drops_id_and_imputes_total_charges():
    df = clean_data(_sample_df())

    assert "customerID" not in df.columns
    assert df["TotalCharges"].isna().sum() == 0
    assert df.loc[1, "TotalCharges"] == df["TotalCharges"].median()


def test_split_features_target_encodes_churn():
    df = clean_data(_sample_df())
    X, y = split_features_target(df)

    assert "Churn" not in X.columns
    assert set(y.unique()) <= {0, 1}
    assert y.tolist() == [0, 0, 1]


def test_get_feature_lists_excludes_target_and_id():
    df = clean_data(_sample_df())
    numeric, categorical = get_feature_lists(df)

    assert "Churn" not in numeric and "Churn" not in categorical
    assert "tenure" in numeric
    assert "Contract" in categorical
    assert "customerID" not in categorical
