from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RANDOM_STATE


def build_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    """Median-impute + scale numeric columns; constant-impute + one-hot categoricals."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0,
    )


def build_model_pipeline(
    numeric_features: list[str], categorical_features: list[str]
) -> ImbPipeline:
    """Preprocessing + SMOTE oversampling + class-weighted Random Forest.

    SMOTE lives inside the pipeline (via imblearn, not sklearn) so it only ever
    resamples the training fold during fit and is a no-op at predict time,
    which avoids leaking synthetic samples into validation/test data.
    """
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    return ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            (
                "clf",
                RandomForestClassifier(
                    class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
                ),
            ),
        ]
    )
