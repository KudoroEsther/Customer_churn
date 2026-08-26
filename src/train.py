"""Train the churn prediction pipeline and persist the fitted model.

Usage:
    python -m src.train
    python -m src.train --n-iter 40 --cv-folds 10
"""
import argparse
import json
import logging
from pathlib import Path

from joblib import dump
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

from src.config import DATA_PATH, MODEL_PATH, RANDOM_STATE
from src.data import clean_data, get_feature_lists, load_data, split_features_target
from src.pipeline import build_model_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PARAM_DISTRIBUTIONS = {
    "clf__n_estimators": [100, 200, 400],
    "clf__max_depth": [None, 10, 20],
    "clf__min_samples_leaf": [1, 2, 4],
    "clf__max_features": ["sqrt", "log2"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the churn prediction pipeline.")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--metrics-path", type=Path, default=None)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--n-iter", type=int, default=20, help="RandomizedSearchCV iterations")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    return parser.parse_args()


def train(args: argparse.Namespace):
    logger.info("Loading data from %s", args.data_path)
    df = clean_data(load_data(args.data_path))
    numeric_features, categorical_features = get_feature_lists(df)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )
    logger.info("Train shape: %s, Test shape: %s", X_train.shape, X_test.shape)

    pipeline = build_model_pipeline(numeric_features, categorical_features)
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=args.n_iter,
        scoring="recall",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=args.random_state,
    )

    logger.info(
        "Starting hyperparameter search (%d candidates x %d folds)", args.n_iter, args.cv_folds
    )
    search.fit(X_train, y_train)
    logger.info("Best CV recall: %.4f", search.best_score_)
    logger.info("Best params: %s", search.best_params_)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "best_cv_recall": search.best_score_,
        "best_params": search.best_params_,
        "test_balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred),
        "test_recall": recall_score(y_test, y_pred),
        "test_f1": f1_score(y_test, y_pred),
        "test_roc_auc": roc_auc_score(y_test, y_proba),
        "test_pr_auc": average_precision_score(y_test, y_proba),
    }
    logger.info("Test metrics:\n%s", json.dumps(metrics, indent=2, default=str))
    logger.info(
        "Classification report:\n%s",
        classification_report(y_test, y_pred, target_names=["No churn", "Churn"]),
    )

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    dump(best_model, args.model_path)
    logger.info("Saved trained pipeline to %s", args.model_path)

    metrics_path = args.metrics_path or args.model_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str))
    logger.info("Saved metrics to %s", metrics_path)

    return best_model, metrics


if __name__ == "__main__":
    train(parse_args())
