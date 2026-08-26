# Customer Churn Prediction

Predicts whether a telecom customer will churn, using the [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset. The project covers the full path from raw CSV to a served model: a reusable preprocessing + training pipeline in `src/`, a Flask API for real-time scoring, a CLI for batch scoring, and a test suite — with the original exploratory notebooks kept for reference.

## Project structure

```
.
├── app.py                    # Flask API (health check + /predict)
├── data/
│   └── Telco_Customer_Churn.csv
├── model/
│   └── churn_pipeline.joblib # trained sklearn/imblearn pipeline
├── notebooks/
│   ├── 01_eda.ipynb          # exploratory data analysis
│   ├── 02_model_training.ipynb  # the training pipeline in notebook form
│   └── archive/              # earlier model-comparison notebooks, kept for history
├── src/
│   ├── config.py             # paths and constants
│   ├── data.py                # load_data / clean_data / split_features_target
│   ├── pipeline.py            # preprocessing + SMOTE + RandomForest pipeline
│   ├── train.py               # CLI: hyperparameter search, evaluation, save model
│   └── predict.py             # CLI + library: load model, score records
├── tests/
├── requirements.txt           # runtime dependencies
├── requirements-dev.txt       # + pytest, jupyter, plotting libs
└── Dockerfile
```

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate on Windows
pip install -r requirements-dev.txt   # or requirements.txt for a runtime-only install
```

## Training

Retrain the pipeline from `data/Telco_Customer_Churn.csv` and overwrite `model/churn_pipeline.joblib`:

```bash
python -m src.train
```

This runs a stratified 70/30 train-test split, a `RandomizedSearchCV` hyperparameter search (5-fold CV, optimizing recall) over a class-weighted Random Forest, evaluates on the held-out test set, and writes both the fitted pipeline and a `model/churn_pipeline.metrics.json` report. Useful flags:

```bash
python -m src.train --n-iter 40 --cv-folds 10 --test-size 0.2
```

Run `python -m src.train --help` for the full list.

## Serving predictions

### API

```bash
python app.py
# or: flask --app app run
```

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
    "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85
  }'
# {"churn_prediction": 1, "churn_probability": 0.636}
```

`POST /predict` also accepts a JSON array of records and returns a JSON array of results. `GET /health` returns `{"status": "ok"}`.

### Batch CLI

```bash
python -m src.predict customers.csv predictions.csv
```

Reads a CSV of raw customer records and writes it back out with `churn_prediction` and `churn_probability` columns appended. Extra columns such as `customerID` are preserved and ignored by the model.

### Docker

```bash
docker build -t churn-prediction .
docker run -p 5000:5000 churn-prediction
```

## Testing

```bash
pytest
```

Covers data cleaning/splitting logic and the end-to-end pipeline (fit, predict, and handling of unseen categorical values at inference time).

## Modeling approach

- **Preprocessing**: median imputation + standard scaling for numeric features (`tenure`, `MonthlyCharges`, `TotalCharges`); constant imputation + one-hot encoding (`handle_unknown="ignore"`) for the remaining categorical features.
- **Class imbalance**: the target is imbalanced (~26.5% churn). The pipeline applies SMOTE oversampling inside an `imblearn` pipeline — so resampling only ever touches the training fold, never validation/test data — combined with `class_weight="balanced"` on the classifier.
- **Model**: `RandomForestClassifier`, tuned via `RandomizedSearchCV` (20 iterations, 5-fold stratified CV, optimizing recall — missed churners are costlier than false alarms here).
- **Held-out test performance** (from the current `model/churn_pipeline.joblib`, threshold 0.5): recall 0.70, precision 0.55, F1 0.62, ROC-AUC 0.84, PR-AUC 0.63. `notebooks/02_model_training.ipynb` also explores lowering the decision threshold to ~0.47, which trades a little precision for recall (F1 0.64); the shipped API uses the standard 0.5 threshold, so revisit `src/predict.py` if the business tradeoff calls for the lower one.

`notebooks/archive/` holds the earlier model comparisons (KNN, SVM, Logistic Regression, AdaBoost, Naive Bayes, a voting classifier, and undersampling/oversampling experiments) that preceded settling on this SMOTE + Random Forest approach.

## Notes

- The model artifact (`model/churn_pipeline.joblib`) is committed directly for simplicity. For a larger project, prefer Git LFS or a model registry (e.g. MLflow) instead of checking binaries into git.
- `data/Telco_Customer_Churn.csv` is the full IBM sample dataset (7,043 rows); no PII beyond a synthetic customer ID.
