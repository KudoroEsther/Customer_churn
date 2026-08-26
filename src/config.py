from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "Telco_Customer_Churn.csv"
MODEL_PATH = ROOT_DIR / "model" / "churn_pipeline.joblib"

ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

RANDOM_STATE = 42
