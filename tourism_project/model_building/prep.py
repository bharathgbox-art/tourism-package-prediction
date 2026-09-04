"""
prep.py
-------
Section 2 – Data Preparation

Loads tourism.csv directly from the repository data folder, performs
cleaning and preprocessing, then saves the train/test splits as CSV files.
The GitHub Actions workflow uploads these split files as an artifact so
the next job (model training) can download and use them.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── Paths ───────────────────────────────────────────────────────────────────
DATA_PATH  = "tourism_project/data/tourism.csv"
RANDOM_STATE = 42
TEST_SIZE    = 0.20   # 80 % train / 20 % test

# ── Columns that must be dropped ────────────────────────────────────────────
# CustomerID is a row identifier with no predictive value.
DROP_COLS = ["CustomerID"]

# ── Categorical columns to label-encode ─────────────────────────────────────
CAT_COLS = [
    "TypeofContact", "Occupation", "Gender",
    "MaritalStatus", "Designation", "ProductPitched",
]


def prepare_data(data_path: str) -> None:
    """Full data preparation pipeline: clean → encode → split → save."""

    # 1. Load ----------------------------------------------------------------
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"  Shape: {df.shape}")

    # 2. Drop unnecessary columns -------------------------------------------
    df.drop(columns=DROP_COLS, inplace=True, errors="ignore")
    print(f"  Dropped columns: {DROP_COLS}")

    # 3. Fix known data-quality issues ----------------------------------------
    # 'Fe Male' is a known typo for 'Female' in the Gender column
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace("Fe Male", "Female")
        print("  Corrected 'Fe Male' → 'Female' in Gender column.")

    # 4. Handle missing values -----------------------------------------------
    # Numeric columns: fill with median (robust to outliers)
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    # Remove target from imputation list
    num_cols = [c for c in num_cols if c != "ProdTaken"]
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Categorical columns: fill with mode
    for col in CAT_COLS:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    print(f"  Missing values after imputation: {df.isnull().sum().sum()}")

    # 5. Label-encode categorical columns ------------------------------------
    le = LabelEncoder()
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
    print(f"  Label-encoded columns: {[c for c in CAT_COLS if c in df.columns]}")

    # 6. Split features / target --------------------------------------------
    X = df.drop(columns=["ProdTaken"])
    y = df["ProdTaken"]
    print(f"  Features shape: {X.shape} | Target shape: {y.shape}")

    # 7. Train / test split (stratified to preserve class balance) -----------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"  Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    # 8. Save splits as CSV files (the workflow uploads these as an artifact) -
    X_train.to_csv("Xtrain.csv", index=False)
    X_test.to_csv("Xtest.csv",  index=False)
    y_train.to_csv("ytrain.csv", index=False)
    y_test.to_csv("ytest.csv",  index=False)
    print("  Saved: Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")
    print("Data preparation complete.")


if __name__ == "__main__":
    prepare_data(DATA_PATH)
