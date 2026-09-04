"""
data_register.py
----------------
Section 1 – Data Registration

Reads tourism.csv from the repository data folder, verifies that all
expected columns are present, and prints a short dataset summary.
The CSV stays inside the GitHub repository – no external data store needed.
"""

import sys
import pandas as pd

# ── Path to the dataset inside the repository ──────────────────────────────
DATA_PATH = "tourism_project/data/tourism.csv"

# ── All 20 columns defined in the data dictionary ──────────────────────────
EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "Occupation", "Gender", "NumberOfPersonVisiting", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "OwnCar",
    "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
    "PitchSatisfactionScore", "ProductPitched", "NumberOfFollowups",
    "DurationOfPitch",
]

def register_dataset(path: str) -> pd.DataFrame:
    """Load the dataset, validate columns, and print a summary."""

    # 1. Load ----------------------------------------------------------------
    print(f"Loading dataset from: {path}")
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"ERROR: File not found at '{path}'.")
        print("Please upload tourism.csv into tourism_project/data/ first.")
        sys.exit(1)

    # 2. Column validation ---------------------------------------------------
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"ERROR: The following expected columns are missing: {missing_cols}")
        sys.exit(1)
    else:
        print("✓  All expected columns are present.")

    # 3. Dataset summary -----------------------------------------------------
    print("\n=== Dataset Summary ===")
    print(f"  Rows    : {df.shape[0]:,}")
    print(f"  Columns : {df.shape[1]}")
    print(f"  Target distribution (ProdTaken):")
    vc = df["ProdTaken"].value_counts()
    for val, cnt in vc.items():
        pct = cnt / len(df) * 100
        label = "Purchased" if val == 1 else "Not Purchased"
        print(f"    {val} ({label}) : {cnt:,}  ({pct:.1f}%)")
    print(f"\n  Missing values per column:")
    missing = df.isnull().sum()
    missing_nonzero = missing[missing > 0]
    if missing_nonzero.empty:
        print("    None")
    else:
        for col, cnt in missing_nonzero.items():
            print(f"    {col}: {cnt}")
    print("\n=== Data Types ===")
    print(df.dtypes.to_string())
    print("\nDataset registration complete.")
    return df


if __name__ == "__main__":
    register_dataset(DATA_PATH)
