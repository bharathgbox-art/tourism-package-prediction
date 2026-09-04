"""
train.py
--------
Section 3 – Model Building with Experimentation Tracking

1. Loads the train/test splits from the workflow artifact.
2. Defines a Random Forest classifier and a hyperparameter grid.
3. Tunes the model with GridSearchCV and logs every trial to MLflow.
4. Evaluates the best model and prints a classification report.
5. Saves the best model to tourism_project/deployment/ so the pipeline
   can commit it back to the repository.
"""

import os
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score,
    roc_auc_score, classification_report,
)

# ── Paths ───────────────────────────────────────────────────────────────────
MODEL_DIR  = "tourism_project/deployment"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
RANDOM_STATE = 42

# ── Hyperparameter grid ──────────────────────────────────────────────────────
# A moderate grid that balances search depth with CI/CD runtime
PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth":    [None, 5, 10],
    "min_samples_split": [2, 5],
    "class_weight": ["balanced"],   # Compensate for the ~81/19 class imbalance
}


def train_model() -> None:
    """Full training pipeline: load splits → tune → log → evaluate → save."""

    # 1. Load data splits from the artifact directory -----------------------
    print("Loading train/test splits...")
    X_train = pd.read_csv("Xtrain.csv")
    X_test  = pd.read_csv("Xtest.csv")
    y_train = pd.read_csv("ytrain.csv").squeeze()  # Series
    y_test  = pd.read_csv("ytest.csv").squeeze()
    print(f"  Train: {X_train.shape} | Test: {X_test.shape}")

    # 2. MLflow experiment setup --------------------------------------------
    mlflow.set_experiment("Tourism-Package-Prediction")

    with mlflow.start_run(run_name="RandomForest_GridSearch"):

        # 3. Define and tune the model --------------------------------------
        base_model = RandomForestClassifier(random_state=RANDOM_STATE)

        # GridSearchCV with 5-fold stratified cross-validation
        # Optimise for recall (reduce false negatives = missed buyers)
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=PARAM_GRID,
            cv=5,
            scoring="recall",
            n_jobs=-1,
            verbose=1,
        )
        print("\nRunning GridSearchCV...")
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_model  = grid_search.best_estimator_
        print(f"  Best parameters: {best_params}")

        # 4. Log all tuned parameters to MLflow ----------------------------
        print("\nLogging parameters and metrics to MLflow...")
        mlflow.log_params(best_params)

        # 5. Evaluate on the held-out test set -----------------------------
        y_pred  = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        acc     = accuracy_score(y_test, y_pred)
        recall  = recall_score(y_test, y_pred)
        f1      = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        # Log metrics to MLflow
        mlflow.log_metric("accuracy",  acc)
        mlflow.log_metric("recall",    recall)
        mlflow.log_metric("f1_score",  f1)
        mlflow.log_metric("roc_auc",   roc_auc)

        print(f"\n=== Model Evaluation ===")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Recall   : {recall:.4f}")
        print(f"  F1 Score : {f1:.4f}")
        print(f"  ROC-AUC  : {roc_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["Not Taken", "Taken"]))

        # 6. Log the model artifact to MLflow ------------------------------
        mlflow.sklearn.log_model(best_model, "random_forest_model")

    # 7. Save the best model to the deployment folder ----------------------
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")
    print("Training complete. The pipeline will now commit this model to the repo.")


if __name__ == "__main__":
    train_model()
