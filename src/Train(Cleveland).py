"""
train_ebm.py
============

Train an Explainable Boosting Machine (EBM) for heart disease prediction
using the Cleveland Heart Disease dataset.

This module:
- Loads the cleaned dataset
- Splits data into Train / Validation / Test sets
- Trains an Explainable Boosting Machine (EBM)
- Evaluates model performance

Author: Swetha M.
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from interpret.glassbox import ExplainableBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    roc_curve,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

RANDOM_STATE = 42

DATA_PATH = "../data/processed/heart_cleveland_clean.csv"

MODEL_PATH = "../models/ebm_heart_cleveland.joblib"

FEATURE_IMPORTANCE_PATH = "../results/ebm_feature_importance.csv"

SHAPE_SUMMARY_PATH = "../results/ebm_shape_summary.csv"

ROC_PATH = "../results/roc_cleveland.png"


# ---------------------------------------------------------
# Main Training Pipeline
# ---------------------------------------------------------

def main():

    # -----------------------------------------------------
    # Load Dataset
    # -----------------------------------------------------

    if not os.path.exists(DATA_PATH):

        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print(f"Loaded dataset: {df.shape}")

    print(df.head(3))

    # -----------------------------------------------------
    # Features and Target
    # -----------------------------------------------------

    X = df.drop(columns=["target"])

    y = df["target"]

    # -----------------------------------------------------
    # Train / Validation / Test Split
    # -----------------------------------------------------

    X_train_temp, X_test, y_train_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_temp,
        y_train_temp,
        test_size=0.1765,
        stratify=y_train_temp,
        random_state=RANDOM_STATE,
    )

    print("\nDataset Split")

    print(
        f"Train : {len(X_train)} "
        f"({100*len(X_train)/len(df):.1f}%)"
    )

    print(
        f"Validation : {len(X_val)} "
        f"({100*len(X_val)/len(df):.1f}%)"
    )

    print(
        f"Test : {len(X_test)} "
        f"({100*len(X_test)/len(df):.1f}%)"
    )

    # -----------------------------------------------------
    # Train EBM
    # -----------------------------------------------------

    print("\nTraining Explainable Boosting Machine...")

    ebm = ExplainableBoostingClassifier(
        random_state=RANDOM_STATE
    )

    ebm.fit(X_train, y_train)

    print("Training completed.")

    # -----------------------------------------------------
    # Validation Evaluation
    # -----------------------------------------------------

    y_val_pred = ebm.predict(X_val)

    y_val_proba = ebm.predict_proba(X_val)[:, 1]

    val_accuracy = accuracy_score(
        y_val,
        y_val_pred,
    )

    val_auc = roc_auc_score(
        y_val,
        y_val_proba,
    )

    print("\nValidation Performance")

    print(f"Accuracy : {val_accuracy:.3f}")

    print(f"AUC      : {val_auc:.3f}")

    # -----------------------------------------------------
    # Test Evaluation
    # -----------------------------------------------------

    y_test_pred = ebm.predict(X_test)

    y_test_proba = ebm.predict_proba(X_test)[:, 1]

    test_accuracy = accuracy_score(
        y_test,
        y_test_pred,
    )

    test_auc = roc_auc_score(
        y_test,
        y_test_proba,
    )

    print("\nFinal Test Performance")

    print(f"Accuracy : {test_accuracy:.3f}")

    print(f"AUC      : {test_auc:.3f}")

    print("\nClassification Report\n")

    print(
        classification_report(
            y_test,
            y_test_pred,
        )
    )

    # -----------------------------------------------------
    # ROC Curves
    # -----------------------------------------------------

    fpr_val, tpr_val, _ = roc_curve(
        y_val,
        y_val_proba,
    )

    fpr_test, tpr_test, _ = roc_curve(
        y_test,
        y_test_proba,
    )
      # -----------------------------------------------------
    # Plot ROC Curves
    # -----------------------------------------------------

    plt.figure(figsize=(6, 5))

    plt.plot(
        fpr_val,
        tpr_val,
        label=f"Validation (AUC = {val_auc:.3f})",
    )

    plt.plot(
        fpr_test,
        tpr_test,
        "--",
        label=f"Test (AUC = {test_auc:.3f})",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        "k--",
        linewidth=0.8,
    )

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title(
        "ROC Curve - Cleveland Heart Disease (EBM)"
    )

    plt.legend()

    plt.grid(True)

    os.makedirs("../results", exist_ok=True)

    plt.savefig(
        ROC_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    # -----------------------------------------------------
    # Global Feature Importance
    # -----------------------------------------------------

    print("\nExtracting Feature Importance...")

    ebm_global = ebm.explain_global()

    feature_importance = pd.DataFrame(
        {
            "Feature": ebm_global.data()["names"],
            "Importance": ebm_global.data()["scores"],
        }
    )

    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False,
    )

    print(
        "\nTop 10 Important Features\n"
    )

    print(feature_importance.head(10))

    feature_importance.to_csv(
        FEATURE_IMPORTANCE_PATH,
        index=False,
    )

    # -----------------------------------------------------
    # Shape Summary
    # -----------------------------------------------------

    shape_summary = pd.DataFrame(
        {
            "Feature": ebm_global.data()["names"],
            "Score": ebm_global.data()["scores"],
        }
    )

    shape_summary.to_csv(
        SHAPE_SUMMARY_PATH,
        index=False,
    )

    # -----------------------------------------------------
    # Save Trained Model
    # -----------------------------------------------------

    os.makedirs("../models", exist_ok=True)

    joblib.dump(
        ebm,
        MODEL_PATH,
    )

    print(
        f"\nModel saved successfully:\n{MODEL_PATH}"
    )

    print(
        "\nTraining pipeline completed successfully."
    )


# ---------------------------------------------------------
# Run Script
# ---------------------------------------------------------

if __name__ == "__main__":

    main()
