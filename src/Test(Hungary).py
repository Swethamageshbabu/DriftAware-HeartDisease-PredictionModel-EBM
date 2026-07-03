#!/usr/bin/main env python3
"""
Biomedical Concept Drift Baseline Model (EBM)
---------------------------------------------
Trains an Explainable Boosting Classifier on pre-split datasets
(Training, Validation, and Testing) and outputs evaluation metrics.
"""

import argparse
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    roc_curve,
)


def parse_arguments():
    """Parses command line arguments for data paths."""
    parser = argparse.ArgumentParser(
        description="Train an EBM model for Biomedical Concept Drift."
    )
    parser.add_argument(
        "--train_path",
        type=str,
        required=True,
        help="Path to the training CSV file",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        required=True,
        help="Path to the validation Excel (.xlsx) file",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        required=True,
        help="Path to the testing CSV file",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="target",
        help="Name of the target label column",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save output files",
    )
    return parser.parse_args()


def load_data(train_path, val_path, test_path):
    """Loads datasets based on their extensions."""
    print("⏳ Loading datasets...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_excel(val_path)
    test_df = pd.read_csv(test_path)

    print(
        f"✅ Loaded datasets:\n"
        f"   - Training:   {train_df.shape}\n"
        f"   - Validation: {val_df.shape}\n"
        f"   - Testing:    {test_df.shape}"
    )
    return train_df, val_df, test_df


def prepare_features(train_df, val_df, test_df, target):
    """Splits target column from features."""
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]

    X_val = val_df.drop(columns=[target])
    y_val = val_df[target]

    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    print(f"Feature count: {X_train.shape[1]} | Target column: '{target}'")
    return X_train, y_train, X_val, y_val, X_test, y_test


def evaluate_model(model, X, y, dataset_name):
    """Evaluates the model and returns predictions, probabilities, accuracy, and AUC."""
    preds = model.predict(X)
    probas = model.predict_proba(X)[:, 1]
    acc = accuracy_score(y, preds)
    auc = roc_auc_score(y, probas)
    print(f"📊 {dataset_name} Results — Accuracy: {acc:.3f}, AUC: {auc:.3f}")
    return preds, probas, acc, auc


def main():
    args = parse_arguments()
    RANDOM_STATE = 42

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load and split datasets
    train_df, val_df, test_df = load_data(
        args.train_path, args.val_path, args.test_path
    )
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_features(
        train_df, val_df, test_df, args.target
    )

    # Train model
    print("\n🚀 Training Explainable Boosting Machine...")
    ebm = ExplainableBoostingClassifier(random_state=RANDOM_STATE)
    ebm.fit(X_train, y_train)
    print("✅ Model training completed.")

    # Evaluate
    print("\n🔍 Evaluating model...")
    _, y_val_proba, _, val_auc = evaluate_model(ebm, X_val, y_val, "Validation")
    y_test_pred, y_test_proba, _, test_auc = evaluate_model(
        ebm, X_test, y_test, "Test"
    )

    print("\nClassification Report (Test):\n", classification_report(y_test, y_test_pred))

    # Save ROC Curve Plot
    fpr_v, tpr_v, _ = roc_curve(y_val, y_val_proba)
    fpr_t, tpr_t, _ = roc_curve(y_test, y_test_proba)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr_v, tpr_v, label=f"Validation (AUC={val_auc:.3f})")
    plt.plot(fpr_t, tpr_t, label=f"Test (AUC={test_auc:.3f})", linestyle="--")
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Heart Disease (EBM)")
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(args.output_dir, "roc_curve.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📈 ROC Curve saved to {plot_path}")

    # Interpretability & Outputs
    ebm_global = ebm.explain_global()
    feature_importance = pd.DataFrame(
        {
            "Feature": ebm_global.data()["names"],
            "Importance": ebm_global.data()["scores"],
        }
    ).sort_values("Importance", ascending=False)

    print("\n🔍 Top 10 Features by EBM Importance:\n", feature_importance.head(10))

    # Save outputs
    feature_importance.to_csv(
        os.path.join(args.output_dir, "ebm_feature_importance.csv"), index=False
    )
    feature_importance.to_csv(
        os.path.join(args.output_dir, "ebm_shape_summary.csv"), index=False
    )
    joblib.dump(ebm, os.path.join(args.output_dir, "ebm_heart_model.joblib"))

    print(f"\n💾 All assets saved successfully to '{args.output_dir}' directory.")


if __name__ == "__main__":
    main()
