#!/usr/bin/env python3
"""
Biomedical Concept Drift Simulation & Evaluation
-------------------------------------------------
Simulates progressive data drift on a baseline test dataset by adding 
Gaussian noise to numeric features, tests the trained EBM model's resilience, 
and measures statistical drift using the Kolmogorov-Smirnov (KS) test.
"""

import argparse
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score, roc_auc_score


def parse_arguments():
    """Parses command line arguments for the drift simulation script."""
    parser = argparse.ArgumentParser(
        description="Simulate feature drift and evaluate model robustness."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained EBM model file (.joblib)",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        required=True,
        help="Path to the baseline testing Excel (.xlsx) file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./drift_outputs",
        help="Directory to save drifted datasets and summary files",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="target",
        help="Name of the target label column",
    )
    parser.add_argument(
        "--baseline_acc",
        type=float,
        default=0.867,
        help="Baseline test accuracy without drift (for plotting)",
    )
    parser.add_argument(
        "--baseline_auc",
        type=float,
        default=0.944,
        help="Baseline test AUC without drift (for plotting)",
    )
    return parser.parse_args()


def add_drift(df, cols, drift_ratio):
    """Simulates numeric drift by adding Gaussian noise relative to standard deviation."""
    drifted = df.copy()
    for c in cols:
        std_dev = drift_ratio * df[c].std()
        noise = np.random.normal(0, std_dev, size=len(df))
        drifted[c] = df[c] + noise
    return drifted


def main():
    args = parse_arguments()
    np.random.seed(42)  # Set seed for reproducible drift simulation

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load trained assets and baseline data
    print(f"⏳ Loading model from {args.model_path}...")
    model = joblib.load(args.model_path)
    
    print(f"⏳ Loading test data from {args.test_path}...")
    test_df = pd.read_excel(args.test_path)

    X_test = test_df.drop(columns=[args.target])
    y_test = test_df[args.target]

    numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    drift_levels = [0.05, 0.10, 0.15]

    print(f"✅ Loaded baseline test data: {X_test.shape}\n")

    # 2. Generate and save progressively drifted datasets
    drifted_sets = {}
    for ratio in drift_levels:
        drifted_df = add_drift(X_test, numeric_cols, ratio)
        drifted_full = drifted_df.copy()
        drifted_full[args.target] = y_test

        name = f"Testing_Cleveland_Drift_{int(ratio * 100)}"
        drifted_sets[name] = drifted_full

        # Save variations to output directory
        csv_path = os.path.join(args.output_dir, f"{name}.csv")
        xlsx_path = os.path.join(args.output_dir, f"{name}.xlsx")
        drifted_full.to_csv(csv_path, index=False)
        drifted_full.to_excel(xlsx_path, index=False)
        print(f"💾 Saved drifted variant: {csv_path}")

    print("\n✅ All drifted datasets stored successfully.")

    # 3. Evaluate model performance & track statistical drift metrics
    print("\n🔍 Evaluating model robustness across scenarios...")
    results = []
    
    for name, df_drifted in drifted_sets.items():
        X_drifted = df_drifted.drop(columns=[args.target])
        y_pred = model.predict(X_drifted)
        y_proba = model.predict_proba(X_drifted)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        # Compute Kolmogorov-Smirnov (KS) drift statistic for continuous features
        ks_vals = {col: ks_2samp(X_test[col], X_drifted[col])[0] for col in numeric_cols}
        avg_ks = np.mean(list(ks_vals.values()))

        results.append({
            "Scenario": name,
            "Accuracy": acc,
            "AUC": auc,
            "Avg_KS_Drift": avg_ks
        })

    drift_summary = pd.DataFrame(results)
    summary_path = os.path.join(args.output_dir, "Performance_vs_Drift_Summary.csv")
    drift_summary.to_csv(summary_path, index=False)

    print("\n📊 Drift Performance Summary:")
    print(drift_summary.to_string(index=False))

    # 4. Generate and save stability curve plot
    plt.figure(figsize=(6, 4))
    drift_percentages = [0, 5, 10, 15]
    accuracies = [args.baseline_acc, *list(drift_summary["Accuracy"])]
    aucs = [args.baseline_auc, *list(drift_summary["AUC"])]

    plt.plot(drift_percentages, accuracies, marker='o', label='Accuracy')
    plt.plot(drift_percentages, aucs, marker='o', label='AUC')
    
    plt.xlabel("Simulated Drift (%)")
    plt.ylabel("Performance Metric Value")
    plt.title("Model Stability Under Progressive Feature Drift")
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(args.output_dir, "drift_stability_curve.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n📈 Stability chart exported to: {plot_path}")
    print(f"💾 Metrics report saved to: {summary_path}")


if __name__ == "__main__":
    main()
