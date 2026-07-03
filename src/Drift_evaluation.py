#!/usr/bin/env python3
"""
Biomedical Interpretability & Feature Drift Analysis
-----------------------------------------------------
Computes Population Stability Index (PSI), Kolmogorov-Smirnov (KS) tests,
and changes in local model feature explanations (interpretability drift) 
between a reference training set and a new evaluation dataset using an EBM.
"""

import argparse
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def parse_arguments():
    """Parses command line arguments for the drift evaluation framework."""
    parser = argparse.ArgumentParser(
        description="Compute data and interpretability drift metrics."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained EBM model (.joblib)",
    )
    parser.add_argument(
        "--reference_path",
        type=str,
        required=True,
        help="Path to the original reference training dataset (CSV or Excel)",
    )
    parser.add_argument(
        "--new_data_path",
        type=str,
        required=True,
        help="Path to the new evaluation dataset (CSV or Excel)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./drift_reports",
        help="Directory to store all generated data drift outputs",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="target",
        help="Name of the target label column",
    )
    parser.add_argument(
        "--psi_moderate",
        type=float,
        default=0.10,
        help="Threshold flag for moderate PSI drift",
    )
    parser.add_argument(
        "--psi_major",
        type=float,
        default=0.25,
        help="Threshold flag for major PSI drift",
    )
    parser.add_argument(
        "--contrib_alert",
        type=float,
        default=0.02,
        help="Absolute delta threshold for feature contribution shifts",
    )
    return parser.parse_args()


def safe_read(path):
    """Safely extracts dataframes from CSV or Excel spreadsheet formats."""
    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv", ".txt"]:
        return pd.read_csv(path)
    elif ext in [".xls", ".xlsx"]:
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type extension encountered: {ext}")


def calculate_psi(expected, actual, bins=10):
    """Calculates Population Stability Index using stable quantile bins."""
    expected = np.array(expected.dropna())
    actual = np.array(actual.dropna())
    if len(expected) == 0 or len(actual) == 0:
        return np.nan
    try:
        breaks = np.unique(np.percentile(expected, np.linspace(0, 100, bins + 1)))
        if len(breaks) <= 1:
            return 0.0
        expected_counts, _ = np.histogram(expected, bins=breaks)
        actual_counts, _ = np.histogram(actual, bins=breaks)
        expected_perc = expected_counts / expected_counts.sum()
        actual_perc = actual_counts / actual_counts.sum()
        
        eps = 1e-8
        expected_perc = np.clip(expected_perc, eps, 1)
        actual_perc = np.clip(actual_perc, eps, 1)
        
        psi_val = np.sum((expected_perc - actual_perc) * np.log(expected_perc / actual_perc))
        return float(psi_val)
    except Exception:
        return np.nan


def safe_ks(a, b):
    """Computes basic two-sample Kolmogorov-Smirnov test statistic."""
    try:
        a = np.array(a.dropna())
        b = np.array(b.dropna())
        if len(a) < 2 or len(b) < 2:
            return np.nan
        stat, _ = ks_2samp(a, b)
        return float(stat)
    except Exception:
        return np.nan


def align_columns(reference_feature_df, new_raw_df, target_col):
    """Aligns feature dimensions by matching structural column layouts."""
    ref_cols = list(reference_feature_df.columns)
    new_aligned_df = new_raw_df.copy()

    for c in ref_cols:
        if c not in new_aligned_df.columns:
            new_aligned_df[c] = np.nan

    extras = [c for c in new_aligned_df.columns if c not in ref_cols and c != target_col]
    if extras:
        new_aligned_df = new_aligned_df.drop(columns=extras)

    return new_aligned_df[ref_cols]


def compute_local_contributions(model, X, y_true):
    """Extracts local score explanations from the Explainable Boosting Machine."""
    preds_proba = model.predict_proba(X)[:, 1]
    preds_class = model.predict(X)

    if y_true is None:
        raise ValueError("Target labels are mandatory for EBM local explanation extraction pipelines.")

    explain_local_obj = model.explain_local(X, y_true)
    contrib_list = []
    explanation_term_names = None

    for i in range(len(X)):
        instance_data = explain_local_obj.data(i)
        if instance_data and 'scores' in instance_data and 'names' in instance_data:
            scores = np.array(instance_data['scores'])
            names = instance_data['names']

            if explanation_term_names is None:
                explanation_term_names = names

            if len(scores) == len(explanation_term_names):
                contrib_list.append(scores)
            else:
                contrib_list.append(np.zeros(len(explanation_term_names)))
        else:
            if explanation_term_names is not None:
                contrib_list.append(np.zeros(len(explanation_term_names)))
            else:
                raise RuntimeError("Failed to resolve baseline features out of initial database indices.")

    if not contrib_list:
        raise RuntimeError("Zero valid contribution footprints located over given matrix structures.")

    return preds_proba, preds_class, np.array(contrib_list), explanation_term_names


def summarize_batch_predictions(X, preds_proba, preds_class, contribs, explanation_term_names):
    """Builds records mapping individual prediction outputs alongside top contributing markers."""
    mean_abs_contrib = np.abs(contribs).mean(axis=0)
    df_mean = pd.DataFrame({"Feature": explanation_term_names, "MeanAbsContribution": mean_abs_contrib})
    df_mean.sort_values("MeanAbsContribution", ascending=False, inplace=True)

    topk_list = []
    for i in range(len(X)):
        row = pd.DataFrame({"Feature": explanation_term_names, "Contribution": contribs[i]})
        row["AbsContribution"] = row["Contribution"].abs()
        row = row.sort_values("AbsContribution", ascending=False)
        topk = row.head(5)[["Feature", "Contribution"]].to_dict(orient="records")
        topk_list.append(topk)

    df_out = X.copy().reset_index(drop=True)
    df_out["pred_proba"] = preds_proba
    df_out["pred_class"] = preds_class
    df_out["top5_contribs"] = topk_list
    return df_out, df_mean


def compute_feature_drift(reference_df, new_df):
    """Calculates PSI and KS across numerical and categorical features."""
    features = reference_df.columns.tolist()
    psi_scores, ks_scores = {}, {}
    
    for f in features:
        a = reference_df[f]
        b = new_df[f] if f in new_df.columns else pd.Series(dtype=object)
        
        if pd.api.types.is_numeric_dtype(a):
            psi_scores[f] = calculate_psi(a, b, bins=10)
            ks_scores[f] = safe_ks(a, b)
        else:
            a_str = a.fillna("nan").astype(str)
            b_str = b.fillna("nan").astype(str)
            cats = sorted(list(set(a_str.unique()).union(set(b_str.unique()))))
            a_counts = a_str.value_counts(normalize=True).reindex(cats, fill_value=0.0).values + 1e-8
            b_counts = b_str.value_counts(normalize=True).reindex(cats, fill_value=0.0).values + 1e-8
            psi_scores[f] = float(np.sum((a_counts - b_counts) * np.log(a_counts / b_counts)))
            ks_scores[f] = np.nan

    df = pd.DataFrame({"Feature": list(psi_scores.keys()), "PSI": list(psi_scores.values()), "KS": [ks_scores[k] for k in psi_scores.keys()]})
    df.sort_values("PSI", ascending=False, inplace=True)
    return df


def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)

    def flag_drift(psi_val):
        if pd.isna(psi_val): return "unknown"
        if psi_val >= args.psi_major: return "major"
        if psi_val >= args.psi_moderate: return "moderate"
        return "none"

    # 1. Loading data structures
    print(f"⏳ Extracting model asset from {args.model_path}...")
    model = joblib.load(args.model_path)
    ref_df = safe_read(args.reference_path)
    new_df = safe_read(args.new_data_path)

    # 2. Extracting target tags and alignments
    for name, df in [("reference", ref_df), ("evaluation", new_df)]:
        if args.target not in df.columns:
            raise KeyError(f"Target column '{args.target}' required but missing inside the {name} data file.")

    ref_target = ref_df[args.target].astype(int)
    ref_X = ref_df.drop(columns=[args.target]).copy()

    new_target = new_df[args.target].astype(int)
    new_X_raw = new_df.drop(columns=[args.target]).copy()
    new_X = align_columns(ref_X, new_X_raw, args.target)

    # 3. Model local predictions
    print("🔍 Computing evaluation dataset local EBM explanation values...")
    preds_proba, preds_class, contribs, new_terms = compute_local_contributions(model, new_X, new_target)
    preds_df, mean_contrib_df = summarize_batch_predictions(new_X, preds_proba, preds_class, contribs, new_terms)

    preds_df.to_csv(os.path.join(args.output_dir, "drift_report_predictions_and_explanations.csv"), index=False)
    mean_contrib_df.to_csv(os.path.join(args.output_dir, "drift_report_mean_contributions.csv"), index=False)

    # 4. Global distribution drifting metrics
    print("🔍 Mapping empirical feature distribution shifts (PSI & KS)...")
    psi_df = compute_feature_drift(ref_X, new_X)
    psi_df.to_csv(os.path.join(args.output_dir, "drift_report_feature_psi_ks.csv"), index=False)

    # 5. Explanatory impact tracking
    print("🔍 Evaluating local contribution delta adjustments against reference framework...")
    _, _, contribs_ref, ref_terms = compute_local_contributions(model, ref_X, ref_target)
    mean_abs_ref = np.abs(contribs_ref).mean(axis=0)
    df_ref_mean_contrib = pd.DataFrame({"Feature": ref_terms, "MeanAbsContribution_Reference": mean_abs_ref})

    merged = df_ref_mean_contrib.merge(mean_contrib_df.rename(columns={"MeanAbsContribution": "MeanAbsContribution_New"}), on="Feature", how="left")
    merged["Delta"] = merged["MeanAbsContribution_New"] - merged["MeanAbsContribution_Reference"]
    merged["AbsDelta"] = merged["Delta"].abs()
    merged = merged.sort_values("AbsDelta", ascending=False)
    merged.to_csv(os.path.join(args.output_dir, "drift_report_interpretability_drift.csv"), index=False)

    # 6. Structuring vulnerability profiles
    combined = merged.merge(psi_df, on="Feature", how="left")
    combined["PSI_flag"] = combined["PSI"].apply(flag_drift)
    combined["Contrib_flag"] = combined["AbsDelta"].apply(lambda v: "alert" if v >= args.contrib_alert else "ok")

    eps = 1e-8
    psi_vals = combined["PSI"].fillna(0).values.astype(float)
    psi_norm = (psi_vals - psi_vals.min()) / (psi_vals.max() - psi_vals.min() + eps)
    delta_vals = combined["AbsDelta"].fillna(0).values.astype(float)
    delta_norm = (delta_vals - delta_vals.min()) / (delta_vals.max() - delta_vals.min() + eps)
    
    combined["RiskScore"] = 0.6 * psi_norm + 0.4 * delta_norm
    combined.sort_values("RiskScore", ascending=False, inplace=True)
    combined.to_csv(os.path.join(args.output_dir, "drift_report_drift_summary.csv"), index=False)

    # 7. Saving warnings logs
    top_warnings = combined[combined["PSI_flag"].isin(["major", "moderate"]) | (combined["Contrib_flag"] == "alert")]
    top_warnings.to_csv(os.path.join(args.output_dir, "drift_report_warnings.csv"), index=False)

    # 8. Generation of analytical plots
    topN = min(8, len(combined))
    
    fig, ax = plt.subplots(figsize=(8, max(4, topN * 0.5)))
    ax.barh(combined["Feature"].head(topN)[::-1], combined["AbsDelta"].head(topN)[::-1])
    ax.set_xlabel("Absolute Change in Mean Contribution")
    ax.set_title("Top Interpretability Drift (New vs Reference)")
    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "drift_report_top_interpretability_drift.png"), dpi=200)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(7, max(4, topN * 0.5)))
    ax2.barh(combined["Feature"].head(topN)[::-1], combined["PSI"].head(topN)[::-1])
    ax2.set_xlabel("Population Stability Index (PSI)")
    ax2.set_title("Top PSI shifts by Feature")
    plt.tight_layout()
    fig2.savefig(os.path.join(args.output_dir, "drift_report_top_psis.png"), dpi=200)
    plt.close(fig2)

    # 9. Clear terminal report representation
    overall_psi_max = psi_df["PSI"].max()
    overall_flag = flag_drift(overall_psi_max)
    print("\n==== Drift Report Summary ====")
    print(f"Overall Max PSI: {overall_psi_max:.4f} -> Global Drift Evaluation Level: {overall_flag.upper()}")
    
    if overall_flag in ("major", "moderate") or len(top_warnings) > 0:
        print("\nTop features showing high risk or explanation anomalies:")
        print(combined[["Feature", "PSI", "AbsDelta", "PSI_flag", "Contrib_flag"]].head(10).to_string(index=False))
    else:
        print("\nNo critical structural feature or representation shifts noted.")
    print(f"\n💾 Analytical data structures exported successfully to: '{args.output_dir}'")


if __name__ == "__main__":
    main()
