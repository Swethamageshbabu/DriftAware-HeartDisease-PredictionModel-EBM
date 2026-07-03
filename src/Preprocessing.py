"""
preprocessing.py
================

Preprocessing pipeline for the UCI Heart Disease dataset used in the
Drift-Aware Explainable Boosting Machine (EBM) framework.

This module performs:
- Cleveland cohort filtering
- Missing value handling
- Categorical encoding
- Feature harmonization
- Target generation

Author: Swetha M.
"""

import os
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

TEXT_COLUMNS = [
    "cp",
    "restecg",
    "slope",
    "thal",
    "sex",
    "fbs",
    "exang",
]

EXPECTED_NUMERIC = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
    "ca",
]

REPLACEMENTS = {
    "reversable defect": "reversible defect",
    "reversable": "reversible",
    "normal ": "normal",
    "non anginal": "non-anginal",
    "non angina": "non-anginal",
}


# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------

def safe_read(path: str) -> pd.DataFrame:
    """
    Read a CSV or Excel dataset.

    Parameters
    ----------
    path : str
        Path to the dataset.

    Returns
    -------
    pandas.DataFrame
    """

    extension = os.path.splitext(path)[1].lower()

    if extension in (".xls", ".xlsx"):
        return pd.read_excel(path)

    if extension in (".csv", ".txt"):
        return pd.read_csv(path)

    raise ValueError(
        "Unsupported file format. Use CSV or Excel."
    )


# ---------------------------------------------------------------------
# Main Preprocessing Function
# ---------------------------------------------------------------------

def preprocess_heart(
    input_path: str,
    output_path: str = "heart_cleveland_clean.csv",
    filter_site: str = "cleveland",
) -> pd.DataFrame:
    """
    Complete preprocessing pipeline for the UCI Heart Disease dataset.

    Parameters
    ----------
    input_path : str
        Raw dataset path.

    output_path : str
        Output CSV filename.

    filter_site : str
        Dataset to retain (default: Cleveland).

    Returns
    -------
    pandas.DataFrame
        Cleaned dataset.
    """

    # ----------------------------------------------------------
    # Load Dataset
    # ----------------------------------------------------------

    df = safe_read(input_path)

    print(f"Loaded dataset shape: {df.shape}")

    df.columns = [column.strip() for column in df.columns]

    # ----------------------------------------------------------
    # Replace missing value placeholders
    # ----------------------------------------------------------

    df.replace("?", np.nan, inplace=True)

    # ----------------------------------------------------------
    # Filter Cleveland Dataset
    # ----------------------------------------------------------

    if "dataset" in [column.lower() for column in df.columns]:

        dataset_column = next(
            column for column in df.columns
            if column.lower() == "dataset"
        )

        df[dataset_column] = (
            df[dataset_column]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        df = df[
            df[dataset_column].str.contains(
                filter_site.lower(),
                na=False
            )
        ].copy()

        print(
            f"Filtered dataset ({filter_site}) : {df.shape}"
        )

    else:

        print(
            "Dataset column not found. "
            "Proceeding without filtering."
        )

    # ----------------------------------------------------------
    # Remove unwanted columns
    # ----------------------------------------------------------

    for column in ["id", "dataset"]:

        if column in df.columns:

            df.drop(columns=column, inplace=True)

    # ----------------------------------------------------------
    # Normalize text columns
    # ----------------------------------------------------------

    for column in TEXT_COLUMNS:

        if column in df.columns:

            df[column] = (
                df[column]
                .astype(str)
                .str.strip()
                .str.lower()
                .replace({"nan": np.nan})
            )

    # ----------------------------------------------------------
    # Fix common naming inconsistencies
    # ----------------------------------------------------------

    for column in ["cp", "restecg", "slope", "thal"]:

        if column in df.columns:

            df[column] = df[column].replace(
                REPLACEMENTS
            )

    # ----------------------------------------------------------
    # Rename alternative column names
    # ----------------------------------------------------------

    if (
        "thalch" in df.columns
        and "thalach" not in df.columns
    ):

        df.rename(
            columns={"thalch": "thalach"},
            inplace=True,
        )

    # ----------------------------------------------------------
    # Category Mapping Dictionaries
    # ----------------------------------------------------------

    mapping_dicts = {}

    if "sex" in df.columns:

        mapping_dicts["sex"] = {
            "male": 1,
            "female": 0,
        }

    if "cp" in df.columns:

        mapping_dicts["cp"] = {
            "typical angina": 0,
            "atypical angina": 1,
            "non-anginal": 2,
            "asymptomatic": 3,
        }

    if "fbs" in df.columns:

        mapping_dicts["fbs"] = {
            "true": 1,
            "false": 0,
            "1": 1,
            "0": 0,
        }

    if "restecg" in df.columns:

        mapping_dicts["restecg"] = {
            "normal": 0,
            "lv hypertrophy": 1,
            "st-t abnormality": 2,
        }

    if "exang" in df.columns:

        mapping_dicts["exang"] = {
            "true": 1,
            "false": 0,
            "1": 1,
            "0": 0,
        }

    if "slope" in df.columns:

        mapping_dicts["slope"] = {
            "upsloping": 0,
            "flat": 1,
            "downsloping": 2,
        }

    if "thal" in df.columns:

        mapping_dicts["thal"] = {
            "normal": 0,
            "fixed defect": 1,
            "reversible defect": 2,
            "reversable defect": 2,
        }

    # ----------------------------------------------------------
    # Display unique values before encoding
    # ----------------------------------------------------------

    for column in mapping_dicts.keys():

        if column in df.columns:

            unique_values = (
                pd.Series(df[column].dropna().unique())
                .astype(str)
                .tolist()
            )

            print(f"\nUnique values for '{column}':")
            print(unique_values)

    # ----------------------------------------------------------
    # Apply categorical mappings
    # ----------------------------------------------------------

    for column, mapping in mapping_dicts.items():

        if column in df.columns:

            df[column] = (
                df[column]
                .map(mapping)
                .astype(float)
            )

    # ----------------------------------------------------------
    # Convert numerical columns
    # ----------------------------------------------------------

    for column in EXPECTED_NUMERIC:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    # ----------------------------------------------------------
    # Identify columns for imputation
    # ----------------------------------------------------------

    numeric_columns = (
        df.select_dtypes(include=[np.number])
        .columns
        .tolist()
    )

    if "num" in numeric_columns:

        numeric_columns.remove("num")

    categorical_columns = [
        column
        for column in mapping_dicts.keys()
        if column in df.columns
    ]

    # ----------------------------------------------------------
    # Missing value report
    # ----------------------------------------------------------

    print("\nMissing values BEFORE imputation:")

    print(
        df[categorical_columns + numeric_columns]
        .isna()
        .sum()
    )

    # ----------------------------------------------------------
    # Median imputation
    # ----------------------------------------------------------

    for column in numeric_columns:

        if df[column].isna().sum() > 0:

            median = df[column].median()

            df[column] = df[column].fillna(
                median
            )

            print(
                f"Imputed '{column}' using median = {median}"
            )

    # ----------------------------------------------------------
    # Mode imputation
    # ----------------------------------------------------------

    for column in categorical_columns:

        if df[column].isna().sum() > 0:

            mode = df[column].mode(dropna=True)

            fill_value = (
                mode.iloc[0]
                if not mode.empty
                else 0
            )

            df[column] = df[column].fillna(
                fill_value
            )

            print(
                f"Imputed '{column}' using mode = {fill_value}"
            )

    # ----------------------------------------------------------
    # Convert categorical columns
    # ----------------------------------------------------------

    for column in categorical_columns:

        df[column] = (
            df[column]
            .astype(int)
            .astype("category")
        )

    # ----------------------------------------------------------
    # Create binary target
    # ----------------------------------------------------------

    if "num" in df.columns:

        df["target"] = (
            df["num"] > 0
        ).astype(int)

        df.drop(
            columns="num",
            inplace=True,
        )

        print("\nTarget distribution:")

        print(df["target"].value_counts())

    else:

        print(
            "Warning: 'num' column not found."
        )

    # ----------------------------------------------------------
    # Final missing value check
    # ----------------------------------------------------------

    print(
        "\nRemaining missing values:",
        df.isna().sum().sum(),
    )

    empty_columns = [
        column
        for column in df.columns
        if df[column].isna().all()
    ]

    if empty_columns:

        df.drop(
            columns=empty_columns,
            inplace=True,
        )

    # ----------------------------------------------------------
    # Reorder columns
    # ----------------------------------------------------------

    ordered_columns = [
        column
        for column in df.columns
        if column != "target"
    ]

    if "target" in df.columns:

        ordered_columns.append("target")

    df = df[ordered_columns]

    # ----------------------------------------------------------
    # Save processed dataset
    # ----------------------------------------------------------

    df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nCleaned dataset saved to: {output_path}"
    )

    print("\nDataset Preview:")

    print(df.head())

    return df


# ---------------------------------------------------------------------
# Run Script
# ---------------------------------------------------------------------

if __name__ == "__main__":

    INPUT_PATH = (
        "../data/raw/heart_disease_uci.csv"
    )

    OUTPUT_PATH = (
        "../data/processed/heart_cleveland_clean.csv"
    )

    if not os.path.exists(INPUT_PATH):

        raise FileNotFoundError(
            f"Dataset not found: {INPUT_PATH}"
        )

    preprocess_heart(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
    )
