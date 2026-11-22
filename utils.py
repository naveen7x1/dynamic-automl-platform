"""
utils.py
--------
Small, shared helper functions used across the app:
- automatic task type detection (classification vs regression)
- target encoding for classification tasks
- missing value summaries
- metric formatting for display
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def detect_task_type(series: pd.Series) -> str:
    """
    Heuristically detect whether a target column should be modeled as
    classification or regression, based on its dtype and cardinality.

    Rules:
      - Non-numeric dtype (object, category, bool) -> classification
      - Numeric, float dtype -> regression, UNLESS very few unique
        values are present (<= 10), in which case it's treated as
        classification (e.g. a float column that's really a coded label).
      - Numeric, integer dtype -> classification if cardinality is low
        (<= 20 unique values), else regression (e.g. counts, prices).

    Args:
        series: the target column (pandas Series).

    Returns:
        "classification" or "regression"
    """
    series = series.dropna()
    if pd.api.types.is_numeric_dtype(series):
        n_unique = series.nunique()
        if pd.api.types.is_float_dtype(series):
            return "classification" if n_unique <= 10 else "regression"
        else:
            return "classification" if n_unique <= 20 else "regression"
    else:
        return "classification"


def encode_target_if_needed(y: pd.Series, task_type: str):
    """
    Label-encode the target for classification tasks so downstream metrics
    (especially ROC-AUC and multi-class handling) work with a consistent
    numeric class representation. Regression targets are returned unchanged.

    Args:
        y: raw target Series.
        task_type: "classification" or "regression".

    Returns:
        (encoded_y, label_encoder_or_None)
    """
    if task_type == "classification":
        le = LabelEncoder()
        y_encoded = le.fit_transform(y.astype(str))
        return pd.Series(y_encoded, index=y.index, name=y.name), le
    return y, None


def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build a per-column summary of dtype, missing count, and missing %."""
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    summary = pd.DataFrame({
        "Column": df.columns,
        "Dtype": df.dtypes.astype(str).values,
        "Missing Count": missing.values,
        "Missing %": pct.values,
    })
    return summary.sort_values("Missing Count", ascending=False).reset_index(drop=True)


def format_metric(value) -> str:
    """Format a numeric metric for display, gracefully handling NaN/None."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"{value:.4f}"
