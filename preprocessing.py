"""
preprocessing.py
-----------------
Builds the Scikit-learn preprocessing pipeline used ahead of every model:
- numeric columns: median imputation + StandardScaler
- categorical columns: most-frequent imputation + OneHotEncoder

The resulting ColumnTransformer is combined with each estimator inside a
Pipeline in models.py, so preprocessing is always fit only on the training
split (no data leakage) and is bundled with the model for export.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_columns(df: pd.DataFrame, target_col: str):
    """
    Split the feature columns (i.e. all columns except the target) into
    numeric and categorical column name lists.

    Args:
        df: full dataframe (including target).
        target_col: name of the target column to exclude.

    Returns:
        (numeric_cols, categorical_cols)
    """
    feature_df = df.drop(columns=[target_col])
    numeric_cols = feature_df.select_dtypes(include="number").columns.tolist()
    categorical_cols = feature_df.select_dtypes(exclude="number").columns.tolist()
    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols, categorical_cols) -> ColumnTransformer:
    """
    Construct the ColumnTransformer that imputes, encodes, and scales
    features ahead of model training.

    Args:
        numeric_cols: list of numeric feature column names.
        categorical_cols: list of categorical feature column names.

    Returns:
        an (unfitted) ColumnTransformer.
    """
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    # remainder="drop" ensures any column not explicitly listed (there
    # shouldn't be any beyond numeric/categorical splits) is safely dropped.
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor
