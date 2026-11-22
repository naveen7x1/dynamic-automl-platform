"""
models.py
---------
The dual-task ML engine: defines baseline models for classification and
regression, trains each (wrapped with a cloned copy of the preprocessing
pipeline) on the training split, evaluates on the held-out test split, and
assembles a benchmarking results table.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    r2_score, mean_absolute_error, mean_squared_error,
)


def get_models(task_type: str, random_state: int = 42) -> dict:
    """
    Return the dictionary of baseline models to benchmark for the given task.

    Args:
        task_type: "classification" or "regression".
        random_state: seed used by every stochastic estimator, for reproducibility.

    Returns:
        dict of {model_name: unfitted estimator}
    """
    if task_type == "classification":
        return {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
            "Decision Tree": DecisionTreeClassifier(random_state=random_state),
            "Random Forest": RandomForestClassifier(random_state=random_state, n_estimators=200),
            "KNN": KNeighborsClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
        }
    elif task_type == "regression":
        return {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(random_state=random_state),
            "Random Forest": RandomForestRegressor(random_state=random_state, n_estimators=200),
            "KNN": KNeighborsRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(random_state=random_state),
        }
    else:
        raise ValueError(f"Unknown task_type: {task_type!r}")


def _safe_roc_auc(y_test, y_proba, classes):
    """Compute ROC-AUC (binary or macro-averaged OVR multi-class), or NaN on failure."""
    try:
        n_classes = len(classes)
        if n_classes == 2:
            return roc_auc_score(y_test, y_proba[:, 1])
        return roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro", labels=classes)
    except Exception:
        return np.nan


def train_and_evaluate(models: dict, preprocessor, X_train, X_test, y_train, y_test, task_type: str):
    """
    Fit each model (bundled with its own cloned preprocessor) on the training
    split and evaluate it on the test split.

    Args:
        models: dict of {name: unfitted estimator}, from get_models().
        preprocessor: unfitted ColumnTransformer, cloned per model to avoid
            any shared, mutable fitted state between pipelines.
        X_train, X_test, y_train, y_test: train/test splits (raw, unprocessed
            features -- the pipeline handles preprocessing internally).
        task_type: "classification" or "regression".

    Returns:
        (results_df, trained_pipelines)
        - results_df: pandas DataFrame, one row per model, with metric columns.
        - trained_pipelines: dict of {model_name: fitted sklearn Pipeline}
          (preprocessor + model bundled together, ready for export/inference).
    """
    rows = []
    trained_pipelines = {}

    for name, model in models.items():
        pipe = Pipeline(steps=[
            ("preprocessor", clone(preprocessor)),
            ("model", model),
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        if task_type == "classification":
            row = {
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
                "Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
                "F1-score": f1_score(y_test, y_pred, average="macro", zero_division=0),
                "ROC-AUC": np.nan,
            }
            if hasattr(pipe.named_steps["model"], "predict_proba"):
                try:
                    y_proba = pipe.predict_proba(X_test)
                    classes = pipe.named_steps["model"].classes_
                    row["ROC-AUC"] = _safe_roc_auc(y_test, y_proba, classes)
                except Exception:
                    pass
        else:
            mse = mean_squared_error(y_test, y_pred)
            row = {
                "Model": name,
                "R2": r2_score(y_test, y_pred),
                "MAE": mean_absolute_error(y_test, y_pred),
                "MSE": mse,
                "RMSE": np.sqrt(mse),
            }

        rows.append(row)
        trained_pipelines[name] = pipe

    results_df = pd.DataFrame(rows)
    return results_df, trained_pipelines


def get_primary_metric(task_type: str) -> str:
    """Return the metric name used to rank models and drive the comparison bar chart."""
    return "Accuracy" if task_type == "classification" else "R2"


def get_best_model_name(results_df: pd.DataFrame, task_type: str) -> str:
    """Return the name of the model with the best score on the primary metric."""
    metric = get_primary_metric(task_type)
    idx = results_df[metric].idxmax()  # higher is better for both Accuracy and R2
    return results_df.loc[idx, "Model"]
