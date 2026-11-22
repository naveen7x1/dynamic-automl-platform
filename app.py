"""
app.py
------
Dynamic AutoML & Interactive EDA Benchmarking Platform
========================================================
A no-code Streamlit app that lets a user upload any tabular CSV, pick a
target column, and get:
  1. Data overview (shape, dtypes, missing values, preview)
  2. Interactive EDA (histograms, correlation heatmap, box plots, target dist.)
  3. Automated preprocessing + multi-model benchmarking (classification or
     regression, auto-detected)
  4. One-click export of the best (or a chosen) trained model + pipeline

Run with:  streamlit run app.py
"""

import io

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.model_selection import train_test_split

from preprocessing import split_columns, build_preprocessor
from eda import plot_histograms, plot_correlation_heatmap, plot_boxplots, plot_target_distribution
from models import get_models, train_and_evaluate, get_primary_metric, get_best_model_name
from utils import detect_task_type, encode_target_if_needed, missing_value_summary, format_metric


# --------------------------------------------------------------------------- #
# Page config & session state
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Dynamic AutoML & Interactive EDA Benchmarking Platform",
    page_icon="🧪",
    layout="wide",
)

DEFAULTS = {
    "df": None,
    "filename": None,
    "target_col": None,
    "task_type": None,
    "results_df": None,
    "trained_pipelines": None,
    "label_encoder": None,
    "numeric_cols": None,
    "categorical_cols": None,
    "has_run": False,
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------------------------------- #
# Sidebar: upload, target selection, run controls
# --------------------------------------------------------------------------- #
st.sidebar.title("🧪 Controls")

uploaded_file = st.sidebar.file_uploader("Upload a CSV dataset", type=["csv"])

if uploaded_file is not None:
    # Only re-read the file if a new one was uploaded, so we don't wipe
    # benchmarking results on every rerun.
    if st.session_state["filename"] != uploaded_file.name:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state["df"] = df
            st.session_state["filename"] = uploaded_file.name
            # Reset downstream state for a freshly uploaded file.
            st.session_state["results_df"] = None
            st.session_state["trained_pipelines"] = None
            st.session_state["has_run"] = False
        except Exception as e:
            st.sidebar.error(f"Could not read CSV: {e}")

df = st.session_state["df"]

target_col = None
test_size = 0.2
run_clicked = False

if df is not None:
    st.sidebar.success(f"Loaded **{st.session_state['filename']}** — {df.shape[0]} rows × {df.shape[1]} cols")

    target_col = st.sidebar.selectbox("🎯 Select target column", df.columns.tolist())

    # Live task-type preview in the sidebar (before running the pipeline)
    if target_col:
        preview_task = detect_task_type(df[target_col])
        st.sidebar.caption(f"Detected task: **{preview_task.capitalize()}**")

    test_size = st.sidebar.slider("Test set size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    random_state = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)

    st.sidebar.markdown("---")
    run_clicked = st.sidebar.button("🚀 Run AutoML", type="primary", use_container_width=True)
else:
    st.sidebar.info("Upload a CSV file to get started.")


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("🧪 Dynamic AutoML & Interactive EDA Benchmarking Platform")
st.caption(
    "Upload a CSV, pick a target column, and get automated preprocessing, "
    "interactive EDA, and multi-model benchmarking — no code required."
)

if df is None:
    st.info("👈 Upload a CSV dataset in the sidebar to begin.")
    st.stop()


# --------------------------------------------------------------------------- #
# Run the AutoML pipeline when the button is clicked
# --------------------------------------------------------------------------- #
if run_clicked and target_col:
    with st.spinner("Running preprocessing and training models..."):
        try:
            # Drop rows where the target itself is missing — can't train/evaluate on those.
            work_df = df.dropna(subset=[target_col]).copy()

            task_type = detect_task_type(work_df[target_col])
            numeric_cols, categorical_cols = split_columns(work_df, target_col)

            X = work_df.drop(columns=[target_col])
            y_raw = work_df[target_col]
            y, label_encoder = encode_target_if_needed(y_raw, task_type)

            stratify = y if task_type == "classification" and y.nunique() > 1 else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=int(random_state), stratify=stratify
            )

            preprocessor = build_preprocessor(numeric_cols, categorical_cols)
            models = get_models(task_type, random_state=int(random_state))
            results_df, trained_pipelines = train_and_evaluate(
                models, preprocessor, X_train, X_test, y_train, y_test, task_type
            )

            # Persist everything needed by the Benchmarking / Export tabs.
            st.session_state["task_type"] = task_type
            st.session_state["target_col"] = target_col
            st.session_state["results_df"] = results_df
            st.session_state["trained_pipelines"] = trained_pipelines
            st.session_state["label_encoder"] = label_encoder
            st.session_state["numeric_cols"] = numeric_cols
            st.session_state["categorical_cols"] = categorical_cols
            st.session_state["has_run"] = True

            st.sidebar.success("✅ Training complete!")
        except Exception as e:
            st.sidebar.error(f"Training failed: {e}")


# --------------------------------------------------------------------------- #
# Main tabs
# --------------------------------------------------------------------------- #
tab_overview, tab_eda, tab_bench, tab_export = st.tabs(
    ["📋 Data Overview", "🔍 EDA", "🏆 Model Benchmarking", "📦 Export"]
)

# ---- Tab 1: Data Overview -------------------------------------------------- #
with tab_overview:
    st.subheader("Dataset Shape")
    c1, c2 = st.columns(2)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])

    st.subheader("Column Dtypes & Missing Values")
    st.dataframe(missing_value_summary(df), use_container_width=True)

    st.subheader("Data Preview")
    n_rows = st.slider("Rows to preview", min_value=5, max_value=min(100, len(df)), value=min(10, len(df)))
    st.dataframe(df.head(n_rows), use_container_width=True)

# ---- Tab 2: EDA -------------------------------------------------------------- #
with tab_eda:
    if target_col is None:
        st.info("Select a target column in the sidebar to unlock target-aware EDA.")
    numeric_cols_eda, categorical_cols_eda = split_columns(df, target_col) if target_col else (
        df.select_dtypes(include="number").columns.tolist(), []
    )
    all_numeric = df.select_dtypes(include="number").columns.tolist()

    with st.expander("📈 Histograms (numeric columns)", expanded=True):
        if all_numeric:
            hist_figs = plot_histograms(df, all_numeric)
            cols = st.columns(2)
            for i, (col_name, fig) in enumerate(hist_figs.items()):
                cols[i % 2].plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numeric columns found.")

    with st.expander("🌡️ Correlation Heatmap"):
        heatmap_fig = plot_correlation_heatmap(df, all_numeric)
        if heatmap_fig:
            st.plotly_chart(heatmap_fig, use_container_width=True)
        else:
            st.warning("Need at least 2 numeric columns for a correlation heatmap.")

    with st.expander("📦 Box Plots (outlier detection)"):
        if all_numeric:
            box_figs = plot_boxplots(df, all_numeric)
            cols = st.columns(2)
            for i, (col_name, fig) in enumerate(box_figs.items()):
                cols[i % 2].plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numeric columns found.")

    with st.expander("🎯 Target Variable Distribution", expanded=True):
        if target_col:
            target_task_type = detect_task_type(df[target_col])
            target_fig = plot_target_distribution(df, target_col, target_task_type)
            st.plotly_chart(target_fig, use_container_width=True)
        else:
            st.info("Select a target column in the sidebar to view its distribution.")

# ---- Tab 3: Model Benchmarking ---------------------------------------------- #
with tab_bench:
    if not st.session_state["has_run"]:
        st.info("Configure your target column and click **🚀 Run AutoML** in the sidebar to benchmark models.")
    else:
        results_df = st.session_state["results_df"]
        task_type = st.session_state["task_type"]
        primary_metric = get_primary_metric(task_type)

        st.subheader(f"Task Detected: {task_type.capitalize()}")
        st.caption(f"Target column: **{st.session_state['target_col']}**")

        st.subheader("Benchmarking Table")
        sort_col = st.selectbox(
            "Sort by metric", [c for c in results_df.columns if c != "Model"],
            index=[c for c in results_df.columns if c != "Model"].index(primary_metric),
        )
        ascending = sort_col in ("MAE", "MSE", "RMSE")  # lower is better for error metrics
        display_df = results_df.sort_values(sort_col, ascending=ascending).reset_index(drop=True)
        st.dataframe(
            display_df.style.format({c: "{:.4f}" for c in display_df.columns if c != "Model"}),
            use_container_width=True,
        )

        st.subheader(f"Model Comparison — {primary_metric}")
        bar_fig = px.bar(
            results_df.sort_values(primary_metric, ascending=ascending),
            x="Model", y=primary_metric, color="Model",
            title=f"{primary_metric} by Model",
        )
        bar_fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(bar_fig, use_container_width=True)

        best_name = get_best_model_name(results_df, task_type)
        st.success(f"🏆 Best model on **{primary_metric}**: **{best_name}**")

# ---- Tab 4: Export ------------------------------------------------------------ #
with tab_export:
    if not st.session_state["has_run"]:
        st.info("Run AutoML first (sidebar) to unlock model export.")
    else:
        results_df = st.session_state["results_df"]
        trained_pipelines = st.session_state["trained_pipelines"]
        task_type = st.session_state["task_type"]
        best_name = get_best_model_name(results_df, task_type)

        st.subheader("Select a Model to Export")
        model_names = list(trained_pipelines.keys())
        default_idx = model_names.index(best_name)
        selection_mode = st.radio("Selection mode", ["Best model (auto)", "Choose manually"], horizontal=True)

        if selection_mode == "Best model (auto)":
            chosen_model = best_name
            st.info(f"Auto-selected best model: **{chosen_model}**")
        else:
            chosen_model = st.selectbox("Model", model_names, index=default_idx)

        chosen_pipeline = trained_pipelines[chosen_model]

        # Bundle the pipeline together with metadata useful for future inference.
        export_bundle = {
            "pipeline": chosen_pipeline,  # preprocessing + model, ready for .predict()
            "task_type": task_type,
            "target_col": st.session_state["target_col"],
            "label_encoder": st.session_state["label_encoder"],  # None for regression
            "model_name": chosen_model,
        }

        buffer = io.BytesIO()
        joblib.dump(export_bundle, buffer)
        buffer.seek(0)

        st.download_button(
            label=f"⬇️ Download '{chosen_model}' pipeline (.pkl)",
            data=buffer,
            file_name=f"{chosen_model.replace(' ', '_').lower()}_pipeline.pkl",
            mime="application/octet-stream",
            use_container_width=True,
        )

        with st.expander("How to load this model later"):
            st.code(
                "import joblib\n"
                "bundle = joblib.load('model_pipeline.pkl')\n"
                "pipeline = bundle['pipeline']\n"
                "predictions = pipeline.predict(new_data_df)  # new_data_df: raw, unprocessed features",
                language="python",
            )
