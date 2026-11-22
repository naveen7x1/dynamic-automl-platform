# 🧪 Dynamic AutoML & Interactive EDA Benchmarking Platform

A no-code Streamlit app for uploading any tabular CSV dataset and getting
automated preprocessing, interactive EDA, and multi-model benchmarking for
either **classification** or **regression** — auto-detected from your target
column.

## Features

- **CSV Upload & Preview** — shape, dtypes, missing-value counts, data preview.
- **Target Selection & Task Detection** — auto-detects classification vs. regression.
- **Automated Preprocessing** — median/most-frequent imputation, one-hot encoding, scaling, built with `Pipeline` / `ColumnTransformer`.
- **Interactive EDA** — histograms, correlation heatmap, box plots, target distribution (Plotly, tab/expander-based).
- **Dual-Task ML Engine** — trains 5 baseline models per task type on an 80/20 split.
- **Benchmarking Table & Chart** — sortable metrics table plus a bar chart comparing models on the primary metric.
- **Model Export** — download the best (or manually chosen) trained pipeline as a single `.pkl` file.

## Project Structure

```
automl_app/
├── app.py              # Streamlit UI: sidebar controls + 4 main tabs
├── preprocessing.py     # ColumnTransformer / Pipeline builder
├── eda.py               # Plotly chart builders
├── models.py             # Baseline models, training, metrics, benchmarking
├── utils.py               # Task detection, target encoding, helpers
├── requirements.txt
└── README.md
```

## Setup

1. **Create a virtual environment** (recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Run

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`).

## Free deployment

1. Push this project to a **public** GitHub repository.
2. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/) with
   GitHub and select **Create app**.
3. Choose this repository, its branch, and `app.py` as the main file, then
   select **Deploy**.

Streamlit Community Cloud will install the packages listed in
`requirements.txt` and provide a public app URL when the deployment finishes.

## Usage

1. Upload a CSV file in the sidebar.
2. Pick your **target column** — the app previews the detected task type (classification/regression).
3. Adjust the test set size / random seed if desired.
4. Click **🚀 Run AutoML** to preprocess the data and benchmark all baseline models.
5. Explore results across the **Data Overview**, **EDA**, **Model Benchmarking**, and **Export** tabs.
6. In **Export**, pick the best (auto) or a manually chosen model and download the fitted preprocessing + model pipeline as a `.pkl` file.

### Loading an exported model later

```python
import joblib

bundle = joblib.load("model_pipeline.pkl")
pipeline = bundle["pipeline"]          # full preprocessing + model Pipeline
task_type = bundle["task_type"]        # "classification" or "regression"
label_encoder = bundle["label_encoder"]  # None for regression

predictions = pipeline.predict(new_data_df)  # pass raw, unprocessed features
if label_encoder is not None:
    predictions = label_encoder.inverse_transform(predictions)
```

## Models Benchmarked

| Classification         | Regression            |
|-------------------------|------------------------|
| Logistic Regression     | Linear Regression      |
| Decision Tree           | Decision Tree          |
| Random Forest           | Random Forest          |
| KNN                     | KNN                    |
| Gradient Boosting       | Gradient Boosting      |

## Metrics Reported

- **Classification:** Accuracy, Precision (macro), Recall (macro), F1-score (macro), ROC-AUC
- **Regression:** R², MAE, MSE, RMSE

## Notes

- Task detection heuristics live in `utils.detect_task_type` — low-cardinality numeric columns are treated as classification targets; everything else numeric is treated as regression, and non-numeric columns are always classification.
- Preprocessing is fit **only on the training split** inside each model's pipeline, avoiding data leakage, and is bundled with the model on export so it can be reused directly on new raw data.
