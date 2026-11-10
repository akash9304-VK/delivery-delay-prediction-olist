"""
Delivery Delay Prediction - Olist Brazilian E-Commerce
Cleaned end-to-end workflow:
- Load data from MySQL
- Run data checks and create EDA charts
- Engineer prediction-time features
- Create stratified train/validation/test splits
- Tune XGBoost with RandomizedSearchCV
- Select a threshold using validation data
- Evaluate once on the untouched final test set
- Save the model and inference metadata

Run from the project root:
    python src/explore_data.py
"""

from __future__ import annotations

import getpass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sqlalchemy import URL, create_engine, text
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TARGET_COLUMN = "is_late"
DB_USER = "root"
DB_HOST = "localhost"
DB_PORT = 3306
DB_NAME = "delivery_delay"
DB_VIEW = "order_delay_model_data"


# =============================================================================
# 2. LOAD DATA FROM MYSQL
# =============================================================================

def load_data() -> pd.DataFrame:
    """Connect to MySQL and load the modelling view."""
    password = getpass.getpass("Enter MySQL password: ")

    database_url = URL.create(
        "mysql+pymysql",
        username=DB_USER,
        password=password,
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
    )

    engine = create_engine(database_url)

    try:
        with engine.connect() as connection:
            print("Database connection test:", connection.execute(text("SELECT 1")).scalar())

        data = pd.read_sql(f"SELECT * FROM {DB_VIEW}", engine)
    finally:
        engine.dispose()

    for column in [
        "order_purchase_timestamp",
        "order_estimated_delivery_date",
    ]:
        data[column] = pd.to_datetime(data[column], errors="coerce")

    print("\nData loaded successfully.")
    print("Dataset shape:", data.shape)
    return data


# =============================================================================
# 3. DATA QUALITY CHECKS AND EDA
# =============================================================================

def run_data_checks(data: pd.DataFrame) -> None:
    print("\n--- Missing values ---")
    print(data.isna().sum())

    print("\n--- Duplicate rows ---")
    print(data.duplicated().sum())

    print("\n--- Data types ---")
    print(data.dtypes)

    print("\n--- Target distribution ---")
    print(data[TARGET_COLUMN].value_counts())
    print(data[TARGET_COLUMN].value_counts(normalize=True))


def create_eda_charts(data: pd.DataFrame) -> None:
    """Create EDA charts and save them under reports/figures."""
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 100, 120, 140, 160]

    duration_data = data.copy()
    duration_data["delivery_duration_group"] = pd.cut(
        duration_data["estimated_delivery_days"],
        bins=bins,
        include_lowest=True,
    )

    duration_summary = (
        duration_data.groupby("delivery_duration_group", observed=False)
        .agg(
            number_of_orders=(TARGET_COLUMN, "size"),
            late_delivery_rate=(TARGET_COLUMN, "mean"),
        )
        .reset_index()
    )
    duration_summary["late_delivery_rate"] *= 100

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=duration_summary,
        x="delivery_duration_group",
        y="late_delivery_rate",
    )
    plt.xlabel("Estimated delivery duration (days)")
    plt.ylabel("Late-delivery rate (%)")
    plt.title("Late-Delivery Rate by Estimated Delivery Duration")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "late_delivery_rate_by_estimated_days.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    state_summary = (
        data.groupby("customer_state")
        .agg(
            number_of_orders=(TARGET_COLUMN, "size"),
            number_of_late_orders=(TARGET_COLUMN, "sum"),
            late_delivery_rate=(TARGET_COLUMN, "mean"),
        )
        .reset_index()
    )
    state_summary["late_delivery_rate"] *= 100
    top_10_states = state_summary.nlargest(10, "late_delivery_rate")

    print("\n--- Top 10 states by late-delivery rate ---")
    print(top_10_states.to_string(index=False))

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=top_10_states,
        x="late_delivery_rate",
        y="customer_state",
    )
    plt.xlabel("Late-delivery rate (%)")
    plt.ylabel("Customer state")
    plt.title("Top 10 Customer States by Late-Delivery Rate")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "late_delivery_rate_by_state.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"\nEDA charts saved to: {FIGURES_DIR}")

 # =============================================================================
# 4. FEATURE ENGINEERING
# =============================================================================

def build_features(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Build features intended to be available when an order is placed.

    Actual delivery date is not included as a feature. It is used only
    upstream to define the target in the SQL view.
    """
    required_columns = [
        "estimated_delivery_days",
        "item_count",
        "seller_count",
        "total_item_price",
        "total_freight",
        "customer_state",
        "order_purchase_timestamp",
        "order_estimated_delivery_date",
        TARGET_COLUMN,
    ]

    missing = sorted(set(required_columns) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    features = data.copy()

    features["order_purchase_year"] = (
        features["order_purchase_timestamp"].dt.year
    )
    features["order_purchase_month"] = (
        features["order_purchase_timestamp"].dt.month
    )
    features["order_purchase_day_of_week"] = (
        features["order_purchase_timestamp"].dt.dayofweek
    )
    features["order_purchase_hour"] = (
        features["order_purchase_timestamp"].dt.hour
    )
    features["order_estimated_delivery_month"] = (
        features["order_estimated_delivery_date"].dt.month
    )
    features["order_estimated_day_of_week"] = (
        features["order_estimated_delivery_date"].dt.dayofweek
    )

    features = pd.get_dummies(
        features,
        columns=["customer_state"],
        drop_first=True,
        dtype=int,
    )

    base_columns = [
        "estimated_delivery_days",
        "item_count",
        "seller_count",
        "total_item_price",
        "total_freight",
        "order_purchase_year",
        "order_purchase_month",
        "order_purchase_day_of_week",
        "order_purchase_hour",
        "order_estimated_delivery_month",
        "order_estimated_day_of_week",
    ]

    state_columns = sorted(
        column
        for column in features.columns
        if column.startswith("customer_state_")
    )

    feature_columns = base_columns + state_columns
    X = features[feature_columns].copy()
    y = features[TARGET_COLUMN].astype(int).copy()

    if X.isna().any().any():
        missing_counts = X.isna().sum()
        raise ValueError(
            "Missing values found in features:\n"
            f"{missing_counts[missing_counts > 0]}"
        )

    print("\nFeature matrix shape:", X.shape)
    print("Target shape:", y.shape)
    print("Number of features:", len(feature_columns))

    return X, y, feature_columns

# =============================================================================
# 5. STRATIFIED TRAIN / VALIDATION / TEST SPLIT
# =============================================================================

def split_data(X, y):
    """Create 70% training, 15% validation, and 15% final test sets."""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.17647,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    print("\n--- Split shapes ---")
    print("Training:", X_train.shape, y_train.shape)
    print("Validation:", X_val.shape, y_val.shape)
    print("Final test:", X_test.shape, y_test.shape)

    print("\n--- Target proportions ---")
    print("Training:\n", y_train.value_counts(normalize=True))
    print("Validation:\n", y_val.value_counts(normalize=True))
    print("Final test:\n", y_test.value_counts(normalize=True))

    return X_train, X_val, X_test, y_train, y_val, y_test

# =============================================================================
# 6. XGBOOST HYPERPARAMETER TUNING
# =============================================================================

def tune_xgboost(X_train, y_train) -> RandomizedSearchCV:
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    parameter_distributions = {
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "scale_pos_weight": [6, 8, 10, 12],
        "min_child_weight": [1, 3, 5],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }

    base_model = XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=1,
    )

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=parameter_distributions,
        n_iter=30,
        scoring="f1",
        cv=cv,
        verbose=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )

    search.fit(X_train, y_train)

    print("\n--- RandomizedSearchCV results ---")
    print("Best parameters:")
    print(search.best_params_)
    print("Best mean CV F1-score:", search.best_score_)

    return search


# =============================================================================
# 7. SELECT CLASSIFICATION THRESHOLD USING VALIDATION DATA
# =============================================================================

def select_threshold(model, X_val, y_val):
    validation_probabilities = model.predict_proba(X_val)[:, 1]
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    results = []

    for threshold in thresholds:
        predictions = (validation_probabilities >= threshold).astype(int)

        results.append(
            {
                "threshold": threshold,
                "precision": precision_score(
                    y_val, predictions, zero_division=0
                ),
                "recall": recall_score(
                    y_val, predictions, zero_division=0
                ),
                "f1_score": f1_score(
                    y_val, predictions, zero_division=0
                ),
            }
        )

    results_df = pd.DataFrame(results)
    print("\n--- Validation threshold comparison ---")
    print(results_df.to_string(index=False))

    best_row = results_df.loc[results_df["f1_score"].idxmax()]
    selected_threshold = float(best_row["threshold"])

    print(f"\nSelected threshold: {selected_threshold:.2f}")
    print(f"Validation F1-score: {best_row['f1_score']:.4f}")

    return selected_threshold


# =============================================================================
# 8. FINAL TEST EVALUATION
# =============================================================================

def evaluate_final_test(model, X_test, y_test, threshold):
    test_probabilities = model.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= threshold).astype(int)

    print("\n--- FINAL TEST RESULTS ---")
    print(f"Threshold: {threshold:.2f}")
    print("Accuracy:", accuracy_score(y_test, test_predictions))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, test_predictions))

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            test_predictions,
            zero_division=0,
        )
    )

# =============================================================================
# 9. SAVE MODEL AND INFERENCE METADATA
# =============================================================================

def save_model(model, feature_columns, threshold):
    artifact = {
        "model": model,
        "feature_columns": feature_columns,
        "threshold": threshold,
        "target_column": TARGET_COLUMN,
    }

    model_path = MODELS_DIR / "delivery_delay_xgboost.joblib"
    joblib.dump(artifact, model_path)
    print(f"\nModel artifact saved to: {model_path}")


# =============================================================================
# 10. MAIN
# =============================================================================

def main():
    data = load_data()

    run_data_checks(data)
    create_eda_charts(data)

    X, y, feature_columns = build_features(data)

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_data(X, y)

    search = tune_xgboost(X_train, y_train)
    best_model = search.best_estimator_

    selected_threshold = select_threshold(
        best_model,
        X_val,
        y_val,
    )

    # Use the final test set only after model and threshold selection.
    evaluate_final_test(
        best_model,
        X_test,
        y_test,
        selected_threshold,
    )

    save_model(
        best_model,
        feature_columns,
        selected_threshold,
    )

    print("\nWorkflow completed successfully.")


if __name__ == "__main__":
    main()