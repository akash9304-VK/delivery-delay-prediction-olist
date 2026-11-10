
from pathlib import Path

import joblib
import pandas as pd


# Load trained model and metadata
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "delivery_delay_xgboost.joblib"

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
feature_columns = artifact["feature_columns"]
threshold = artifact["threshold"]


REQUIRED_COLUMNS = [
    "item_count",
    "seller_count",
    "total_item_price",
    "total_freight",
    "order_purchase_timestamp",
    "order_estimated_delivery_date",
    "customer_state",
]


def prepare_features(data):

    data = data.copy()

    missing = set(REQUIRED_COLUMNS) - set(data.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # Convert dates
    data["order_purchase_timestamp"] = pd.to_datetime(
        data["order_purchase_timestamp"],
        errors="raise"
    )

    data["order_estimated_delivery_date"] = pd.to_datetime(
        data["order_estimated_delivery_date"],
        errors="raise"
    )

    # Calculate estimated delivery duration
    data["estimated_delivery_days"] = (
        data["order_estimated_delivery_date"].dt.normalize()
        - data["order_purchase_timestamp"].dt.normalize()
    ).dt.days

    if (data["estimated_delivery_days"] <= 0).any():
        raise ValueError("Estimated delivery date must be after purchase date.")

    # Date features
    purchase_date = data["order_purchase_timestamp"]
    estimated_date = data["order_estimated_delivery_date"]

    data["order_purchase_year"] = purchase_date.dt.year
    data["order_purchase_month"] = purchase_date.dt.month
    data["order_purchase_day_of_week"] = purchase_date.dt.dayofweek
    data["order_purchase_hour"] = purchase_date.dt.hour

    data["order_estimated_delivery_month"] = estimated_date.dt.month
    data["order_estimated_day_of_week"] = estimated_date.dt.dayofweek

    # Validate state categories
    state_features = [
        col for col in feature_columns
        if col.startswith("customer_state_")
    ]

    encoded_states = {
        col.replace("customer_state_", "")
        for col in state_features
    }

    # AC is the reference state from the training encoding
    known_states = encoded_states | {"AC"}

    data["customer_state"] = (
        data["customer_state"].astype(str).str.upper()
    )

    unknown_states = set(data["customer_state"]) - known_states

    if unknown_states:
        raise ValueError(
            f"Unknown customer states: {sorted(unknown_states)}"
        )

    # One-hot encoding
    data = pd.get_dummies(
        data,
        columns=["customer_state"],
        dtype=int
    )

    # Match exact training feature structure
    X = data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    return X


def predict_orders(data):

    X = prepare_features(data)

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (probabilities >= threshold).astype(int)

    results = data.copy()

    results["estimated_delivery_days"] = X["estimated_delivery_days"]
    results["late_delivery_probability"] = probabilities
    results["prediction"] = predictions

    results["risk_status"] = results["prediction"].map({
        0: "Likely On Time",
        1: "Potential Late Delivery"
    })

    return results
