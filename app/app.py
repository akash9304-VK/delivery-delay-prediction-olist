
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay

# Allow imports from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_orders  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "models" / "delivery_delay_xgboost.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"

GITHUB_URL = "https://github.com/akash9304-VK"
LINKEDIN_URL = "https://www.linkedin.com/in/akashsonwalkar/"

REQUIRED_COLUMNS = [
    "item_count",
    "seller_count",
    "total_item_price",
    "total_freight",
    "order_purchase_timestamp",
    "order_estimated_delivery_date",
    "customer_state",
]

STATES = [
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS",
    "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC",
    "SE", "SP", "TO","KA",
]

FEATURE_LABELS = {
    "order_purchase_month": "Purchase month",
    "order_purchase_year": "Purchase year",
    "order_purchase_day": "Purchase day",
    "order_purchase_hour": "Purchase hour",
    "order_purchase_dayofweek": "Purchase weekday",
    "order_estimated_delivery_month": "Estimated delivery month",
    "order_estimated_delivery_year": "Estimated delivery year",
    "order_estimated_delivery_day": "Estimated delivery day",
    "seller_count": "Number of sellers",
    "item_count": "Number of items",
    "total_item_price": "Total item price",
    "total_freight": "Total freight",
}

# ----------------------------------------------------------
# Page setup and styling
# ----------------------------------------------------------
st.set_page_config(page_title="Delivery Delay Predictor", page_icon="📦", layout="wide")

st.markdown(
    """
<style>
.stApp { background-color: #F5F7FB; }
.block-container { max-width: 1250px; padding-top: 2rem; padding-bottom: 3rem; }
h1 { color: #172B4D !important; font-weight: 750 !important; }
h2, h3 { color: #253858 !important; font-weight: 650 !important; }

[data-testid="stMetric"] {
    background: white; border: 1px solid #E3E8F0; padding: 18px;
    border-radius: 12px; box-shadow: 0 2px 8px rgba(20, 40, 80, 0.04);
}
[data-testid="stMetricLabel"] { color: #64748B; }
[data-testid="stMetricValue"] { color: #172B4D; font-weight: 700; }

.stTabs [data-baseweb="tab-list"] { gap: 24px; border-bottom: 1px solid #E2E8F0; }
.stTabs [data-baseweb="tab"] { color: #475569; font-weight: 600; }

.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
    border-radius: 8px; font-weight: 600; padding: 10px 20px;
}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primaryFormSubmit"] {
    background-color: #2563EB; color: white; border: none;
}
.stButton > button[kind="primary"]:hover,
.stFormSubmitButton > button[kind="primaryFormSubmit"]:hover {
    background-color: #1D4ED8; color: white; border: none;
}

[data-testid="stAlert"] { border-radius: 10px; }
hr { border-color: #E2E8F0; }
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------
@st.cache_resource
def load_artifact(path: Path):
    return joblib.load(path)


@st.cache_data
def load_metrics(path: Path):
    """Read metrics saved at training time; fall back to the last known values."""
    default = {
        "accuracy": 0.8563,
        "precision_late": 0.28,
        "recall_late": 0.49,
        "f1_late": 0.36,
        "roc_auc": None,
        "confusion_matrix": [[11803, 1494], [593, 581]],
        "from_file": False,
    }
    if path.exists():
        try:
            data = json.loads(path.read_text())
            default.update(data)
            default["from_file"] = True
        except (OSError, json.JSONDecodeError):
            pass
    return default


def group_importance(columns, importances) -> pd.DataFrame:
    """Merge one-hot columns (customer_state_SP, ...) into one bar per original feature."""
    df = pd.DataFrame({"raw": columns, "Importance": importances})
    df["Feature"] = df["raw"].apply(
        lambda c: "Customer state"
        if c.startswith("customer_state_")
        else FEATURE_LABELS.get(c, c.replace("_", " ").capitalize())
    )
    return (
        df.groupby("Feature", as_index=False)["Importance"]
        .sum()
        .sort_values("Importance", ascending=False)
    )


def risk_label(probability: float, threshold: float) -> str:
    return "Late risk" if probability >= threshold else "On time"


# ----------------------------------------------------------
# Model check
# ----------------------------------------------------------
if not MODEL_PATH.exists():
    st.error(
        f"Trained model not found at `{MODEL_PATH}`. Train the model first, then reload."
    )
    st.stop()

artifact = load_artifact(MODEL_PATH)
model = artifact["model"]
feature_columns = artifact["feature_columns"]
metrics = load_metrics(METRICS_PATH)

# ----------------------------------------------------------
# Sidebar
# ----------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.write(
        "Predicts whether an order on a Brazilian e-commerce marketplace "
        "will arrive after its estimated delivery date."
    )
    st.markdown(
        "**How it works**  \n"
        "An XGBoost classifier uses order size, price, freight, customer state "
        "and purchase and delivery dates."
    )
    st.markdown("**Dataset**  \nOlist Brazilian E-Commerce (public).")

    st.divider()
    st.subheader("Decision threshold")
    default_threshold = float(artifact.get("threshold", 0.5))
    threshold = st.slider(
        "Flag an order as late when probability is at least",
        min_value=0.05,
        max_value=0.95,
        value=min(max(default_threshold, 0.05), 0.95),
        step=0.01,
        help="Lower values catch more late orders but raise more false alarms.",
    )

    st.divider()
    st.markdown(f"[GitHub]({GITHUB_URL})")
    if LINKEDIN_URL:
        st.markdown(f"[LinkedIn]({LINKEDIN_URL})")

# ----------------------------------------------------------
# Header
# ----------------------------------------------------------
st.title("📦 Delivery Intelligence Platform")
st.caption("Delivery risk analytics for Brazilian e-commerce orders")

predict_tab, bulk_tab, insights_tab = st.tabs(
    ["📝 Single Order", "📂 Bulk CSV", "📊 Model Insights"]
)

# ==========================================================
# SINGLE ORDER
# ==========================================================
with predict_tab:
    st.subheader("Check one order")

    with st.form("single_order_form"):
        left, right = st.columns(2)

        with left:
            item_count = st.number_input("Number of items", min_value=1, value=2)
            seller_count = st.number_input("Number of sellers", min_value=1, value=1)
            total_item_price = st.number_input(
                "Total item price", min_value=0.01, value=250.0
            )
            total_freight = st.number_input("Total freight", min_value=0.0, value=35.0)

        with right:
            purchase_date = st.date_input("Order purchase date")
            purchase_time = st.time_input("Order purchase time")
            estimated_date = st.date_input(
                "Estimated delivery date",
                value=pd.Timestamp.today().date() + pd.Timedelta(days=10),
            )
            customer_state = st.selectbox("Customer state", STATES, index=STATES.index("SP"))

        submitted = st.form_submit_button(
            "Predict delivery risk", type="primary", use_container_width=True
        )

    if submitted:
        purchase_timestamp = pd.Timestamp.combine(purchase_date, purchase_time)
        estimated_timestamp = pd.Timestamp(estimated_date)

        if estimated_timestamp <= purchase_timestamp.normalize():
            st.error("The estimated delivery date must be after the purchase date.")
        else:
            order = pd.DataFrame(
                [
                    {
                        "item_count": item_count,
                        "seller_count": seller_count,
                        "total_item_price": total_item_price,
                        "total_freight": total_freight,
                        "order_purchase_timestamp": purchase_timestamp,
                        "order_estimated_delivery_date": estimated_timestamp,
                        "customer_state": customer_state,
                    }
                ]
            )

            try:
                result = predict_orders(order)
                probability = float(result["late_delivery_probability"].iloc[0])
                label = risk_label(probability, threshold)
                window_days = (estimated_timestamp - purchase_timestamp.normalize()).days

                st.divider()
                st.subheader("Result")

                m1, m2, m3 = st.columns(3)
                m1.metric("Late probability", f"{probability * 100:.1f}%")
                m2.metric("Promised window", f"{window_days} days")
                m3.metric("Prediction", label)

                st.progress(min(max(probability, 0.0), 1.0))
                st.caption(f"Flagged as late at or above {threshold * 100:.0f}% (set in the sidebar).")

                if label == "Late risk":
                    st.warning(
                        "This order is likely to miss its estimated delivery date. "
                        "Consider contacting the seller or the customer early."
                    )
                else:
                    st.success("This order is likely to arrive on time.")

                if window_days <= 3:
                    st.info(
                        "The promised window is very short, which usually raises late risk."
                    )
            except Exception as error:
                st.error(f"Prediction failed: {error}")

# ==========================================================
# BULK CSV
# ==========================================================
with bulk_tab:
    st.subheader("Check many orders")
    st.write("Upload a CSV with one row per order.")

    sample = pd.DataFrame(
        [
            [2, 1, 250.0, 35.0, "2026-11-10 14:30:00", "2026-11-22 00:00:00", "SP"],
            [1, 1, 89.9, 21.5, "2026-11-10 09:05:00", "2026-11-18 00:00:00", "RJ"],
            [3, 2, 540.0, 78.0, "2026-11-11 18:45:00", "2026-11-17 00:00:00", "BA"],
        ],
        columns=REQUIRED_COLUMNS,
    )

    st.info("Required columns: " + ", ".join(f"`{c}`" for c in REQUIRED_COLUMNS))
    st.download_button(
        "Download sample CSV",
        data=sample.to_csv(index=False).encode("utf-8"),
        file_name="sample_orders.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            uploaded_data = pd.read_csv(uploaded_file)
            missing = [c for c in REQUIRED_COLUMNS if c not in uploaded_data.columns]

            st.write("Preview")
            st.dataframe(uploaded_data.head(), use_container_width=True)

            if missing:
                st.error("Missing columns: " + ", ".join(missing))
            elif st.button("Predict all orders", type="primary"):
                with st.spinner("Generating predictions..."):
                    results = predict_orders(uploaded_data)

                if "late_delivery_probability" in results.columns:
                    results["risk_flag"] = np.where(
                        results["late_delivery_probability"] >= threshold,
                        "Late risk",
                        "On time",
                    )
                    late_orders = int((results["risk_flag"] == "Late risk").sum())
                else:
                    late_orders = int((results["prediction"] == 1).sum())

                total_orders = len(results)

                c1, c2, c3 = st.columns(3)
                c1.metric("Total orders", f"{total_orders:,}")
                c2.metric("Late risk", f"{late_orders:,}")
                c3.metric("On time", f"{total_orders - late_orders:,}")

                st.subheader("Results")
                st.dataframe(results, use_container_width=True)

                st.download_button(
                    "Download results",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name="delivery_predictions.csv",
                    mime="text/csv",
                )
        except Exception as error:
            st.error(f"Could not process the CSV: {error}")

# ==========================================================
# MODEL INSIGHTS
# ==========================================================
with insights_tab:
    st.subheader("How well does the model perform?")
    st.caption("Results on the held-out test set")

    cm = np.array(metrics["confusion_matrix"])
    total = cm.sum()
    late_total = cm[1].sum()
    late_share = late_total / total
    baseline_acc = 1 - late_share

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    k2.metric("Precision (late)", f"{metrics['precision_late'] * 100:.0f}%")
    k3.metric("Recall (late)", f"{metrics['recall_late'] * 100:.0f}%")
    k4.metric("F1 (late)", f"{metrics['f1_late'] * 100:.0f}%")

    if metrics.get("roc_auc"):
        st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

    st.info(
        f"Only {late_share * 100:.1f}% of test orders ({late_total:,} of {total:,}) were late. "
        f"A model that always predicts 'on time' would score {baseline_acc * 100:.1f}% accuracy, "
        "so accuracy alone is not the right yardstick here. "
        "This model trades precision for recall: it flags more orders so fewer late ones slip through. "
        "Use the threshold slider in the sidebar to see how that trade-off shifts."
    )

    if not metrics["from_file"]:
        st.caption(
            "Metrics shown are static values. Save `models/metrics.json` during training "
            "so this page updates when you retrain."
        )

    st.divider()

    chart_left, chart_right = st.columns(2)

    with chart_left:
        st.markdown("**Confusion matrix**")
        fig, ax = plt.subplots(figsize=(4.5, 3.6), dpi=100)
        ConfusionMatrixDisplay(cm, display_labels=["On time", "Late"]).plot(
            cmap="Blues", values_format="d", ax=ax, colorbar=False
        )
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Actual", fontsize=9)
        ax.tick_params(labelsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption("Rows are actual classes; columns are predicted classes.")

    with chart_right:
        st.markdown("**What the model relies on**")
        imp = group_importance(feature_columns, model.feature_importances_).head(10)

        fig, ax = plt.subplots(figsize=(4.5, 3.6), dpi=100)
        ax.barh(imp["Feature"], imp["Importance"], color="#2563EB")
        ax.invert_yaxis()
        ax.set_xlabel("Summed importance", fontsize=9)
        ax.tick_params(labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption(
            "State columns are combined into one bar. Importance shows reliance, "
            "not causation."
        )

    st.warning(
        "Limitation: purchase year ranks among the top features. A model that leans on the "
        "calendar year may not hold up on future dates."
    )

st.divider()
st.caption(
    "Built with Python, pandas, scikit-learn, XGBoost and Streamlit. "
    "Predictions are estimates, not delivery guarantees."
)