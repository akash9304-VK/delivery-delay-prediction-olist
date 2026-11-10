import pandas as pd
from src.predict import predict_orders

# Create sample order
sample_order = pd.DataFrame([{
    "estimated_delivery_days": 15,
    "item_count": 2,
    "seller_count": 1,
    "total_item_price": 250.00,
    "total_freight": 35.00,
    "order_purchase_timestamp": "2018-06-15 14:30:00",
    "order_estimated_delivery_date": "2018-06-30",
    "customer_state": "SP"
}])

# Generate prediction
result = predict_orders(sample_order)

print("\nPrediction Result:")
print(result[[
    "late_delivery_probability",
    "prediction",
    "risk_status"
]])