
USE delivery_delay;

-- Create the dataset used for delivery delay prediction

CREATE OR REPLACE VIEW order_delay_model_data AS

WITH item_summary AS (
    SELECT
        order_id,
        COUNT(order_item_id) AS item_count,
        COUNT(DISTINCT seller_id) AS seller_count,
        SUM(price) AS total_item_price,
        SUM(freight_value) AS total_freight
    FROM olist_order_items_dataset
    GROUP BY order_id
)

SELECT
    o.order_id,
    o.order_purchase_timestamp,
    o.order_estimated_delivery_date,

    DATEDIFF(
        o.order_estimated_delivery_date,
        o.order_purchase_timestamp
    ) AS estimated_delivery_days,

    CASE
        WHEN o.order_delivered_customer_date >
             o.order_estimated_delivery_date
        THEN 1
        ELSE 0
    END AS is_late,

    i.item_count,
    i.seller_count,
    i.total_item_price,
    i.total_freight,
    c.customer_state

FROM olist_orders_dataset o

LEFT JOIN item_summary i
    ON o.order_id = i.order_id

LEFT JOIN olist_customers_dataset c
    ON o.customer_id = c.customer_id

WHERE
    o.order_status = 'delivered'
    AND o.order_delivered_customer_date IS NOT NULL;
