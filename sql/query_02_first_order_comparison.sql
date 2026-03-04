
WITH first_orders AS (
  SELECT 
    CustomerID,
    MIN(PARSE_TIMESTAMP('%m/%d/%y %H:%M', InvoiceDate)) AS first_purchase_date,
    MIN(InvoiceNo) AS first_invoice
  FROM `starry-axis-487018-u9.Online_retail.Online_retail`
  WHERE 
    CustomerID IS NOT NULL
    AND CustomerID != ''
    AND Quantity > 0
    AND UnitPrice > 0
  GROUP BY CustomerID
),

repeat_status AS (
  SELECT 
    fo.CustomerID,
    CASE 
      WHEN COUNT(DISTINCT o.InvoiceNo) > 0 THEN 1 
      ELSE 0 
    END AS is_repeat_buyer
  FROM first_orders fo
  LEFT JOIN `starry-axis-487018-u9.Online_retail.Online_retail` o
    ON fo.CustomerID = o.CustomerID
    AND o.InvoiceNo != fo.first_invoice
    AND PARSE_TIMESTAMP('%m/%d/%y %H:%M', o.InvoiceDate) > fo.first_purchase_date
    AND o.Quantity > 0
    AND o.UnitPrice > 0
  GROUP BY fo.CustomerID
),

first_order_metrics AS (
  SELECT 
    fo.CustomerID,
    rs.is_repeat_buyer,
    COUNT(*) AS num_items,
    COUNT(DISTINCT o.StockCode) AS num_unique_products,
    SUM(o.Quantity) AS total_quantity,
    SUM(o.Quantity * o.UnitPrice) AS order_value,
    AVG(o.UnitPrice) AS avg_item_price
  FROM first_orders fo
  INNER JOIN repeat_status rs ON fo.CustomerID = rs.CustomerID
  INNER JOIN `starry-axis-487018-u9.Online_retail.Online_retail` o
    ON fo.CustomerID = o.CustomerID 
    AND fo.first_invoice = o.InvoiceNo
  WHERE o.Quantity > 0 AND o.UnitPrice > 0
  GROUP BY fo.CustomerID, rs.is_repeat_buyer
)


SELECT 
  CASE WHEN is_repeat_buyer = 1 THEN 'Repeat Buyer' ELSE 'One-time Buyer' END AS customer_type,
  COUNT(DISTINCT CustomerID) AS num_customers,
  ROUND(AVG(order_value), 2) AS avg_order_value,
  ROUND(AVG(num_items), 1) AS avg_num_items,
  ROUND(AVG(num_unique_products), 1) AS avg_unique_products,
  ROUND(AVG(total_quantity), 1) AS avg_total_quantity,
  ROUND(AVG(avg_item_price), 2) AS avg_item_price
FROM first_order_metrics
GROUP BY is_repeat_buyer
ORDER BY is_repeat_buyer DESC;