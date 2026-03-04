
WITH first_orders AS (
  SELECT 
    CustomerID,
    MIN(PARSE_TIMESTAMP('%m/%d/%y %H:%M', InvoiceDate)) AS first_purchase_date
  FROM `starry-axis-487018-u9.Online_retail.Online_retail`
  WHERE 
    CustomerID IS NOT NULL
    AND CustomerID != ''
    AND Quantity > 0
    AND UnitPrice > 0
  GROUP BY CustomerID
),

repeat_purchases AS (
  SELECT 
    fo.CustomerID,
    MIN(PARSE_TIMESTAMP('%m/%d/%y %H:%M', o.InvoiceDate)) AS second_purchase_date
  FROM first_orders fo
  INNER JOIN `starry-axis-487018-u9.Online_retail.Online_retail` o
    ON fo.CustomerID = o.CustomerID
    AND PARSE_TIMESTAMP('%m/%d/%y %H:%M', o.InvoiceDate) > fo.first_purchase_date
    AND o.Quantity > 0
    AND o.UnitPrice > 0
  GROUP BY fo.CustomerID
)

SELECT 
  EXTRACT(MONTH FROM fo.first_purchase_date) AS month,
  COUNT(DISTINCT fo.CustomerID) AS total_customers,
  COUNT(DISTINCT rp.CustomerID) AS repeat_customers,
  ROUND(COUNT(DISTINCT rp.CustomerID) * 100.0 / COUNT(DISTINCT fo.CustomerID), 2) AS repeat_rate,
  COUNT(DISTINCT fo.CustomerID) - COUNT(DISTINCT rp.CustomerID) AS one_time_customers
FROM first_orders fo
LEFT JOIN repeat_purchases rp ON fo.CustomerID = rp.CustomerID
GROUP BY month
ORDER BY month;