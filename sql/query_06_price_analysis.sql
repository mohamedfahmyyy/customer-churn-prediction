
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

first_order_metrics AS (
  SELECT 
    fo.CustomerID,
    AVG(o.UnitPrice) AS avg_price,
    STDDEV(o.UnitPrice) AS price_stddev,
    MAX(o.UnitPrice) AS max_price,
    MIN(o.UnitPrice) AS min_price,
    MAX(o.UnitPrice) - MIN(o.UnitPrice) AS price_range
  FROM first_orders fo
  INNER JOIN `starry-axis-487018-u9.Online_retail.Online_retail` o
    ON fo.CustomerID = o.CustomerID 
    AND fo.first_invoice = o.InvoiceNo
  WHERE o.Quantity > 0 AND o.UnitPrice > 0
  GROUP BY fo.CustomerID
),

repeat_purchases AS (
  SELECT 
    fo.CustomerID,
    CASE WHEN COUNT(DISTINCT o.InvoiceNo) > 0 THEN 1 ELSE 0 END AS is_repeat
  FROM first_orders fo
  LEFT JOIN `starry-axis-487018-u9.Online_retail.Online_retail` o
    ON fo.CustomerID = o.CustomerID
    AND o.InvoiceNo != fo.first_invoice
    AND PARSE_TIMESTAMP('%m/%d/%y %H:%M', o.InvoiceDate) > fo.first_purchase_date
    AND o.Quantity > 0
    AND o.UnitPrice > 0
  GROUP BY fo.CustomerID
)

SELECT 
  CASE 
    WHEN fom.price_range < 5 THEN 'Low Variance (<$5)'
    WHEN fom.price_range < 10 THEN 'Medium Variance ($5-$10)'
    ELSE 'High Variance (>$10)'
  END AS price_variance_category,
  COUNT(DISTINCT fom.CustomerID) AS total_customers,
  SUM(rp.is_repeat) AS repeat_customers,
  ROUND(SUM(rp.is_repeat) * 100.0 / COUNT(DISTINCT fom.CustomerID), 2) AS repeat_rate
FROM first_order_metrics fom
INNER JOIN repeat_purchases rp ON fom.CustomerID = rp.CustomerID
GROUP BY price_variance_category
ORDER BY repeat_rate DESC;