
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
    END AS made_repeat_purchase
  FROM first_orders fo
  LEFT JOIN `starry-axis-487018-u9.Online_retail.Online_retail` o
    ON fo.CustomerID = o.CustomerID
    AND o.InvoiceNo != fo.first_invoice
    AND PARSE_TIMESTAMP('%m/%d/%y %H:%M', o.InvoiceDate) > fo.first_purchase_date
    AND o.Quantity > 0
    AND o.UnitPrice > 0
  GROUP BY fo.CustomerID
),

product_repeat_rates AS (
  SELECT 
    o.StockCode,
    o.Description,
    COUNT(DISTINCT fo.CustomerID) AS total_customers,
    SUM(rs.made_repeat_purchase) AS repeat_customers,
    ROUND(SUM(rs.made_repeat_purchase) * 100.0 / COUNT(DISTINCT fo.CustomerID), 2) AS repeat_rate
  FROM first_orders fo
  INNER JOIN `starry-axis-487018-u9.Online_retail.Online_retail` o
    ON fo.CustomerID = o.CustomerID 
    AND fo.first_invoice = o.InvoiceNo
  INNER JOIN repeat_status rs ON fo.CustomerID = rs.CustomerID
  WHERE o.Quantity > 0 AND o.UnitPrice > 0
  GROUP BY o.StockCode, o.Description
  HAVING COUNT(DISTINCT fo.CustomerID) >= 10
)


SELECT 
  StockCode,
  Description,
  total_customers,
  repeat_customers,
  repeat_rate
FROM product_repeat_rates
ORDER BY repeat_rate DESC, total_customers DESC
LIMIT 50;