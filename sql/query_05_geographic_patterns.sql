
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

first_order_country AS (
  SELECT 
    fo.CustomerID,
    fo.first_purchase_date,
    o.Country
  FROM first_orders fo
  INNER JOIN `starry-axis-487018-u9.Online_retail.Online_retail` o
    ON fo.CustomerID = o.CustomerID 
    AND PARSE_TIMESTAMP('%m/%d/%y %H:%M', o.InvoiceDate) = fo.first_purchase_date
  WHERE o.Quantity > 0 AND o.UnitPrice > 0
  GROUP BY fo.CustomerID, fo.first_purchase_date, o.Country
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
  foc.Country,
  COUNT(DISTINCT foc.CustomerID) AS total_customers,
  COUNT(DISTINCT rp.CustomerID) AS repeat_customers,
  ROUND(COUNT(DISTINCT rp.CustomerID) * 100.0 / COUNT(DISTINCT foc.CustomerID), 2) AS repeat_rate,
  COUNT(DISTINCT foc.CustomerID) - COUNT(DISTINCT rp.CustomerID) AS one_time_customers
FROM first_order_country foc
LEFT JOIN repeat_purchases rp ON foc.CustomerID = rp.CustomerID
GROUP BY foc.Country
HAVING COUNT(DISTINCT foc.CustomerID) >= 10
ORDER BY repeat_rate DESC;