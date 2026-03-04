
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
    MIN(PARSE_TIMESTAMP('%m/%d/%y %H:%M', o.InvoiceDate)) AS second_purchase_date,
    TIMESTAMP_DIFF(
      MIN(PARSE_TIMESTAMP('%m/%d/%y %H:%M', o.InvoiceDate)),
      fo.first_purchase_date,
      DAY
    ) AS days_to_second_purchase
  FROM first_orders fo
  INNER JOIN `starry-axis-487018-u9.Online_retail.Online_retail` o
    ON fo.CustomerID = o.CustomerID
    AND PARSE_TIMESTAMP('%m/%d/%y %H:%M', o.InvoiceDate) > fo.first_purchase_date
    AND o.Quantity > 0
    AND o.UnitPrice > 0
  GROUP BY fo.CustomerID, fo.first_purchase_date
)


SELECT 
  COUNT(DISTINCT fo.CustomerID) AS total_customers,
  COUNT(DISTINCT rp.CustomerID) AS repeat_customers,
  COUNT(DISTINCT fo.CustomerID) - COUNT(DISTINCT rp.CustomerID) AS one_time_customers,
  ROUND(COUNT(DISTINCT rp.CustomerID) * 100.0 / COUNT(DISTINCT fo.CustomerID), 2) AS repeat_rate,
  ROUND((COUNT(DISTINCT fo.CustomerID) - COUNT(DISTINCT rp.CustomerID)) * 100.0 / COUNT(DISTINCT fo.CustomerID), 2) AS one_time_rate,
  ROUND(AVG(rp.days_to_second_purchase), 0) AS avg_days_to_second_purchase,
  APPROX_QUANTILES(rp.days_to_second_purchase, 100)[OFFSET(50)] AS median_days_to_second_purchase
FROM first_orders fo
LEFT JOIN repeat_purchases rp ON fo.CustomerID = rp.CustomerID;