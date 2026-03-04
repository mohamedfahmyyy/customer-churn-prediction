
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
    MIN(PARSE_TIMESTAMP('%m/%d/%y %H:%M', o.InvoiceDate)) AS second_purchase_date,
    TIMESTAMP_DIFF(
      MIN(PARSE_TIMESTAMP('%m/%d/%y %H:%M', o.InvoiceDate)),
      fo.first_purchase_date,
      DAY
    ) AS days_to_second_purchase
  FROM first_orders fo
  LEFT JOIN `starry-axis-487018-u9.Online_retail.Online_retail` o
    ON fo.CustomerID = o.CustomerID
    AND o.InvoiceNo != fo.first_invoice
    AND PARSE_TIMESTAMP('%m/%d/%y %H:%M', o.InvoiceDate) > fo.first_purchase_date
    AND o.Quantity > 0
    AND o.UnitPrice > 0
  GROUP BY fo.CustomerID, fo.first_purchase_date
),

first_order_features AS (
  SELECT 
    fo.CustomerID,
    fo.first_purchase_date,
    o.Country,
    
    
    COUNT(*) AS num_items,
    COUNT(DISTINCT o.StockCode) AS num_unique_products,
    SUM(o.Quantity) AS total_quantity,
    SUM(o.Quantity * o.UnitPrice) AS order_value,
    AVG(o.UnitPrice) AS avg_item_price,
    
    
    EXTRACT(DAYOFWEEK FROM fo.first_purchase_date) AS day_of_week,
    EXTRACT(MONTH FROM fo.first_purchase_date) AS month,
    EXTRACT(HOUR FROM fo.first_purchase_date) AS hour,
    CASE WHEN EXTRACT(DAYOFWEEK FROM fo.first_purchase_date) IN (1, 7) THEN 1 ELSE 0 END AS is_weekend,
    CASE WHEN EXTRACT(HOUR FROM fo.first_purchase_date) BETWEEN 9 AND 17 THEN 1 ELSE 0 END AS is_business_hours,
    
 
    DATE_DIFF(
      DATE(fo.first_purchase_date),
      DATE('2010-12-01'),
      DAY
    ) AS days_from_start,
    
   
    ARRAY_AGG(o.StockCode) AS products_purchased
    
  FROM first_orders fo
  INNER JOIN `starry-axis-487018-u9.Online_retail.Online_retail` o
    ON fo.CustomerID = o.CustomerID 
    AND fo.first_invoice = o.InvoiceNo
  WHERE o.Quantity > 0 AND o.UnitPrice > 0
  GROUP BY fo.CustomerID, fo.first_purchase_date, o.Country
),


product_repeat_rates AS (
  SELECT 
    StockCode,
    AVG(CASE WHEN made_second_purchase = 1 THEN 1.0 ELSE 0.0 END) AS product_repeat_rate,
    COUNT(DISTINCT CustomerID) AS num_customers
  FROM (
    SELECT 
      o.StockCode,
      fo.CustomerID,
      MAX(CASE 
        WHEN o2.InvoiceNo IS NOT NULL THEN 1 
        ELSE 0 
      END) AS made_second_purchase
    FROM first_orders fo
    INNER JOIN `starry-axis-487018-u9.Online_retail.Online_retail` o
      ON fo.CustomerID = o.CustomerID 
      AND fo.first_invoice = o.InvoiceNo
    LEFT JOIN `starry-axis-487018-u9.Online_retail.Online_retail` o2
      ON fo.CustomerID = o2.CustomerID 
      AND o2.InvoiceNo != fo.first_invoice
      AND PARSE_TIMESTAMP('%m/%d/%y %H:%M', o2.InvoiceDate) > fo.first_purchase_date
      AND o2.Quantity > 0
    WHERE o.Quantity > 0
    GROUP BY o.StockCode, fo.CustomerID
  )
  GROUP BY StockCode
  HAVING COUNT(DISTINCT CustomerID) >= 5
),


basket_scores AS (
  SELECT 
    fof.CustomerID,
    AVG(prr.product_repeat_rate) AS avg_product_repeat_score,
    MAX(prr.product_repeat_rate) AS max_product_repeat_score,
    COUNT(prr.StockCode) AS num_products_with_history
  FROM first_order_features fof
  CROSS JOIN UNNEST(fof.products_purchased) AS product_code
  LEFT JOIN product_repeat_rates prr ON product_code = prr.StockCode
  GROUP BY fof.CustomerID
),


country_repeat_rates AS (
  SELECT 
    fof.Country,
    AVG(CASE WHEN rs.second_purchase_date IS NOT NULL THEN 1.0 ELSE 0.0 END) AS country_repeat_rate,
    COUNT(*) AS country_customer_count
  FROM first_order_features fof
  INNER JOIN repeat_status rs ON fof.CustomerID = rs.CustomerID
  GROUP BY fof.Country
),


month_repeat_rates AS (
  SELECT 
    fof.month,
    AVG(CASE WHEN rs.second_purchase_date IS NOT NULL THEN 1.0 ELSE 0.0 END) AS month_repeat_rate,
    COUNT(*) AS month_customer_count
  FROM first_order_features fof
  INNER JOIN repeat_status rs ON fof.CustomerID = rs.CustomerID
  GROUP BY fof.month
)


SELECT 
  fof.CustomerID,
  

  CASE WHEN rs.second_purchase_date IS NOT NULL THEN 1 ELSE 0 END AS target,
  
  
  rs.days_to_second_purchase,
  
  
  ROUND(fof.order_value, 2) AS order_value,
  fof.num_items,
  fof.num_unique_products,
  fof.total_quantity,
  ROUND(fof.avg_item_price, 2) AS avg_item_price,
  
  
  ROUND(fof.num_unique_products * 1.0 / fof.num_items, 4) AS product_diversity_ratio,
  ROUND((fof.num_items * fof.num_unique_products * 1.0) / NULLIF(fof.order_value, 0), 4) AS order_complexity_score,
  
  
  fof.day_of_week,
  fof.month,
  fof.hour,
  fof.is_weekend,
  fof.is_business_hours,
  fof.days_from_start,
  
  
  fof.Country,
  ROUND(COALESCE(crr.country_repeat_rate, 0.5), 4) AS country_repeat_rate,
  
 
  ROUND(COALESCE(mrr.month_repeat_rate, 0.5), 4) AS month_repeat_rate,
  

  ROUND(COALESCE(bs.avg_product_repeat_score, 0.5), 4) AS basket_repeat_score,
  ROUND(COALESCE(bs.max_product_repeat_score, 0.5), 4) AS best_product_repeat_score,
  COALESCE(bs.num_products_with_history, 0) AS products_with_history_count,
  

  ROUND(PERCENT_RANK() OVER (ORDER BY fof.order_value) * 100, 2) AS order_value_percentile,
  ROUND(PERCENT_RANK() OVER (ORDER BY fof.num_items) * 100, 2) AS num_items_percentile

FROM first_order_features fof
INNER JOIN repeat_status rs ON fof.CustomerID = rs.CustomerID
LEFT JOIN basket_scores bs ON fof.CustomerID = bs.CustomerID
LEFT JOIN country_repeat_rates crr ON fof.Country = crr.Country
LEFT JOIN month_repeat_rates mrr ON fof.month = mrr.month
ORDER BY fof.CustomerID;