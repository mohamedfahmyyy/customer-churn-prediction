# Second Purchase Prediction for Online Retail

> Built to understand what drives customer loyalty in e-commerce and translate ML insights into actionable business strategies.

**Author:** [Mohamed Hassan]  
**Contact:** [LinkedIn](www.linkedin.com/in/mohamed-fahmy-1911b6399) | [Email](moh365fahmey@gmail.com)  
**Live Demo:** [Power BI Dashboard](link-if-published)

---




Predicting first-time customer churn to enable targeted retention campaigns with 49x ROI.

## Overview

Built an end-to-end machine learning system to identify which first-time customers will never make a second purchase. The model achieves 0.821 ROC-AUC and enables targeted retention campaigns generating $104K net profit on a $2.1K investment.

## Dashboard Preview

### Executive Dashboard
Key metrics, customer distribution, and top countries.

[Executive Dashboard](outputs/figures/dashboard_executive.png)

### Customer Segmentation
Risk analysis, product quality patterns, and geographic distribution.

[Customer Segmentation](outputs/figures/dashboard_segmentation.png)

### Model Performance
Feature importance, accuracy over time, and prediction breakdown.

[Model Performance](outputs/figures/dashboard_performance.png)

### Campaign Targeting
Interactive targeting simulator with ROI calculations.

[Campaign Targeting](outputs/figures/dashboard_targeting.png)

**Note:** Power BI dashboard file (`.pbix`) included in repository. Download to explore interactively.

## Business Problem

Analysis of 4,338 customers revealed that 34.5% make only one purchase and never return. This represents significant lost revenue opportunity. The challenge: identify these at-risk customers within 30-60 days of their first purchase to enable timely intervention.

## Key Findings

**Product Categories Drive Loyalty:**
- Customers purchasing home decor/kitchen items show 80-90% repeat rates
- Generic product buyers show only 22% repeat rates
- Product intelligence features account for 40% of model's predictive power

**Temporal Patterns:**
- December first-time buyers: 71% repeat rate
- September first-time buyers: 22% repeat rate
- 49 percentage point spread driven by seasonality and purchase intent

**Geographic Variation:**
- Belgium/France/Germany: 53-54% repeat rates
- United Kingdom (90% of customers): 46.5% repeat rate

## Technical Approach

### Data Engineering (SQL/BigQuery)
- Analyzed 541,909 transactions across 4,338 customers (Dec 2010 - Sep 2011)
- Created 24 engineered features including:
  - Product intelligence scores (historical repeat rates by product)
  - Temporal features (month, time period, days from dataset start)
  - Order characteristics (value, basket size, product diversity)
  - Geographic patterns (country-level repeat rates)

### Machine Learning (Python)

**Models Tested:**
- Logistic Regression (baseline): 0.796 ROC-AUC
- Random Forest: 0.817 ROC-AUC
- XGBoost (default): 0.808 ROC-AUC
- **XGBoost (tuned): 0.821 ROC-AUC** ← Final model

**Hyperparameter Optimization:**
- Grid search across 243 parameter combinations
- 3-fold cross-validation
- Optimized for ROC-AUC score
- Final parameters: learning_rate=0.05, max_depth=4, subsample=0.8

**Top 3 Features:**
1. basket_repeat_score - 26.0% importance
2. best_product_repeat_score - 10.5% importance
3. days_from_start - 8.7% importance

**Model Performance (Test Set - 868 customers):**
- ROC-AUC: 0.821
- Overall Accuracy: 73.7%
- Precision @ 10% targeting: 89.6%
- Recall @ 10% targeting: 55.7%

### Model Interpretation (SHAP Analysis)
- Conducted SHAP analysis to explain individual predictions
- Identified systematic error patterns:
  - False positives: Early joiners with good products who didn't return (likely holiday gift buyers)
  - False negatives: Late joiners with mediocre products who did return (brand loyal despite product mix)
- Discovered temporal bias: Model error rate increases from 16% (December) to 48% (August)
- Root cause: December buyers fundamentally different behavior - cannot be fixed with feature engineering alone

### Visualization (Power BI)
Built interactive 4-page dashboard:
- **Executive Summary:** KPIs, target distribution, top countries
- **Customer Segmentation:** Risk categories, product quality analysis, geographic patterns
- **Model Performance:** Feature importance, accuracy over time, prediction breakdown
- **Campaign Targeting:** Interactive threshold selector, ROI calculator, high-risk customer list

## Business Impact

**Optimal Strategy: Target top 10% highest-risk customers**

| Metric | Value |
|--------|-------|
| Customers Targeted | 433 |
| Precision | 89.6% |
| One-time Buyers Caught | 388 |
| Campaign Cost | $2,165 |
| Expected Revenue | $106,157 |
| Net Profit | $103,992 |
| ROI | 49.0x |

**Deployment Approach:**
1. Score all first-time customers 30-60 days post-purchase
2. Target bottom 10% predicted probability (highest churn risk)
3. Send personalized retention offers via email ($5 per customer)
4. Expected conversion: 20% of targeted one-time buyers become repeat customers
5. Lifetime value per converted customer: $1,368 (3 additional orders × $456 average)

**Why 10% targeting?**
- Higher thresholds (20%, 30%) catch more churners but lower precision
- 10% threshold maximizes ROI (49x) while maintaining 89.6% precision
- Enables focused, high-quality campaigns rather than spray-and-pray approach

## Project Structure
```
second-purchase-prediction/
├── data/
│   ├── raw/                      # BigQuery exports
│   │   └── TRdata.csv           # Full dataset (4,338 customers)
│   └── processed/                # Power BI optimized files
│       ├── powerbi_customers.csv
│       ├── powerbi_metrics.csv
│       ├── powerbi_feature_importance.csv
│       └── powerbi_business_scenarios.csv
├── sql/
│   ├── query_01_repeat_rate_analysis.sql
│   ├── query_02_first_order_comparison.sql
│   ├── query_03_product_patterns.sql
│   ├── query_04_temporal_patterns.sql
│   ├── query_05_geographic_patterns.sql
│   └── query_08_ml_dataset_final.sql
├── src/
│   ├── 01_data_validation.py
│   ├── 02_eda_correlations.py
│   ├── 03_baseline_models.py
│   ├── 04_advanced_models.py
│   ├── 05_feature_importance.py
│   ├── 06_model_evaluation.py
│   ├── 07_model_interpretation.py
│   ├── 08_error_deep_dive.py
│   ├── 09_model_improvement.py
│   └── 10_export_for_powerbi.py
├── outputs/
│   ├── figures/                  # 15+ visualizations
│   │   ├── roc_curve.png
│   │   ├── feature_importance.png
│   │   ├── shap_summary_plot.png
│   │   └── business_metrics.png
│   └── models/                   # Trained models
│       ├── xgboost_tuned.pkl
│       ├── random_forest.pkl
│       └── logistic_regression.pkl
├── second-purchase-prediction.pbix  # Power BI dashboard
├── requirements.txt
└── README.md
```

## Tech Stack

- **Database:** Google BigQuery
- **Languages:** SQL, Python, DAX
- **ML Libraries:** scikit-learn, XGBoost, SHAP, pandas, numpy
- **Visualization:** Power BI, matplotlib, seaborn
- **Techniques:** 
  - Feature engineering with window functions
  - Hyperparameter tuning via grid search
  - SHAP explainability analysis
  - Business metrics optimization

## Results Summary

| Model | Test ROC-AUC | Overall Accuracy | Precision @ 10% | Net Profit |
|-------|--------------|------------------|-----------------|------------|
| Logistic Regression | 0.796 | 73.0% | 80.2% | Baseline |
| Random Forest | 0.817 | 73.7% | 80.2% | $103K |
| **XGBoost (tuned)** | **0.821** | **73.7%** | **89.6%** | **$104K** |

**Model Validation:**
- 80/20 train-test split (stratified)
- 5-fold cross-validation: 0.819 ± 0.013
- Test set: 868 customers (unseen data)
- No data leakage (temporal features calculated on training data only)

**Key Insight:** Precision @ 10% (89.6%) far exceeds overall accuracy (73.7%) - the model is significantly better at identifying the highest-risk customers, which is exactly what the business needs.

## Key Learnings

1. **Product intelligence beats demographics:** Product category features accounted for 40% of model importance, far outweighing geographic (3%) or standard order metrics (26%). The breakthrough was engineering historical repeat rates by product.

2. **Temporal bias cannot be fixed with features alone:** Attempted to address December vs. August error rate gap (16% vs 48%) by adding interaction features and seasonal adjustments. Result: +0.0006 ROC-AUC improvement (essentially zero). Conclusion: December buyers are fundamentally different (gift shoppers) - requires business rules or segmented models, not better ML.

3. **Business metrics matter more than model metrics:** A model with 0.821 ROC-AUC that achieves 89.6% precision at the business decision point (10% targeting) is far more valuable than a 0.85 ROC-AUC model with 75% precision at that threshold.

4. **Error analysis drives insights:** SHAP analysis revealed that misclassifications weren't random - they clustered around early joiners with good products (false positives) and late joiners with poor products (false negatives). This led to actionable recommendations for segmented strategies.

## Future Improvements

**Technical:**
1. Segment models by time period (Dec-Mar vs Apr-Sep) to address temporal bias
2. Add customer demographics and marketing channel data
3. Build real-time API for production deployment
4. Implement model monitoring and drift detection

**Business:**
1. A/B test campaign effectiveness with control group
2. Develop tiered intervention strategies (10% = aggressive, 20% = moderate, 30% = light touch)
3. Create product category-specific retention offers
4. Build closed-loop system to retrain on campaign results

## How to Reproduce
```bash
# 1. Clone repository
git clone https://github.com/yourusername/second-purchase-prediction.git
cd second-purchase-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Data validation
python src/01_data_validation.py

# 4. Exploratory analysis
python src/02_eda_correlations.py

# 5. Train models
python src/03_baseline_models.py
python src/04_advanced_models.py

# 6. Model interpretation
python src/07_model_interpretation.py

# 7. Export for Power BI
python src/10_export_for_powerbi.py

# 8. Open Power BI dashboard
# Open: second-purchase-prediction.pbix
```

## Dataset

**Source:** UCI Machine Learning Repository - Online Retail Dataset
- **Period:** December 2010 - September 2011
- **Transactions:** 541,909
- **Customers:** 4,338 (after filtering for valid CustomerID)
- **Countries:** 37
- **Products:** 3,665 unique stock codes

**Data Filters Applied:**
- CustomerID IS NOT NULL
- Quantity > 0
- UnitPrice > 0

## Model Deployment Considerations

**Scoring Frequency:** Monthly batch scoring 30 days post-first-purchase

**Infrastructure Requirements:**
- Python 3.8+ runtime
- Model artifact: 2.3 MB (XGBoost pkl file)
- Inference time: <100ms per customer
- Storage: BigQuery for customer data, Cloud Storage for model artifacts

**Monitoring:**
- Track precision @ 10% over time (target: >85%)
- Monitor feature drift (especially basket_repeat_score distribution)
- Measure actual campaign conversion rates vs. 20% assumption
- Alert if ROC-AUC drops below 0.80

## Interview Talking Points

**"Walk me through your project":**
*"I built a churn prediction system for online retail using 540K transactions. Through SQL analysis in BigQuery, I discovered product category was the strongest predictor - home decor buyers had 80% repeat rates versus 22% for generic items. I engineered product intelligence scores based on historical patterns, trained multiple models, and used XGBoost to achieve 0.821 ROC-AUC. The key business insight: targeting just the top 10% highest-risk customers yields 89.6% precision and 49x ROI, generating $104K profit on a $2K campaign."*

**"What was your biggest challenge?":**
*"Error analysis revealed a temporal bias - December buyers had 16% error rate while August buyers had 48%. I hypothesized feature engineering could fix it, so I built 6 new features including seasonal adjustments and time-based interactions. Result: +0.0006 improvement, essentially nothing. This taught me that not all problems are fixable with better features. December buyers are fundamentally different - many are one-time gift purchasers. The solution isn't better ML, it's segmented business strategies."*

**"How would this be deployed?":**
*"Monthly batch scoring 30 days post-first-purchase. New customers get risk scores, bottom 10% receive targeted retention emails with product category-specific offers. The model retrains quarterly on fresh data. We'd run an A/B test first - control group gets no intervention, treatment group gets campaigns. Measure actual conversion rates vs our 20% assumption and adjust targeting threshold accordingly. Success metrics: precision stays above 85%, ROI stays above 30x."*

## Author

Portfolio project demonstrating end-to-end data science capabilities for data analyst roles.

Skills demonstrated: SQL (BigQuery), Python (ML pipeline), Statistical Analysis, Feature Engineering, Model Interpretation (SHAP), Business Metrics, Data Visualization (Power BI)

## License

MIT License - Feel free to use this project as reference for your own work.

---

**Questions or feedback?** Open an issue or reach out via [[My LinkedIn Profile](https://www.linkedin.com/in/mohamed-hassan-27901330a)/[E-mail]moh365fahmey@gmail.com].