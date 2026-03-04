"""
Script 10: Export Data for Power BI Dashboard
Prepares datasets optimized for Power BI visualization
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split

# Set up paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "outputs" / "models"

# Create processed directory if it doesn't exist
PROCESSED_PATH.mkdir(exist_ok=True)

print("="*80)
print("EXPORTING DATA FOR POWER BI")
print("="*80)

# Load data
csv_file = list(DATA_PATH.glob("*.csv"))[0]
df = pd.read_csv(csv_file)

print(f"\nOriginal dataset: {df.shape[0]} customers, {df.shape[1]} columns")

# ============================================================================
# Dataset 1: Full Customer Data with Predictions
# ============================================================================
print("\n" + "="*80)
print("DATASET 1: CUSTOMER-LEVEL DATA WITH PREDICTIONS")
print("="*80)

# Prepare features
features_to_drop = ['CustomerID', 'days_to_second_purchase']
X = df.drop(columns=features_to_drop + ['target'])
y = df['target']

# Get customer IDs for later
customer_ids = df['CustomerID'].values

# One-hot encode Country
X_encoded = pd.get_dummies(X, columns=['Country'], drop_first=True)

# Split into train/test (same as model training)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# Load best model and get predictions on FULL dataset (for business projections)
xgb_model = joblib.load(MODEL_PATH / 'xgboost_tuned.pkl')
predictions_proba = xgb_model.predict_proba(X_encoded)[:, 1]
predictions_label = xgb_model.predict(X_encoded)

# Also get test set predictions (for accurate metrics)
predictions_proba_test = xgb_model.predict_proba(X_test)[:, 1]
predictions_label_test = xgb_model.predict(X_test)

# Create comprehensive dataset
powerbi_customers = pd.DataFrame({
    'CustomerID': customer_ids,
    'ActualLabel': y.values,
    'ActualLabel_Text': y.map({0: 'One-time Buyer', 1: 'Repeat Buyer'}),
    'PredictedProbability': predictions_proba,
    'PredictedLabel': predictions_label,
    'PredictedLabel_Text': pd.Series(predictions_label).map({0: 'One-time Buyer', 1: 'Repeat Buyer'}),
    'ChurnRisk': 1 - predictions_proba,
    
    # Key features
    'OrderValue': df['order_value'].values,
    'NumItems': df['num_items'].values,
    'NumUniqueProducts': df['num_unique_products'].values,
    'BasketRepeatScore': df['basket_repeat_score'].values,
    'BestProductRepeatScore': df['best_product_repeat_score'].values,
    
    # Temporal features
    'Month': df['month'].values,
    'MonthName': pd.to_datetime(df['month'], format='%m').dt.month_name(),
    'DaysFromStart': df['days_from_start'].values,
    'DayOfWeek': df['day_of_week'].values,
    'IsWeekend': df['is_weekend'].values,
    'IsBusinessHours': df['is_business_hours'].values,
    
    # Geographic
    'Country': df['Country'].values,
    'CountryRepeatRate': df['country_repeat_rate'].values,
    'MonthRepeatRate': df['month_repeat_rate'].values,
    
    # Derived metrics
    'AvgItemPrice': df['avg_item_price'].values,
    'OrderValuePercentile': df['order_value_percentile'].values,
    'NumItemsPercentile': df['num_items_percentile'].values,
})

# Add segments
powerbi_customers['TimePeriod'] = pd.cut(
    powerbi_customers['DaysFromStart'], 
    bins=[-1, 60, 120, 180, 240, 400],
    labels=['Dec-Jan', 'Feb-Mar', 'Apr-May', 'Jun-Jul', 'Aug-Sep']
)

powerbi_customers['ProductQuality'] = pd.cut(
    powerbi_customers['BasketRepeatScore'], 
    bins=[0, 0.60, 0.68, 0.75, 1.0],
    labels=['Low', 'Medium', 'High', 'Very High']
)

powerbi_customers['RiskCategory'] = pd.cut(
    powerbi_customers['ChurnRisk'],
    bins=[0, 0.25, 0.50, 0.75, 1.0],
    labels=['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']
)

powerbi_customers['OrderValueCategory'] = pd.cut(
    powerbi_customers['OrderValue'],
    bins=[0, 200, 400, 800, 100000],
    labels=['Small (<$200)', 'Medium ($200-$400)', 'Large ($400-$800)', 'Very Large (>$800)']
)

# Prediction accuracy
powerbi_customers['PredictionCorrect'] = (
    powerbi_customers['ActualLabel'] == powerbi_customers['PredictedLabel']
)
powerbi_customers['PredictionType'] = 'Correct'
powerbi_customers.loc[
    (powerbi_customers['ActualLabel'] == 0) & (powerbi_customers['PredictedLabel'] == 1),
    'PredictionType'
] = 'False Positive'
powerbi_customers.loc[
    (powerbi_customers['ActualLabel'] == 1) & (powerbi_customers['PredictedLabel'] == 0),
    'PredictionType'
] = 'False Negative'

print(f"\nCustomer dataset created: {powerbi_customers.shape}")
print(f"Columns: {powerbi_customers.shape[1]}")

# Export
customer_export_path = PROCESSED_PATH / 'powerbi_customers.csv'
powerbi_customers.to_csv(customer_export_path, index=False)
print(f"Exported: {customer_export_path}")

# ============================================================================
# Dataset 2: Model Performance Metrics (TEST SET ONLY)
# ============================================================================
print("\n" + "="*80)
print("DATASET 2: MODEL PERFORMANCE METRICS (TEST SET)")
print("="*80)

from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

# Calculate metrics on TEST SET only for accuracy
overall_metrics = pd.DataFrame({
    'Metric': [
        'ROC-AUC Score',
        'Precision (One-time)',
        'Recall (One-time)',
        'F1-Score (One-time)',
        'Precision (Repeat)',
        'Recall (Repeat)',
        'F1-Score (Repeat)',
        'Overall Accuracy',
        'Total Customers',
        'Repeat Buyers',
        'One-time Buyers',
        'Test Set Size',
        'Train Set Size'
    ],
    'Value': [
        roc_auc_score(y_test, predictions_proba_test),
        precision_score(y_test, predictions_label_test, pos_label=0),
        recall_score(y_test, predictions_label_test, pos_label=0),
        f1_score(y_test, predictions_label_test, pos_label=0),
        precision_score(y_test, predictions_label_test, pos_label=1),
        recall_score(y_test, predictions_label_test, pos_label=1),
        f1_score(y_test, predictions_label_test, pos_label=1),
        (y_test == predictions_label_test).mean(),
        len(y),
        y.sum(),
        (1 - y).sum(),
        len(y_test),
        len(y_train)
    ]
})

metrics_export_path = PROCESSED_PATH / 'powerbi_metrics.csv'
overall_metrics.to_csv(metrics_export_path, index=False)
print(f"Exported: {metrics_export_path}")
print(f"\nTest set metrics (honest performance):")
print(f"  ROC-AUC: {overall_metrics[overall_metrics['Metric']=='ROC-AUC Score']['Value'].values[0]:.4f}")
print(f"  Accuracy: {overall_metrics[overall_metrics['Metric']=='Overall Accuracy']['Value'].values[0]:.4f}")

# ============================================================================
# Dataset 3: Feature Importance
# ============================================================================
print("\n" + "="*80)
print("DATASET 3: FEATURE IMPORTANCE")
print("="*80)

# Get feature importances
feature_names = X_encoded.columns.tolist()
importances = xgb_model.feature_importances_

# Create feature importance dataframe
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances,
    'ImportancePercent': importances * 100
}).sort_values('Importance', ascending=False).reset_index(drop=True)

# Add rank
feature_importance_df['Rank'] = range(1, len(feature_importance_df) + 1)

# Categorize features
def categorize_feature(feature_name):
    if 'Country_' in feature_name:
        return 'Geographic'
    elif feature_name in ['basket_repeat_score', 'best_product_repeat_score', 'products_with_history_count']:
        return 'Product Intelligence'
    elif feature_name in ['month', 'month_repeat_rate', 'days_from_start', 'day_of_week', 'hour', 'is_weekend', 'is_business_hours']:
        return 'Temporal'
    elif feature_name in ['order_value', 'num_items', 'num_unique_products', 'total_quantity', 'avg_item_price', 
                          'product_diversity_ratio', 'order_complexity_score', 'order_value_percentile', 'num_items_percentile']:
        return 'Order Characteristics'
    else:
        return 'Other'

feature_importance_df['Category'] = feature_importance_df['Feature'].apply(categorize_feature)

importance_export_path = PROCESSED_PATH / 'powerbi_feature_importance.csv'
feature_importance_df.to_csv(importance_export_path, index=False)
print(f"Exported: {importance_export_path}")

print(f"\nTop 10 features:")
print(feature_importance_df.head(10)[['Feature', 'ImportancePercent', 'Category']].to_string(index=False))

# ============================================================================
# Dataset 4: Business Impact by Targeting Strategy
# ============================================================================
print("\n" + "="*80)
print("DATASET 4: BUSINESS IMPACT SCENARIOS (FULL DATASET PROJECTION)")
print("="*80)

# Calculate metrics for different targeting thresholds on FULL dataset
thresholds = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
business_scenarios = []

for pct in thresholds:
    n_target = int(len(powerbi_customers) * pct / 100)
    
    # Sort by churn risk (descending)
    sorted_customers = powerbi_customers.sort_values('ChurnRisk', ascending=False)
    targeted = sorted_customers.head(n_target)
    
    # Metrics
    actual_onetime = (targeted['ActualLabel'] == 0).sum()
    total_onetime = (powerbi_customers['ActualLabel'] == 0).sum()
    
    precision = actual_onetime / n_target if n_target > 0 else 0
    recall = actual_onetime / total_onetime if total_onetime > 0 else 0
    
    # Business calculations
    campaign_cost_per_customer = 5
    conversion_rate = 0.20
    avg_repeat_order_value = 456
    num_future_orders = 3
    clv_per_converted = avg_repeat_order_value * num_future_orders
    
    total_cost = n_target * campaign_cost_per_customer
    customers_converted = actual_onetime * conversion_rate
    total_revenue = customers_converted * clv_per_converted
    net_profit = total_revenue - total_cost
    roi = net_profit / total_cost if total_cost > 0 else 0
    
    business_scenarios.append({
        'TargetingThreshold': pct,
        'CustomersTargeted': n_target,
        'OneTimeBuyersCaught': actual_onetime,
        'Precision': precision,
        'Recall': recall,
        'CampaignCost': total_cost,
        'ExpectedRevenue': total_revenue,
        'NetProfit': net_profit,
        'ROI': roi,
        'ROI_Multiple': roi + 1
    })

business_df = pd.DataFrame(business_scenarios)

business_export_path = PROCESSED_PATH / 'powerbi_business_scenarios.csv'
business_df.to_csv(business_export_path, index=False)
print(f"Exported: {business_export_path}")

print(f"\nBusiness scenarios created for {len(thresholds)} targeting strategies")
print(f"Note: Based on full dataset deployment (4,338 customers)")

# ============================================================================
# Dataset 5: Temporal Analysis
# ============================================================================
print("\n" + "="*80)
print("DATASET 5: TEMPORAL ANALYSIS")
print("="*80)

# Aggregate by time period
temporal_analysis = powerbi_customers.groupby('TimePeriod', observed=True).agg({
    'CustomerID': 'count',
    'ActualLabel': 'mean',
    'PredictedProbability': 'mean',
    'ChurnRisk': 'mean',
    'PredictionCorrect': 'mean',
    'OrderValue': 'mean',
    'BasketRepeatScore': 'mean'
}).reset_index()

temporal_analysis.columns = [
    'TimePeriod', 'CustomerCount', 'ActualRepeatRate', 
    'PredictedRepeatRate', 'AvgChurnRisk', 'AccuracyRate',
    'AvgOrderValue', 'AvgBasketScore'
]

temporal_export_path = PROCESSED_PATH / 'powerbi_temporal_analysis.csv'
temporal_analysis.to_csv(temporal_export_path, index=False)
print(f"Exported: {temporal_export_path}")

# ============================================================================
# Dataset 6: Product Quality Analysis
# ============================================================================
print("\n" + "="*80)
print("DATASET 6: PRODUCT QUALITY ANALYSIS")
print("="*80)

product_analysis = powerbi_customers.groupby('ProductQuality', observed=True).agg({
    'CustomerID': 'count',
    'ActualLabel': 'mean',
    'PredictedProbability': 'mean',
    'ChurnRisk': 'mean',
    'OrderValue': 'mean',
    'NumItems': 'mean'
}).reset_index()

product_analysis.columns = [
    'ProductQuality', 'CustomerCount', 'ActualRepeatRate',
    'PredictedRepeatRate', 'AvgChurnRisk', 'AvgOrderValue', 'AvgNumItems'
]

product_export_path = PROCESSED_PATH / 'powerbi_product_analysis.csv'
product_analysis.to_csv(product_export_path, index=False)
print(f"Exported: {product_export_path}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("EXPORT COMPLETE")
print("="*80)

print(f"\nExported 6 datasets to {PROCESSED_PATH}:")
print(f"\n1. powerbi_customers.csv - {len(powerbi_customers)} rows, {len(powerbi_customers.columns)} columns")
print(f"2. powerbi_metrics.csv - {len(overall_metrics)} rows (TEST SET metrics)")
print(f"3. powerbi_feature_importance.csv - {len(feature_importance_df)} rows")
print(f"4. powerbi_business_scenarios.csv - {len(business_df)} rows (FULL dataset projection)")
print(f"5. powerbi_temporal_analysis.csv - {len(temporal_analysis)} rows")
print(f"6. powerbi_product_analysis.csv - {len(product_analysis)} rows")

print(f"\nKey distinction:")
print(f"  Model metrics (ROC-AUC, Accuracy): Test set only (honest performance)")
print(f"  Business projections (Revenue, ROI): Full dataset (deployment estimate)")
