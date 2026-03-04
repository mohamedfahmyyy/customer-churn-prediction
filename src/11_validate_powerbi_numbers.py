"""
Script 11: Validate Power BI Numbers
Checks all key metrics match between Python calculations and Power BI exports
"""

import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

# Set up paths
PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"

print("="*80)
print("POWER BI DATA VALIDATION")
print("="*80)

# Load datasets
customers = pd.read_csv(PROCESSED_PATH / 'powerbi_customers.csv')
metrics = pd.read_csv(PROCESSED_PATH / 'powerbi_metrics.csv')
business = pd.read_csv(PROCESSED_PATH / 'powerbi_business_scenarios.csv')
temporal = pd.read_csv(PROCESSED_PATH / 'powerbi_temporal_analysis.csv')
product = pd.read_csv(PROCESSED_PATH / 'powerbi_product_analysis.csv')
features = pd.read_csv(PROCESSED_PATH / 'powerbi_feature_importance.csv')

print(f"\nDatasets loaded successfully")

# ============================================================================
# PAGE 1 VALIDATION: EXECUTIVE DASHBOARD
# ============================================================================
print("\n" + "="*80)
print("PAGE 1: EXECUTIVE DASHBOARD")
print("="*80)

print("\nKPI Cards:")

# Total Customers
total_customers = len(customers)
print(f"1. Total Customers: {total_customers:,}")
print(f"   Expected: 4,338")
print(f"   Match: {'✓' if total_customers == 4338 else '✗'}")

# Repeat Rate
repeat_rate = (customers['ActualLabel'] == 1).mean()
print(f"\n2. Repeat Rate: {repeat_rate:.1%}")
print(f"   Expected: ~65.5%")
print(f"   Match: {'✓' if 0.654 <= repeat_rate <= 0.656 else '✗'}")

# ROC-AUC
roc_auc = metrics[metrics['Metric'] == 'ROC-AUC Score']['Value'].values[0]
print(f"\n3. ROC-AUC: {roc_auc:.4f}")
print(f"   Expected: ~0.821")
print(f"   Match: {'✓' if 0.820 <= roc_auc <= 0.822 else '✗'}")

# Net Profit (at 10% threshold)
net_profit_10 = business[business['TargetingThreshold'] == 10]['NetProfit'].values[0]
print(f"\n4. Net Profit (10% threshold): ${net_profit_10:,.2f}")
print(f"   Expected: ~$104,000")
print(f"   Match: {'✓' if 103000 <= net_profit_10 <= 105000 else '✗'}")

# Model Accuracy
accuracy = metrics[metrics['Metric'] == 'Overall Accuracy']['Value'].values[0]
print(f"\n5. Model Accuracy: {accuracy:.1%}")
print(f"   Expected: ~73.7%")
print(f"   Match: {'✓' if 0.736 <= accuracy <= 0.738 else '✗'}")

# Distribution checks
print("\nDonut Chart - Actual Distribution:")
actual_dist = customers['ActualLabel_Text'].value_counts()
print(actual_dist)
print(f"Repeat Buyers: {actual_dist.get('Repeat Buyer', 0):,}")
print(f"One-time Buyers: {actual_dist.get('One-time Buyer', 0):,}")

print("\nDonut Chart - Predicted Distribution:")
pred_dist = customers['PredictedLabel_Text'].value_counts()
print(pred_dist)

# Top 5 Countries
print("\nTop 5 Countries by Customer Count:")
top_countries = customers['Country'].value_counts().head(5)
print(top_countries)

# ============================================================================
# PAGE 2 VALIDATION: CUSTOMER SEGMENTATION
# ============================================================================
print("\n" + "="*80)
print("PAGE 2: CUSTOMER SEGMENTATION")
print("="*80)

print("\nScatter Plot Data Ranges:")
print(f"BasketRepeatScore: {customers['BasketRepeatScore'].min():.3f} to {customers['BasketRepeatScore'].max():.3f}")
print(f"OrderValue: ${customers['OrderValue'].min():.2f} to ${customers['OrderValue'].max():.2f}")
print(f"NumItems: {customers['NumItems'].min()} to {customers['NumItems'].max()}")

print("\nRisk Category Distribution:")
print(customers['RiskCategory'].value_counts().sort_index())

print("\nProduct Quality Distribution:")
print(customers['ProductQuality'].value_counts().sort_index())

print("\nTime Period Distribution:")
print(customers['TimePeriod'].value_counts().sort_index())

# ============================================================================
# PAGE 3 VALIDATION: MODEL PERFORMANCE
# ============================================================================
print("\n" + "="*80)
print("PAGE 3: MODEL PERFORMANCE")
print("="*80)

print("\nTop 10 Features by Importance:")
top_features = features.nlargest(10, 'ImportancePercent')[['Feature', 'ImportancePercent', 'Category']]
print(top_features.to_string(index=False))

print("\nFeature Category Totals:")
category_totals = features.groupby('Category')['ImportancePercent'].sum().sort_values(ascending=False)
print(category_totals)
print(f"\nProduct Intelligence should be ~42%: {category_totals.get('Product Intelligence', 0):.1f}%")

print("\nPrediction Type Distribution:")
print(customers['PredictionType'].value_counts())

print("\nTemporal Analysis - Error Rates:")
print(temporal[['TimePeriod', 'AccuracyRate']])
print("\nDec-Jan should have highest accuracy (~84%)")
print("Aug-Sep should have lowest accuracy (~52%)")

# ============================================================================
# PAGE 4 VALIDATION: CAMPAIGN TARGETING
# ============================================================================
print("\n" + "="*80)
print("PAGE 4: CAMPAIGN TARGETING")
print("="*80)

print("\nBusiness Scenarios - All Thresholds:")
print(business[['TargetingThreshold', 'CustomersTargeted', 'Precision', 
               'CampaignCost', 'NetProfit', 'ROI_Multiple']].to_string(index=False))

print("\nKey Thresholds to Check:")
for threshold in [10, 20, 30]:
    row = business[business['TargetingThreshold'] == threshold].iloc[0]
    print(f"\n{threshold}% Threshold:")
    print(f"  Customers Targeted: {row['CustomersTargeted']:,}")
    print(f"  Precision: {row['Precision']:.1%}")
    print(f"  Campaign Cost: ${row['CampaignCost']:,.0f}")
    print(f"  Net Profit: ${row['NetProfit']:,.0f}")
    print(f"  ROI: {row['ROI_Multiple']:.1f}x")

print("\nOptimal Threshold (highest ROI):")
optimal_idx = business['ROI'].idxmax()
optimal = business.loc[optimal_idx]
print(f"Threshold: {optimal['TargetingThreshold']:.0f}%")
print(f"ROI: {optimal['ROI_Multiple']:.1f}x")
print(f"Net Profit: ${optimal['NetProfit']:,.0f}")

# High-risk customer sample
print("\n" + "="*80)
print("HIGH-RISK CUSTOMERS (Top 10)")
print("="*80)
high_risk = customers.nlargest(10, 'ChurnRisk')[
    ['CustomerID', 'ChurnRisk', 'OrderValue', 'BasketRepeatScore', 'Country', 'ActualLabel_Text']
]
print(high_risk.to_string(index=False))

# ============================================================================
# CROSS-VALIDATION CHECKS
# ============================================================================
print("\n" + "="*80)
print("CROSS-VALIDATION CHECKS")
print("="*80)

# Check if predictions match actual labels for correct predictions
correct_predictions = customers[customers['PredictionCorrect'] == True]
mismatches = ((correct_predictions['ActualLabel'] == correct_predictions['PredictedLabel']) == False).sum()
print(f"\nPrediction Consistency Check:")
print(f"Mismatches in 'PredictionCorrect' flag: {mismatches}")
print(f"Should be 0: {'✓' if mismatches == 0 else '✗'}")

# Check if ChurnRisk = 1 - PredictedProbability
churn_risk_check = (customers['ChurnRisk'] - (1 - customers['PredictedProbability'])).abs().max()
print(f"\nChurnRisk Calculation Check:")
print(f"Max difference: {churn_risk_check:.10f}")
print(f"Should be ~0: {'✓' if churn_risk_check < 0.0001 else '✗'}")

# Check temporal analysis aggregations
print(f"\nTemporal Analysis Row Count: {len(temporal)}")
print(f"Should be 5 (one per time period): {'✓' if len(temporal) == 5 else '✗'}")

# Check product analysis aggregations
print(f"\nProduct Analysis Row Count: {len(product)}")
print(f"Should be 4 (one per quality level): {'✓' if len(product) == 4 else '✗'}")

# Check business scenarios
print(f"\nBusiness Scenarios Row Count: {len(business)}")
print(f"Should be 10 (thresholds 5%-50%): {'✓' if len(business) == 10 else '✗'}")

print("\n" + "="*80)
print("VALIDATION COMPLETE")
print("="*80)