"""
Script 6: Model Evaluation & Business Impact
Final evaluation metrics and business case analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    precision_recall_curve, average_precision_score
)

# Set up paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw"
MODEL_PATH = PROJECT_ROOT / "outputs" / "models"
FIGURE_PATH = PROJECT_ROOT / "outputs" / "figures"

# Load data
csv_file = list(DATA_PATH.glob("*.csv"))[0]
df = pd.read_csv(csv_file)

# Prepare features
features_to_drop = ['CustomerID', 'days_to_second_purchase']
X = df.drop(columns=features_to_drop + ['target'])
y = df['target']

X_encoded = pd.get_dummies(X, columns=['Country'], drop_first=True)

# Train-test split (same as training)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# Load best model (Random Forest)
rf_model = joblib.load(MODEL_PATH / 'random_forest.pkl')
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
y_pred = rf_model.predict(X_test)

print("="*80)
print("FINAL MODEL EVALUATION - RANDOM FOREST")
print("="*80)

# Overall metrics
test_auc = roc_auc_score(y_test, y_pred_proba)
avg_precision = average_precision_score(y_test, y_pred_proba)

print(f"\nOverall Performance:")
print(f"  ROC-AUC Score: {test_auc:.4f}")
print(f"  Average Precision: {avg_precision:.4f}")
print(f"  Test Set Size: {len(y_test)} customers")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\nConfusion Matrix:")
print(f"  True Negatives (correctly identified one-time): {tn}")
print(f"  False Positives (predicted one-time, actually repeat): {fp}")
print(f"  False Negatives (predicted repeat, actually one-time): {fn}")
print(f"  True Positives (correctly identified repeat): {tp}")

# ============================================================================
# Business Impact Analysis - Multiple Targeting Strategies
# ============================================================================
print("\n" + "="*80)
print("BUSINESS IMPACT ANALYSIS")
print("="*80)

# Sort customers by risk (ascending probability = higher risk of not repeating)
risk_scores = pd.DataFrame({
    'customer_idx': range(len(y_test)),
    'true_label': y_test.values,
    'predicted_proba': y_pred_proba,
    'risk_score': 1 - y_pred_proba  # Higher score = higher churn risk
})
risk_scores = risk_scores.sort_values('risk_score', ascending=False).reset_index(drop=True)

# Calculate metrics at different targeting thresholds
thresholds = [10, 20, 30, 40, 50]
results_by_threshold = []

for pct in thresholds:
    n_target = int(len(risk_scores) * pct / 100)
    targeted = risk_scores.head(n_target)
    
    # How many actual one-time buyers did we catch?
    actual_onetime = (targeted['true_label'] == 0).sum()
    total_onetime = (y_test == 0).sum()
    
    # Metrics
    precision = actual_onetime / n_target
    recall = actual_onetime / total_onetime
    
    # Business calculations
    campaign_cost_per_customer = 5
    conversion_rate = 0.20  # 20% of one-time buyers will convert
    avg_repeat_order_value = 456
    num_future_orders = 3
    clv_per_converted = avg_repeat_order_value * num_future_orders
    
    total_cost = n_target * campaign_cost_per_customer
    customers_converted = actual_onetime * conversion_rate
    total_revenue = customers_converted * clv_per_converted
    net_profit = total_revenue - total_cost
    roi = net_profit / total_cost if total_cost > 0 else 0
    
    results_by_threshold.append({
        'Target %': pct,
        'Customers': n_target,
        'Caught One-time': actual_onetime,
        'Precision': precision,
        'Recall': recall,
        'Cost': total_cost,
        'Revenue': total_revenue,
        'Net Profit': net_profit,
        'ROI': roi
    })

results_df = pd.DataFrame(results_by_threshold)

print("\nCampaign Performance by Targeting Strategy:")
print("\n" + results_df.to_string(index=False))

# Optimal strategy
best_roi_idx = results_df['ROI'].idxmax()
best_strategy = results_df.iloc[best_roi_idx]

print(f"\n" + "="*80)
print("RECOMMENDED STRATEGY")
print("="*80)
print(f"\nTarget the top {best_strategy['Target %']:.0f}% highest-risk customers")
print(f"  Customers to target: {best_strategy['Customers']:.0f}")
print(f"  Expected one-time buyers identified: {best_strategy['Caught One-time']:.0f}")
print(f"  Precision: {best_strategy['Precision']:.1%}")
print(f"  Recall: {best_strategy['Recall']:.1%}")
print(f"  Campaign cost: ${best_strategy['Cost']:,.0f}")
print(f"  Expected revenue: ${best_strategy['Revenue']:,.0f}")
print(f"  Net profit: ${best_strategy['Net Profit']:,.0f}")
print(f"  ROI: {best_strategy['ROI']:.1f}x")

# Scale to full dataset
full_dataset_size = len(df)
scale_factor = full_dataset_size / len(y_test)

print(f"\n" + "="*80)
print("FULL DATASET PROJECTION")
print("="*80)
print(f"\nScaling test set results ({len(y_test)} customers) to full dataset ({full_dataset_size} customers)")
print(f"\nProjected campaign at {best_strategy['Target %']:.0f}% targeting:")
print(f"  Customers to target: {best_strategy['Customers'] * scale_factor:.0f}")
print(f"  Campaign cost: ${best_strategy['Cost'] * scale_factor:,.0f}")
print(f"  Expected revenue: ${best_strategy['Revenue'] * scale_factor:,.0f}")
print(f"  Net profit: ${best_strategy['Net Profit'] * scale_factor:,.0f}")
print(f"  ROI: {best_strategy['ROI']:.1f}x")

# ============================================================================
# Visualizations
# ============================================================================
print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

# Plot 1: Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['One-time', 'Repeat'],
            yticklabels=['One-time', 'Repeat'])
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
ax.set_title('Confusion Matrix - Random Forest', fontsize=14, pad=20)
plt.tight_layout()

cm_path = FIGURE_PATH / 'confusion_matrix.png'
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
print(f"Saved: {cm_path}")
plt.close()

# Plot 2: Business Metrics by Threshold
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Campaign Performance by Targeting Strategy', fontsize=16, y=1.00)

# Precision and Recall
ax1.plot(results_df['Target %'], results_df['Precision'], 
         marker='o', linewidth=2, color='#2ecc71', label='Precision')
ax1.plot(results_df['Target %'], results_df['Recall'], 
         marker='s', linewidth=2, color='#e74c3c', label='Recall')
ax1.set_xlabel('Targeting Threshold (%)', fontsize=11)
ax1.set_ylabel('Score', fontsize=11)
ax1.set_title('Precision vs Recall Trade-off', fontsize=12)
ax1.legend()
ax1.grid(alpha=0.3)

# ROI
ax2.bar(results_df['Target %'], results_df['ROI'], color='#3498db', alpha=0.7)
ax2.axhline(y=best_strategy['ROI'], color='#e74c3c', linestyle='--', 
            linewidth=2, label=f"Best: {best_strategy['ROI']:.1f}x at {best_strategy['Target %']:.0f}%")
ax2.set_xlabel('Targeting Threshold (%)', fontsize=11)
ax2.set_ylabel('ROI (times)', fontsize=11)
ax2.set_title('Return on Investment', fontsize=12)
ax2.legend()
ax2.grid(alpha=0.3)

# Net Profit
ax3.bar(results_df['Target %'], results_df['Net Profit'], color='#2ecc71', alpha=0.7)
ax3.set_xlabel('Targeting Threshold (%)', fontsize=11)
ax3.set_ylabel('Net Profit ($)', fontsize=11)
ax3.set_title('Campaign Net Profit', fontsize=12)
ax3.grid(alpha=0.3)

# Cost vs Revenue
width = 2
x_pos = results_df['Target %']
ax4.bar(x_pos - width/2, results_df['Cost'], width, label='Cost', color='#e74c3c', alpha=0.7)
ax4.bar(x_pos + width/2, results_df['Revenue'], width, label='Revenue', color='#2ecc71', alpha=0.7)
ax4.set_xlabel('Targeting Threshold (%)', fontsize=11)
ax4.set_ylabel('Amount ($)', fontsize=11)
ax4.set_title('Cost vs Revenue', fontsize=12)
ax4.legend()
ax4.grid(alpha=0.3)

plt.tight_layout()
business_path = FIGURE_PATH / 'business_metrics.png'
plt.savefig(business_path, dpi=300, bbox_inches='tight')
print(f"Saved: {business_path}")
plt.close()

# Plot 3: Risk Distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Risk scores by actual label
repeat_risks = risk_scores[risk_scores['true_label'] == 1]['risk_score']
onetime_risks = risk_scores[risk_scores['true_label'] == 0]['risk_score']

ax1.hist(repeat_risks, bins=30, alpha=0.6, label='Repeat Buyers', color='green')
ax1.hist(onetime_risks, bins=30, alpha=0.6, label='One-time Buyers', color='red')
ax1.set_xlabel('Risk Score (higher = more likely to churn)', fontsize=11)
ax1.set_ylabel('Number of Customers', fontsize=11)
ax1.set_title('Risk Score Distribution by Customer Type', fontsize=12)
ax1.legend()
ax1.grid(alpha=0.3)

# Cumulative customers by risk
ax2.plot(range(len(risk_scores)), 
         (risk_scores['true_label'] == 0).cumsum() / (y_test == 0).sum(),
         linewidth=2, color='#e74c3c')
ax2.axhline(y=best_strategy['Recall'], color='#95a5a6', linestyle='--', 
            linewidth=1, label=f"Optimal: {best_strategy['Recall']:.1%} recall at {best_strategy['Target %']:.0f}%")
ax2.axvline(x=best_strategy['Customers'], color='#95a5a6', linestyle='--', linewidth=1)
ax2.set_xlabel('Number of Customers Targeted', fontsize=11)
ax2.set_ylabel('Cumulative Recall', fontsize=11)
ax2.set_title('One-time Buyer Capture Rate', fontsize=12)
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
risk_path = FIGURE_PATH / 'risk_analysis.png'
plt.savefig(risk_path, dpi=300, bbox_inches='tight')
print(f"Saved: {risk_path}")
plt.close()

print("\n" + "="*80)
print("MODEL EVALUATION COMPLETE")
print("="*80)
print("\nProject Summary:")
print(f"  Dataset: {len(df)} customers, {X_encoded.shape[1]} features")
print(f"  Best Model: Random Forest (ROC-AUC: {test_auc:.3f})")
print(f"  Top Feature: basket_repeat_score (26.9% importance)")
print(f"  Business Impact: ${best_strategy['Net Profit'] * scale_factor:,.0f} net profit at {best_strategy['ROI']:.1f}x ROI")
print(f"\nAll outputs saved to: {PROJECT_ROOT / 'outputs'}")