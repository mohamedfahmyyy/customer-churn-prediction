"""
Script 9: Model Improvements Based on Error Analysis
Implements recommended fixes: interaction features and seasonal flags
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb

# Set up paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw"
MODEL_PATH = PROJECT_ROOT / "outputs" / "models"
FIGURE_PATH = PROJECT_ROOT / "outputs" / "figures"

print("="*80)
print("MODEL IMPROVEMENT - ADDRESSING TEMPORAL BIAS")
print("="*80)

# Load data
csv_file = list(DATA_PATH.glob("*.csv"))[0]
df = pd.read_csv(csv_file)

print(f"\nOriginal features: {df.shape[1] - 1}")  # -1 for target

# ============================================================================
# NEW FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*80)
print("CREATING IMPROVED FEATURES")
print("="*80)

# 1. Early joiner flag (Dec-Mar = days 0-120)
df['is_early_joiner'] = (df['days_from_start'] <= 120).astype(int)
print(f"\n1. is_early_joiner: {df['is_early_joiner'].sum()} early joiners ({df['is_early_joiner'].mean():.1%})")

# 2. December buyer flag (highest risk for false positives)
df['is_december_buyer'] = (df['month'] == 12).astype(int)
print(f"2. is_december_buyer: {df['is_december_buyer'].sum()} December buyers ({df['is_december_buyer'].mean():.1%})")

# 3. Interaction: product score × early joiner
df['product_time_interaction'] = df['basket_repeat_score'] * df['is_early_joiner']
print(f"3. product_time_interaction: Created (basket_repeat_score × early joiner flag)")

# 4. Seasonal risk score (products matter less for December buyers)
df['seasonal_risk_adjustment'] = df['basket_repeat_score'] * (1 - 0.3 * df['is_december_buyer'])
print(f"4. seasonal_risk_adjustment: Reduces product score weight for December buyers")


# 5. Time period buckets (better than continuous days_from_start)
df['time_period_bucket'] = pd.cut(df['days_from_start'], 
                                   bins=[-1, 60, 120, 180, 240, 400],  # Expanded range
                                   labels=[1, 2, 3, 4, 5])
df['time_period_bucket'] = df['time_period_bucket'].cat.codes + 1  # Convert to int safely
print(f"5. time_period_bucket: 5 distinct time periods")

# 6. Late joiner with low products (high churn risk)
df['high_churn_segment'] = ((df['days_from_start'] > 180) & 
                             (df['basket_repeat_score'] < 0.65)).astype(int)
print(f"6. high_churn_segment: {df['high_churn_segment'].sum()} high-risk late joiners")

print(f"\nTotal features after engineering: {df.shape[1] - 3}")  # -3 for CustomerID, days_to_second_purchase, target

# ============================================================================
# PREPARE DATA FOR MODELING
# ============================================================================
print("\n" + "="*80)
print("PREPARING IMPROVED DATASET")
print("="*80)

# Drop columns
features_to_drop = ['CustomerID', 'days_to_second_purchase']
X = df.drop(columns=features_to_drop + ['target'])
y = df['target']

# One-hot encode Country
X_encoded = pd.get_dummies(X, columns=['Country'], drop_first=True)

print(f"Final feature count: {X_encoded.shape[1]}")
print(f"New features added: 6")
print(f"  - is_early_joiner")
print(f"  - is_december_buyer")
print(f"  - product_time_interaction")
print(f"  - seasonal_risk_adjustment")
print(f"  - time_period_bucket")
print(f"  - high_churn_segment")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# ============================================================================
# TRAIN IMPROVED MODEL
# ============================================================================
print("\n" + "="*80)
print("TRAINING IMPROVED XGBOOST MODEL")
print("="*80)

# Use same hyperparameters as tuned model, but with new features
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

improved_model = xgb.XGBClassifier(
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=1,
    n_estimators=100,
    subsample=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='auc'
)

print("\nTraining improved model...")
improved_model.fit(X_train, y_train)

# Predictions
improved_train_proba = improved_model.predict_proba(X_train)[:, 1]
improved_test_proba = improved_model.predict_proba(X_test)[:, 1]
improved_test_pred = improved_model.predict(X_test)

# Metrics
improved_train_auc = roc_auc_score(y_train, improved_train_proba)
improved_test_auc = roc_auc_score(y_test, improved_test_proba)

print(f"\nImproved Model Performance:")
print(f"  Training ROC-AUC: {improved_train_auc:.4f}")
print(f"  Test ROC-AUC: {improved_test_auc:.4f}")
print(f"  Overfit gap: {improved_train_auc - improved_test_auc:.4f}")

# Cross-validation
cv_scores = cross_val_score(improved_model, X_train, y_train, cv=5, scoring='roc_auc')
print(f"  5-Fold CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

print("\nTest Set Classification Report:")
print(classification_report(y_test, improved_test_pred, target_names=['One-time', 'Repeat']))

# Save improved model
joblib.dump(improved_model, MODEL_PATH / 'xgboost_improved.pkl')
print(f"\nSaved: xgboost_improved.pkl")

# ============================================================================
# COMPARE WITH ORIGINAL MODEL
# ============================================================================
print("\n" + "="*80)
print("COMPARISON: ORIGINAL vs IMPROVED MODEL")
print("="*80)

# Load original model and get predictions on original features
original_features = ['order_value', 'num_items', 'num_unique_products', 'total_quantity',
                     'avg_item_price', 'product_diversity_ratio', 'order_complexity_score',
                     'day_of_week', 'month', 'hour', 'is_weekend', 'is_business_hours',
                     'days_from_start', 'country_repeat_rate', 'month_repeat_rate',
                     'basket_repeat_score', 'best_product_repeat_score', 
                     'products_with_history_count', 'order_value_percentile', 
                     'num_items_percentile']

# Load original model and get its feature names in the correct order
original_model = joblib.load(MODEL_PATH / 'xgboost_tuned.pkl')
original_feature_names = original_model.get_booster().feature_names

# Ensure we use the exact features in the exact order the model expects
X_test_original = X_test[original_feature_names]
original_test_proba = original_model.predict_proba(X_test_original)[:, 1]
original_test_auc = roc_auc_score(y_test, original_test_proba)

comparison = pd.DataFrame({
    'Model': ['Original XGBoost', 'Improved XGBoost'],
    'Test ROC-AUC': [original_test_auc, improved_test_auc],
    'Improvement': [0, improved_test_auc - original_test_auc]
})

print("\n", comparison.to_string(index=False))

if improved_test_auc > original_test_auc:
    print(f"\n✓ Improvement achieved: +{(improved_test_auc - original_test_auc):.4f} ROC-AUC")
else:
    print(f"\n✗ No improvement: {(improved_test_auc - original_test_auc):.4f} ROC-AUC")

# ============================================================================
# TEMPORAL ERROR ANALYSIS - DID WE FIX IT?
# ============================================================================
print("\n" + "="*80)
print("TEMPORAL ERROR ANALYSIS - IMPROVED MODEL")
print("="*80)

# Recreate analysis dataframe
analysis_improved = X_test.copy()
analysis_improved['true_label'] = y_test.values
analysis_improved['original_proba'] = original_test_proba
analysis_improved['improved_proba'] = improved_test_proba
analysis_improved['original_pred'] = (original_test_proba >= 0.5).astype(int)
analysis_improved['improved_pred'] = improved_test_pred

# Error rates by time period
analysis_improved['time_period'] = pd.cut(analysis_improved['days_from_start'], 
                                          bins=[0, 60, 120, 180, 240, 300],
                                          labels=['Dec-Jan', 'Feb-Mar', 'Apr-May', 
                                                 'Jun-Jul', 'Aug-Sep'])

print("\nError rates by time period:")
print("\nOriginal Model:")
original_errors = analysis_improved.groupby('time_period').apply(
    lambda x: ((x['true_label'] != x['original_pred']).sum() / len(x) * 100)
)
print(original_errors.round(1))

print("\nImproved Model:")
improved_errors = analysis_improved.groupby('time_period').apply(
    lambda x: ((x['true_label'] != x['improved_pred']).sum() / len(x) * 100)
)
print(improved_errors.round(1))

print("\nError Rate Reduction:")
error_reduction = original_errors - improved_errors
print(error_reduction.round(1))

# ============================================================================
# FALSE POSITIVE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("FALSE POSITIVE COMPARISON")
print("="*80)

original_fp = ((analysis_improved['true_label'] == 0) & 
               (analysis_improved['original_pred'] == 1)).sum()
improved_fp = ((analysis_improved['true_label'] == 0) & 
               (analysis_improved['improved_pred'] == 1)).sum()

print(f"\nOriginal model False Positives: {original_fp}")
print(f"Improved model False Positives: {improved_fp}")
print(f"Reduction: {original_fp - improved_fp} ({(original_fp - improved_fp)/original_fp*100:.1f}%)")

# December-specific false positives
december_mask = analysis_improved['month'] == 12
original_fp_dec = ((analysis_improved[december_mask]['true_label'] == 0) & 
                   (analysis_improved[december_mask]['original_pred'] == 1)).sum()
improved_fp_dec = ((analysis_improved[december_mask]['true_label'] == 0) & 
                   (analysis_improved[december_mask]['improved_pred'] == 1)).sum()

print(f"\nDecember False Positives:")
print(f"  Original: {original_fp_dec}")
print(f"  Improved: {improved_fp_dec}")
print(f"  Reduction: {original_fp_dec - improved_fp_dec}")

# ============================================================================
# BUSINESS IMPACT
# ============================================================================
print("\n" + "="*80)
print("BUSINESS IMPACT - IMPROVED MODEL")
print("="*80)

# Calculate precision at 30%
sorted_indices = np.argsort(improved_test_proba)
top_30_pct = int(len(sorted_indices) * 0.30)
high_risk_indices = sorted_indices[:top_30_pct]

targeted_labels = y_test.iloc[high_risk_indices]
precision_at_30 = (targeted_labels == 0).sum() / len(targeted_labels)
caught_onetime = (targeted_labels == 0).sum()
total_onetime = (y_test == 0).sum()
recall_at_30 = caught_onetime / total_onetime

print(f"\nTargeting top 30% highest-risk customers:")
print(f"  Customers targeted: {len(high_risk_indices)}")
print(f"  Actual one-time buyers caught: {caught_onetime}")
print(f"  Precision @ 30%: {precision_at_30:.1%}")
print(f"  Recall @ 30%: {recall_at_30:.1%}")
print(f"  Expected campaign ROI: {precision_at_30 * 0.20 * 1368 * len(high_risk_indices) / (5 * len(high_risk_indices)):.1f}x")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("CREATING COMPARISON VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Error rates over time
ax = axes[0, 0]
x = range(len(original_errors))
width = 0.35
ax.bar([i - width/2 for i in x], original_errors.values, width, 
       label='Original', color='#e74c3c', alpha=0.7)
ax.bar([i + width/2 for i in x], improved_errors.values, width, 
       label='Improved', color='#2ecc71', alpha=0.7)
ax.set_ylabel('Error Rate (%)', fontsize=11)
ax.set_title('Error Rates by Time Period', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(original_errors.index, rotation=45, ha='right')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# 2. ROC curves
from sklearn.metrics import roc_curve
fpr_orig, tpr_orig, _ = roc_curve(y_test, original_test_proba)
fpr_imp, tpr_imp, _ = roc_curve(y_test, improved_test_proba)

ax = axes[0, 1]
ax.plot(fpr_orig, tpr_orig, label=f'Original (AUC={original_test_auc:.3f})', 
        linewidth=2, color='#e74c3c')
ax.plot(fpr_imp, tpr_imp, label=f'Improved (AUC={improved_test_auc:.3f})', 
        linewidth=2, color='#2ecc71')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title('ROC Curve Comparison', fontsize=12)
ax.legend()
ax.grid(alpha=0.3)

# 3. Feature importance comparison
ax = axes[1, 0]
new_features = ['is_early_joiner', 'is_december_buyer', 'product_time_interaction',
                'seasonal_risk_adjustment', 'time_period_bucket', 'high_churn_segment']
feature_names = X_encoded.columns.tolist()
importances = improved_model.feature_importances_

# Get indices of new features
new_feature_importance = {}
for feat in new_features:
    if feat in feature_names:
        idx = feature_names.index(feat)
        new_feature_importance[feat] = importances[idx]

if new_feature_importance:
    ax.barh(range(len(new_feature_importance)), list(new_feature_importance.values()), 
            color='#3498db', alpha=0.7)
    ax.set_yticks(range(len(new_feature_importance)))
    ax.set_yticklabels(list(new_feature_importance.keys()))
    ax.set_xlabel('Importance Score', fontsize=11)
    ax.set_title('New Feature Importance', fontsize=12)
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis='x')

# 4. False positive reduction
ax = axes[1, 1]
categories = ['Overall', 'December Only']
original_fps = [original_fp, original_fp_dec]
improved_fps = [improved_fp, improved_fp_dec]
x = range(len(categories))
width = 0.35
ax.bar([i - width/2 for i in x], original_fps, width, 
       label='Original', color='#e74c3c', alpha=0.7)
ax.bar([i + width/2 for i in x], improved_fps, width, 
       label='Improved', color='#2ecc71', alpha=0.7)
ax.set_ylabel('False Positive Count', fontsize=11)
ax.set_title('False Positive Reduction', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
improvement_path = FIGURE_PATH / 'model_improvement_comparison.png'
plt.savefig(improvement_path, dpi=300, bbox_inches='tight')
print(f"Saved: {improvement_path}")
plt.close()

print("\n" + "="*80)
print("MODEL IMPROVEMENT COMPLETE")
print("="*80)

print("\nSummary:")
print(f"1. Added 6 new features targeting temporal bias")
print(f"2. ROC-AUC change: {original_test_auc:.4f} → {improved_test_auc:.4f} ({improved_test_auc - original_test_auc:+.4f})")
print(f"3. False positives: {original_fp} → {improved_fp} ({original_fp - improved_fp:+d})")
print(f"4. August error rate reduced by {original_errors.iloc[-1] - improved_errors.iloc[-1]:.1f} percentage points")
print("\nNext step: Days 11-14 - Power BI Dashboard")