"""
Script 8: Deep Error Analysis
Investigates why the model makes mistakes and how to improve
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split

# Set up paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw"
MODEL_PATH = PROJECT_ROOT / "outputs" / "models"
FIGURE_PATH = PROJECT_ROOT / "outputs" / "figures"

print("="*80)
print("DEEP ERROR ANALYSIS")
print("="*80)

# Load data
csv_file = list(DATA_PATH.glob("*.csv"))[0]
df = pd.read_csv(csv_file)

# Prepare features
features_to_drop = ['CustomerID', 'days_to_second_purchase']
X = df.drop(columns=features_to_drop + ['target'])
y = df['target']

X_encoded = pd.get_dummies(X, columns=['Country'], drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# Load model
xgb_model = joblib.load(MODEL_PATH / 'xgboost_tuned.pkl')
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
y_pred = xgb_model.predict(X_test)

# Create analysis dataframe
analysis_df = X_test.copy()
analysis_df['true_label'] = y_test.values
analysis_df['predicted_proba'] = y_pred_proba
analysis_df['predicted_label'] = y_pred

# Classify predictions
analysis_df['error_type'] = 'Correct'
analysis_df.loc[(analysis_df['true_label'] == 0) & (analysis_df['predicted_label'] == 1), 'error_type'] = 'False Positive'
analysis_df.loc[(analysis_df['true_label'] == 1) & (analysis_df['predicted_label'] == 0), 'error_type'] = 'False Negative'

print(f"\nTotal test samples: {len(analysis_df)}")
print(f"Correct predictions: {(analysis_df['error_type'] == 'Correct').sum()} ({(analysis_df['error_type'] == 'Correct').mean():.1%})")
print(f"False Positives: {(analysis_df['error_type'] == 'False Positive').sum()}")
print(f"False Negatives: {(analysis_df['error_type'] == 'False Negative').sum()}")

# ============================================================================
# Temporal Pattern Analysis
# ============================================================================
print("\n" + "="*80)
print("TEMPORAL PATTERN IN ERRORS")
print("="*80)

# Analyze by days_from_start
analysis_df['time_period'] = pd.cut(analysis_df['days_from_start'], 
                                     bins=[0, 60, 120, 180, 240, 300],
                                     labels=['Dec-Jan (0-60)', 'Feb-Mar (60-120)', 
                                            'Apr-May (120-180)', 'Jun-Jul (180-240)', 
                                            'Aug-Sep (240-300)'])

print("\nError rates by time period:")
temporal_analysis = analysis_df.groupby('time_period').agg({
    'error_type': lambda x: (x != 'Correct').sum(),
    'true_label': 'count'
})
temporal_analysis['error_rate'] = temporal_analysis['error_type'] / temporal_analysis['true_label']
temporal_analysis.columns = ['Errors', 'Total', 'Error Rate']
print(temporal_analysis)

print("\nBreakdown by error type and time period:")
error_temporal = pd.crosstab(analysis_df['time_period'], analysis_df['error_type'], 
                             normalize='index') * 100
print(error_temporal.round(1))

# ============================================================================
# Product Score vs Time Interaction
# ============================================================================
print("\n" + "="*80)
print("PRODUCT SCORE vs TIME PERIOD INTERACTION")
print("="*80)

# Create segments
analysis_df['product_quality'] = pd.cut(analysis_df['basket_repeat_score'], 
                                        bins=[0, 0.60, 0.68, 0.75, 1.0],
                                        labels=['Low (<0.60)', 'Medium (0.60-0.68)', 
                                               'High (0.68-0.75)', 'Very High (>0.75)'])

analysis_df['time_segment'] = pd.cut(analysis_df['days_from_start'], 
                                     bins=[0, 120, 300],
                                     labels=['Early Joiners (Dec-Mar)', 
                                            'Late Joiners (Apr-Sep)'])

print("\nActual repeat rate by Product Quality × Time:")
interaction = pd.crosstab(analysis_df['product_quality'], 
                          analysis_df['time_segment'], 
                          values=analysis_df['true_label'], 
                          aggfunc='mean') * 100
print(interaction.round(1))

print("\nError rate by Product Quality × Time:")
interaction_errors = pd.crosstab(analysis_df['product_quality'], 
                                 analysis_df['time_segment'], 
                                 values=(analysis_df['error_type'] != 'Correct'), 
                                 aggfunc='mean') * 100
print(interaction_errors.round(1))

# ============================================================================
# Misclassification Patterns
# ============================================================================
print("\n" + "="*80)
print("SYSTEMATIC MISCLASSIFICATION PATTERNS")
print("="*80)

# Pattern 1: Early joiners with good products who didn't return (False Positives)
fp = analysis_df[analysis_df['error_type'] == 'False Positive']
pattern1 = fp[(fp['basket_repeat_score'] > 0.68) & (fp['days_from_start'] < 120)]
print(f"\nPattern 1: High product score + Early joiner → Predicted Repeat, Actually One-time")
print(f"  Count: {len(pattern1)} customers ({len(pattern1)/len(fp)*100:.1f}% of false positives)")
print(f"  Average basket_repeat_score: {pattern1['basket_repeat_score'].mean():.3f}")
print(f"  Average days_from_start: {pattern1['days_from_start'].mean():.0f}")
print(f"  Likely reason: December holiday/gift buyers who never intended to return")

# Pattern 2: Late joiners with mediocre products who returned (False Negatives)
fn = analysis_df[analysis_df['error_type'] == 'False Negative']
pattern2 = fn[(fn['basket_repeat_score'] < 0.68) & (fn['days_from_start'] > 180)]
print(f"\nPattern 2: Low product score + Late joiner → Predicted One-time, Actually Repeat")
print(f"  Count: {len(pattern2)} customers ({len(pattern2)/len(fn)*100:.1f}% of false negatives)")
print(f"  Average basket_repeat_score: {pattern2['basket_repeat_score'].mean():.3f}")
print(f"  Average days_from_start: {pattern2['days_from_start'].mean():.0f}")
print(f"  Likely reason: Brand discovery late in dataset, loyal despite product mix")

# Pattern 3: Edge cases around decision boundary
pattern3 = analysis_df[(analysis_df['predicted_proba'] > 0.45) & 
                       (analysis_df['predicted_proba'] < 0.55) &
                       (analysis_df['error_type'] != 'Correct')]
print(f"\nPattern 3: Uncertain predictions (0.45-0.55 probability) that were wrong")
print(f"  Count: {len(pattern3)} customers")
print(f"  These are genuinely hard to classify - features give mixed signals")

# ============================================================================
# Feature Distributions by Error Type
# ============================================================================
print("\n" + "="*80)
print("FEATURE DISTRIBUTIONS BY ERROR TYPE")
print("="*80)

correct = analysis_df[analysis_df['error_type'] == 'Correct']
fp_all = analysis_df[analysis_df['error_type'] == 'False Positive']
fn_all = analysis_df[analysis_df['error_type'] == 'False Negative']

key_features = ['basket_repeat_score', 'best_product_repeat_score', 
                'order_value', 'num_items', 'days_from_start', 'month']

print("\nFeature comparison:")
comparison = pd.DataFrame({
    'Feature': key_features,
    'Correct': [correct[f].mean() for f in key_features],
    'False Positive': [fp_all[f].mean() for f in key_features],
    'False Negative': [fn_all[f].mean() for f in key_features]
})
print(comparison.round(3).to_string(index=False))

# ============================================================================
# Visualizations
# ============================================================================
print("\n" + "="*80)
print("CREATING ERROR ANALYSIS VISUALIZATIONS")
print("="*80)

# 1. Error rates over time
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Time period error rates
ax = axes[0, 0]
temporal_plot = analysis_df.groupby('time_period')['error_type'].apply(
    lambda x: (x != 'Correct').sum() / len(x) * 100
)
ax.bar(range(len(temporal_plot)), temporal_plot.values, color='#e74c3c', alpha=0.7)
ax.set_xticks(range(len(temporal_plot)))
ax.set_xticklabels(temporal_plot.index, rotation=45, ha='right')
ax.set_ylabel('Error Rate (%)', fontsize=11)
ax.set_title('Error Rate by Time Period', fontsize=12)
ax.grid(alpha=0.3, axis='y')

# Product score vs days_from_start (colored by error)
ax = axes[0, 1]
for error_type, color, marker in [('Correct', '#2ecc71', '.'), 
                                    ('False Positive', '#e74c3c', 'x'),
                                    ('False Negative', '#f39c12', '^')]:
    subset = analysis_df[analysis_df['error_type'] == error_type]
    ax.scatter(subset['days_from_start'], subset['basket_repeat_score'], 
              c=color, marker=marker, alpha=0.6, s=30, label=error_type)
ax.set_xlabel('Days from Start', fontsize=11)
ax.set_ylabel('Basket Repeat Score', fontsize=11)
ax.set_title('Product Score vs Time (by Prediction Type)', fontsize=12)
ax.legend()
ax.grid(alpha=0.3)

# Error type distribution by product quality
ax = axes[1, 0]
error_by_quality = pd.crosstab(analysis_df['product_quality'], 
                               analysis_df['error_type'])
error_by_quality.plot(kind='bar', stacked=True, ax=ax, 
                     color=['#2ecc71', '#e74c3c', '#f39c12'])
ax.set_xlabel('Product Quality', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('Error Distribution by Product Quality', fontsize=12)
ax.legend(title='Prediction Type')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

# Interaction heatmap
ax = axes[1, 1]
interaction_matrix = pd.crosstab(analysis_df['product_quality'], 
                                 analysis_df['time_segment'], 
                                 values=(analysis_df['error_type'] != 'Correct'), 
                                 aggfunc='mean') * 100
sns.heatmap(interaction_matrix, annot=True, fmt='.1f', cmap='Reds', 
           ax=ax, cbar_kws={'label': 'Error Rate (%)'})
ax.set_title('Error Rate Heatmap: Product Quality × Time Period', fontsize=12)
ax.set_ylabel('Product Quality', fontsize=11)
ax.set_xlabel('Time Period', fontsize=11)

plt.tight_layout()
error_viz_path = FIGURE_PATH / 'error_analysis_detailed.png'
plt.savefig(error_viz_path, dpi=300, bbox_inches='tight')
print(f"Saved: {error_viz_path}")
plt.close()

# 2. False positive deep dive
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# FP distribution
fp_time = fp_all.groupby(pd.cut(fp_all['days_from_start'], bins=10)).size()
ax1.bar(range(len(fp_time)), fp_time.values, color='#e74c3c', alpha=0.7)
ax1.set_xlabel('Time Period (binned)', fontsize=11)
ax1.set_ylabel('False Positive Count', fontsize=11)
ax1.set_title('False Positives Over Time', fontsize=12)
ax1.grid(alpha=0.3, axis='y')

# FP by month
fp_month = fp_all.groupby('month').size()
all_month = analysis_df.groupby('month').size()
fp_rate_month = (fp_month / all_month * 100).fillna(0)
ax2.bar(fp_rate_month.index, fp_rate_month.values, color='#e74c3c', alpha=0.7)
ax2.set_xlabel('Month', fontsize=11)
ax2.set_ylabel('False Positive Rate (%)', fontsize=11)
ax2.set_title('False Positive Rate by Month', fontsize=12)
ax2.set_xticks(range(1, 13))
ax2.grid(alpha=0.3, axis='y')

plt.tight_layout()
fp_path = FIGURE_PATH / 'false_positive_analysis.png'
plt.savefig(fp_path, dpi=300, bbox_inches='tight')
print(f"Saved: {fp_path}")
plt.close()

# ============================================================================
# Recommendations for Model Improvement
# ============================================================================
print("\n" + "="*80)
print("RECOMMENDATIONS FOR MODEL IMPROVEMENT")
print("="*80)

print("\n1. FEATURE ENGINEERING:")
print("   - Create interaction feature: basket_repeat_score × early_joiner_flag")
print("   - Add seasonal flag: is_december_buyer (captures holiday shoppers)")
print("   - Add time_since_dataset_start buckets instead of continuous days")

print("\n2. SEGMENTED MODELS:")
print("   - Train separate models for early joiners (Dec-Mar) vs late joiners (Apr-Sep)")
print("   - Early joiners need stronger product signals to overcome seasonality")
print("   - Late joiners may have different loyalty drivers")

print("\n3. ADDITIONAL DATA NEEDED:")
print("   - Customer demographics (age, location detail)")
print("   - Marketing channel (how they found the store)")
print("   - Product categories (home goods vs fashion vs gifts)")
print("   - Purchase context (gift vs personal)")

print("\n4. BUSINESS RULES:")
print("   - Flag December buyers for special treatment (gift buyer risk)")
print("   - Apply higher threshold for early joiners with good product scores")
print("   - Consider time-decay: older customers may have left naturally")

print("\n5. MODEL ENSEMBLE:")
print("   - Combine XGBoost with a model that explicitly handles time periods")
print("   - Use stacking: one model for product signals, another for temporal")

print("\n" + "="*80)
print("ERROR ANALYSIS COMPLETE")
print("="*80)

print("\nMain Findings:")
print("1. Temporal bias: Model struggles with December holiday buyers")
print("2. Product scores work well but miss seasonal/gift purchase intent")
print(f"3. {len(pattern1)} false positives are early joiners (likely gift buyers)")
print(f"4. {len(pattern2)} false negatives are late joiners (brand loyal)")
print("5. Model needs interaction features or segmentation by time period")

print("\nNext step: Implement improvements or move to visualization (Power BI)")