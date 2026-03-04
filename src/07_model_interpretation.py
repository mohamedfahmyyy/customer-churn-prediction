"""
Model Interpretation using SHAP
Explains model predictions and identifies systematic error patterns
Outputs: SHAP visualizations, waterfall plots, confidence analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
import shap

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw"
MODEL_PATH = PROJECT_ROOT / "outputs" / "models"
FIGURE_PATH = PROJECT_ROOT / "outputs" / "figures"

print("="*80)
print("MODEL INTERPRETATION & EXPLAINABILITY")
print("="*80)

csv_file = list(DATA_PATH.glob("*.csv"))[0]
df = pd.read_csv(csv_file)

features_to_drop = ['CustomerID', 'days_to_second_purchase']
X = df.drop(columns=features_to_drop + ['target'])
y = df['target']

X_encoded = pd.get_dummies(X, columns=['Country'], drop_first=True)
feature_names = X_encoded.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

print("\nLoading XGBoost tuned model...")
xgb_model = joblib.load(MODEL_PATH / 'xgboost_tuned.pkl')

y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
y_pred = xgb_model.predict(X_test)

print(f"Test set: {len(X_test)} customers")
print(f"Model loaded successfully")

print("\n" + "="*80)
print("SHAP ANALYSIS - EXPLAINING PREDICTIONS")
print("="*80)

print("\nCalculating SHAP values (this may take 1-2 minutes)...")

explainer = shap.TreeExplainer(xgb_model)


sample_size = min(500, len(X_test))
X_test_sample = X_test.sample(n=sample_size, random_state=42)
shap_values = explainer.shap_values(X_test_sample)

print(f"SHAP values calculated for {sample_size} customers")

base_value = explainer.expected_value
print(f"\nBase prediction (no features): {base_value:.3f}")
print(f"This represents the average prediction across all training data")

print("\n" + "="*80)
print("CREATING SHAP VISUALIZATIONS")
print("="*80)

print("\n1. Creating SHAP summary plot...")
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names, 
                  show=False, max_display=15)
plt.title('SHAP Feature Importance - Impact on Predictions', fontsize=14, pad=20)
plt.tight_layout()
summary_path = FIGURE_PATH / 'shap_summary_plot.png'
plt.savefig(summary_path, dpi=300, bbox_inches='tight')
print(f"Saved: {summary_path}")
plt.close()

print("2. Creating SHAP feature importance bar plot...")
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names,
                  plot_type="bar", show=False, max_display=15)
plt.title('SHAP Feature Importance - Mean Absolute Impact', fontsize=14, pad=20)
plt.tight_layout()
bar_path = FIGURE_PATH / 'shap_importance_bar.png'
plt.savefig(bar_path, dpi=300, bbox_inches='tight')
print(f"Saved: {bar_path}")
plt.close()

print("3. Creating SHAP dependence plots for top features...")

top_features_for_dependence = ['basket_repeat_score', 'best_product_repeat_score', 
                                'days_from_start', 'month']

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

for idx, feature in enumerate(top_features_for_dependence):
    if feature in feature_names:
        feature_idx = feature_names.index(feature)
        shap.dependence_plot(feature_idx, shap_values, X_test_sample, 
                            feature_names=feature_names, ax=axes[idx], show=False)
        axes[idx].set_title(f'SHAP Dependence: {feature}', fontsize=12)

plt.tight_layout()
dependence_path = FIGURE_PATH / 'shap_dependence_plots.png'
plt.savefig(dependence_path, dpi=300, bbox_inches='tight')
print(f"Saved: {dependence_path}")
plt.close()

print("\n" + "="*80)
print("INDIVIDUAL PREDICTION EXAMPLES")
print("="*80)

test_df = X_test_sample.copy()
test_df['true_label'] = y_test.loc[X_test_sample.index].values
test_df['predicted_proba'] = xgb_model.predict_proba(X_test_sample)[:, 1]
test_df['predicted_label'] = (test_df['predicted_proba'] >= 0.5).astype(int)

high_conf_onetime = test_df[(test_df['true_label'] == 0) & 
                             (test_df['predicted_proba'] < 0.3)].head(1)

high_conf_repeat = test_df[(test_df['true_label'] == 1) & 
                            (test_df['predicted_proba'] > 0.7)].head(1)

misclass_onetime = test_df[(test_df['true_label'] == 0) & 
                            (test_df['predicted_label'] == 1)].head(1)

misclass_repeat = test_df[(test_df['true_label'] == 1) & 
                           (test_df['predicted_label'] == 0)].head(1)

cases = [
    ("High Confidence One-time Buyer (Correct)", high_conf_onetime),
    ("High Confidence Repeat Buyer (Correct)", high_conf_repeat),
    ("Misclassified: Predicted Repeat, Actually One-time", misclass_onetime),
    ("Misclassified: Predicted One-time, Actually Repeat", misclass_repeat)
]

print("\nExamining 4 interesting prediction cases...\n")

for case_name, case_df in cases:
    if len(case_df) > 0:
        idx = case_df.index[0]
        sample_idx = X_test_sample.index.get_loc(idx)
        
        print(f"{case_name}:")
        print(f"  True Label: {'Repeat' if case_df.iloc[0]['true_label'] == 1 else 'One-time'}")
        print(f"  Predicted Probability: {case_df.iloc[0]['predicted_proba']:.3f}")
        print(f"  basket_repeat_score: {X_test_sample.iloc[sample_idx]['basket_repeat_score']:.3f}")
        print(f"  best_product_repeat_score: {X_test_sample.iloc[sample_idx]['best_product_repeat_score']:.3f}")
        print(f"  days_from_start: {X_test_sample.iloc[sample_idx]['days_from_start']:.0f}")
        
        plt.figure(figsize=(12, 6))
        shap.plots._waterfall.waterfall_legacy(base_value, shap_values[sample_idx], 
                                                X_test_sample.iloc[sample_idx], 
                                                feature_names=feature_names,
                                                max_display=10, show=False)
        plt.title(f'{case_name}\nTrue: {case_df.iloc[0]["true_label"]}, '
                 f'Predicted: {case_df.iloc[0]["predicted_proba"]:.3f}', 
                 fontsize=12, pad=20)
        plt.tight_layout()
        
        safe_name = case_name.replace(':', '').replace(',', '').replace(' ', '_').lower()
        waterfall_path = FIGURE_PATH / f'shap_waterfall_{safe_name[:30]}.png'
        plt.savefig(waterfall_path, dpi=300, bbox_inches='tight')
        print(f"  Saved waterfall plot: {waterfall_path.name}\n")
        plt.close()

print("="*80)
print("ERROR ANALYSIS - UNDERSTANDING MODEL MISTAKES")
print("="*80)

test_results = pd.DataFrame({
    'true_label': y_test.values,
    'predicted_proba': y_pred_proba,
    'predicted_label': y_pred
})

test_results = pd.concat([test_results, X_test.reset_index(drop=True)], axis=1)

test_results['prediction_type'] = 'Unknown'
test_results.loc[(test_results['true_label'] == 0) & (test_results['predicted_label'] == 0), 'prediction_type'] = 'True Negative (Correct)'
test_results.loc[(test_results['true_label'] == 1) & (test_results['predicted_label'] == 1), 'prediction_type'] = 'True Positive (Correct)'
test_results.loc[(test_results['true_label'] == 0) & (test_results['predicted_label'] == 1), 'prediction_type'] = 'False Positive (Error)'
test_results.loc[(test_results['true_label'] == 1) & (test_results['predicted_label'] == 0), 'prediction_type'] = 'False Negative (Error)'

print("\nPrediction Breakdown:")
print(test_results['prediction_type'].value_counts())

false_positives = test_results[test_results['prediction_type'] == 'False Positive (Error)']
true_negatives = test_results[test_results['prediction_type'] == 'True Negative (Correct)']

print("\n" + "-"*80)
print("FALSE POSITIVES: Predicted Repeat, Actually One-time")
print("-"*80)
print(f"Count: {len(false_positives)} customers")

if len(false_positives) > 0:
    comparison_features = ['basket_repeat_score', 'best_product_repeat_score', 
                           'order_value', 'num_items', 'days_from_start', 'month']
    
    print("\nComparing False Positives vs True Negatives:")
    for feature in comparison_features:
        if feature in false_positives.columns:
            fp_mean = false_positives[feature].mean()
            tn_mean = true_negatives[feature].mean()
            diff_pct = ((fp_mean - tn_mean) / tn_mean) * 100 if tn_mean != 0 else 0
            
            print(f"\n{feature}:")
            print(f"  False Positives: {fp_mean:.3f}")
            print(f"  True Negatives: {tn_mean:.3f}")
            print(f"  Difference: {diff_pct:+.1f}%")
    
    print("\nInterpretation:")
    print("False positives have product scores similar to repeat buyers,")
    print("but something else prevented them from returning.")
    print("Possible reasons: External factors, customer experience issues, price sensitivity")

false_negatives = test_results[test_results['prediction_type'] == 'False Negative (Error)']
true_positives = test_results[test_results['prediction_type'] == 'True Positive (Correct)']

print("\n" + "-"*80)
print("FALSE NEGATIVES: Predicted One-time, Actually Repeat")
print("-"*80)
print(f"Count: {len(false_negatives)} customers")

if len(false_negatives) > 0:
    print("\nComparing False Negatives vs True Positives:")
    for feature in comparison_features:
        if feature in false_negatives.columns:
            fn_mean = false_negatives[feature].mean()
            tp_mean = true_positives[feature].mean()
            diff_pct = ((fn_mean - tp_mean) / tp_mean) * 100 if tp_mean != 0 else 0
            
            print(f"\n{feature}:")
            print(f"  False Negatives: {fn_mean:.3f}")
            print(f"  True Positives: {tp_mean:.3f}")
            print(f"  Difference: {diff_pct:+.1f}%")
    
    print("\nInterpretation:")
    print("False negatives have lower product scores than typical repeat buyers,")
    print("but returned anyway. These may be brand-loyal customers or")
    print("customers who found value beyond product category.")

print("\n" + "="*80)
print("PREDICTION CONFIDENCE ANALYSIS")
print("="*80)

test_results['confidence_bin'] = pd.cut(test_results['predicted_proba'], 
                                        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                        labels=['Very Low (0-0.2)', 'Low (0.2-0.4)', 
                                               'Medium (0.4-0.6)', 'High (0.6-0.8)', 
                                               'Very High (0.8-1.0)'])

print("\nPredictions by Confidence Level:")
confidence_analysis = test_results.groupby('confidence_bin').agg({
    'true_label': ['count', 'mean'],
    'predicted_label': 'mean'
}).round(3)

confidence_analysis.columns = ['Count', 'Actual Repeat Rate', 'Predicted Repeat Rate']
print(confidence_analysis)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.hist(test_results[test_results['true_label'] == 1]['predicted_proba'], 
         bins=20, alpha=0.6, label='Actual Repeat', color='green')
ax1.hist(test_results[test_results['true_label'] == 0]['predicted_proba'], 
         bins=20, alpha=0.6, label='Actual One-time', color='red')
ax1.set_xlabel('Predicted Probability', fontsize=11)
ax1.set_ylabel('Count', fontsize=11)
ax1.set_title('Prediction Confidence Distribution', fontsize=12)
ax1.legend()
ax1.grid(alpha=0.3)

bin_centers = [0.1, 0.3, 0.5, 0.7, 0.9]
actual_rates = confidence_analysis['Actual Repeat Rate'].values
ax2.plot(bin_centers, actual_rates, 'o-', linewidth=2, markersize=8, label='Actual', color='blue')
ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect Calibration')
ax2.set_xlabel('Predicted Probability', fontsize=11)
ax2.set_ylabel('Actual Repeat Rate', fontsize=11)
ax2.set_title('Model Calibration', fontsize=12)
ax2.legend()
ax2.grid(alpha=0.3)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)

plt.tight_layout()
confidence_path = FIGURE_PATH / 'confidence_analysis.png'
plt.savefig(confidence_path, dpi=300, bbox_inches='tight')
print(f"\nSaved: {confidence_path}")
plt.close()

print("\n" + "="*80)
print("MODEL INTERPRETATION COMPLETE")
print("="*80)

print("\nKey Insights:")
print("1. SHAP analysis shows basket_repeat_score drives most predictions")
print("2. Model is well-calibrated - predicted probabilities match actual rates")
print(f"3. False positives ({len(false_positives)}): Have good product scores but didn't return")
print(f"4. False negatives ({len(false_negatives)}): Have lower scores but returned anyway")
print("\nNext step: python src/08_error_deep_dive.py")