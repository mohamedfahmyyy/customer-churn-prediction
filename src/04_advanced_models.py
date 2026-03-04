"""
Script 4: Advanced Models
Trains XGBoost and optimizes hyperparameters
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb
import matplotlib.pyplot as plt
import joblib

# Set up paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = PROJECT_ROOT / "outputs"
MODEL_PATH = OUTPUT_PATH / "models"
FIGURE_PATH = OUTPUT_PATH / "figures"

# Load data
csv_file = list(DATA_PATH.glob("*.csv"))[0]
df = pd.read_csv(csv_file)

print("="*80)
print("ADVANCED MODEL TRAINING - XGBOOST")
print("="*80)

# Prepare features
features_to_drop = ['CustomerID', 'days_to_second_purchase']
X = df.drop(columns=features_to_drop + ['target'])
y = df['target']

# One-hot encode Country
X_encoded = pd.get_dummies(X, columns=['Country'], drop_first=True)

print(f"\nDataset: {X_encoded.shape[0]} samples, {X_encoded.shape[1]} features")
print(f"Target: {y.sum()} repeat buyers ({y.mean():.1%}), {(1-y).sum()} one-time buyers ({1-y.mean():.1%})")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# ============================================================================
# Model 1: XGBoost with Default Parameters
# ============================================================================
print("\n" + "="*80)
print("XGBOOST - DEFAULT PARAMETERS")
print("="*80)

# Calculate scale_pos_weight for class imbalance
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_default = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='auc'
)

xgb_default.fit(X_train, y_train)

# Predictions
xgb_train_proba = xgb_default.predict_proba(X_train)[:, 1]
xgb_test_proba = xgb_default.predict_proba(X_test)[:, 1]
xgb_test_pred = xgb_default.predict(X_test)

# Metrics
xgb_train_auc = roc_auc_score(y_train, xgb_train_proba)
xgb_test_auc = roc_auc_score(y_test, xgb_test_proba)

print(f"\nTraining ROC-AUC: {xgb_train_auc:.4f}")
print(f"Test ROC-AUC: {xgb_test_auc:.4f}")
print(f"Overfit gap: {xgb_train_auc - xgb_test_auc:.4f}")

print("\nTest Set Classification Report:")
print(classification_report(y_test, xgb_test_pred, target_names=['One-time', 'Repeat']))

# Cross-validation
cv_scores = cross_val_score(xgb_default, X_train, y_train, cv=5, scoring='roc_auc')
print(f"\n5-Fold CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ============================================================================
# Model 2: XGBoost with Hyperparameter Tuning
# ============================================================================
print("\n" + "="*80)
print("XGBOOST - HYPERPARAMETER TUNING")
print("="*80)

print("\nSearching optimal hyperparameters (this may take 2-3 minutes)...")

param_grid = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1, 0.15],
    'n_estimators': [100, 150, 200],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 0.9, 1.0]
}

xgb_base = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='auc'
)

grid_search = GridSearchCV(
    xgb_base,
    param_grid,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"\nBest parameters found:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

# Best model
xgb_tuned = grid_search.best_estimator_

# Predictions
xgb_tuned_train_proba = xgb_tuned.predict_proba(X_train)[:, 1]
xgb_tuned_test_proba = xgb_tuned.predict_proba(X_test)[:, 1]
xgb_tuned_test_pred = xgb_tuned.predict(X_test)

# Metrics
xgb_tuned_train_auc = roc_auc_score(y_train, xgb_tuned_train_proba)
xgb_tuned_test_auc = roc_auc_score(y_test, xgb_tuned_test_proba)

print(f"\nTuned Model Performance:")
print(f"  Training ROC-AUC: {xgb_tuned_train_auc:.4f}")
print(f"  Test ROC-AUC: {xgb_tuned_test_auc:.4f}")
print(f"  Overfit gap: {xgb_tuned_train_auc - xgb_tuned_test_auc:.4f}")

print("\nTest Set Classification Report:")
print(classification_report(y_test, xgb_tuned_test_pred, target_names=['One-time', 'Repeat']))

# Save best model
joblib.dump(xgb_tuned, MODEL_PATH / 'xgboost_tuned.pkl')
print(f"\nSaved: xgboost_tuned.pkl")

# ============================================================================
# Model Comparison with Previous Models
# ============================================================================
print("\n" + "="*80)
print("ALL MODELS COMPARISON")
print("="*80)

# Load previous models
lr_model = joblib.load(MODEL_PATH / 'logistic_regression.pkl')
rf_model = joblib.load(MODEL_PATH / 'random_forest.pkl')
scaler = joblib.load(MODEL_PATH / 'scaler.pkl')

X_test_scaled = scaler.transform(X_test)

# Get predictions from all models
lr_test_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
rf_test_proba = rf_model.predict_proba(X_test)[:, 1]

# Compare
comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost (default)', 'XGBoost (tuned)'],
    'Test ROC-AUC': [
        roc_auc_score(y_test, lr_test_proba),
        roc_auc_score(y_test, rf_test_proba),
        xgb_test_auc,
        xgb_tuned_test_auc
    ]
})

print("\n", comparison.to_string(index=False))

best_model_name = comparison.loc[comparison['Test ROC-AUC'].idxmax(), 'Model']
best_auc = comparison['Test ROC-AUC'].max()

print(f"\nBest performing model: {best_model_name} (ROC-AUC: {best_auc:.4f})")

# ============================================================================
# Business Metrics for XGBoost
# ============================================================================
print("\n" + "="*80)
print("BUSINESS METRICS - XGBOOST TUNED MODEL")
print("="*80)

# Sort by risk (lowest probability = highest churn risk)
sorted_indices = np.argsort(xgb_tuned_test_proba)
top_30_pct = int(len(sorted_indices) * 0.30)
high_risk_indices = sorted_indices[:top_30_pct]

# Calculate metrics
targeted_labels = y_test.iloc[high_risk_indices]
precision_at_30 = (targeted_labels == 0).sum() / len(targeted_labels)
total_onetime = (y_test == 0).sum()
caught_onetime = (targeted_labels == 0).sum()
recall_at_30 = caught_onetime / total_onetime

print(f"\nTargeting top 30% highest-risk customers:")
print(f"  Customers targeted: {len(high_risk_indices)}")
print(f"  Actual one-time buyers caught: {caught_onetime}")
print(f"  Precision @ 30%: {precision_at_30:.1%}")
print(f"  Recall @ 30%: {recall_at_30:.1%}")
print(f"  Expected campaign ROI: {precision_at_30 * 0.20 * 1368 * len(high_risk_indices) / (5 * len(high_risk_indices)):.1f}x")

# ============================================================================
# Feature Importance Comparison
# ============================================================================
print("\n" + "="*80)
print("FEATURE IMPORTANCE - XGBOOST vs RANDOM FOREST")
print("="*80)

# Get feature importances
xgb_importance = xgb_tuned.feature_importances_
rf_importance = rf_model.feature_importances_
feature_names = X_encoded.columns.tolist()

# Create comparison dataframe
importance_comparison = pd.DataFrame({
    'Feature': feature_names,
    'XGBoost': xgb_importance,
    'Random Forest': rf_importance
})

# Sort by XGBoost importance
importance_comparison = importance_comparison.sort_values('XGBoost', ascending=False)

print("\nTop 10 Features Comparison:")
print(importance_comparison.head(10).to_string(index=False))

# Visualize comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

top_n = 12
top_features_xgb = importance_comparison.head(top_n)

# XGBoost
ax1.barh(range(len(top_features_xgb)), top_features_xgb['XGBoost'], color='#9b59b6')
ax1.set_yticks(range(len(top_features_xgb)))
ax1.set_yticklabels(top_features_xgb['Feature'])
ax1.set_xlabel('Importance Score', fontsize=11)
ax1.set_title('XGBoost Feature Importance', fontsize=13)
ax1.invert_yaxis()
ax1.grid(alpha=0.3)

# Random Forest
top_features_rf = importance_comparison.sort_values('Random Forest', ascending=False).head(top_n)
ax2.barh(range(len(top_features_rf)), top_features_rf['Random Forest'], color='#2ecc71')
ax2.set_yticks(range(len(top_features_rf)))
ax2.set_yticklabels(top_features_rf['Feature'])
ax2.set_xlabel('Importance Score', fontsize=11)
ax2.set_title('Random Forest Feature Importance', fontsize=13)
ax2.invert_yaxis()
ax2.grid(alpha=0.3)

plt.tight_layout()
comparison_path = FIGURE_PATH / 'xgboost_vs_rf_importance.png'
plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
print(f"\nSaved: {comparison_path}")
plt.close()

print("\n" + "="*80)
print("ADVANCED MODELS COMPLETE")
print("="*80)