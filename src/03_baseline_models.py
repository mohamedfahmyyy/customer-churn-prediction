"""
Script 3: Baseline Models
Trains Logistic Regression and Random Forest classifiers
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, 
    confusion_matrix, classification_report, precision_score, recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns
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
print("BASELINE MODEL TRAINING")
print("="*80)

# Prepare features
features_to_drop = ['CustomerID', 'days_to_second_purchase']
X = df.drop(columns=features_to_drop + ['target'])
y = df['target']

# One-hot encode Country
X_encoded = pd.get_dummies(X, columns=['Country'], drop_first=True)
feature_names = X_encoded.columns.tolist()

print(f"\nDataset: {X_encoded.shape[0]} samples, {X_encoded.shape[1]} features")
print(f"Target: {y.sum()} repeat buyers ({y.mean():.1%}), {(1-y).sum()} one-time buyers ({1-y.mean():.1%})")

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Scale features for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Store results
results = {}

# ============================================================================
# Model 1: Logistic Regression
# ============================================================================
print("\n" + "="*80)
print("MODEL 1: LOGISTIC REGRESSION")
print("="*80)

lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr_model.fit(X_train_scaled, y_train)

# Predictions
lr_train_pred = lr_model.predict(X_train_scaled)
lr_test_pred = lr_model.predict(X_test_scaled)
lr_train_proba = lr_model.predict_proba(X_train_scaled)[:, 1]
lr_test_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

# Metrics
lr_train_auc = roc_auc_score(y_train, lr_train_proba)
lr_test_auc = roc_auc_score(y_test, lr_test_proba)

print(f"\nTraining ROC-AUC: {lr_train_auc:.4f}")
print(f"Test ROC-AUC: {lr_test_auc:.4f}")

print("\nTest Set Classification Report:")
print(classification_report(y_test, lr_test_pred, target_names=['One-time', 'Repeat']))

# Cross-validation
cv_scores = cross_val_score(lr_model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
print(f"\n5-Fold CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

results['Logistic Regression'] = {
    'model': lr_model,
    'train_auc': lr_train_auc,
    'test_auc': lr_test_auc,
    'test_proba': lr_test_proba,
    'cv_mean': cv_scores.mean(),
    'cv_std': cv_scores.std()
}

# Save model
joblib.dump(lr_model, MODEL_PATH / 'logistic_regression.pkl')
joblib.dump(scaler, MODEL_PATH / 'scaler.pkl')
print(f"\nSaved: logistic_regression.pkl")

# ============================================================================
# Model 2: Random Forest
# ============================================================================
print("\n" + "="*80)
print("MODEL 2: RANDOM FOREST")
print("="*80)

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# Predictions
rf_train_pred = rf_model.predict(X_train)
rf_test_pred = rf_model.predict(X_test)
rf_train_proba = rf_model.predict_proba(X_train)[:, 1]
rf_test_proba = rf_model.predict_proba(X_test)[:, 1]

# Metrics
rf_train_auc = roc_auc_score(y_train, rf_train_proba)
rf_test_auc = roc_auc_score(y_test, rf_test_proba)

print(f"\nTraining ROC-AUC: {rf_train_auc:.4f}")
print(f"Test ROC-AUC: {rf_test_auc:.4f}")

print("\nTest Set Classification Report:")
print(classification_report(y_test, rf_test_pred, target_names=['One-time', 'Repeat']))

# Cross-validation
cv_scores_rf = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='roc_auc')
print(f"\n5-Fold CV ROC-AUC: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std():.4f})")

results['Random Forest'] = {
    'model': rf_model,
    'train_auc': rf_train_auc,
    'test_auc': rf_test_auc,
    'test_proba': rf_test_proba,
    'cv_mean': cv_scores_rf.mean(),
    'cv_std': cv_scores_rf.std()
}

# Save model
joblib.dump(rf_model, MODEL_PATH / 'random_forest.pkl')
print(f"\nSaved: random_forest.pkl")

# ============================================================================
# Model Comparison
# ============================================================================
print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Train AUC': [r['train_auc'] for r in results.values()],
    'Test AUC': [r['test_auc'] for r in results.values()],
    'CV Mean': [r['cv_mean'] for r in results.values()],
    'CV Std': [r['cv_std'] for r in results.values()],
    'Overfit': [r['train_auc'] - r['test_auc'] for r in results.values()]
})

print("\n", comparison_df.to_string(index=False))

# ============================================================================
# Business Metrics: Precision at Top 30%
# ============================================================================
print("\n" + "="*80)
print("BUSINESS METRICS: TARGETING TOP 30% HIGHEST RISK")
print("="*80)

for model_name, result in results.items():
    proba = result['test_proba']
    
    # Sort by probability (ascending - lowest probability = highest risk of NOT repeating)
    sorted_indices = np.argsort(proba)
    top_30_pct = int(len(sorted_indices) * 0.30)
    high_risk_indices = sorted_indices[:top_30_pct]
    
    # Calculate precision (what % of targeted customers are actually one-time buyers)
    targeted_labels = y_test.iloc[high_risk_indices]
    precision_at_30 = (targeted_labels == 0).sum() / len(targeted_labels)
    
    # Calculate recall (what % of all one-time buyers did we catch)
    total_onetime = (y_test == 0).sum()
    caught_onetime = (targeted_labels == 0).sum()
    recall_at_30 = caught_onetime / total_onetime
    
    print(f"\n{model_name}:")
    print(f"  Customers targeted: {len(high_risk_indices)} (30% of test set)")
    print(f"  Actual one-time buyers caught: {caught_onetime}")
    print(f"  Precision @ 30%: {precision_at_30:.1%}")
    print(f"  Recall @ 30%: {recall_at_30:.1%}")
    print(f"  Expected campaign ROI: {precision_at_30 * 0.20 * 1368 * len(high_risk_indices) / (5 * len(high_risk_indices)):.1f}x")

# ============================================================================
# Visualizations
# ============================================================================
print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

# ROC Curves
plt.figure(figsize=(10, 8))
for model_name, result in results.items():
    fpr, tpr, _ = roc_curve(y_test, result['test_proba'])
    auc = result['test_auc']
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Model Comparison', fontsize=14, pad=20)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()

roc_path = FIGURE_PATH / 'roc_curve.png'
plt.savefig(roc_path, dpi=300, bbox_inches='tight')
print(f"Saved: {roc_path}")
plt.close()

# Precision-Recall Curves
plt.figure(figsize=(10, 8))
for model_name, result in results.items():
    precision, recall, _ = precision_recall_curve(y_test, result['test_proba'])
    plt.plot(recall, precision, label=model_name, linewidth=2)

plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curves', fontsize=14, pad=20)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()

pr_path = FIGURE_PATH / 'precision_recall_curve.png'
plt.savefig(pr_path, dpi=300, bbox_inches='tight')
print(f"Saved: {pr_path}")
plt.close()

print("\n" + "="*80)
print("BASELINE MODELS COMPLETE")
print("="*80)
print(f"\nBest model: {comparison_df.loc[comparison_df['Test AUC'].idxmax(), 'Model']}")
print(f"Next step: python src/04_advanced_models.py")