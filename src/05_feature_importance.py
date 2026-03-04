"""
Script 5: Feature Importance Analysis
Analyzes which features drive model predictions
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Set up paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw"
MODEL_PATH = PROJECT_ROOT / "outputs" / "models"
FIGURE_PATH = PROJECT_ROOT / "outputs" / "figures"

# Load data and model
csv_file = list(DATA_PATH.glob("*.csv"))[0]
df = pd.read_csv(csv_file)

# Prepare features (same as training)
features_to_drop = ['CustomerID', 'days_to_second_purchase']
X = df.drop(columns=features_to_drop + ['target'])
y = df['target']

X_encoded = pd.get_dummies(X, columns=['Country'], drop_first=True)
feature_names = X_encoded.columns.tolist()

# Load trained Random Forest model
rf_model = joblib.load(MODEL_PATH / 'random_forest.pkl')

print("="*80)
print("FEATURE IMPORTANCE ANALYSIS - RANDOM FOREST")
print("="*80)

# Get feature importances
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

# Create dataframe
importance_df = pd.DataFrame({
    'Feature': [feature_names[i] for i in indices],
    'Importance': [importances[i] for i in indices]
})

print("\nTop 20 Most Important Features:")
print(importance_df.head(20).to_string(index=False))

# Calculate cumulative importance
importance_df['Cumulative'] = importance_df['Importance'].cumsum()
features_for_80pct = (importance_df['Cumulative'] <= 0.80).sum()
features_for_90pct = (importance_df['Cumulative'] <= 0.90).sum()

print(f"\nFeatures needed for 80% of predictive power: {features_for_80pct}")
print(f"Features needed for 90% of predictive power: {features_for_90pct}")

# Group country features
country_features = importance_df[importance_df['Feature'].str.startswith('Country_')]
other_features = importance_df[~importance_df['Feature'].str.startswith('Country_')]

print(f"\nCountry features total importance: {country_features['Importance'].sum():.3f}")
print(f"Other features total importance: {other_features['Importance'].sum():.3f}")

# Top non-country features
print("\nTop 10 Non-Country Features:")
print(other_features.head(10).to_string(index=False))

# ============================================================================
# Visualizations
# ============================================================================
print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

# Plot 1: Top 15 features
plt.figure(figsize=(12, 8))
top_15 = importance_df.head(15)
colors = ['#2ecc71' if not f.startswith('Country_') else '#3498db' 
          for f in top_15['Feature']]

plt.barh(range(len(top_15)), top_15['Importance'], color=colors)
plt.yticks(range(len(top_15)), top_15['Feature'])
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Top 15 Most Important Features - Random Forest', fontsize=14, pad=20)
plt.gca().invert_yaxis()

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ecc71', label='Core Features'),
    Patch(facecolor='#3498db', label='Country Features')
]
plt.legend(handles=legend_elements, loc='lower right')
plt.tight_layout()

imp_path = FIGURE_PATH / 'feature_importance.png'
plt.savefig(imp_path, dpi=300, bbox_inches='tight')
print(f"Saved: {imp_path}")
plt.close()

# Plot 2: Cumulative importance
plt.figure(figsize=(12, 8))
plt.plot(range(1, len(importance_df) + 1), importance_df['Cumulative'], 
         linewidth=2, color='#e74c3c')
plt.axhline(y=0.80, color='#95a5a6', linestyle='--', linewidth=1, label='80% threshold')
plt.axhline(y=0.90, color='#95a5a6', linestyle='--', linewidth=1, label='90% threshold')
plt.axvline(x=features_for_80pct, color='#95a5a6', linestyle=':', linewidth=1)
plt.axvline(x=features_for_90pct, color='#95a5a6', linestyle=':', linewidth=1)

plt.xlabel('Number of Features', fontsize=12)
plt.ylabel('Cumulative Importance', fontsize=12)
plt.title('Cumulative Feature Importance', fontsize=14, pad=20)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

cumulative_path = FIGURE_PATH / 'cumulative_importance.png'
plt.savefig(cumulative_path, dpi=300, bbox_inches='tight')
print(f"Saved: {cumulative_path}")
plt.close()

# Plot 3: Feature categories comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Group features by category
categories = {
    'Product Intelligence': ['basket_repeat_score', 'best_product_repeat_score', 
                             'products_with_history_count'],
    'Temporal': ['month_repeat_rate', 'month', 'days_from_start', 'day_of_week', 
                 'hour', 'is_weekend', 'is_business_hours'],
    'Order Characteristics': ['order_value', 'num_items', 'num_unique_products', 
                              'total_quantity', 'avg_item_price', 
                              'product_diversity_ratio', 'order_complexity_score',
                              'order_value_percentile', 'num_items_percentile'],
    'Geographic': ['country_repeat_rate'] + [f for f in feature_names if f.startswith('Country_')]
}

category_importance = {}
for category, features in categories.items():
    total_imp = sum(importances[feature_names.index(f)] 
                   for f in features if f in feature_names)
    category_importance[category] = total_imp

# Bar chart
ax1.bar(category_importance.keys(), category_importance.values(), 
        color=['#2ecc71', '#e74c3c', '#3498db', '#f39c12'])
ax1.set_ylabel('Total Importance', fontsize=11)
ax1.set_title('Feature Importance by Category', fontsize=13)
ax1.tick_params(axis='x', rotation=45)

# Pie chart
ax2.pie(category_importance.values(), labels=category_importance.keys(), 
        autopct='%1.1f%%', startangle=90, 
        colors=['#2ecc71', '#e74c3c', '#3498db', '#f39c12'])
ax2.set_title('Feature Importance Distribution', fontsize=13)

plt.tight_layout()
category_path = FIGURE_PATH / 'importance_by_category.png'
plt.savefig(category_path, dpi=300, bbox_inches='tight')
print(f"Saved: {category_path}")
plt.close()

print("\n" + "="*80)
print("FEATURE IMPORTANCE ANALYSIS COMPLETE")
print("="*80)

# Key insights
top_3_features = importance_df.head(3)['Feature'].tolist()
print(f"\nKey Insights:")
print(f"1. Top 3 features: {', '.join(top_3_features)}")
print(f"2. Product intelligence features account for {sum(importance_df[importance_df['Feature'].isin(['basket_repeat_score', 'best_product_repeat_score'])]['Importance']):.1%} of importance")
print(f"3. Just {features_for_80pct} features explain 80% of model predictions")
