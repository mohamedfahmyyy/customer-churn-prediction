"""
Script 2: Exploratory Data Analysis & Correlation Analysis
Analyzes feature distributions and correlations with target variable
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "figures"

csv_file = list(DATA_PATH.glob("*.csv"))[0]
df = pd.read_csv(csv_file)

print("="*80)
print("EXPLORATORY DATA ANALYSIS")
print("="*80)

features_to_drop = ['CustomerID', 'days_to_second_purchase']
X = df.drop(columns=features_to_drop + ['target'])
y = df['target']

print(f"\nDataset shape: {df.shape}")
print(f"Features for modeling: {X.shape[1]}")
print(f"Target distribution: {y.value_counts()[1]} repeat buyers, {y.value_counts()[0]} one-time buyers")

print(f"\nEncoding categorical variables...")
print(f"Countries before encoding: {X['Country'].nunique()}")
X_encoded = pd.get_dummies(X, columns=['Country'], drop_first=True)
print(f"Total features after encoding: {X_encoded.shape[1]}")

print("\n" + "="*80)
print("TOP CORRELATIONS WITH TARGET VARIABLE")
print("="*80)

analysis_df = X_encoded.copy()
analysis_df['target'] = y.values

correlations = analysis_df.corr()['target'].drop('target').sort_values(ascending=False)

print("\nTop 15 Positive Correlations:")
print(correlations.head(15))

print("\nTop 15 Negative Correlations:")
print(correlations.tail(15))

print("\n" + "="*80)
print("KEY FEATURE STATISTICS")
print("="*80)

key_features = ['order_value', 'num_items', 'basket_repeat_score', 
                'best_product_repeat_score', 'month', 'country_repeat_rate']

for feature in key_features:
    if feature in df.columns:
        print(f"\n{feature}:")
        print(f"  Mean: {df[feature].mean():.3f}")
        print(f"  Median: {df[feature].median():.3f}")
        print(f"  Std: {df[feature].std():.3f}")
        print(f"  Min: {df[feature].min():.3f}")
        print(f"  Max: {df[feature].max():.3f}")

print("\n" + "="*80)
print("REPEAT vs ONE-TIME BUYER COMPARISON")
print("="*80)

repeat_buyers = df[df['target'] == 1]
onetime_buyers = df[df['target'] == 0]

comparison_features = ['order_value', 'num_items', 'basket_repeat_score', 
                       'best_product_repeat_score', 'month']

for feature in comparison_features:
    if feature in df.columns:
        repeat_mean = repeat_buyers[feature].mean()
        onetime_mean = onetime_buyers[feature].mean()
        diff_pct = ((repeat_mean - onetime_mean) / onetime_mean) * 100
        
        print(f"\n{feature}:")
        print(f"  Repeat buyers: {repeat_mean:.3f}")
        print(f"  One-time buyers: {onetime_mean:.3f}")
        print(f"  Difference: {diff_pct:+.1f}%")

print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

top_features = correlations.abs().nlargest(10).index.tolist() + ['target']
heatmap_data = analysis_df[top_features].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap - Top 10 Features', fontsize=16, pad=20)
plt.tight_layout()

heatmap_path = OUTPUT_PATH / 'correlation_heatmap.png'
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
print(f"Saved: {heatmap_path}")
plt.close()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Key Feature Distributions by Customer Type', fontsize=16)

axes[0, 0].hist(repeat_buyers['order_value'], bins=50, alpha=0.6, label='Repeat', color='green')
axes[0, 0].hist(onetime_buyers['order_value'], bins=50, alpha=0.6, label='One-time', color='red')
axes[0, 0].set_xlabel('Order Value ($)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Order Value Distribution')
axes[0, 0].legend()
axes[0, 0].set_xlim(0, 2000)

axes[0, 1].hist(repeat_buyers['basket_repeat_score'], bins=30, alpha=0.6, label='Repeat', color='green')
axes[0, 1].hist(onetime_buyers['basket_repeat_score'], bins=30, alpha=0.6, label='One-time', color='red')
axes[0, 1].set_xlabel('Basket Repeat Score')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Basket Repeat Score Distribution')
axes[0, 1].legend()

axes[1, 0].hist(repeat_buyers['num_items'], bins=50, alpha=0.6, label='Repeat', color='green')
axes[1, 0].hist(onetime_buyers['num_items'], bins=50, alpha=0.6, label='One-time', color='red')
axes[1, 0].set_xlabel('Number of Items')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Basket Size Distribution')
axes[1, 0].legend()
axes[1, 0].set_xlim(0, 150)

month_repeat = repeat_buyers['month'].value_counts().sort_index()
month_onetime = onetime_buyers['month'].value_counts().sort_index()
x = np.arange(1, 13)
width = 0.35
axes[1, 1].bar(x - width/2, month_repeat.reindex(range(1, 13), fill_value=0), 
               width, label='Repeat', color='green', alpha=0.6)
axes[1, 1].bar(x + width/2, month_onetime.reindex(range(1, 13), fill_value=0), 
               width, label='One-time', color='red', alpha=0.6)
axes[1, 1].set_xlabel('Month')
axes[1, 1].set_ylabel('Number of Customers')
axes[1, 1].set_title('First Purchase Month Distribution')
axes[1, 1].legend()
axes[1, 1].set_xticks(range(1, 13))

plt.tight_layout()
dist_path = OUTPUT_PATH / 'feature_distributions.png'
plt.savefig(dist_path, dpi=300, bbox_inches='tight')
print(f"Saved: {dist_path}")
plt.close()

print("\n" + "="*80)
print("EDA COMPLETE")
print("="*80)
print(f"\nKey Insights:")
print(f"1. basket_repeat_score has correlation of {correlations['basket_repeat_score']:.3f} with target")
print(f"2. Repeat buyers spend {((repeat_buyers['order_value'].mean() - onetime_buyers['order_value'].mean()) / onetime_buyers['order_value'].mean() * 100):.1f}% more")
print(f"3. Top country correlation: {correlations.filter(like='Country').abs().max():.3f}")
print(f"\nNext step: python src/03_baseline_models.py")