
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# --- LOAD DATA ---
data = pd.read_csv('data/Loan_Data.csv')

# --- 1. MSE/K-Means Quantization ---
fico_scores = data['fico_score'].values.reshape(-1, 1)
num_buckets = 5

kmeans = KMeans(n_clusters=num_buckets, random_state=42).fit(fico_scores)
data['fico_bucket_mse'] = kmeans.labels_

# Plot and save FICO score histogram by bucket
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='fico_score', hue='fico_bucket_mse', multiple='stack', palette='tab10')
plt.title('FICO Score Distribution by Risk Bucket')
plt.xlabel('FICO Score')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig("figures/ficobuckets_hist.png")
plt.close()

bucket_centers = sorted(kmeans.cluster_centers_.flatten())
print("MSE Bucket centroids:", bucket_centers)
print(data.groupby('fico_bucket_mse')['fico_score'].mean())

# --- 2. Log-Likelihood Quantization ---
def compute_log_likelihood(data, bucket_edges):
    log_likelihood = 0
    for i in range(len(bucket_edges) - 1):
        bucket_data = data[(data['fico_score'] >= bucket_edges[i]) & (data['fico_score'] < bucket_edges[i+1])]
        n_i = len(bucket_data)
        k_i = bucket_data['default'].sum()
        if n_i == 0:
            continue
        p_i = k_i / n_i
        if p_i == 0 or p_i == 1:
            continue
        log_likelihood += k_i * np.log(p_i) + (n_i - k_i) * np.log(1 - p_i)
    return log_likelihood

bucket_edges = np.linspace(data['fico_score'].min(), data['fico_score'].max() + 1, num_buckets + 1)
ll = compute_log_likelihood(data, bucket_edges)
print(f"\nLog-likelihood: {ll:.3f}")
print("Log-likelihood bucket boundaries:", bucket_edges)
data['fico_bucket_ll'] = pd.cut(data['fico_score'], bins=bucket_edges, labels=False, include_lowest=True)

# Plot and save histogram for Log-Likelihood buckets
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='fico_score', hue='fico_bucket_ll', multiple='stack', palette='tab10')
plt.title('FICO Score Distribution by Log-Likelihood Buckets')
plt.xlabel('FICO Score')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig("figures/ficobuckets_ll_hist.png")
plt.close()

print(data.groupby('fico_bucket_ll')['default'].mean())

# --- Start Visualization Add-On ---
import os
os.makedirs('figures', exist_ok=True)

# Assign buckets (or reuse your earlier bucket column)
def assign_fico_bucket(fico_score):
    if fico_score >= 760:
        return 'Excellent'
    elif fico_score >= 700:
        return 'Good'
    elif fico_score >= 650:
        return 'Fair'
    elif fico_score >= 600:
        return 'Poor'
    else:
        return 'Very Poor'


df = pd.read_csv('data/Loan_Data.csv')
df['Bucket'] = df['fico_score'].apply(assign_fico_bucket)

import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='fico_score', hue='Bucket', multiple='stack', bins=30, palette='muted')
plt.title('FICO Score Distribution by Risk Bucket')
plt.xlabel('FICO Score')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('figures/fico_bucket_distribution.png')
plt.show()
