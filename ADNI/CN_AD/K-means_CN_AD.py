import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    adjusted_rand_score, normalized_mutual_info_score,
    adjusted_mutual_info_score, silhouette_score, davies_bouldin_score
)
import warnings
warnings.filterwarnings('ignore')

print("=" * 72)
print("K-MEANS BASELINE – CN vs AD (10 runs with std)")
print("=" * 72)

# Load data – adjust file paths if needed
cn_data = np.load("/home/snu/Downloads/Histogram_CN_FA_20bin_updated.npy", allow_pickle=True)
ad_data = np.load("/home/snu/Downloads/Histogram_AD_FA_20bin_updated.npy", allow_pickle=True)

X = np.vstack([cn_data, ad_data])
y = np.hstack([np.zeros(cn_data.shape[0], dtype=np.int64),
               np.ones(ad_data.shape[0], dtype=np.int64)])

# Fixed permutation for reproducibility (same as other methods)
np.random.seed(42)
perm = np.random.permutation(X.shape[0])
X, y = X[perm], y[perm]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.astype(np.float32))

n_runs = 10
results = {
    'accuracy': [], 'precision': [], 'recall': [], 'f1': [],
    'nmi': [], 'ari': [], 'ami': [], 'silhouette': [], 'davies_bouldin': []
}

for run in range(n_runs):
    seed = 42 + run
    kmeans = KMeans(n_clusters=2, random_state=seed, max_iter=5000, n_init=10)
    kmeans.fit(X_scaled)
    yp = kmeans.labels_

    # Align cluster labels to majority diagnosis (post‑hoc)
    acc = accuracy_score(y, yp)
    acc_inv = accuracy_score(y, 1 - yp)
    if acc_inv > acc:
        yp = 1 - yp
        acc = acc_inv

    # Weighted metrics (averaged across both classes)
    precision = precision_score(y, yp, average='weighted', zero_division=0)
    recall = recall_score(y, yp, average='weighted', zero_division=0)
    f1 = f1_score(y, yp, average='weighted', zero_division=0)
    nmi = normalized_mutual_info_score(y, yp, average_method='arithmetic')
    ari = adjusted_rand_score(y, yp)
    ami = adjusted_mutual_info_score(y, yp, average_method='arithmetic')

    # Silhouette and Davies‑Bouldin (require at least 2 samples per cluster)
    if len(np.unique(yp)) > 1 and min(np.bincount(yp)) >= 2:
        sil = silhouette_score(X_scaled, yp, metric='euclidean')
        db = davies_bouldin_score(X_scaled, yp)
    else:
        sil = np.nan
        db = np.nan

    results['accuracy'].append(acc)
    results['precision'].append(precision)
    results['recall'].append(recall)
    results['f1'].append(f1)
    results['nmi'].append(nmi)
    results['ari'].append(ari)
    results['ami'].append(ami)
    results['silhouette'].append(sil)
    results['davies_bouldin'].append(db)

# Compute mean and std over 10 runs
mean_std = {}
for metric in results:
    arr = np.array(results[metric])
    mean_std[metric] = (np.nanmean(arr), np.nanstd(arr))

print("\n" + "-" * 100)
print("K-means results (10 runs) – mean ± std")
print("-" * 100)
print(f"Method\tAccuracy\tPrecision\tRecall\tF1\tNMI\tARI\tAMI\tSilhouette\tDavies-Bouldin")
print(f"K-means\t{mean_std['accuracy'][0]:.4f}±{mean_std['accuracy'][1]:.4f}\t"
      f"{mean_std['precision'][0]:.4f}±{mean_std['precision'][1]:.4f}\t"
      f"{mean_std['recall'][0]:.4f}±{mean_std['recall'][1]:.4f}\t"
      f"{mean_std['f1'][0]:.4f}±{mean_std['f1'][1]:.4f}\t"
      f"{mean_std['nmi'][0]:.4f}±{mean_std['nmi'][1]:.4f}\t"
      f"{mean_std['ari'][0]:.4f}±{mean_std['ari'][1]:.4f}\t"
      f"{mean_std['ami'][0]:.4f}±{mean_std['ami'][1]:.4f}\t"
      f"{mean_std['silhouette'][0]:.4f}±{mean_std['silhouette'][1]:.4f}\t"
      f"{mean_std['davies_bouldin'][0]:.4f}±{mean_std['davies_bouldin'][1]:.4f}")

# Save detailed results to CSV
df_results = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1', 'NMI', 'ARI', 'AMI', 'Silhouette', 'Davies-Bouldin'],
    'Mean': [mean_std['accuracy'][0], mean_std['precision'][0], mean_std['recall'][0],
             mean_std['f1'][0], mean_std['nmi'][0], mean_std['ari'][0], mean_std['ami'][0],
             mean_std['silhouette'][0], mean_std['davies_bouldin'][0]],
    'Std': [mean_std['accuracy'][1], mean_std['precision'][1], mean_std['recall'][1],
            mean_std['f1'][1], mean_std['nmi'][1], mean_std['ari'][1], mean_std['ami'][1],
            mean_std['silhouette'][1], mean_std['davies_bouldin'][1]]
})
df_results.to_csv("kmeans_cn_vs_ad_10run_summary.csv", index=False, float_format="%.4f")
print("\nSaved results to: kmeans_cn_vs_ad_10run_summary.csv")
