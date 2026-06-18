#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, log_loss, adjusted_rand_score,
                             normalized_mutual_info_score, adjusted_mutual_info_score,
                             silhouette_score, davies_bouldin_score)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader, Subset
from torchvision import transforms
import timm
import random
import pandas as pd
from scipy import stats

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("CUDA available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

RADIODINO_MODEL = "hf_hub:Snarcy/RadioDino-s16"   # outputs 384‑dim features
NUM_CLUSTERS = 2
NUM_RUNS = 10
SEED_BASE = 42

# ─── LOAD PNEUMONIAMNIST DATA ────────────────────────────────────────────────
data_npz = np.load('/home/snu/Downloads/pneumoniamnist_224.npz', allow_pickle=True)

all_images = np.concatenate([data_npz['train_images'],
                              data_npz['val_images'],
                              data_npz['test_images']], axis=0)
all_labels = np.concatenate([data_npz['train_labels'],
                              data_npz['val_labels'],
                              data_npz['test_labels']], axis=0).squeeze()

images = all_images.astype(np.float32) / 255.0
images = np.repeat(images[:, None, :, :], 3, axis=1)   # (N, 3, 224, 224)
X_img = torch.tensor(images)
y_img = torch.tensor(all_labels).long()
print("Images shape:", X_img.shape, "  Labels shape:", y_img.shape)

# ── Subsample: up to 2000 per class (PneumoniaMNIST is larger than BreastMNIST) ──
dataset = TensorDataset(X_img, y_img)
class0_indices = [i for i in range(len(y_img)) if y_img[i] == 0]  # Normal
class1_indices = [i for i in range(len(y_img)) if y_img[i] == 1]  # Pneumonia
random.seed(SEED_BASE)
sampled_class0 = random.sample(class0_indices, min(2000, len(class0_indices)))
sampled_class1 = random.sample(class1_indices, min(2000, len(class1_indices)))
combined_indices = sampled_class0 + sampled_class1
random.shuffle(combined_indices)
final_dataset = Subset(dataset, combined_indices)
final_loader = DataLoader(final_dataset, batch_size=64, shuffle=False)

# ─── EXTRACT RADIODINO FEATURES ───────────────────────────────────────────────
print("\nLoading RadioDINO model:", RADIODINO_MODEL)
radiodino = timm.create_model(RADIODINO_MODEL, pretrained=True)
radiodino.eval().to(DEVICE)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

rd_feats, y_list = [], []
with torch.no_grad():
    for imgs, lbls in final_loader:
        imgs = imgs.to(DEVICE)
        imgs_norm = normalize(imgs)
        feats = radiodino(imgs_norm)          # CLS token, shape (batch, 384)
        rd_feats.append(feats.cpu())
        y_list.extend(lbls.cpu().tolist())

features_np = torch.cat(rd_feats, dim=0).numpy().astype(np.float32)
y_labels = np.array(y_list, dtype=np.int64)

# Shuffle once for reproducibility
np.random.seed(SEED_BASE)
perm = np.random.permutation(features_np.shape[0])
features_np = features_np[perm]
y_labels = y_labels[perm]

print(f"Features shape: {features_np.shape}, Labels shape: {y_labels.shape}")
print(f"Normal (class 0): {np.sum(y_labels==0)}, Pneumonia (class 1): {np.sum(y_labels==1)}")
print("NOTE: Diagnostic labels are held out and used ONLY for post-hoc external validation.\n")

# Standardize features (important for K-means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features_np)

# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────
def softmax_np(logits):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def entropy_bits(p):
    p = np.clip(p, 1e-12, 1.0)
    return -np.sum(p * np.log2(p), axis=1)

# ─── 10‑RUN EVALUATION WITH WEIGHTED METRICS ─────────────────────────────────
results_records = []

for run in range(NUM_RUNS):
    print(f"\n--- Run {run+1}/{NUM_RUNS} ---")
    np.random.seed(run)
    torch.manual_seed(run)

    # K-means clustering
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=run, max_iter=5000, n_init=10)
    kmeans.fit(X_scaled)
    yp = kmeans.labels_

    # Soft assignments using temperature-scaled squared Euclidean distances
    distances = kmeans.transform(X_scaled)                       # (N, K)
    tau = np.std(distances) + 1e-12
    logits = -(distances ** 2) / tau
    soft_assignments = softmax_np(logits)

    # Align cluster labels to majority diagnosis (post-hoc)
    acc = accuracy_score(y_labels, yp)
    acc_inv = accuracy_score(y_labels, 1 - yp)
    if acc_inv > acc:
        yp = 1 - yp
        soft_assignments = soft_assignments[:, ::-1].copy()
        acc = acc_inv

    # Weighted classification metrics (average over both classes)
    prec_w = precision_score(y_labels, yp, average='weighted', zero_division=0)
    rec_w  = recall_score(y_labels, yp, average='weighted', zero_division=0)
    f1_w   = f1_score(y_labels, yp, average='weighted', zero_division=0)
    ll = log_loss(y_labels, soft_assignments[:, 1])

    # Clustering metrics (using standardized features)
    ari = adjusted_rand_score(y_labels, yp)
    nmi = normalized_mutual_info_score(y_labels, yp, average_method='arithmetic')
    ami = adjusted_mutual_info_score(y_labels, yp, average_method='arithmetic')
    if len(np.unique(yp)) >= 2:
        sil = silhouette_score(X_scaled, yp, metric='euclidean')
        db  = davies_bouldin_score(X_scaled, yp)
    else:
        sil, db = np.nan, np.nan

    # Assignment entropy
    H_assign = entropy_bits(soft_assignments)

    # Store all metrics
    results_records.append({
        'run': run,
        'acc': acc,
        'prec_weighted': prec_w,
        'rec_weighted': rec_w,
        'f1_weighted': f1_w,
        'log_loss': ll,
        'ari': ari,
        'nmi': nmi,
        'ami': ami,
        'silhouette': sil,
        'davies_bouldin': db,
        'mean_entropy': H_assign.mean(),
        'std_entropy': H_assign.std(),
    })

    print(f"  Acc: {acc:.4f} | Prec_w: {prec_w:.4f} | Rec_w: {rec_w:.4f} | F1_w: {f1_w:.4f} | "
          f"ARI: {ari:.4f} | MeanEntropy: {H_assign.mean():.4f}")

# ─── HORIZONTAL TABLE FOR MANUSCRIPT (mean ± std over 10 runs) ───────────────
print("\n" + "=" * 72)
print("  HORIZONTAL TABLE FOR MANUSCRIPT (mean ± std over 10 runs)")
print("  Metrics: Accuracy, Weighted Precision, Weighted Recall, Weighted F1, NMI, ARI, AMI, Silhouette, Davies‑Bouldin")
print("=" * 72)

acc_vals   = np.array([r['acc'] for r in results_records])
prec_vals  = np.array([r['prec_weighted'] for r in results_records])
rec_vals   = np.array([r['rec_weighted'] for r in results_records])
f1_vals    = np.array([r['f1_weighted'] for r in results_records])
nmi_vals   = np.array([r['nmi'] for r in results_records])
ari_vals   = np.array([r['ari'] for r in results_records])
ami_vals   = np.array([r['ami'] for r in results_records])
sil_vals   = np.array([r['silhouette'] for r in results_records])
db_vals    = np.array([r['davies_bouldin'] for r in results_records])

metrics_names = [
    "Accuracy", "Prec (weighted)", "Recall (weighted)", "F1 (weighted)",
    "NMI", "ARI", "AMI", "Silhouette", "Davies‑Bouldin"
]
means = [
    acc_vals.mean(), prec_vals.mean(), rec_vals.mean(), f1_vals.mean(),
    nmi_vals.mean(), ari_vals.mean(), ami_vals.mean(),
    np.nanmean(sil_vals), np.nanmean(db_vals)
]
stds = [
    acc_vals.std(), prec_vals.std(), rec_vals.std(), f1_vals.std(),
    nmi_vals.std(), ari_vals.std(), ami_vals.std(),
    np.nanstd(sil_vals), np.nanstd(db_vals)
]

# Tab‑separated line
print("\nMethod\t" + "\t".join(metrics_names))
row = "K-means (RadioDINO)"
for m, s in zip(means, stds):
    if np.isnan(m):
        row += "\tN/A±N/A"
    else:
        row += f"\t{m:.4f}±{s:.4f}"
print(row)

# LaTeX version
print("\n" + "-" * 72)
print("  LaTeX code for the horizontal table (copy the line below):")
print("-" * 72)
latex_row = "K-means (RadioDINO)"
for m, s in zip(means, stds):
    if np.isnan(m):
        latex_row += " & N/A±N/A"
    else:
        latex_row += f" & ${m:.4f}\\pm{s:.4f}$"
latex_row += " \\\\"
print(latex_row)

# ─── DETAILED ANALYSIS FOR SEED 0 (same as ARMA Part G) ───────────────────────
print("\n" + "═" * 80)
print("  DETAILED ANALYSIS FOR SEED 0")
print("═" * 80)

# Re-run K-means with seed 0
kmeans_final = KMeans(n_clusters=NUM_CLUSTERS, random_state=0, max_iter=5000, n_init=10)
kmeans_final.fit(X_scaled)
yp_final = kmeans_final.labels_

# Soft assignments for seed 0
dist_final = kmeans_final.transform(X_scaled)
tau_final = np.std(dist_final) + 1e-12
logits_final = -(dist_final ** 2) / tau_final
soft_final = softmax_np(logits_final)

# Align clusters
acc_final = accuracy_score(y_labels, yp_final)
acc_inv_final = accuracy_score(y_labels, 1 - yp_final)
if acc_inv_final > acc_final:
    yp_final = 1 - yp_final
    soft_final = soft_final[:, ::-1].copy()
    acc_final = acc_inv_final

LABEL_MAP = {0: "Normal", 1: "Pneumonia"}
cluster_assignments = yp_final
cluster0_mask = (cluster_assignments == 0)
cluster1_mask = (cluster_assignments == 1)

print(f"\n  Cluster sizes: Cluster 0 = {cluster0_mask.sum()} subjects, "
      f"Cluster 1 = {cluster1_mask.sum()} subjects")

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm_diag = confusion_matrix(y_labels, cluster_assignments)
print(f"\n  Confusion Matrix (rows=true label, cols=cluster):")
print(f"                     Cluster 0    Cluster 1")
print(f"    Normal (n={(y_labels==0).sum()})      {cm_diag[0,0]:>6}       {cm_diag[0,1]:>6}")
print(f"    Pneumonia (n={(y_labels==1).sum()})      {cm_diag[1,0]:>6}       {cm_diag[1,1]:>6}")

pct_normal_in_c0 = (y_labels[cluster0_mask] == 0).mean() * 100
pct_pneum_in_c1  = (y_labels[cluster1_mask] == 1).mean() * 100
agreement = (cluster_assignments == y_labels).mean() * 100
print(f"\n  Cluster composition:")
print(f"    Cluster 0: {pct_normal_in_c0:.1f}% Normal, {100-pct_normal_in_c0:.1f}% Pneumonia")
print(f"    Cluster 1: {100-pct_pneum_in_c1:.1f}% Normal, {pct_pneum_in_c1:.1f}% Pneumonia")
print(f"\n  Overall label‑cluster agreement: {agreement:.1f}%")

# Feature statistics
feats_c0 = features_np[cluster0_mask]
feats_c1 = features_np[cluster1_mask]
norm_c0 = np.linalg.norm(feats_c0, axis=1)
norm_c1 = np.linalg.norm(feats_c1, axis=1)
mean_c0 = feats_c0.mean(axis=1)
mean_c1 = feats_c1.mean(axis=1)

t_norm, p_norm = stats.ttest_ind(norm_c0, norm_c1, equal_var=False)
t_mean, p_mean = stats.ttest_ind(mean_c0, mean_c1, equal_var=False)

print(f"\n  RadioDINO feature statistics:")
print(f"    L2 norm: C0 = {norm_c0.mean():.3f}±{norm_c0.std():.3f}, C1 = {norm_c1.mean():.3f}±{norm_c1.std():.3f}, p={p_norm:.4f}")
print(f"    Mean activation: C0 = {mean_c0.mean():.4f}±{mean_c0.std():.4f}, C1 = {mean_c1.mean():.4f}±{mean_c1.std():.4f}, p={p_mean:.4f}")

# Cosine similarity within/between clusters
def cosine_similarity_matrix(H):
    n = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-12)
    return n @ n.T

sim = cosine_similarity_matrix(features_np)
N = len(y_labels)
mask = np.triu(np.ones((N, N), dtype=bool), k=1)
same_cluster = (cluster_assignments[:, None] == cluster_assignments[None, :]) & mask
diff_cluster = (cluster_assignments[:, None] != cluster_assignments[None, :]) & mask

sim_within  = sim[same_cluster]
sim_between = sim[diff_cluster]

print(f"\n  Intra‑cluster cosine similarity: mean={sim_within.mean():.4f} ± {sim_within.std():.4f}")
print(f"  Inter‑cluster cosine similarity: mean={sim_between.mean():.4f} ± {sim_between.std():.4f}")
sep_score = sim_within.mean() - sim_between.mean()
print(f"  Separation score (within − between): {sep_score:.4f}")

# Top discriminating dimensions
mean_diff = feats_c1.mean(axis=0) - feats_c0.mean(axis=0)
top10_idx = np.argsort(np.abs(mean_diff))[-10:][::-1]
print(f"\n  Top‑10 discriminating feature dimensions (C1 - C0):")
print(f"    Dim    Diff")
for idx in top10_idx:
    print(f"    {idx:3d}   {mean_diff[idx]:+8.4f}")

# ─── EXPORT RESULTS FOR MANUSCRIPT ───────────────────────────────────────────
print("\n" + "─" * 72)
print("  EXPORT RESULTS FOR MANUSCRIPT")
print("─" * 72)

paper_results = {
    "Metric": [
        "Method",
        "Accuracy (ext. validation)",
        "Weighted Precision",
        "Weighted Recall",
        "Weighted F1",
        "Log Loss",
        "ARI",
        "NMI",
        "AMI",
        "Silhouette",
        "Davies-Bouldin",
        "Mean assignment entropy (bits)",
        "Cluster 0 size",
        "Cluster 1 size",
        "% Normal in Cluster 0",
        "% Pneumonia in Cluster 1",
        "Intra‑cluster cosine sim.",
        "Inter‑cluster cosine sim.",
        "Separation score",
    ],
    "Value": [
        "K-means (RadioDINO features)",
        f"{acc_final:.4f}",
        f"{prec_vals[0]:.4f}",
        f"{rec_vals[0]:.4f}",
        f"{f1_vals[0]:.4f}",
        f"{log_loss(y_labels, soft_final[:,1]):.4f}",
        f"{ari_vals[0]:.4f}",
        f"{nmi_vals[0]:.4f}",
        f"{ami_vals[0]:.4f}",
        f"{sil_vals[0]:.4f}" if not np.isnan(sil_vals[0]) else "N/A",
        f"{db_vals[0]:.4f}" if not np.isnan(db_vals[0]) else "N/A",
        f"{entropy_bits(soft_final).mean():.4f} ± {entropy_bits(soft_final).std():.4f}",
        f"{cluster0_mask.sum()}",
        f"{cluster1_mask.sum()}",
        f"{pct_normal_in_c0:.1f}%",
        f"{pct_pneum_in_c1:.1f}%",
        f"{sim_within.mean():.4f}",
        f"{sim_between.mean():.4f}",
        f"{sep_score:.4f}",
    ]
}

df_paper = pd.DataFrame(paper_results)
print("\n", df_paper.to_string(index=False))
df_paper.to_csv("kmeans_radiodino_pneumoniamnist_summary.csv", index=False)
print("\n  ✓ Saved to: kmeans_radiodino_pneumoniamnist_summary.csv")

# Save per-run results
df_results = pd.DataFrame(results_records)
df_results.to_csv("kmeans_radiodino_pneumoniamnist_per_run.csv", index=False)
print("  ✓ Saved per‑run results to: kmeans_radiodino_pneumoniamnist_per_run.csv")

print("\n" + "═" * 80)
print("  ANALYSIS COMPLETE")
print("═" * 80)

