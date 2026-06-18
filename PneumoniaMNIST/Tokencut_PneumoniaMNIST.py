#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Subset
from torchvision import transforms
import timm
import random
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, log_loss,
    normalized_mutual_info_score, adjusted_rand_score,
    adjusted_mutual_info_score, silhouette_score, davies_bouldin_score
)
from scipy import sparse
from scipy.sparse.linalg import eigsh
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADING – PNEUMONIAMNIST + RADIODINO FEATURES
# ═══════════════════════════════════════════════════════════════════════════
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("CUDA available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

RADIODINO_MODEL = "hf_hub:Snarcy/RadioDino-s16"
SEED_BASE = 42

# Load PneumoniaMNIST data
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

# Subsample: up to 2000 per class (PneumoniaMNIST is larger than BreastMNIST)
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

# Extract RadioDINO features
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

X = torch.cat(rd_feats, dim=0).numpy().astype(np.float32)
y = np.array(y_list, dtype=np.int64)

# Shuffle once (fixed permutation for reproducibility)
np.random.seed(SEED_BASE)
perm = np.random.permutation(X.shape[0])
X, y = X[perm], y[perm]

print(f"Features: {X.shape}, Labels: {y.shape} (Normal: {np.sum(y==0)}, Pneumonia: {np.sum(y==1)})")
print("NOTE: Diagnostic labels used ONLY for post-hoc external validation.\n")

# ═══════════════════════════════════════════════════════════════════════════
#  TOKENCUT CLUSTERING FUNCTION (identical logic)
# ═══════════════════════════════════════════════════════════════════════════
def tokencut_on_features(F_array, alpha=1e-6):
    """
    Apply TokenCut clustering to feature matrix F_array (N × D).
    Returns binary labels (0/1) and the Fiedler vector.
    """
    N, D = F_array.shape

    # 1. Normalize features row-wise
    norms = np.linalg.norm(F_array, axis=1, keepdims=True) + 1e-10
    F_norm = F_array / norms

    # 2. Construct cosine similarity matrix (fully connected)
    W = np.dot(F_norm, F_norm.T)
    W = W + alpha  # stabilizer

    # 3. Normalized Laplacian: L = I - D^{-1/2} W D^{-1/2}
    d = np.sum(W, axis=1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(d + 1e-10))
    L = np.eye(N) - d_inv_sqrt @ W @ d_inv_sqrt

    # Sparse for efficiency
    L_sparse = sparse.csr_matrix(L)

    # 4. Compute the Fiedler vector (second smallest eigenvector)
    vals, vecs = eigsh(L_sparse, k=2, which='SM')
    fiedler = vecs[:, 1]

    # 5. Threshold by mean
    threshold = fiedler.mean()
    labels = (fiedler > threshold).astype(np.int64)

    return labels, fiedler

# ═══════════════════════════════════════════════════════════════════════════
#  EVALUATION LOOP (10 runs with different permutations)
# ═══════════════════════════════════════════════════════════════════════════
num_runs = 10
all_metrics = {
    "accuracy": [], "precision": [], "recall": [], "f1": [],
    "nmi": [], "ari": [], "ami": [], "silhouette": [], "davies_bouldin": [], "log_loss": []
}
# Additional storage for weighted metrics
weighted_prec = []
weighted_rec = []
weighted_f1 = []

print("═" * 72)
print("  TOKENCUT BASELINE – PneumoniaMNIST (RadioDINO features)")
print("  NOTE: Diagnostic labels used ONLY for post-hoc external validation.")
print("═" * 72)

print(f"\n  {'Run':>4}  {'Acc':>7}  {'Prec_w':>8}  {'Rec_w':>8}  {'F1_w':>8}  "
      f"{'NMI':>7}  {'ARI':>7}  {'LogLoss':>9}")
print("  " + "─" * 90)

for run in range(num_runs):
    # Permute data for each run (TokenCut is permutation‑invariant, but we follow protocol)
    np.random.seed(run)
    perm_run = np.random.permutation(X.shape[0])
    X_run = X[perm_run]
    y_run = y[perm_run]

    # Run TokenCut
    labels, fiedler = tokencut_on_features(X_run)

    # Align clusters to majority diagnosis (post‑hoc)
    y_pred = labels
    acc = accuracy_score(y_run, y_pred)
    acc_inv = accuracy_score(y_run, 1 - y_pred)
    if acc_inv > acc:
        y_pred = 1 - y_pred
        acc = acc_inv

    # Weighted metrics (averaged across both classes)
    prec_w = precision_score(y_run, y_pred, average='weighted', zero_division=0)
    rec_w  = recall_score(y_run, y_pred, average='weighted', zero_division=0)
    f1_w   = f1_score(y_run, y_pred, average='weighted', zero_division=0)

    # Binary metrics (positive class only) – kept for backwards compatibility
    prec_bin = precision_score(y_run, y_pred, zero_division=0)
    rec_bin = recall_score(y_run, y_pred, zero_division=0)
    f1_bin = f1_score(y_run, y_pred, zero_division=0)

    # Clustering metrics (external validation)
    nmi = normalized_mutual_info_score(y_run, y_pred, average_method='arithmetic')
    ari = adjusted_rand_score(y_run, y_pred)
    ami = adjusted_mutual_info_score(y_run, y_pred, average_method='arithmetic')

    # Geometric metrics on original feature space (use X_run)
    unique_preds = np.unique(y_pred)
    can_geom = (len(unique_preds) >= 2) and all((y_pred == c).sum() >= 2 for c in unique_preds)
    if can_geom:
        sil = silhouette_score(X_run, y_pred, metric='euclidean')
        db = davies_bouldin_score(X_run, y_pred)
    else:
        sil, db = np.nan, np.nan

    # Probability estimate from normalized Fiedler vector (for log loss)
    probs = (fiedler - fiedler.min()) / (fiedler.max() - fiedler.min() + 1e-10)
    # Ensure probs correspond to cluster 1 after alignment
    if y_pred.mean() > 0.5:  # if cluster 1 is majority after alignment
        probs = 1 - probs
    logloss = log_loss(y_run, probs)

    # Store
    all_metrics["accuracy"].append(acc)
    all_metrics["precision"].append(prec_bin)  # binary precision (original)
    all_metrics["recall"].append(rec_bin)      # binary recall
    all_metrics["f1"].append(f1_bin)           # binary F1
    all_metrics["nmi"].append(nmi)
    all_metrics["ari"].append(ari)
    all_metrics["ami"].append(ami)
    all_metrics["silhouette"].append(sil)
    all_metrics["davies_bouldin"].append(db)
    all_metrics["log_loss"].append(logloss)

    weighted_prec.append(prec_w)
    weighted_rec.append(rec_w)
    weighted_f1.append(f1_w)

    print(f"  {run+1:>4}  {acc:>7.4f}  {prec_w:>8.4f}  {rec_w:>8.4f}  {f1_w:>8.4f}  "
          f"{nmi:>7.4f}  {ari:>7.4f}  {logloss:>9.6f}")

# ═══════════════════════════════════════════════════════════════════════════
#  HORIZONTAL TABLE FOR MANUSCRIPT (mean ± std over 10 runs)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 72)
print("  HORIZONTAL TABLE FOR MANUSCRIPT (mean ± std over 10 runs)")
print("  Metrics: Accuracy, Weighted Precision, Weighted Recall, Weighted F1, NMI, ARI, AMI, Silhouette, Davies‑Bouldin")
print("═" * 72)

acc_vals = np.array(all_metrics["accuracy"])
prec_vals = np.array(weighted_prec)
rec_vals = np.array(weighted_rec)
f1_vals = np.array(weighted_f1)
nmi_vals = np.array(all_metrics["nmi"])
ari_vals = np.array(all_metrics["ari"])
ami_vals = np.array(all_metrics["ami"])
sil_vals = np.array(all_metrics["silhouette"])
db_vals = np.array(all_metrics["davies_bouldin"])

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

print("\nMethod\t" + "\t".join(metrics_names))
row = "TokenCut"
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
latex_row = "TokenCut"
for m, s in zip(means, stds):
    if np.isnan(m):
        latex_row += " & N/A±N/A"
    else:
        latex_row += f" & ${m:.4f}\\pm{s:.4f}$"
latex_row += " \\\\"
print(latex_row)

# ═══════════════════════════════════════════════════════════════════════════
#  SUMMARY TABLES (original, using weighted metrics in the main summary)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 72)
print("  10-RUN SUMMARY (mean ± std)                         [TokenCut, PneumoniaMNIST]")
print("  NOTE: Diagnostic labels used only for post-hoc external validation.")
print("═" * 72)

# External validation metrics (use weighted for precision/recall/f1)
print("\n  A.  EXTERNAL VALIDATION METRICS (weighted)")
print(f"  {'Metric':<20}  {'Mean':>9}  {'Std':>9}  {'Min':>9}  {'Max':>9}")
print("  " + "─" * 60)
for name, vals in [
    ("Accuracy", acc_vals),
    ("Weighted Precision", prec_vals),
    ("Weighted Recall", rec_vals),
    ("Weighted F1", f1_vals),
    ("NMI", nmi_vals),
    ("ARI", ari_vals),
    ("AMI", ami_vals),
    ("Log Loss", np.array(all_metrics["log_loss"]))
]:
    print(f"  {name:<20}  {vals.mean():>9.4f}  {vals.std():>9.4f}  "
          f"{vals.min():>9.4f}  {vals.max():>9.4f}")

# Geometric metrics (silhouette, DB)
print("\n  B.  GEOMETRIC CLUSTERING METRICS (on original feature space)")
print(f"  {'Metric':<20}  {'Mean':>9}  {'Std':>9}  {'Min':>9}  {'Max':>9}  {'Valid runs':>10}")
print("  " + "─" * 70)
for name, vals in [("Silhouette", sil_vals), ("Davies‑Bouldin", db_vals)]:
    valid = vals[~np.isnan(vals)]
    if len(valid) > 0:
        print(f"  {name:<20}  {valid.mean():>9.4f}  {valid.std():>9.4f}  "
              f"{valid.min():>9.4f}  {valid.max():>9.4f}  {len(valid):>10}")
    else:
        print(f"  {name:<20}  {'N/A':>9}  {'N/A':>9}  {'N/A':>9}  {'N/A':>9}  {0:>10}")

# Cluster stability
print("\n  C.  CLUSTER STABILITY ACROSS 10 RUNS")
print(f"  {'Metric':<22}  {'Mean':>9}  {'Std':>9}  Note")
print("  " + "─" * 55)
print(f"  {'Accuracy':<22}  {acc_vals.mean():>9.4f}  {acc_vals.std():>9.4f}  "
      f"Consistency of label‑cluster alignment")
print(f"  {'ARI':<22}  {ari_vals.mean():>9.4f}  {ari_vals.std():>9.4f}  "
      f"Adjusted Rand Index stability")
print("  " + "─" * 55)

# ═══════════════════════════════════════════════════════════════════════════
#  EXPORT RESULTS
# ═══════════════════════════════════════════════════════════════════════════
df_summary = pd.DataFrame(all_metrics)
df_summary.to_csv("tokencut_pneumoniamnist_10run_summary.csv", index=False, float_format="%.6f")
print("\n  ✓ Saved 10‑run summary to tokencut_pneumoniamnist_10run_summary.csv")

# Final summary line
print(f"\n{'='*60}")
print(f"  TokenCut — Final Summary (10 runs, weighted metrics)")
print(f"{'='*60}")
print(f"  ACC : {acc_vals.mean():.4f} ± {acc_vals.std():.4f}")
print(f"  Weighted F1 : {f1_vals.mean():.4f} ± {f1_vals.std():.4f}")
print(f"  NMI : {nmi_vals.mean():.4f} ± {nmi_vals.std():.4f}")
print(f"  ARI : {ari_vals.mean():.4f} ± {ari_vals.std():.4f}")
print(f"  Silhouette: {np.nanmean(sil_vals):.4f} ± {np.nanstd(sil_vals):.4f}")
print(f"  Log Loss  : {np.mean(all_metrics['log_loss']):.4f} ± {np.std(all_metrics['log_loss']):.4f}")
print(f"\n{'='*60}\n  ANALYSIS COMPLETE\n{'='*60}")

