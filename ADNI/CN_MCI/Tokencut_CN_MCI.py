import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, log_loss,
    normalized_mutual_info_score, adjusted_rand_score,
    adjusted_mutual_info_score, silhouette_score, davies_bouldin_score
)
from scipy import sparse
from scipy.sparse.linalg import eigsh
import pandas as pd

fa_cn_path = "/home/snu/Downloads/Histogram_CN_FA_20bin_updated.npy"
fa_mci_path = "/home/snu/Downloads/Histogram_MCI_FA_20bin_updated.npy"

cn_data = np.load(fa_cn_path, allow_pickle=True)
mci_data = np.load(fa_mci_path, allow_pickle=True)

X = np.vstack([cn_data, mci_data]).astype(np.float32)
y = np.hstack([
    np.zeros(len(cn_data), dtype=np.int64),
    np.ones(len(mci_data), dtype=np.int64)
])

np.random.seed(42)
perm = np.random.permutation(X.shape[0])
X, y = X[perm], y[perm]

print(f"Features: {X.shape}, Labels: {y.shape} (CN: {np.sum(y==0)}, MCI: {np.sum(y==1)})")
print("NOTE: Diagnostic labels used ONLY for post-hoc external validation.\n")

def tokencut_on_features(F_array, alpha=1e-6):
    N, D = F_array.shape

    norms = np.linalg.norm(F_array, axis=1, keepdims=True) + 1e-10
    F_norm = F_array / norms

    W = np.dot(F_norm, F_norm.T)
    W = W + alpha

    d = np.sum(W, axis=1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(d + 1e-10))
    L = np.eye(N) - d_inv_sqrt @ W @ d_inv_sqrt

    L_sparse = sparse.csr_matrix(L)

    vals, vecs = eigsh(L_sparse, k=2, which='SM')
    fiedler = vecs[:, 1]

    threshold = fiedler.mean()
    labels = (fiedler > threshold).astype(np.int64)

    return labels, fiedler

num_runs = 10
all_metrics = {
    "accuracy": [],
    "prec_weighted": [],
    "recall_weighted": [],
    "f1_weighted": [],
    "f1_mci": [],
    "nmi": [], "ari": [], "ami": [],
    "silhouette": [], "davies_bouldin": [], "log_loss": []
}

print("═" * 72)
print("  TOKENCUT BASELINE – CN vs MCI (weighted metrics for fair comparison)")
print("  NOTE: Diagnostic labels used ONLY for post-hoc external validation.")
print("═" * 72)

print(f"\n  {'Run':>4}  {'Acc':>7}  {'Prec_w':>8}  {'Rec_w':>8}  {'F1_w':>8}  " \
      f"{'F1_MCI':>8}  {'NMI':>7}  {'ARI':>7}  {'LogLoss':>9}")
print("  " + "─" * 95)

for run in range(num_runs):
    np.random.seed(run)
    perm_run = np.random.permutation(X.shape[0])
    X_run = X[perm_run]
    y_run = y[perm_run]

    labels, fiedler = tokencut_on_features(X_run)

    y_pred = labels
    acc = accuracy_score(y_run, y_pred)
    acc_inv = accuracy_score(y_run, 1 - y_pred)
    if acc_inv > acc:
        y_pred = 1 - y_pred
        acc = acc_inv

    prec_w = precision_score(y_run, y_pred, average='weighted', zero_division=0)
    rec_w  = recall_score(y_run, y_pred, average='weighted', zero_division=0)
    f1_w   = f1_score(y_run, y_pred, average='weighted', zero_division=0)

    f1_mci = f1_score(y_run, y_pred, pos_label=1, zero_division=0)

    nmi = normalized_mutual_info_score(y_run, y_pred, average_method='arithmetic')
    ari = adjusted_rand_score(y_run, y_pred)
    ami = adjusted_mutual_info_score(y_run, y_pred, average_method='arithmetic')

    unique_preds = np.unique(y_pred)
    can_geom = (len(unique_preds) >= 2) and all((y_pred == c).sum() >= 2 for c in unique_preds)
    if can_geom:
        sil = silhouette_score(X_run, y_pred, metric='euclidean')
        db = davies_bouldin_score(X_run, y_pred)
    else:
        sil, db = np.nan, np.nan

    probs = (fiedler - fiedler.min()) / (fiedler.max() - fiedler.min() + 1e-10)
    if y_pred.mean() > 0.5:
        probs = 1 - probs
    logloss = log_loss(y_run, probs)

    all_metrics["accuracy"].append(acc)
    all_metrics["prec_weighted"].append(prec_w)
    all_metrics["recall_weighted"].append(rec_w)
    all_metrics["f1_weighted"].append(f1_w)
    all_metrics["f1_mci"].append(f1_mci)
    all_metrics["nmi"].append(nmi)
    all_metrics["ari"].append(ari)
    all_metrics["ami"].append(ami)
    all_metrics["silhouette"].append(sil)
    all_metrics["davies_bouldin"].append(db)
    all_metrics["log_loss"].append(logloss)

    print(f"  {run+1:>4}  {acc:>7.4f}  {prec_w:>8.4f}  {rec_w:>8.4f}  {f1_w:>8.4f}  " \
          f"{f1_mci:>8.4f}  {nmi:>7.4f}  {ari:>7.4f}  {logloss:>9.6f}")

print("\n" + "═" * 72)
print("  HORIZONTAL TABLE FOR MANUSCRIPT (mean ± std over 10 runs)")
print("  Metrics: Accuracy, Weighted Precision, Weighted Recall, Weighted F1, " \
      "F1(MCI), NMI, ARI, AMI, Silhouette, Davies‑Bouldin")
print("═" * 72)

manuscript_metrics = [
    "accuracy", "prec_weighted", "recall_weighted", "f1_weighted", "f1_mci",
    "nmi", "ari", "ami", "silhouette", "davies_bouldin"
]
manuscript_labels = [
    "Accuracy", "Prec (weighted)", "Recall (weighted)", "F1 (weighted)", "F1 (MCI)",
    "NMI", "ARI", "AMI", "Silhouette", "Davies‑Bouldin"
]

means = []
stds = []
for m in manuscript_metrics:
    vals = np.array(all_metrics[m])
    if m in ["silhouette", "davies_bouldin"]:
        valid = vals[~np.isnan(vals)]
        if len(valid) > 0:
            means.append(valid.mean())
            stds.append(valid.std())
        else:
            means.append(np.nan)
            stds.append(np.nan)
    else:
        means.append(vals.mean())
        stds.append(vals.std())

print("\n" + "Method\t" + "\t".join(manuscript_labels))
row = "TokenCut"
for mean_val, std_val in zip(means, stds):
    if np.isnan(mean_val):
        row += "\tN/A±N/A"
    else:
        row += f"\t{mean_val:.4f}±{std_val:.4f}"
print(row)

print("\n" + "─" * 72)
print("  LaTeX code for the horizontal table (copy the line below):")
print("─" * 72)
latex_row = "TokenCut"
for mean_val, std_val in zip(means, stds):
    if np.isnan(mean_val):
        latex_row += " & N/A±N/A"
    else:
        latex_row += f" & ${mean_val:.4f}\\pm{std_val:.4f}$"
latex_row += " \\"
print(latex_row)

print("\n" + "═" * 72)
print("  DETAILED 10‑RUN SUMMARY (mean ± std)")
print("═" * 72)

ext_metrics = ["accuracy", "prec_weighted", "recall_weighted", "f1_weighted", "f1_mci", "nmi", "ari", "ami", "log_loss"]
print("\n  A.  EXTERNAL VALIDATION METRICS")
print(f"  {'Metric':<20}  {'Mean':>9}  {'Std':>9}  {'Min':>9}  {'Max':>9}")
print("  " + "─" * 60)
for m in ext_metrics:
    vals = np.array(all_metrics[m])
    print(f"  {m.replace('_',' ').capitalize():<20}  {vals.mean():>9.4f}  {vals.std():>9.4f}  " \
          f"{vals.min():>9.4f}  {vals.max():>9.4f}")

geo_metrics = ["silhouette", "davies_bouldin"]
print("\n  B.  GEOMETRIC CLUSTERING METRICS (on original feature space)")
print(f"  {'Metric':<20}  {'Mean':>9}  {'Std':>9}  {'Min':>9}  {'Max':>9}  {'Valid runs':>10}")
print("  " + "─" * 70)
for m in geo_metrics:
    vals = np.array(all_metrics[m])
    valid = vals[~np.isnan(vals)]
    if len(valid) > 0:
        print(f"  {m.capitalize():<20}  {valid.mean():>9.4f}  {valid.std():>9.4f}  " \
              f"{valid.min():>9.4f}  {valid.max():>9.4f}  {len(valid):>10}")
    else:
        print(f"  {m.capitalize():<20}  {'N/A':>9}  {'N/A':>9}  {'N/A':>9}  {'N/A':>9}  {0:>10}")

print("\n  C.  CLUSTER STABILITY ACROSS 10 RUNS")
print(f"  {'Metric':<25}  {'Mean':>9}  {'Std':>9}  Note")
print("  " + "─" * 60)
print(f"  {'Accuracy':<25}  {np.mean(all_metrics['accuracy']):>9.4f}  " \
      f"{np.std(all_metrics['accuracy']):>9.4f}  Consistency of label‑cluster alignment")
print(f"  {'Weighted F1':<25}  {np.mean(all_metrics['f1_weighted']):>9.4f}  " \
      f"{np.std(all_metrics['f1_weighted']):>9.4f}  Weighted F1 stability")
print(f"  {'ARI':<25}  {np.mean(all_metrics['ari']):>9.4f}  " \
      f"{np.std(all_metrics['ari']):>9.4f}  Adjusted Rand Index stability")
print("  " + "─" * 60)

df_summary = pd.DataFrame(all_metrics)
df_summary.to_csv("tokencut_10run_summary.csv", index=False, float_format="%.6f")
print("\n  ✓ Saved 10‑run summary to tokencut_10run_summary.csv")

print(f"\n{'='*60}")
print(f"  TokenCut — Final Summary (10 runs)")
print(f"{'='*60}")
print(f"  ACC (unweighted)        : {np.mean(all_metrics['accuracy']):.4f} ± {np.std(all_metrics['accuracy']):.4f}")
print(f"  Weighted Precision      : {np.mean(all_metrics['prec_weighted']):.4f} ± {np.std(all_metrics['prec_weighted']):.4f}")
print(f"  Weighted Recall         : {np.mean(all_metrics['recall_weighted']):.4f} ± {np.std(all_metrics['recall_weighted']):.4f}")
print(f"  Weighted F1             : {np.mean(all_metrics['f1_weighted']):.4f} ± {np.std(all_metrics['f1_weighted']):.4f}")
print(f"  F1 (MCI)                : {np.mean(all_metrics['f1_mci']):.4f} ± {np.std(all_metrics['f1_mci']):.4f}")
print(f"  NMI                     : {np.mean(all_metrics['nmi']):.4f} ± {np.std(all_metrics['nmi']):.4f}")
print(f"  ARI                     : {np.mean(all_metrics['ari']):.4f} ± {np.std(all_metrics['ari']):.4f}")
print(f"  Silhouette              : {np.nanmean(all_metrics['silhouette']):.4f} ± {np.nanstd(all_metrics['silhouette']):.4f}")
print(f"  Log Loss                : {np.mean(all_metrics['log_loss']):.4f} ± {np.std(all_metrics['log_loss']):.4f}")
print(f"\n{'='*60}\n  ANALYSIS COMPLETE\n{'='*60}")
