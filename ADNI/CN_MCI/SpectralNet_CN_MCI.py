import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, normalized_mutual_info_score,
    adjusted_rand_score, adjusted_mutual_info_score,
    silhouette_score, davies_bouldin_score
)
from munkres import Munkres
import random
import pandas as pd

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_CLUSTERS     = 2
HIDDEN_DIM     = 256
BATCH_SIZE     = 32
N_NEIGHBORS    = 20
SIAMESE_EPOCHS = 50
SPECTRAL_EPOCHS= 50
N_RUNS         = 10
N_MC_PASSES    = 30
FEAT_DROP      = 0.05
FEAT_NOISE_SD  = 0.01
TAU_FACTOR     = 0.05

fa_cn  = np.load("/home/snu/Downloads/Histogram_CN_FA_20bin_updated.npy",  allow_pickle=True)
fa_mci = np.load("/home/snu/Downloads/Histogram_MCI_FA_20bin_updated.npy", allow_pickle=True)

X = np.vstack([fa_cn, fa_mci]).astype(np.float32)
y = np.hstack([np.zeros(len(fa_cn), dtype=np.int64),
               np.ones(len(fa_mci), dtype=np.int64)])

np.random.seed(42)
perm = np.random.permutation(X.shape[0])
X, y = X[perm], y[perm]

N, F_DIM = X.shape
print(f"Nodes: {N}   Features: {F_DIM}   CN: {(y==0).sum()}   MCI: {(y==1).sum()}")
print("NOTE: Labels held out — used ONLY for post-hoc external validation.\n")

class PairDataset(Dataset):
    def __init__(self, pairs, labels):
        self.pairs  = pairs
        self.labels = labels
    def __len__(self):  return len(self.pairs)
    def __getitem__(self, idx):
        return self.pairs[idx][0], self.pairs[idx][1], self.labels[idx]

class SiameseNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
    def forward_once(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        return F.pairwise_distance(out1, out2), out1, out2

class OrthoLinear(nn.Module):
    def __init__(self, in_dim, out_dim, eps=1e-4):
        super().__init__()
        self.fc  = nn.Linear(in_dim, out_dim)
        self.eps = eps
    def forward(self, x):
        Y_tilde = self.fc(x)
        gram    = Y_tilde.T @ Y_tilde + self.eps * torch.eye(Y_tilde.shape[1], device=x.device)
        L       = torch.linalg.cholesky(gram)
        L_inv   = torch.inverse(L)
        return Y_tilde @ L_inv.T

class SpectralNet(nn.Module):
    def __init__(self, input_dim, n_clusters, hidden_dim=256):
        super().__init__()
        self.fc1   = nn.Linear(input_dim, hidden_dim)
        self.fc2   = nn.Linear(hidden_dim, hidden_dim)
        self.ortho = OrthoLinear(hidden_dim, n_clusters)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.ortho(x)

def contrastive_loss(dist, labels, margin=1.0):
    pos = labels * dist.pow(2)
    neg = (1 - labels) * F.relu(margin - dist).pow(2)
    return (pos + neg).mean()

def spectral_loss(Y, W):
    D   = torch.diag(W.sum(dim=1))
    L   = D - W
    num = torch.trace(Y.T @ L @ Y)
    den = torch.trace(Y.T @ D @ Y)
    return num / (den + 1e-12)

def compute_scale(feats, n_neighbors=20):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(feats)
    dists, _ = nbrs.kneighbors(feats)
    return np.median(dists[:, -1])

def compute_affinity(feats, scale_val, n_neighbors=20):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(feats)
    dists, indices = nbrs.kneighbors(feats)
    dists, indices = dists[:, 1:], indices[:, 1:]
    W = np.zeros((len(feats), len(feats)), dtype=np.float32)
    for i in range(len(feats)):
        for j in range(n_neighbors):
            w = np.exp(-dists[i, j] ** 2 / (2 * scale_val ** 2))
            W[i, indices[i, j]] = w
            W[indices[i, j], i] = w
    return W

def hungarian_accuracy(y_pred, y_true, n_clusters):
    cm   = confusion_matrix(y_true, y_pred)
    cost = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):
            cost[i, j] = cm[:, j].sum() - cm[i, j]
    mapping    = Munkres().compute(cost.tolist())
    new_labels = np.zeros_like(y_pred)
    for row, col in mapping:
        new_labels[y_pred == row] = col
    return new_labels, (new_labels == y_true).mean()

def train_siamese(siamese, dataloader, epochs=50, lr=1e-3):
    siamese.to(device)
    opt = optim.Adam(siamese.parameters(), lr=lr)
    for ep in range(epochs):
        siamese.train()
        total = 0.0
        for x1, x2, lbl in dataloader:
            x1, x2, lbl = x1.to(device).float(), x2.to(device).float(), lbl.to(device).float()
            dist, _, _ = siamese(x1, x2)
            loss = contrastive_loss(dist, lbl)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * x1.size(0)
        if (ep + 1) % 10 == 0:
            print(f"  [Siamese] Epoch {ep+1:3d}/{epochs}  loss={total/len(dataloader.dataset):.6f}")
    siamese.to('cpu')
    return siamese

def train_spectral(spectral, X_train, W_np, epochs=50, lr=1e-3, tol=1e-6):
    spectral.to(device)
    opt = optim.Adam(spectral.parameters(), lr=lr)
    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    W_t = torch.tensor(W_np, dtype=torch.float32, device=device)
    prev_loss = float('inf')
    for ep in range(epochs):
        spectral.train()
        Y = spectral(X_t)
        loss = spectral_loss(Y, W_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (ep + 1) % 10 == 0:
            print(f"  [SpectralNet] Epoch {ep+1:3d}/{epochs}  loss={loss.item():.8f}")
        if abs(prev_loss - loss.item()) < tol:
            print(f"  SpectralNet converged at epoch {ep+1}.")
            break
        prev_loss = loss.item()
    spectral.to('cpu')
    return spectral

def calibrate_tau(centers, tau_factor=TAU_FACTOR):
    if centers.shape[0] == 2:
        d_between = np.linalg.norm(centers[0] - centers[1])
    else:
        dists = [np.linalg.norm(centers[i] - centers[j])
                 for i in range(len(centers)) for j in range(i+1, len(centers))]
        d_between = np.mean(dists)
    tau = tau_factor * d_between
    return max(tau, 1e-8)

def _soft_probs_from_centers(Y_np, centers, tau):
    dists = np.linalg.norm(Y_np[:, None, :] - centers[None, :, :], axis=2)
    logits = -(dists ** 2) / tau
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_l = np.exp(logits)
    return exp_l / exp_l.sum(axis=1, keepdims=True)

def inspect_tau_calibration(Y_np, centers, tau):
    d_between = np.linalg.norm(centers[0] - centers[1]) if len(centers) == 2 else float('nan')
    probs = _soft_probs_from_centers(Y_np, centers, tau)
    p_max = probs.max(axis=1)
    sep = "─" * 60
    print(f"\n{sep}\n  TAU CALIBRATION DIAGNOSTIC\n{sep}")
    print(f"  Inter-centroid distance : {d_between:.6f}")
    print(f"  τ (= {TAU_FACTOR} × d)  : {tau:.6f}")
    print(f"\n  Soft-assignment confidence  p(best cluster):")
    print(f"    Mean  : {p_max.mean():.4f}")
    print(f"    Std   : {p_max.std():.4f}")
    print(f"    Min   : {p_max.min():.4f}")
    print(f"    Max   : {p_max.max():.4f}")
    print(f"{sep}\n")
    return p_max

def mc_uncertainty(spectral_net, X_np, kmeans_centers,
                   n_passes=N_MC_PASSES, seed_base=9999,
                   feat_drop=FEAT_DROP, feat_noise_sd=FEAT_NOISE_SD,
                   verbose_calibration=False):
    spectral_net.eval()
    tau = calibrate_tau(kmeans_centers, tau_factor=TAU_FACTOR)
    if verbose_calibration:
        with torch.no_grad():
            Y_det = spectral_net(torch.tensor(X_np, dtype=torch.float32)).numpy()
        inspect_tau_calibration(Y_det, kmeans_centers, tau)
    all_log_probs = []
    with torch.no_grad():
        for i in range(n_passes):
            rng = np.random.default_rng(seed_base + i)
            X_mc = X_np.copy()
            if i % 2 == 0:
                mask = rng.random(X_mc.shape) < feat_drop
                X_mc[mask] = 0.0
            else:
                sigma = feat_noise_sd * float(np.std(X_np))
                X_mc += rng.normal(0.0, sigma, X_mc.shape).astype(np.float32)
            Y_mc = spectral_net(torch.tensor(X_mc, dtype=torch.float32)).numpy()
            probs = _soft_probs_from_centers(Y_mc, kmeans_centers, tau)
            log_probs = np.log(probs + 1e-12)
            all_log_probs.append(log_probs)
    logits_stack = np.stack(all_log_probs, axis=2)
    logits_mean  = logits_stack.mean(axis=2)
    yp = np.argmax(logits_mean, axis=1)

    a  = accuracy_score(y, yp)
    ai = accuracy_score(y, 1 - yp)
    if ai > a:
        yp = 1 - yp
        logits_mean = logits_mean[:, ::-1].copy()
        logits_stack = logits_stack[:, ::-1, :].copy()
    with torch.no_grad():
        Y_det = spectral_net(torch.tensor(X_np, dtype=torch.float32)).numpy()
    ypp = _soft_probs_from_centers(Y_det, kmeans_centers, tau)
    return yp, ypp, logits_stack, logits_mean, tau

def _softmax_np(logits):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def entropy_bits(p):
    p = np.clip(p, 1e-12, 1.0)
    return -np.sum(p * np.log2(p), axis=1)

def compute_cluster_assignment_uncertainty(logits_stack):
    N, K_dim, P = logits_stack.shape
    probs = np.stack([_softmax_np(logits_stack[:, :, p]) for p in range(P)], axis=2)
    p_mean = probs.mean(axis=2)
    H_assign = entropy_bits(p_mean)
    H_aleat = np.stack([entropy_bits(probs[:, :, p]) for p in range(P)], axis=1).mean(axis=1)
    MI = np.clip(H_assign - H_aleat, 0, None)
    return H_assign, H_aleat, MI, p_mean

def print_cluster_uncertainty_report(logits_stack, yp, y_true, n_passes, tau, sample_ids=None):
    sep = "─" * 72
    H_assign, H_aleat, MI, p_mean = compute_cluster_assignment_uncertainty(logits_stack)
    label_map = {0: "CN", 1: "MCI"}
    if sample_ids is None:
        sample_ids = list(range(len(y_true)))
    df = pd.DataFrame({
        "sample_id": sample_ids,
        "true_label": [label_map[int(l)] for l in y_true],
        "cluster_assignment": [label_map[int(l)] for l in yp],
        "correct_ext_valid": (yp == y_true).astype(int),
        "p_cluster0": p_mean[:, 0],
        "p_cluster1": p_mean[:, 1],
        "entropy_assignment": H_assign,
        "entropy_aleatoric": H_aleat,
        "model_uncertainty": MI,
    })
    print(f"\n{sep}\n  CLUSTER ASSIGNMENT UNCERTAINTY (SpectralNet)\n{sep}")
    print(f"  ({n_passes} MC perturbation passes | τ = {tau:.6f})\n")
    print(f"  {'Metric':<35} {'Mean':>9} {'Std':>9} {'Min':>9} {'Max':>9}")
    print(f"  {'─'*70}")
    for col, label in [
        ("entropy_assignment", "Assignment entropy H[p̄] (bits)"),
        ("entropy_aleatoric",  "Aleatoric entropy E[H] (bits)"),
        ("model_uncertainty",  "Perturbation disagreement MI (bits)"),
        ("p_cluster1",        "Soft assignment p(cluster=1)"),
    ]:
        vals = df[col].values
        print(f"  {label:<35}  {vals.mean():>9.4f}  {vals.std():>9.4f}  " \
              f"{vals.min():>9.4f}  {vals.max():>9.4f}")
    df_sorted = df.sort_values("entropy_assignment", ascending=False).reset_index(drop=True)
    print(f"\n  Top-10 most uncertain cluster assignments:")
    cols_show = ["sample_id", "true_label", "cluster_assignment",
                 "p_cluster0", "p_cluster1", "entropy_assignment", "model_uncertainty"]
    print(df_sorted.head(10)[cols_show].to_string(index=True))
    print(f"\n  Top-10 most confident cluster assignments:")
    df_conf = df.sort_values("entropy_assignment", ascending=True).reset_index(drop=True)
    print(df_conf.head(10)[cols_show].to_string(index=True))
    low_unc = (H_assign < 0.3).sum()
    high_unc = (H_assign > 0.7).sum()
    print(f"\n  Assignment confidence summary:")
    print(f"    High-confidence (H < 0.3 bits) : {low_unc:4d} / {len(y_true)} ({100*low_unc/len(y_true):.1f}%)")
    print(f"    Ambiguous       (H > 0.7 bits) : {high_unc:4d} / {len(y_true)} ({100*high_unc/len(y_true):.1f}%)")
    print(f"\n  Mean assignment entropy  : {H_assign.mean():.4f} ± {H_assign.std():.4f} bits")
    print(f"  Max possible (K=2)       : {np.log2(2):.4f} bits")
    print(f"  Temperature τ used       : {tau:.6f}  (= {TAU_FACTOR} × inter-centroid dist)")
    print(f"  NOTE: 'Perturbation disagreement MI' is NOT Bayesian epistemic")
    print(f"        uncertainty. It reflects sensitivity to input perturbations.")
    print(f"{sep}")
    return df

def compute_clustering_metrics(embeddings, pred_labels, true_labels, space_name=""):
    unique_preds = np.unique(pred_labels)
    n_valid = sum((pred_labels == c).sum() >= 2 for c in unique_preds)
    can_geom = (len(unique_preds) >= 2) and (n_valid == len(unique_preds))
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels, average_method='arithmetic')
    ami = adjusted_mutual_info_score(true_labels, pred_labels, average_method='arithmetic')
    sil = silhouette_score(embeddings, pred_labels) if can_geom else float('nan')
    db = davies_bouldin_score(embeddings, pred_labels) if can_geom else float('nan')
    return dict(space=space_name, ARI=ari, NMI=nmi, AMI=ami,
                Silhouette=sil, DaviesBouldin=db)

def evaluate_clustering(Y_np, yp, logits_mean, y_true, prefix=""):
    results = [
        compute_clustering_metrics(logits_mean, yp, y_true, space_name="logit (MC avg)"),
        compute_clustering_metrics(Y_np, yp, y_true, space_name="spectral embed"),
    ]
    sep = "─" * 72
    header = (f"  {'Space':<22} {'ARI':>8} {'NMI':>8} {'AMI':>8} " \
              f"{'Silhouette':>12} {'DaviesBouldin':>14}")
    print(f"\n{sep}\n  CLUSTERING METRICS{' '+prefix if prefix else ''}\n{sep}")
    print(header)
    print(f"  {sep}")
    for r in results:
        sil_s = f"{r['Silhouette']:>12.4f}" if not np.isnan(r['Silhouette']) else "         N/A"
        db_s = f"{r['DaviesBouldin']:>14.4f}" if not np.isnan(r['DaviesBouldin']) else "           N/A"
        print(f"  {r['space']:<22} {r['ARI']:>8.4f} {r['NMI']:>8.4f} {r['AMI']:>8.4f}{sil_s}{db_s}")
    print(f"  {sep}")
    return results

def print_clustering_summary(all_records, label="SpectralNet"):
    sep = "─" * 90
    print(f"\n{sep}")
    print(f"  CLUSTERING METRICS SUMMARY  [{label}]  (mean ± std, {len(all_records)} runs)")
    print(f"{sep}")
    spaces = [("logit (MC avg)", "Logit space (MC-averaged, K-dim)"),
              ("spectral embed", "Spectral embedding space")]
    for space_key, space_label in spaces:
        print(f"\n  {space_label}")
        print(f"  {'Metric':<16} {'Mean':>9} {'Std':>9} {'Min':>9} {'Max':>9}")
        print(f"  {'─'*55}")
        for metric, hl in [("ARI","↑"),("NMI","↑"),("AMI","↑"),
                           ("Silhouette","↑"),("DaviesBouldin","↓")]:
            key = f"{metric}_{space_key}"
            vals = np.array([r.get(key, np.nan) for r in all_records])
            valid = vals[~np.isnan(vals)]
            if len(valid) == 0:
                print(f"  {metric+' '+hl:<16}  {'N/A':>9}")
                continue
            print(f"  {metric+' '+hl:<16}  {valid.mean():>9.4f}  {valid.std():>9.4f}" \
                  f"  {valid.min():>9.4f}  {valid.max():>9.4f}")
    print(f"\n{sep}")

SEP    = "═" * 72
sep100 = "─" * 100

print(f"\n{SEP}")
print(f"  SpectralNet (Siamese + SpectralNet + KMeans)")
print(f"  {N_RUNS} runs × {N_MC_PASSES} MC perturbation passes")
print(f"  τ = {TAU_FACTOR} × inter-centroid distance  (recalibrated per run)")
print(f"  Labels used ONLY for post-hoc external validation.")
print(f"{SEP}\n")

col_w = "─" * 110
print(f"  {'Run':>3}  {'Acc':>7}  {'Prec_w':>8}  {'Rec_w':>8}  {'F1_w':>8}  " \
      f"{'NMI':>7}  {'MeanEntropy':>13}  {'StdEntropy':>11}  " \
      f"{'TN':>5}  {'FP':>5}  {'FN':>5}  {'TP':>5}")
print(f"  {col_w}")

all_acc = []
all_prec_weighted = []
all_recall_weighted = []
all_f1_weighted = []
all_nmi = []
all_ent = []
all_std_ent = []
all_tau = []
calibration_rows = []
seed_clustering_records = []

last_logits_stack = None
last_yp = None
last_tau = None

for run in range(N_RUNS):
    setup_seed(42 + run)
    print(f"\n  ── Run {run+1}/{N_RUNS} ──────────────────────────────────────")

    pairs, pair_labels = [], []
    nbrs = NearestNeighbors(n_neighbors=N_NEIGHBORS + 1).fit(X)
    _, indices = nbrs.kneighbors(X)
    for i in range(N):
        for j in indices[i, 1:]:
            pairs.append([X[i], X[j]]); pair_labels.append(1)
        non_nb = list(set(range(N)) - set(indices[i, 1:]) - {i})
        j = np.random.choice(non_nb)
        pairs.append([X[i], X[j]]); pair_labels.append(0)
    dataset = PairDataset(pairs, pair_labels)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    siamese = SiameseNet(F_DIM, HIDDEN_DIM)
    siamese = train_siamese(siamese, dataloader, epochs=SIAMESE_EPOCHS)
    siamese.eval()
    with torch.no_grad():
        X_embed = siamese.forward_once(torch.tensor(X, dtype=torch.float32)).numpy()

    scale_val = compute_scale(X_embed, N_NEIGHBORS)
    W_np = compute_affinity(X_embed, scale_val, N_NEIGHBORS)
    print(f"  Affinity edges: {np.count_nonzero(W_np) // 2}")

    spectral = SpectralNet(F_DIM, N_CLUSTERS, HIDDEN_DIM)
    spectral = train_spectral(spectral, X, W_np, epochs=SPECTRAL_EPOCHS)

    spectral.eval()
    with torch.no_grad():
        Y_det = spectral(torch.tensor(X, dtype=torch.float32)).numpy()
    kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=20, random_state=run)
    kmeans.fit(Y_det)
    centers = kmeans.cluster_centers_

    yp_i, ypp_i, logits_stack_i, logits_mean_i, tau_i = mc_uncertainty(
        spectral, X, centers,
        n_passes=N_MC_PASSES, seed_base=9999 + run,
        verbose_calibration=(run == 0)
    )

    acc_i = accuracy_score(y, yp_i)
    prec_w = precision_score(y, yp_i, average='weighted', zero_division=0)
    rec_w  = recall_score(y, yp_i, average='weighted', zero_division=0)
    f1_w   = f1_score(y, yp_i, average='weighted', zero_division=0)
    nmi_i  = normalized_mutual_info_score(y, yp_i, average_method='arithmetic')

    H_assign_i, _, _, _ = compute_cluster_assignment_uncertainty(logits_stack_i)
    mean_ent_i = float(H_assign_i.mean())
    std_ent_i  = float(H_assign_i.std())

    tn_i, fp_i, fn_i, tp_i = confusion_matrix(y, yp_i).ravel()

    cl_res_i = evaluate_clustering(Y_det, yp_i, logits_mean_i, y,
                                   prefix=f"Run {run+1}")
    cl_flat = {}
    for r in cl_res_i:
        sp = r['space']
        for m in ['ARI','NMI','AMI','Silhouette','DaviesBouldin']:
            cl_flat[f"{m}_{sp}"] = r[m]
    seed_clustering_records.append(cl_flat)

    all_acc.append(acc_i)
    all_prec_weighted.append(prec_w)
    all_recall_weighted.append(rec_w)
    all_f1_weighted.append(f1_w)
    all_nmi.append(nmi_i)
    all_ent.append(mean_ent_i)
    all_std_ent.append(std_ent_i)
    all_tau.append(tau_i)

    calibration_rows.append(dict(
        run=run, acc=acc_i, prec_weighted=prec_w, rec_weighted=rec_w, f1_weighted=f1_w, nmi=nmi_i,
        mean_entropy=mean_ent_i, std_entropy=std_ent_i, tau=tau_i,
        tn=int(tn_i), fp=int(fp_i), fn=int(fn_i), tp=int(tp_i),
    ))

    print(f"  {run+1:>3}  {acc_i:>7.4f}  {prec_w:>8.4f}  {rec_w:>8.4f}  {f1_w:>8.4f}  " \
          f"{nmi_i:>7.4f}  {mean_ent_i:>13.4f}  {std_ent_i:>11.4f}  " \
          f"{tn_i:>5}  {fp_i:>5}  {fn_i:>5}  {tp_i:>5}")

    if run == 0:
        last_logits_stack = logits_stack_i
        last_yp = yp_i
        last_tau = tau_i

print(f"\n{SEP}\n  DETAILED UNCERTAINTY REPORT — Run 1\n{SEP}")
df_unc = print_cluster_uncertainty_report(
    last_logits_stack, last_yp, y,
    n_passes=N_MC_PASSES, tau=last_tau,
    sample_ids=list(range(N))
)
df_unc.to_csv("spectralnet_cluster_uncertainty_run1.csv", index=False, float_format="%.6f")
print(f"\n  CSV → spectralnet_cluster_uncertainty_run1.csv  ({len(df_unc)} subjects)")

print(f"\n{SEP}")
print("  HORIZONTAL TABLE FOR MANUSCRIPT (mean ± std over 10 runs)")
print("  Metrics: Accuracy, Weighted Precision, Weighted Recall, Weighted F1, NMI, ARI, AMI, Silhouette, Davies‑Bouldin")
print(SEP)

metrics_names = ["Accuracy", "Prec (weighted)", "Recall (weighted)", "F1 (weighted)", "NMI", "ARI", "AMI", "Silhouette", "Davies‑Bouldin"]
acc_vals = np.array(all_acc)
prec_vals = np.array(all_prec_weighted)
rec_vals = np.array(all_recall_weighted)
f1_vals = np.array(all_f1_weighted)
nmi_vals = np.array(all_nmi)

ari_list = []
ami_list = []
sil_list = []
db_list = []
for rec in seed_clustering_records:
    ari_list.append(rec.get("ARI_logit (MC avg)", np.nan))
    ami_list.append(rec.get("AMI_logit (MC avg)", np.nan))
    sil_list.append(rec.get("Silhouette_logit (MC avg)", np.nan))
    db_list.append(rec.get("DaviesBouldin_logit (MC avg)", np.nan))

ari_vals = np.array(ari_list)
ami_vals = np.array(ami_list)
sil_vals = np.array(sil_list)
db_vals = np.array(db_list)

means = [acc_vals.mean(), prec_vals.mean(), rec_vals.mean(), f1_vals.mean(),
         nmi_vals.mean(), ari_vals.mean(), ami_vals.mean(),
         np.nanmean(sil_vals), np.nanmean(db_vals)]
stds = [acc_vals.std(), prec_vals.std(), rec_vals.std(), f1_vals.std(),
        nmi_vals.std(), ari_vals.std(), ami_vals.std(),
        np.nanstd(sil_vals), np.nanstd(db_vals)]

print("\nMethod\t" + "\t".join(metrics_names))
row = "SpectralNet"
for m, s in zip(means, stds):
    if np.isnan(m):
        row += "\tN/A±N/A"
    else:
        row += f"\t{m:.4f}±{s:.4f}"
print(row)

print("\n" + "─" * 72)
print("  LaTeX code for the horizontal table (copy the line below):")
print("─" * 72)
latex_row = "SpectralNet"
for m, s in zip(means, stds):
    if np.isnan(m):
        latex_row += " & N/A±N/A"
    else:
        latex_row += f" & ${m:.4f}\\pm{s:.4f}$"
latex_row += " \\"
print(latex_row)

print(f"\n{sep100}")
print(f"  10-RUN SUMMARY  (mean ± std)                [SpectralNet, CN vs MCI]")
print(f"  NOTE: Labels used only for post-hoc external validation.")
print(f"{sep100}")

ext_labels = [
    ("Accuracy", "Weighted accuracy (unweighted also same here)"),
    ("Weighted Precision", "Post-hoc external validation"),
    ("Weighted Recall", "Post-hoc external validation"),
    ("Weighted F1", "Post-hoc external validation"),
    ("NMI", "Post-hoc external validation"),
    ("Mean Entropy", "Mean cluster assignment entropy (bits)"),
    ("Std Entropy", "Std of assignment entropy across subjects"),
]
print(f"\n  A.  EXTERNAL VALIDATION & ASSIGNMENT UNCERTAINTY")
print(f"  {'Metric':<20}  {'Mean':>9}  {'Std':>9}  {'Min':>9}  {'Max':>9}  Note")
print(f"  {sep100}")
ext_means = [means[0], means[1], means[2], means[3], means[4]]
ext_stds = [stds[0], stds[1], stds[2], stds[3], stds[4]]
ext_min = [np.min(acc_vals), np.min(prec_vals), np.min(rec_vals), np.min(f1_vals), np.min(nmi_vals)]
ext_max = [np.max(acc_vals), np.max(prec_vals), np.max(rec_vals), np.max(f1_vals), np.max(nmi_vals)]
for idx, (label, note) in enumerate(ext_labels[:5]):
    print(f"  {label:<20}  {ext_means[idx]:>9.4f}  {ext_stds[idx]:>9.4f}  " \
          f"{ext_min[idx]:>9.4f}  {ext_max[idx]:>9.4f}  {note}")

print(f"\n  τ across runs: mean={np.mean(all_tau):.6f}  " \
      f"std={np.std(all_tau):.6f}  min={np.min(all_tau):.6f}  max={np.max(all_tau):.6f}")

counts_np = np.array([[r['tn'], r['fp'], r['fn'], r['tp']] for r in calibration_rows], dtype=float)
print(f"\n  B.  CONFUSION MATRIX COUNTS (post-hoc external validation only)")
print(f"  {'Group':<15}  {'Mean':>7}  {'Std':>7}  {'Min':>5}  {'Max':>5}  Note")
print(f"  {sep100}")
count_labels = [
    ("TN  (CN→C0)",  "CN assigned to Cluster 0"),
    ("FP  (CN→C1)",  "CN assigned to Cluster 1"),
    ("FN  (MCI→C0)", "MCI assigned to Cluster 0"),
    ("TP  (MCI→C1)", "MCI assigned to Cluster 1"),
]
for idx, (label, note) in enumerate(count_labels):
    v = counts_np[:, idx]
    print(f"  {label:<15}  {v.mean():>7.1f}  {v.std():>7.2f}  " \
          f"{v.min():>5.0f}  {v.max():>5.0f}  {note}")

print(f"\n  C.  CLUSTER STABILITY ACROSS {N_RUNS} RUNS")
print(f"  {'Metric':<22}  {'Mean':>9}  {'Std':>9}  Note")
print(f"  {sep100}")
print(f"  {'Weighted F1':<22}  {np.mean(all_f1_weighted):>9.4f}  {np.std(all_f1_weighted):>9.4f}" \
      f"  Consistency of weighted F1")
print(f"  {'Assignment Entropy':<22}  {np.mean(all_ent):>9.4f}  {np.std(all_ent):>9.4f}" \
      f"  Low std = stable confidence")
print(f"  {sep100}\n")

print(f"\n{SEP}\n  CLUSTERING METRICS SUMMARY ({N_RUNS} runs)\n{SEP}")
print_clustering_summary(seed_clustering_records, label="SpectralNet — 10 runs")

print("\n===== Weighted F1 Scores Across Runs =====")
for idx, s in enumerate(all_f1_weighted):
    print(f"  {idx+1}. {s:.4f}")

print(f"\n{'='*60}")
print(f"  SpectralNet — Final Summary ({N_RUNS} runs, weighted metrics)")
print(f"{'='*60}")
print(f"  ACC (unweighted) : {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
print(f"  Weighted PREC    : {np.mean(all_prec_weighted):.4f} ± {np.std(all_prec_weighted):.4f}")
print(f"  Weighted REC     : {np.mean(all_recall_weighted):.4f} ± {np.std(all_recall_weighted):.4f}")
print(f"  Weighted F1      : {np.mean(all_f1_weighted):.4f} ± {np.std(all_f1_weighted):.4f}")
print(f"  NMI              : {np.mean(all_nmi):.4f} ± {np.std(all_nmi):.4f}")
print(f"  Mean Assignment Entropy : {np.mean(all_ent):.4f} ± {np.std(all_ent):.4f} bits")
print(f"  (Max possible for K=2   : {np.log2(N_CLUSTERS):.4f} bits)")
print(f"\n  Uncertainty method: perturbation-based MC ({N_MC_PASSES} passes)")
print(f"    Feature masking  (p={FEAT_DROP}) alternated with Gaussian noise (σ={FEAT_NOISE_SD}×std(X))")
print(f"    τ calibration    : {TAU_FACTOR} × inter-centroid distance per run")
print(f"    Soft logits      : -(d²)/τ → softmax → entropy decomposition")
print(f"    Mean τ           : {np.mean(all_tau):.6f} ± {np.std(all_tau):.6f}")

df_summary = pd.DataFrame(calibration_rows)
df_summary.to_csv("spectralnet_10run_summary.csv", index=False, float_format="%.6f")
print(f"\n  ✓ Saved → spectralnet_10run_summary.csv")
print(f"\n{'='*60}\n  ANALYSIS COMPLETE\n{'='*60}")
