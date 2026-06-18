import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected
from torch_sparse import SparseTensor
from sklearn.metrics import (
    accuracy_score, f1_score, normalized_mutual_info_score,
    adjusted_rand_score, precision_score, recall_score,
    silhouette_score, davies_bouldin_score,
    adjusted_mutual_info_score, confusion_matrix
)
from sklearn.cluster import KMeans
import pandas as pd

# =============================================================================
# CONFIGURATION (faithful MAGI, same as BreastMNIST version)
# =============================================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("CUDA available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

N_EVAL_PASSES = 30          # MC perturbation passes
K = 2                       # number of clusters
FEAT_NOISE = 0.20           # fraction of features zeroed per MC pass
EDGE_DROP_MIN = 0.15
EDGE_DROP_STEP = 0.10
ALPHA = 0.83                # graph adjacency threshold (cosine similarity)
EPOCHS = 150                # training epochs
N_RUNS = 10                 # number of runs

# MAGI hyperparameters
BATCH_ROOT_NODES = 128      # number of root nodes sampled per batch
WALKS_PER_ROOT = 10         # number of random walks per root
WALK_DEPTH = 4              # walk length
TAU = 0.3                   # temperature for SimCLR loss

# =============================================================================
# REPRODUCIBILITY
# =============================================================================
def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(42)

# =============================================================================
# DATA: CN vs AD (FA histograms, 20 bins)
# =============================================================================
fa_cn  = np.load("/home/snu/Downloads/Histogram_CN_FA_20bin_updated.npy", allow_pickle=True)
fa_ad = np.load("/home/snu/Downloads/Histogram_AD_FA_20bin_updated.npy", allow_pickle=True)

X = np.vstack([fa_cn, fa_ad]).astype(np.float32)
y = np.hstack([np.zeros(len(fa_cn), dtype=np.int64),
               np.ones(len(fa_ad), dtype=np.int64)])

# Shuffle once
np.random.seed(42)
perm = np.random.permutation(X.shape[0])
X, y = X[perm], y[perm]

N, F_dim = X.shape
print(f"Nodes: {N}  Features: {F_dim}  (CN: {(y==0).sum()}  AD: {(y==1).sum()})")
print("NOTE: Diagnostic labels are used ONLY for post‑hoc external validation.\n")

x_torch_full = torch.from_numpy(X).float().to(DEVICE)
y_torch_full = torch.from_numpy(y).long().to(DEVICE)

# =============================================================================
# GRAPH CONSTRUCTION (cosine similarity threshold)
# =============================================================================
def create_adj(features, alpha=0.83):
    f = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)
    W = np.dot(f, f.T)
    return (W >= alpha).astype(np.float32)

W = create_adj(X, alpha=ALPHA)
rows, cols = np.nonzero(W)
edge_index_np = np.vstack([rows, cols]).astype(np.int64)

edge_index_full = torch.tensor(edge_index_np, dtype=torch.long)
edge_index_full = to_undirected(edge_index_full).to(DEVICE)

adj_full = SparseTensor(
    row=edge_index_full[0],
    col=edge_index_full[1],
    sparse_sizes=(N, N)
).fill_value(1.).to(DEVICE)

print(f"Graph built: {edge_index_full.shape[1]} edges (undirected)")

# =============================================================================
# OPTIMIZED MAGI STAGE 1 (batched random walks)
# =============================================================================
def stage1_sample_communities_batched(root_nodes, t, l, adj_sparse, N_total):
    """
    Batched Stage‑1: fully vectorized counting per root using bincount with offsets.
    Returns batch B as a tensor of unique node ids.
    """
    R = len(root_nodes)
    start = root_nodes.repeat_interleave(t)
    walks = adj_sparse.random_walk(start, l)               # (R*t, l+1)
    visited = walks[:, 1:].reshape(-1)                     # (R*t*l)
    root_idx = torch.arange(R, device=DEVICE).repeat_interleave(t * l)
    indices = root_idx * N_total + visited
    counts = torch.bincount(indices, minlength=R * N_total).view(R, N_total)
    visited_mask = counts > 0
    mean_counts = torch.zeros(R, device=DEVICE)
    for i in range(R):
        if visited_mask[i].any():
            mean_counts[i] = counts[i, visited_mask[i]].float().mean()
    selected_mask = (counts > mean_counts.unsqueeze(1)) & visited_mask
    batch_list = [torch.nonzero(selected_mask[i]).flatten() for i in range(R)]
    if batch_list:
        batch = torch.cat(batch_list).unique()
    else:
        batch = torch.tensor([], device=DEVICE, dtype=torch.long)
    return batch

# =============================================================================
# OPTIMIZED MAGI STAGE 2 (modularity matrix with batched walks)
# =============================================================================
def stage2_build_modularity_batched(batch_nodes, t, l, adj_sparse):
    b = len(batch_nodes)
    if b == 0:
        return torch.zeros((0, 0), device=DEVICE)
    start = batch_nodes.repeat_interleave(t)
    walks = adj_sparse.random_walk(start, l)               # (b*t, l+1)
    visited = walks[:, 1:].reshape(b, t, l)                # (b, t, l)
    node_to_idx = {int(n.item()): i for i, n in enumerate(batch_nodes)}
    S = torch.zeros((b, b), device=DEVICE)
    for i in range(b):
        targets = visited[i].reshape(-1)                   # (t*l)
        uniq, cnt = torch.unique(targets, return_counts=True)
        in_batch = torch.isin(uniq, batch_nodes)
        uniq_in = uniq[in_batch]
        cnt_in = cnt[in_batch]
        if len(uniq_in) == 0:
            continue
        j_indices = torch.tensor([node_to_idx[int(u.item())] for u in uniq_in], device=DEVICE)
        S[i, j_indices] = cnt_in.float()
    row_sum = S.sum(dim=1, keepdim=True)
    S = S / (row_sum + 1e-12)
    B_mat = S - 1.0 / b
    return B_mat

# =============================================================================
# GNN ENCODER (GCN)
# =============================================================================
class Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=256):
        super().__init__()
        self.conv = GCNConv(in_dim, hidden_dim)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = F.leaky_relu(x, 0.5)
        return x

# =============================================================================
# MAGI CONTRASTIVE LOSS (SimCLR with modularity signs)
# =============================================================================
def magi_contrastive_loss(z_batch, B_mat, tau=TAU):
    b = z_batch.shape[0]
    if b <= 1:
        return torch.tensor(0.0, device=z_batch.device)
    z_norm = F.normalize(z_batch, dim=1)
    sim = z_norm @ z_norm.T
    sim = sim / tau
    exp_sim = torch.exp(sim)
    pos_mask = (B_mat > 0).float()
    neg_mask = (B_mat <= 0).float()
    pos_sum = (exp_sim * pos_mask).sum(dim=1)
    all_sum = (exp_sim * (pos_mask + neg_mask)).sum(dim=1)
    loss = -torch.log(pos_sum / (all_sum + 1e-12) + 1e-12)
    valid = (pos_sum > 0).float()
    if valid.sum() == 0:
        return torch.tensor(0.0, device=z_batch.device)
    loss = (loss * valid).sum() / valid.sum()
    return loss

def scale(z):
    zmin = z.min(dim=1, keepdim=True)[0]
    zmax = z.max(dim=1, keepdim=True)[0]
    return (z - zmin) / (zmax - zmin + 1e-12)

# =============================================================================
# MC EVALUATION (edge/feature dropout, temperature‑scaled soft logits)
# =============================================================================
def aug_edge(ei_np, drop=0.2, seed=None):
    rng = np.random.default_rng(seed)
    mask = rng.random(ei_np.shape[1]) >= drop
    return ei_np[:, mask]

def aug_feats(feats_np, drop=0.2, seed=None):
    rng = np.random.default_rng(seed)
    noise = (rng.random(feats_np.shape) >= drop).astype(np.float32)
    return feats_np * noise

def np_to_edge_index(ei_np):
    return torch.from_numpy(ei_np.astype(np.int64)).long().to(DEVICE)

def _soft_logits_from_kmeans(z_np, centers):
    dists = np.linalg.norm(z_np[:, None, :] - centers[None, :, :], axis=2)
    tau = np.std(dists) + 1e-12
    logits = -(dists ** 2) / tau
    return logits

def _softmax_np(logits):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def entropy_bits(p):
    p = np.clip(p, 1e-12, 1.0)
    return -np.sum(p * np.log2(p), axis=1)

def compute_cluster_assignment_uncertainty(logits_stack):
    N_nodes, K_dim, P = logits_stack.shape
    probs = np.stack([_softmax_np(logits_stack[:, :, p]) for p in range(P)], axis=2)
    p_mean = probs.mean(axis=2)
    H_assign = entropy_bits(p_mean)
    H_aleat = np.stack([entropy_bits(probs[:, :, p]) for p in range(P)], axis=1).mean(axis=1)
    MI = np.clip(H_assign - H_aleat, 0, None)
    return H_assign, H_aleat, MI, p_mean

def evaluate_magi_mc(encoder, feats_np, edge_index_np_base, y_true, kmeans_centers,
                     n_passes=N_EVAL_PASSES, seed_base=9999):
    encoder.eval()
    all_logits = []
    with torch.no_grad():
        for i in range(n_passes):
            drop_rate = EDGE_DROP_MIN + EDGE_DROP_STEP * (i % 3)
            fa = aug_feats(feats_np, drop=FEAT_NOISE, seed=seed_base + i)
            ei = aug_edge(edge_index_np_base, drop=drop_rate, seed=seed_base + i)
            x_t = torch.from_numpy(fa).float().to(DEVICE)
            ei_t = np_to_edge_index(ei)
            z = encoder(x_t, ei_t)
            z = scale(z)
            z = F.normalize(z, dim=1).cpu().numpy()
            logits = _soft_logits_from_kmeans(z, kmeans_centers)
            all_logits.append(logits)
    logits_stack = np.stack(all_logits, axis=2)
    logits_mean = logits_stack.mean(axis=2)
    yp = np.argmax(logits_mean, axis=1)
    # Align clusters to diagnostic labels (majority voting)
    a = accuracy_score(y_true, yp)
    ai = accuracy_score(y_true, 1 - yp)
    if ai > a:
        yp = 1 - yp
        logits_mean = logits_mean[:, ::-1].copy()
        logits_stack = logits_stack[:, ::-1, :].copy()
    ypp = _softmax_np(logits_mean)
    return yp, ypp, logits_stack, logits_mean

# =============================================================================
# CLUSTERING METRICS (same as ARMA)
# =============================================================================
def compute_clustering_metrics(embeddings, pred_labels, true_labels, space_name=""):
    unique_preds = np.unique(pred_labels)
    n_valid = sum((pred_labels == c).sum() >= 2 for c in unique_preds)
    can_geom = (len(unique_preds) >= 2) and (n_valid == len(unique_preds))
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels, average_method='arithmetic')
    ami = adjusted_mutual_info_score(true_labels, pred_labels, average_method='arithmetic')
    if can_geom:
        sil = silhouette_score(embeddings, pred_labels, metric='euclidean')
        db = davies_bouldin_score(embeddings, pred_labels)
    else:
        sil = float('nan')
        db = float('nan')
    return dict(space=space_name, ARI=ari, NMI=nmi, AMI=ami,
                Silhouette=sil, DaviesBouldin=db)

def evaluate_clustering_mc(embeddings_np, yp, logits_mean, y_true, prefix=""):
    results = [
        compute_clustering_metrics(logits_mean, yp, y_true, space_name="logit (MC avg)"),
        compute_clustering_metrics(embeddings_np, yp, y_true, space_name="embedding (GCN out)"),
    ]
    sep = "─" * 72
    header = (f"  {'Space':<24} {'ARI':>8} {'NMI':>8} {'AMI':>8} "
              f"{'Silhouette':>12} {'DaviesBouldin':>14}")
    print(f"\n{sep}\n  CLUSTERING METRICS{' '+prefix if prefix else ''}\n{sep}")
    print(header)
    print(f"  {sep}")
    for r in results:
        sil_s = f"{r['Silhouette']:>12.4f}" if not np.isnan(r['Silhouette']) else "         N/A"
        db_s = f"{r['DaviesBouldin']:>14.4f}" if not np.isnan(r['DaviesBouldin']) else "           N/A"
        print(f"  {r['space']:<24} {r['ARI']:>8.4f} {r['NMI']:>8.4f} "
              f"{r['AMI']:>8.4f}{sil_s}{db_s}")
    print(f"  {sep}")
    return results

def print_cluster_uncertainty_report(logits_stack, yp, y_true, n_passes, sample_ids=None):
    sep = "─" * 72
    H_assign, H_aleat, MI, p_mean = compute_cluster_assignment_uncertainty(logits_stack)
    label_map = {0: "CN", 1: "AD"}
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
    print(f"\n{sep}\n  CLUSTER ASSIGNMENT UNCERTAINTY (MAGI)\n{sep}")
    print(f"  (Entropy of soft cluster assignments — {n_passes} MC perturbation passes)\n")
    print(f"  {'Metric':<30} {'Mean':>9} {'Std':>9} {'Min':>9} {'Max':>9}")
    print(f"  {'─'*65}")
    for col, label in [
        ("entropy_assignment", "Assignment entropy H[p̄] (bits)"),
        ("entropy_aleatoric",  "Aleatoric entropy E[H] (bits)"),
        ("model_uncertainty",  "Model uncertainty MI (bits)"),
        ("p_cluster1",         "Soft assignment p(cluster=1)"),
    ]:
        vals = df[col].values
        print(f"  {label:<30}  {vals.mean():>9.4f}  {vals.std():>9.4f}  "
              f"{vals.min():>9.4f}  {vals.max():>9.4f}")
    df_sorted = df.sort_values("entropy_assignment", ascending=False).reset_index(drop=True)
    print(f"\n  Top-10 most uncertain cluster assignments:")
    cols_show = ["sample_id", "true_label", "cluster_assignment",
                 "p_cluster0", "p_cluster1", "entropy_assignment", "model_uncertainty"]
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.width", 160)
    print(df_sorted.head(10)[cols_show].to_string(index=True))
    low_unc = (H_assign < 0.3).sum()
    high_unc = (H_assign > 0.7).sum()
    print(f"\n  Assignment confidence summary:")
    print(f"    Low entropy  (< 0.3 bits, high-confidence): {low_unc}/{len(y_true)} "
          f"({100*low_unc/len(y_true):.1f}%)")
    print(f"    High entropy (> 0.7 bits, ambiguous):        {high_unc}/{len(y_true)} "
          f"({100*high_unc/len(y_true):.1f}%)")
    print(f"\n  Mean assignment entropy: {H_assign.mean():.4f} ± {H_assign.std():.4f} bits")
    print(f"  (Max possible entropy for K={K}: {np.log2(K):.4f} bits)")
    print(f"{sep}")
    return df

def print_clustering_summary(all_records, depth_label="MAGI GCN"):
    sep = "─" * 90
    print(f"\n{sep}")
    print(f"  CLUSTERING METRICS SUMMARY  [{depth_label}]  "
          f"(mean ± std, {len(all_records)} seeds)")
    print(f"{sep}")
    spaces = [("logit (MC avg)",      "Logit space (MC-averaged, K-dim)"),
              ("embedding (GCN out)", "Embedding space (256-dim GCN output)")]
    for space_key, space_label in spaces:
        print(f"\n  {space_label}")
        print(f"  {'Metric':<16} {'Mean':>9} {'Std':>9} {'Min':>9} {'Max':>9}")
        print(f"  {'─'*55}")
        for metric, hi_lo in [("ARI", "↑"), ("NMI", "↑"), ("AMI", "↑"),
                               ("Silhouette", "↑"), ("DaviesBouldin", "↓")]:
            key = f"{metric}_{space_key}"
            vals = np.array([r.get(key, np.nan) for r in all_records])
            valid = vals[~np.isnan(vals)]
            if len(valid) == 0:
                print(f"  {metric+' '+hi_lo:<16}  {'N/A':>9}")
                continue
            print(f"  {metric+' '+hi_lo:<16}  {valid.mean():>9.4f}  {valid.std():>9.4f}"
                  f"  {valid.min():>9.4f}  {valid.max():>9.4f}")
    print(f"\n{sep}")

# =============================================================================
# MAIN TRAINING LOOP (FAITHFUL MAGI WITH MINI‑BATCHES)
# =============================================================================
print(f"\n{'═'*72}")
print(f"  MAGI (faithful) + K‑Means — CN vs AD (α={ALPHA})")
print(f"  {N_RUNS} runs, {EPOCHS} epochs each, {N_EVAL_PASSES} MC passes")
print(f"  Roots={BATCH_ROOT_NODES}, walks={WALKS_PER_ROOT}, depth={WALK_DEPTH}")
print(f"{'═'*72}\n")

# We will collect weighted metrics across runs
all_acc = []
all_prec_weighted = []
all_recall_weighted = []
all_f1_weighted = []
all_nmi = []
all_ari = []
all_ami = []
all_silhouette = []
all_davies_bouldin = []
all_mean_entropy = []
all_std_entropy = []

# For final uncertainty report (run 1)
last_logits_stack = None
last_yp = None

col_w = "─" * 120
print(f"  {'Run':>3}  {'Acc':>7}  {'Prec_w':>8}  {'Rec_w':>8}  {'F1_w':>8}  "
      f"{'NMI':>7}  {'ARI':>7}  {'MeanEntropy':>13}  {'StdEntropy':>11}")
print(f"  {col_w}")

for run in range(N_RUNS):
    setup_seed(42 + run)

    encoder = Encoder(F_dim, 256).to(DEVICE)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=5e-4, weight_decay=1e-3)

    for epoch in range(EPOCHS):
        encoder.train()
        optimizer.zero_grad()

        # Stage 1: sample root nodes and build batch B
        root_nodes = torch.randint(0, N, (BATCH_ROOT_NODES,), device=DEVICE)
        batch_nodes = stage1_sample_communities_batched(
            root_nodes, WALKS_PER_ROOT, WALK_DEPTH, adj_full, N
        )
        if batch_nodes.numel() == 0:
            continue

        # Build subgraph for this batch
        row, col = edge_index_full
        mask = torch.isin(row, batch_nodes) | torch.isin(col, batch_nodes)
        sub_edge = edge_index_full[:, mask]
        sub_nodes = torch.unique(torch.cat([batch_nodes, sub_edge.view(-1)]))
        node_to_local = {int(n.item()): i for i, n in enumerate(sub_nodes)}
        local_batch = torch.tensor([node_to_local[int(n.item())] for n in batch_nodes], device=DEVICE)
        x_sub = x_torch_full[sub_nodes]
        sub_edge_local = torch.stack([
            torch.tensor([node_to_local[int(n.item())] for n in sub_edge[0]], device=DEVICE),
            torch.tensor([node_to_local[int(n.item())] for n in sub_edge[1]], device=DEVICE)
        ])

        # Forward pass on subgraph
        z_sub = encoder(x_sub, sub_edge_local)
        z_sub = scale(z_sub)
        z_sub = F.normalize(z_sub, dim=1)
        z_batch = z_sub[local_batch]

        # Stage 2: modularity matrix for batch
        B_mat = stage2_build_modularity_batched(batch_nodes, WALKS_PER_ROOT, WALK_DEPTH, adj_full)

        # Loss and backprop
        loss = magi_contrastive_loss(z_batch, B_mat, tau=TAU)
        if loss.item() > 0:
            loss.backward()
            optimizer.step()

    # — Evaluation after training (full graph) —
    encoder.eval()
    with torch.no_grad():
        z_full = encoder(x_torch_full, edge_index_full)
        z_full = scale(z_full)
        z_full = F.normalize(z_full, dim=1).cpu().numpy()

    kmeans = KMeans(n_clusters=K, n_init=20, random_state=run)
    kmeans.fit(z_full)
    centers = kmeans.cluster_centers_

    yp_i, _, logits_stack_i, logits_mean_i = evaluate_magi_mc(
        encoder, X, edge_index_np, y, centers,
        n_passes=N_EVAL_PASSES, seed_base=9999 + run
    )

    # — Weighted metrics (averaged across both classes) —
    acc_i = accuracy_score(y, yp_i)
    prec_w = precision_score(y, yp_i, average='weighted', zero_division=0)
    rec_w = recall_score(y, yp_i, average='weighted', zero_division=0)
    f1_w = f1_score(y, yp_i, average='weighted', zero_division=0)
    nmi_i = normalized_mutual_info_score(y, yp_i, average_method='arithmetic')
    ari_i = adjusted_rand_score(y, yp_i)
    ami_i = adjusted_mutual_info_score(y, yp_i, average_method='arithmetic')

    # Geometric metrics on original feature space
    # Standardize features for silhouette/DB (important)
    X_scaled = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    unique_preds = np.unique(yp_i)
    can_geom = (len(unique_preds) >= 2) and all((yp_i == c).sum() >= 2 for c in unique_preds)
    if can_geom:
        sil_i = silhouette_score(X_scaled, yp_i, metric='euclidean')
        db_i = davies_bouldin_score(X_scaled, yp_i)
    else:
        sil_i, db_i = np.nan, np.nan

    # Entropy metrics
    H_assign_i, _, _, _ = compute_cluster_assignment_uncertainty(logits_stack_i)
    mean_ent_i = float(H_assign_i.mean())
    std_ent_i = float(H_assign_i.std())

    # Store
    all_acc.append(acc_i)
    all_prec_weighted.append(prec_w)
    all_recall_weighted.append(rec_w)
    all_f1_weighted.append(f1_w)
    all_nmi.append(nmi_i)
    all_ari.append(ari_i)
    all_ami.append(ami_i)
    all_silhouette.append(sil_i)
    all_davies_bouldin.append(db_i)
    all_mean_entropy.append(mean_ent_i)
    all_std_entropy.append(std_ent_i)

    print(f"  {run+1:>3}  {acc_i:>7.4f}  {prec_w:>8.4f}  {rec_w:>8.4f}  {f1_w:>8.4f}  "
          f"{nmi_i:>7.4f}  {ari_i:>7.4f}  {mean_ent_i:>13.4f}  {std_ent_i:>11.4f}")

    if run == 0:
        last_logits_stack = logits_stack_i
        last_yp = yp_i

# =============================================================================
# HORIZONTAL TABLE WITH MEAN ± STD (weighted metrics)
# =============================================================================
print(f"\n{'═'*72}")
print("  HORIZONTAL TABLE FOR MANUSCRIPT (mean ± std over 10 runs)")
print("  Metrics: Accuracy, Weighted Precision, Weighted Recall, Weighted F1, NMI, ARI, AMI, Silhouette, Davies‑Bouldin")
print('═'*72)

metrics_names = [
    "Accuracy", "Prec (weighted)", "Recall (weighted)", "F1 (weighted)",
    "NMI", "ARI", "AMI", "Silhouette", "Davies‑Bouldin"
]
means = [
    np.mean(all_acc), np.mean(all_prec_weighted), np.mean(all_recall_weighted), np.mean(all_f1_weighted),
    np.mean(all_nmi), np.mean(all_ari), np.mean(all_ami),
    np.nanmean(all_silhouette), np.nanmean(all_davies_bouldin)
]
stds = [
    np.std(all_acc), np.std(all_prec_weighted), np.std(all_recall_weighted), np.std(all_f1_weighted),
    np.std(all_nmi), np.std(all_ari), np.std(all_ami),
    np.nanstd(all_silhouette), np.nanstd(all_davies_bouldin)
]

# Tab‑separated line (ready to copy)
print("\nMethod\t" + "\t".join(metrics_names))
row = "MAGI"
for m, s in zip(means, stds):
    if np.isnan(m):
        row += "\tN/A±N/A"
    else:
        row += f"\t{m:.4f}±{s:.4f}"
print(row)

# LaTeX version
print("\n" + "─" * 72)
print("  LaTeX code for the horizontal table (copy the line below):")
print("─" * 72)
latex_row = "MAGI"
for m, s in zip(means, stds):
    if np.isnan(m):
        latex_row += " & N/A±N/A"
    else:
        latex_row += f" & ${m:.4f}\\pm{s:.4f}$"
latex_row += " \\\\"
print(latex_row)

# =============================================================================
# DETAILED UNCERTAINTY REPORT (Run 1)
# =============================================================================
print(f"\n{'═'*72}\n  DETAILED UNCERTAINTY REPORT — Run 1\n{'═'*72}")
df_unc = print_cluster_uncertainty_report(
    last_logits_stack, last_yp, y,
    n_passes=N_EVAL_PASSES,
    sample_ids=list(range(len(y)))
)
df_unc.to_csv("magi_cn_ad_uncertainty_run1.csv", index=False, float_format="%.6f")
print(f"\n  CSV → magi_cn_ad_uncertainty_run1.csv")

# =============================================================================
# 10‑RUN SUMMARY (mean ± std)
# =============================================================================
sep100 = "─" * 100
print(f"\n{sep100}")
print(f"  {N_RUNS}‑RUN SUMMARY (mean ± std) — MAGI (CN vs AD, α={ALPHA})")
print(f"{sep100}")

print("\n  A. EXTERNAL VALIDATION METRICS (weighted)")
print(f"  {'Metric':<20}  {'Mean':>9}  {'Std':>9}  {'Min':>9}  {'Max':>9}  Note")
print(f"  {sep100}")
ext_metrics = [
    ("Accuracy", all_acc, ""),
    ("Weighted Precision", all_prec_weighted, ""),
    ("Weighted Recall", all_recall_weighted, ""),
    ("Weighted F1", all_f1_weighted, ""),
    ("NMI", all_nmi, ""),
    ("ARI", all_ari, ""),
    ("AMI", all_ami, ""),
]
for name, vals, _ in ext_metrics:
    print(f"  {name:<20}  {np.mean(vals):>9.4f}  {np.std(vals):>9.4f}  "
          f"{np.min(vals):>9.4f}  {np.max(vals):>9.4f}")

print("\n  B. GEOMETRIC CLUSTERING METRICS")
print(f"  {'Metric':<20}  {'Mean':>9}  {'Std':>9}  {'Min':>9}  {'Max':>9}  {'Valid runs':>10}")
print(f"  {sep100}")
geo_metrics = [("Silhouette", all_silhouette), ("Davies‑Bouldin", all_davies_bouldin)]
for name, vals in geo_metrics:
    valid = [v for v in vals if not np.isnan(v)]
    if len(valid) > 0:
        print(f"  {name:<20}  {np.mean(valid):>9.4f}  {np.std(valid):>9.4f}  "
              f"{np.min(valid):>9.4f}  {np.max(valid):>9.4f}  {len(valid):>10}")
    else:
        print(f"  {name:<20}  {'N/A':>9}  {'N/A':>9}  {'N/A':>9}  {'N/A':>9}  {0:>10}")

print("\n  C. ASSIGNMENT ENTROPY")
print(f"  {'Metric':<20}  {'Mean':>9}  {'Std':>9}  {'Min':>9}  {'Max':>9}")
print(f"  {sep100}")
print(f"  {'Mean entropy (bits)':<20}  {np.mean(all_mean_entropy):>9.4f}  {np.std(all_mean_entropy):>9.4f}  "
      f"{np.min(all_mean_entropy):>9.4f}  {np.max(all_mean_entropy):>9.4f}")
print(f"  {'Std entropy (bits)':<20}  {np.mean(all_std_entropy):>9.4f}  {np.std(all_std_entropy):>9.4f}  "
      f"{np.min(all_std_entropy):>9.4f}  {np.max(all_std_entropy):>9.4f}")

print("\n  D. CLUSTER STABILITY")
print(f"  {'Weighted F1':<20}  {np.mean(all_f1_weighted):>9.4f}  {np.std(all_f1_weighted):>9.4f}")
print(f"  {'Assignment Entropy':<20}  {np.mean(all_mean_entropy):>9.4f}  {np.std(all_mean_entropy):>9.4f}")
print(f"{sep100}")

# Export weighted summary CSV
df_weighted_summary = pd.DataFrame({
    "accuracy": all_acc,
    "prec_weighted": all_prec_weighted,
    "recall_weighted": all_recall_weighted,
    "f1_weighted": all_f1_weighted,
    "nmi": all_nmi,
    "ari": all_ari,
    "ami": all_ami,
    "silhouette": all_silhouette,
    "davies_bouldin": all_davies_bouldin,
    "mean_entropy": all_mean_entropy,
    "std_entropy": all_std_entropy,
})
df_weighted_summary.to_csv("magi_cn_ad_weighted_summary.csv", index=False, float_format="%.6f")
print(f"\n  ✓ Saved weighted summary → magi_cn_ad_weighted_summary.csv")

print(f"\n{'═'*60}")
print(f"  MAGI + K‑Means ({N_RUNS} Runs) — Final Summary (weighted metrics)")
print(f"{'═'*60}")
print(f"  ACC (unweighted)        : {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
print(f"  Weighted Precision      : {np.mean(all_prec_weighted):.4f} ± {np.std(all_prec_weighted):.4f}")
print(f"  Weighted Recall         : {np.mean(all_recall_weighted):.4f} ± {np.std(all_recall_weighted):.4f}")
print(f"  Weighted F1             : {np.mean(all_f1_weighted):.4f} ± {np.std(all_f1_weighted):.4f}")
print(f"  NMI                     : {np.mean(all_nmi):.4f} ± {np.std(all_nmi):.4f}")
print(f"  ARI                     : {np.mean(all_ari):.4f} ± {np.std(all_ari):.4f}")
print(f"  Mean Assignment Entropy : {np.mean(all_mean_entropy):.4f} ± {np.std(all_mean_entropy):.4f} bits")
print(f"  (Max possible for K=2   : {np.log2(K):.4f} bits)")

print(f"\n{'═'*60}\n  MAGI ANALYSIS COMPLETE (weighted metrics)\n{'═'*60}")
