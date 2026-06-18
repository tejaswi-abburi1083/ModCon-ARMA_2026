#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


get_ipython().system('pip install -q torch_geometric')
get_ipython().system('pip install -q class_resolver')
get_ipython().system('pip3 install pymatting')


# In[3]:


get_ipython().system('pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-2.0.0+cu118.html')


# In[2]:


get_ipython().system('pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html')


# In[4]:


import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected
from torch_sparse import SparseTensor
from torch.utils.data import TensorDataset, DataLoader, Subset
from torchvision import transforms
import timm
from sklearn.metrics import (
    accuracy_score, f1_score, normalized_mutual_info_score,
    adjusted_rand_score, precision_score, recall_score,
    silhouette_score, davies_bouldin_score,
    adjusted_mutual_info_score, confusion_matrix
)
from sklearn.cluster import KMeans
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("CUDA available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

N_EVAL_PASSES = 30
K = 2
FEAT_NOISE = 0.20
EDGE_DROP_MIN = 0.15
EDGE_DROP_STEP = 0.10
ALPHA = 0.92
EPOCHS = 200
N_RUNS = 10

BATCH_ROOT_NODES = 128
WALKS_PER_ROOT = 10
WALK_DEPTH = 4
TAU = 0.3

RADIODINO_MODEL = "hf_hub:Snarcy/RadioDino-s16"

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
# DATA: PNEUMONIAMNIST + RADIODINO FEATURES
# =============================================================================
data_npz = np.load('/content/drive/MyDrive/TejaswiAbburi_va797/Dataset/Medmnist_data/pneumoniamnist_224.npz', allow_pickle=True)
# data_npz = np.load('/home/snu/Downloads/pneumoniamnist_224.npz', allow_pickle=True)

all_images = np.concatenate([data_npz['train_images'],
                              data_npz['val_images'],
                              data_npz['test_images']], axis=0)
all_labels = np.concatenate([data_npz['train_labels'],
                              data_npz['val_labels'],
                              data_npz['test_labels']], axis=0).squeeze()

images = all_images.astype(np.float32) / 255.0
images = np.repeat(images[:, None, :, :], 3, axis=1)
X_img = torch.tensor(images)
y_img = torch.tensor(all_labels).long()

dataset = TensorDataset(X_img, y_img)
class0_indices = [i for i in range(len(y_img)) if y_img[i] == 0]  # Normal
class1_indices = [i for i in range(len(y_img)) if y_img[i] == 1]  # Pneumonia
random.seed(42)
sampled_class0 = random.sample(class0_indices, min(2000, len(class0_indices)))
sampled_class1 = random.sample(class1_indices, min(2000, len(class1_indices)))
combined_indices = sampled_class0 + sampled_class1
random.shuffle(combined_indices)
final_dataset = Subset(dataset, combined_indices)
final_loader = DataLoader(final_dataset, batch_size=64, shuffle=False)

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
        feats = radiodino(imgs_norm)
        rd_feats.append(feats.cpu())
        y_list.extend(lbls.cpu().tolist())

X = torch.cat(rd_feats, dim=0).numpy().astype(np.float32)
y = np.array(y_list, dtype=np.int64)

np.random.seed(42)
perm = np.random.permutation(X.shape[0])
X, y = X[perm], y[perm]

N, F_dim = X.shape
print(f"Nodes: {N}  Features: {F_dim}  (Normal: {(y==0).sum()}  Pneumonia: {(y==1).sum()})")
print("NOTE: Diagnostic labels used ONLY for post‑hoc external validation.\n")

x_torch_full = torch.from_numpy(X).float().to(DEVICE)
y_torch_full = torch.from_numpy(y).long().to(DEVICE)

# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================
def create_adj(features, alpha=0.6):
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
# MAGI STAGE 1 & 2 (unchanged)
# =============================================================================
def stage1_sample_communities_batched(root_nodes, t, l, adj_sparse, N_total):
    R = len(root_nodes)
    start = root_nodes.repeat_interleave(t)
    walks = adj_sparse.random_walk(start, l)
    visited = walks[:, 1:].reshape(-1)
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

def stage2_build_modularity_batched(batch_nodes, t, l, adj_sparse):
    b = len(batch_nodes)
    if b == 0:
        return torch.zeros((0, 0), device=DEVICE)
    start = batch_nodes.repeat_interleave(t)
    walks = adj_sparse.random_walk(start, l)
    visited = walks[:, 1:].reshape(b, t, l)
    node_to_idx = {int(n.item()): i for i, n in enumerate(batch_nodes)}
    S = torch.zeros((b, b), device=DEVICE)
    for i in range(b):
        targets = visited[i].reshape(-1)
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

class Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()
        self.conv = GCNConv(in_dim, hidden_dim)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = F.leaky_relu(x, 0.5)
        return x

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
# MC EVALUATION
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
    a = accuracy_score(y_true, yp)
    ai = accuracy_score(y_true, 1 - yp)
    if ai > a:
        yp = 1 - yp
        logits_mean = logits_mean[:, ::-1].copy()
        logits_stack = logits_stack[:, ::-1, :].copy()
    ypp = _softmax_np(logits_mean)
    return yp, ypp, logits_stack, logits_mean

def print_cluster_uncertainty_report(logits_stack, yp, y_true, n_passes, sample_ids=None):
    sep = "─" * 72
    H_assign, H_aleat, MI, p_mean = compute_cluster_assignment_uncertainty(logits_stack)
    label_map = {0: "Normal", 1: "Pneumonia"}
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
# MAIN TRAINING LOOP (WITH WEIGHTED METRICS)
# =============================================================================
print(f"\n{'═'*72}")
print(f"  MAGI (faithful) + K‑Means — PneumoniaMNIST (α={ALPHA})")
print(f"  {N_RUNS} runs, {EPOCHS} epochs each, {N_EVAL_PASSES} MC passes")
print(f"  Roots={BATCH_ROOT_NODES}, walks={WALKS_PER_ROOT}, depth={WALK_DEPTH}")
print(f"{'═'*72}\n")

col_w = "─" * 120
print(f"  {'Run':>3}  {'Acc':>7}  {'Prec_w':>8}  {'Rec_w':>8}  {'F1_w':>8}  "
      f"{'NMI':>7}  {'MeanEntropy':>13}  {'StdEntropy':>11}  "
      f"{'TN':>5}  {'FP':>5}  {'FN':>5}  {'TP':>5}")
print(f"  {col_w}")

# Storage for weighted metrics
all_acc = []
all_prec_weighted = []
all_recall_weighted = []
all_f1_weighted = []
all_nmi = []
all_ent = []
all_std_ent = []
seed_clustering_records = []
calibration_rows = []
last_logits_stack = None
last_yp = None

for run in range(N_RUNS):
    setup_seed(42 + run)

    encoder = Encoder(F_dim, 256).to(DEVICE)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=5e-4, weight_decay=1e-3)

    for epoch in range(EPOCHS):
        encoder.train()
        optimizer.zero_grad()

        root_nodes = torch.randint(0, N, (BATCH_ROOT_NODES,), device=DEVICE)
        batch_nodes = stage1_sample_communities_batched(
            root_nodes, WALKS_PER_ROOT, WALK_DEPTH, adj_full, N
        )
        if batch_nodes.numel() == 0:
            continue

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

        z_sub = encoder(x_sub, sub_edge_local)
        z_sub = scale(z_sub)
        z_sub = F.normalize(z_sub, dim=1)
        z_batch = z_sub[local_batch]

        B_mat = stage2_build_modularity_batched(batch_nodes, WALKS_PER_ROOT, WALK_DEPTH, adj_full)

        loss = magi_contrastive_loss(z_batch, B_mat, tau=TAU)
        if loss.item() > 0:
            loss.backward()
            optimizer.step()

    # Evaluation after training
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

    # Weighted metrics (averaged across both classes)
    acc_i = accuracy_score(y, yp_i)
    prec_w = precision_score(y, yp_i, average='weighted', zero_division=0)
    rec_w  = recall_score(y, yp_i, average='weighted', zero_division=0)
    f1_w   = f1_score(y, yp_i, average='weighted', zero_division=0)
    nmi_i  = normalized_mutual_info_score(y, yp_i, average_method='arithmetic')

    H_assign_i, _, _, _ = compute_cluster_assignment_uncertainty(logits_stack_i)
    mean_ent_i = float(H_assign_i.mean())
    std_ent_i  = float(H_assign_i.std())

    tn_i, fp_i, fn_i, tp_i = confusion_matrix(y, yp_i).ravel()

    cl_res_i = evaluate_clustering_mc(z_full, yp_i, logits_mean_i, y,
                                      prefix=f"Run {run+1} (quiet)")
    cl_flat_i = {}
    for r in cl_res_i:
        sp = r['space']
        for metric in ['ARI', 'NMI', 'AMI', 'Silhouette', 'DaviesBouldin']:
            cl_flat_i[f"{metric}_{sp}"] = r[metric]
    seed_clustering_records.append(cl_flat_i)

    all_acc.append(acc_i)
    all_prec_weighted.append(prec_w)
    all_recall_weighted.append(rec_w)
    all_f1_weighted.append(f1_w)
    all_nmi.append(nmi_i)
    all_ent.append(mean_ent_i)
    all_std_ent.append(std_ent_i)

    calibration_rows.append(dict(
        run=run, acc=acc_i, prec_weighted=prec_w, rec_weighted=rec_w, f1_weighted=f1_w, nmi=nmi_i,
        mean_entropy=mean_ent_i, std_entropy=std_ent_i,
        tn=int(tn_i), fp=int(fp_i), fn=int(fn_i), tp=int(tp_i),
    ))

    print(f"  {run+1:>3}  {acc_i:>7.4f}  {prec_w:>8.4f}  {rec_w:>8.4f}  {f1_w:>8.4f}  "
          f"{nmi_i:>7.4f}  {mean_ent_i:>13.4f}  {std_ent_i:>11.4f}  "
          f"{tn_i:>5}  {fp_i:>5}  {fn_i:>5}  {tp_i:>5}")

    if run == 0:
        last_logits_stack = logits_stack_i
        last_yp = yp_i

print(f"\n{'═'*72}\n  DETAILED UNCERTAINTY REPORT — Run 1\n{'═'*72}")
df_unc = print_cluster_uncertainty_report(
    last_logits_stack, last_yp, y,
    n_passes=N_EVAL_PASSES,
    sample_ids=list(range(len(y)))
)
df_unc.to_csv("magi_pneumoniamnist_uncertainty_run1.csv", index=False, float_format="%.6f")
print(f"\n  CSV → magi_pneumoniamnist_uncertainty_run1.csv")

# =============================================================================
#  HORIZONTAL TABLE FOR MANUSCRIPT (mean ± std over 10 runs)
# =============================================================================
print("\n" + "=" * 72)
print("  HORIZONTAL TABLE FOR MANUSCRIPT (mean ± std over 10 runs)")
print("  Metrics: Accuracy, Weighted Precision, Weighted Recall, Weighted F1, NMI, ARI, AMI, Silhouette, Davies‑Bouldin")
print("=" * 72)

# Extract weighted classification metrics
acc_vals = np.array(all_acc)
prec_vals = np.array(all_prec_weighted)
rec_vals = np.array(all_recall_weighted)
f1_vals = np.array(all_f1_weighted)
nmi_vals = np.array(all_nmi)

# Extract clustering metrics from seed_clustering_records (logit space)
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
row = "MAGI"
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
latex_row = "MAGI"
for m, s in zip(means, stds):
    if np.isnan(m):
        latex_row += " & N/A±N/A"
    else:
        latex_row += f" & ${m:.4f}\\pm{s:.4f}$"
latex_row += " \\\\"
print(latex_row)

# =============================================================================
#  ORIGINAL SUMMARY (adapted to show weighted metrics)
# =============================================================================
results_np = np.array([[r['acc'], r['prec_weighted'], r['rec_weighted'], r['f1_weighted'],
                        r['nmi'], r['mean_entropy'], r['std_entropy']]
                       for r in calibration_rows])
counts_np = np.array([[r['tn'], r['fp'], r['fn'], r['tp']]
                      for r in calibration_rows], dtype=float)

sep100 = "─" * 100
print(f"\n{sep100}")
print(f"  {N_RUNS}-RUN SUMMARY (mean ± std) — MAGI (PneumoniaMNIST, α={ALPHA})")
print(f"{sep100}")

ext_val_labels = [
    ("Accuracy", "External validation (weighted)"),
    ("Weighted Precision", "External validation"),
    ("Weighted Recall", "External validation"),
    ("Weighted F1", "External validation"),
    ("NMI", "External validation"),
    ("Mean Entropy", "Assignment entropy (bits)"),
    ("Std Entropy", "Assignment entropy std"),
]
print(f"\n  A. EXTERNAL VALIDATION METRICS & UNCERTAINTY")
print(f"  {'Metric':<20}  {'Mean':>9}  {'Std':>9}  {'Min':>9}  {'Max':>9}  Note")
print(f"  {sep100}")
for col_idx, (label, note) in enumerate(ext_val_labels):
    vals = results_np[:, col_idx]
    print(f"  {label:<20}  {vals.mean():>9.4f}  {vals.std():>9.4f}  "
          f"{vals.min():>9.4f}  {vals.max():>9.4f}  {note}")

print(f"\n  B. CONFUSION MATRIX COUNTS")
print(f"  {'Group':<20}  {'Mean':>7}  {'Std':>7}  {'Min':>5}  {'Max':>5}")
print(f"  {sep100}")
count_labels = [
    ("TN (Normal→C0)", "Normal→Cluster0"),
    ("FP (Normal→C1)", "Normal→Cluster1"),
    ("FN (Pneumonia→C0)",  "Pneumonia→Cluster0"),
    ("TP (Pneumonia→C1)",  "Pneumonia→Cluster1"),
]
for col_idx, (label, _) in enumerate(count_labels):
    vals = counts_np[:, col_idx]
    print(f"  {label:<20}  {vals.mean():>7.1f}  {vals.std():>7.2f}  "
          f"{vals.min():>5.0f}  {vals.max():>5.0f}")

print(f"\n  C. CLUSTER STABILITY")
print(f"  {'Weighted F1':<20}  {results_np[:,3].mean():>9.4f}  {results_np[:,3].std():>9.4f}")
print(f"  {'Assignment Entropy':<20}  {results_np[:,5].mean():>9.4f}  {results_np[:,5].std():>9.4f}")
print(f"{sep100}")

print(f"\n{'═'*72}\n  CLUSTERING METRICS SUMMARY\n{'═'*72}")
print_clustering_summary(seed_clustering_records, depth_label="MAGI GCN — PneumoniaMNIST (RadioDINO)")

print("\n===== Weighted F1 Scores Across Runs =====")
for idx, score in enumerate(all_f1_weighted):
    print(f"  {idx+1}. {score:.4f}")

print(f"\n{'='*60}")
print(f"  MAGI + K‑Means ({N_RUNS} Runs) — Final Summary (weighted metrics)")
print(f"{'='*60}")
print(f"  ACC (unweighted) : {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
print(f"  Weighted PREC    : {np.mean(all_prec_weighted):.4f} ± {np.std(all_prec_weighted):.4f}")
print(f"  Weighted REC     : {np.mean(all_recall_weighted):.4f} ± {np.std(all_recall_weighted):.4f}")
print(f"  Weighted F1      : {np.mean(all_f1_weighted):.4f} ± {np.std(all_f1_weighted):.4f}")
print(f"  NMI              : {np.mean(all_nmi):.4f} ± {np.std(all_nmi):.4f}")
print(f"  Mean Assignment Entropy : {np.mean(all_ent):.4f} ± {np.std(all_ent):.4f} bits")
print(f"  (Max possible for K=2   : {np.log2(K):.4f} bits)")

df_summary = pd.DataFrame(calibration_rows)
df_summary.to_csv("magi_pneumoniamnist_summary.csv", index=False, float_format="%.6f")
print(f"\n  ✓ Saved summary → magi_pneumoniamnist_summary.csv")
print(f"\n{'='*60}\n  MAGI ANALYSIS COMPLETE\n{'='*60}")

