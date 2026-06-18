import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnFn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import Data
from torch_geometric.nn import ARMAConv
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, confusion_matrix, normalized_mutual_info_score,
    adjusted_rand_score, adjusted_mutual_info_score,
    silhouette_score, davies_bouldin_score
)
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
from contextlib import contextmanager
import random
import copy
import pandas as pd


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FEATS_DIM = 180
K = 2
ALPHA = 0.92
NUM_EPOCHS = 5000
N_EVAL_PASSES = 30
NUM_RUNS = 10


cn_path = "/home/snu/Downloads/Histogram_CN_FA_20bin_updated.npy"
mci_path = "/home/snu/Downloads/Histogram_MCI_FA_20bin_updated.npy"

cn_data = np.load(cn_path, allow_pickle=True)
mci_data = np.load(mci_path, allow_pickle=True)

X = np.vstack([cn_data, mci_data]).astype(np.float32)
y = np.hstack([np.zeros(len(cn_data), dtype=np.int64),
               np.ones(len(mci_data), dtype=np.int64)])

np.random.seed(42)
perm = np.random.permutation(X.shape[0])
X, y = X[perm], y[perm]

print(f"Features: {X.shape}, Labels: {y.shape} (CN: {np.sum(y==0)}, MCI: {np.sum(y==1)})")
print("NOTE: Diagnostic labels are held out and used ONLY for post-hoc external validation.\n")


def create_adj(features, cut, alpha=1.0):
    F_norm = features / np.linalg.norm(features, axis=1, keepdims=True)
    W = np.dot(F_norm, F_norm.T)
    if cut == 0:
        W = np.where(W >= alpha, 1, 0).astype(np.float32)
        W = (W / W.max()).astype(np.float32)
    else:
        W = (W * (W >= alpha)).astype(np.float32)
    return W

def edge_index_from_dense(W):
    r, c = np.nonzero(W > 0)
    return np.vstack([r, c]).astype(np.int64), W[r, c].astype(np.float32)

def aug_random_edge_edge_index(edge_index_np, drop_percent=0.2, seed=None):
    rng = np.random.default_rng(seed)
    keep = rng.random(edge_index_np.shape[1]) >= drop_percent
    return edge_index_np[:, keep]

def to_data(node_feats_np, edge_index_np, device):
    node_feats = torch.from_numpy(node_feats_np).float().to(device)
    edge_index = torch.from_numpy(edge_index_np.astype(np.int64)).long().to(device)
    return Data(x=node_feats, edge_index=edge_index)

def hungarian_map(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).T
    new_pred = np.zeros_like(y_pred, dtype=np.int64)
    for i, j in ind:
        new_pred[y_pred == i] = j
    return new_pred


class MLP(nn.Module):
    def __init__(self, inp_size, outp_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, outp_size)
        )
    def forward(self, x):
        return self.net(x)

class ARMAEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, device, activ, stacks=1, layers=1):
        super(ARMAEncoder, self).__init__()
        self.device = device
        self.arma = ARMAConv(input_dim, hidden_dim, num_stacks=stacks, num_layers=layers)
        self.batchnorm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.mlp = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.arma(x, edge_index)
        x = self.dropout(x)
        x = self.batchnorm(x)
        logits = self.mlp(x)
        return logits

class ARMA(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_clusters, device, activ):
        super(ARMA, self).__init__()
        self.device = device
        self.num_clusters = num_clusters
        self.online_encoder = ARMAEncoder(input_dim, hidden_dim, device, activ)
        self.online_predictor = MLP(hidden_dim, num_clusters, hidden_dim)
        self.loss = self.modularity_loss

    def forward(self, data):
        x = self.online_encoder(data)
        logits = self.online_predictor(x)
        S = nnFn.softmax(logits, dim=1)
        return S, logits

    def modularity_loss(self, A, S):
        C = nnFn.softmax(S, dim=1)
        d = torch.sum(A, dim=1)
        m = torch.sum(A)
        B = A - torch.ger(d, d) / (2 * m)
        k = torch.tensor(self.num_clusters, device=self.device, dtype=torch.float32)
        n = S.shape[0]
        modularity_term = (-1 / (2 * m)) * torch.trace(C.T @ B @ C)
        collapse_reg_term = (torch.sqrt(k) / n) * torch.norm(C.sum(dim=0), p='fro') - 1
        return modularity_term + collapse_reg_term


@contextmanager
def mc_dropout_mode(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    try:
        yield model
    finally:
        model.eval()


def train_once(seed, X_data, y_data, verbose=True):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


    perm_run = np.random.permutation(X_data.shape[0])
    features = X_data[perm_run].astype(np.float32)
    labels = y_data[perm_run]


    W0 = create_adj(features, 0, ALPHA)
    edge_index_np, _ = edge_index_from_dense(W0)
    A1 = torch.from_numpy(W0).float().to(DEVICE)
    data0 = to_data(features, edge_index_np, DEVICE)

    model = ARMA(FEATS_DIM, 256, K, DEVICE, "ELU").to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=200, gamma=0.5)

    for epoch in range(NUM_EPOCHS):
        model.train()
        optimizer.zero_grad()
        S, logits = model(data0)
        loss = model.loss(A1, logits)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if verbose and epoch % 1000 == 0:
            print(f"  Epoch {epoch:4d} | Loss: {loss.item():.4f}")

    return model, edge_index_np, features, labels


def evaluate_mc(model, features, edge_index_np, n_passes=30, seed_base=42):
    model.eval()
    all_logits = []
    ref_labels = None

    with mc_dropout_mode(model):
        with torch.no_grad():
            for i in range(n_passes):
                rng = np.random.default_rng(seed_base + i)


                mask = rng.random(features.shape) >= 0.2
                feats_mc = features * mask.astype(np.float32)


                ei_mc = aug_random_edge_edge_index(edge_index_np, drop_percent=0.2, seed=seed_base + i)

                data_mc = to_data(feats_mc, ei_mc, DEVICE)


                _, logits = model(data_mc)
                logits = logits.cpu().numpy()


                labels_mc = np.argmax(logits, axis=1)

                if i == 0:
                    ref_labels = labels_mc.copy()
                else:

                    labels_aligned = hungarian_map(ref_labels, labels_mc)
                    if (labels_aligned != labels_mc).any():

                        logits = logits[:, ::-1].copy()
                all_logits.append(logits)

    logits_stack = np.stack(all_logits, axis=2)
    logits_mean = logits_stack.mean(axis=2)
    yp = np.argmax(logits_mean, axis=1)


    acc = accuracy_score(labels, yp)
    acc_inv = accuracy_score(labels, 1 - yp)
    if acc_inv > acc:
        yp = 1 - yp
        logits_mean = logits_mean[:, ::-1].copy()
        logits_stack = logits_stack[:, ::-1, :].copy()


    ypp = nnFn.softmax(torch.from_numpy(logits_mean), dim=1).numpy()
    return yp, ypp, logits_stack, logits_mean


def entropy_bits(p):
    p = np.clip(p, 1e-12, 1.0)
    return -np.sum(p * np.log2(p), axis=1)

def compute_cluster_assignment_uncertainty(logits_stack):
    N, K, P = logits_stack.shape
    probs = np.stack([nnFn.softmax(torch.from_numpy(logits_stack[:, :, p]), dim=1).numpy()
                      for p in range(P)], axis=2)
    p_mean = probs.mean(axis=2)
    H_assign = entropy_bits(p_mean)
    H_aleat = np.stack([entropy_bits(probs[:, :, p]) for p in range(P)], axis=1).mean(axis=1)
    MI = np.clip(H_assign - H_aleat, 0, None)
    return H_assign, H_aleat, MI, p_mean

def compute_all_metrics(yp, ypp, logits_stack, features, labels):
    """Compute all evaluation metrics, including weighted precision/recall/f1."""

    acc = accuracy_score(labels, yp)
    prec_weighted = precision_score(labels, yp, average='weighted', zero_division=0)
    rec_weighted  = recall_score(labels, yp, average='weighted', zero_division=0)
    f1_weighted   = f1_score(labels, yp, average='weighted', zero_division=0)
    ll = log_loss(labels, ypp[:, 1])


    nmi = normalized_mutual_info_score(labels, yp, average_method='arithmetic')
    ari = adjusted_rand_score(labels, yp)
    ami = adjusted_mutual_info_score(labels, yp, average_method='arithmetic')


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    unique_preds = np.unique(yp)
    can_geom = (len(unique_preds) >= 2) and all((yp == c).sum() >= 2 for c in unique_preds)
    if can_geom:
        sil = silhouette_score(X_scaled, yp, metric='euclidean')
        db = davies_bouldin_score(X_scaled, yp)
    else:
        sil, db = np.nan, np.nan


    H_assign, H_aleat, MI, _ = compute_cluster_assignment_uncertainty(logits_stack)
    mean_ent = H_assign.mean()
    std_ent = H_assign.std()

    return {
        'acc': acc,
        'prec_weighted': prec_weighted,
        'rec_weighted': rec_weighted,
        'f1_weighted': f1_weighted,
        'nmi': nmi,
        'ari': ari,
        'ami': ami,
        'silhouette': sil,
        'db': db,
        'log_loss': ll,
        'mean_entropy': mean_ent,
        'std_entropy': std_ent,
        'H_assign': H_assign,
        'yp': yp
    }


all_results = []
last_yp = None
last_H_assign = None
last_logits_stack = None
last_true_labels = None

print("========================================================================")
print("  ARMA (modularity only) with MC uncertainty")
print(f"  {NUM_RUNS} runs x {N_EVAL_PASSES} MC passes (Dropout only, BatchNorm frozen)")
print("  NOTE: Diagnostic labels used ONLY for post-hoc external validation.")
print("========================================================================")
print(f"\n  {'Run':>4}  {'Acc':>7}  {'Prec_w':>8}  {'Rec_w':>8}  {'F1_w':>8}  "
      f"{'NMI':>7}  {'ARI':>7}  {'MeanEntropy':>13}  {'LogLoss':>9}")
print("  " + "-" * 95)

for run in range(NUM_RUNS):
    print(f"  {run:>4}  ...", end="", flush=True)
    model, edge_index_np, features, labels = train_once(seed=42 + run, X_data=X, y_data=y, verbose=False)
    yp, ypp, logits_stack, _ = evaluate_mc(
        model, features, edge_index_np, n_passes=N_EVAL_PASSES, seed_base=9999 + run
    )
    metrics = compute_all_metrics(yp, ypp, logits_stack, features, labels)

    all_results.append(metrics)
    print(f" {metrics['acc']:7.4f}  {metrics['prec_weighted']:8.4f}  {metrics['rec_weighted']:8.4f}  "
          f"{metrics['f1_weighted']:8.4f}  {metrics['nmi']:7.4f}  {metrics['ari']:7.4f}  "
          f"{metrics['mean_entropy']:13.4f}  {metrics['log_loss']:9.6f}")

    if run == NUM_RUNS - 1:
        last_yp = metrics['yp']
        last_H_assign = metrics['H_assign']
        last_logits_stack = logits_stack
        last_true_labels = labels


print("\n" + "========================================================================")
print("  HORIZONTAL TABLE FOR MANUSCRIPT (mean +- std over 10 runs)")
print("  Metrics: Accuracy, Weighted Precision, Weighted Recall, Weighted F1, NMI, ARI, AMI, Silhouette, Davies-Bouldin")
print("========================================================================")


acc_vals = np.array([r['acc'] for r in all_results])
prec_w_vals = np.array([r['prec_weighted'] for r in all_results])
rec_w_vals = np.array([r['rec_weighted'] for r in all_results])
f1_w_vals = np.array([r['f1_weighted'] for r in all_results])
nmi_vals = np.array([r['nmi'] for r in all_results])
ari_vals = np.array([r['ari'] for r in all_results])
ami_vals = np.array([r['ami'] for r in all_results])
sil_vals = np.array([r['silhouette'] for r in all_results])
db_vals = np.array([r['db'] for r in all_results])

metrics_names = [
    "Accuracy", "Prec (weighted)", "Recall (weighted)", "F1 (weighted)",
    "NMI", "ARI", "AMI", "Silhouette", "Davies-Bouldin"
]
means = [
    acc_vals.mean(), prec_w_vals.mean(), rec_w_vals.mean(), f1_w_vals.mean(),
    nmi_vals.mean(), ari_vals.mean(), ami_vals.mean(),
    np.nanmean(sil_vals), np.nanmean(db_vals)
]
stds = [
    acc_vals.std(), prec_w_vals.std(), rec_w_vals.std(), f1_w_vals.std(),
    nmi_vals.std(), ari_vals.std(), ami_vals.std(),
    np.nanstd(sil_vals), np.nanstd(db_vals)
]


print("\nMethod\t" + "\t".join(metrics_names))
row = "ARMA"
for m, s in zip(means, stds):
    if np.isnan(m):
        row += "\tN/A\xc2\xb1N/A"
    else:
        row += f"\t{m:.4f}\xc2\xb1{s:.4f}"
print(row)


print("\n" + "-" * 72)
print("  LaTeX code for the horizontal table (copy the line below):")
print("-" * 72)
latex_row = "ARMA"
for m, s in zip(means, stds):
    if np.isnan(m):
        latex_row += " & N/A\xc2\xb1N/A"
    else:
        latex_row += f" & ${m:.4f}\\pm{s:.4f}$"
latex_row += " \\\\"
print(latex_row)


if last_logits_stack is not None:
    sep = "-" * 72
    H_assign, H_aleat, MI, p_mean = compute_cluster_assignment_uncertainty(last_logits_stack)
    label_map = {0: "CN", 1: "MCI"}
    df = pd.DataFrame({
        "sample_id": range(len(last_true_labels)),
        "true_label": [label_map[int(l)] for l in last_true_labels],
        "cluster_assignment": [label_map[int(l)] for l in last_yp],
        "correct_ext_valid": (last_yp == last_true_labels).astype(int),
        "p_cluster0": p_mean[:, 0],
        "p_cluster1": p_mean[:, 1],
        "entropy_assignment": H_assign,
        "entropy_aleatoric": H_aleat,
        "model_uncertainty": MI,
    })
    print(f"\n{sep}\n  DETAILED UNCERTAINTY REPORT (last run)\n{sep}")
    print(f"  (Based on {N_EVAL_PASSES} MC passes, Dropout only + label alignment)\n")
    print(f"  {'Metric':<30} {'Mean':>9} {'Std':>9} {'Min':>9} {'Max':>9}")
    print(f"  {'='*65}")
    for col, label in [
        ("entropy_assignment", "Assignment entropy H[p\xc2\xaf] (bits)"),
        ("entropy_aleatoric",  "Aleatoric entropy E[H] (bits)"),
        ("model_uncertainty",  "Mutual information (bits)"),
        ("p_cluster1",         "Soft assignment p(cluster=1)"),
    ]:
        vals = df[col].values
        print(f"  {label:<30}  {vals.mean():>9.4f}  {vals.std():>9.4f}  "
              f"{vals.min():>9.4f}  {vals.max():>9.4f}")

    df_sorted = df.sort_values("entropy_assignment", ascending=False).reset_index(drop=True)
    print(f"\n  Top-10 most uncertain subjects:")
    print(df_sorted.head(10)[["sample_id", "true_label", "cluster_assignment",
                              "p_cluster0", "p_cluster1", "entropy_assignment"]].to_string(index=False))

    df.to_csv("arma_modularity_uncertainty_last_run.csv", index=False, float_format="%.6f")
    print(f"\n   Saved per-subject uncertainty to arma_modularity_uncertainty_last_run.csv")


print(f"\n{'='*60}")
print(f"  ARMA (modularity only) - Final Summary ({NUM_RUNS} runs, weighted metrics)")
print(f"{'='*60}")
print(f"  ACC (unweighted)        : {acc_vals.mean():.4f} \xc2\xb1 {acc_vals.std():.4f}")
print(f"  Weighted Precision      : {prec_w_vals.mean():.4f} \xc2\xb1 {prec_w_vals.std():.4f}")
print(f"  Weighted Recall         : {rec_w_vals.mean():.4f} \xc2\xb1 {rec_w_vals.std():.4f}")
print(f"  Weighted F1             : {f1_w_vals.mean():.4f} \xc2\xb1 {f1_w_vals.std():.4f}")
print(f"  NMI                     : {nmi_vals.mean():.4f} \xc2\xb1 {nmi_vals.std():.4f}")
print(f"  ARI                     : {ari_vals.mean():.4f} \xc2\xb1 {ari_vals.std():.4f}")
print(f"  Mean Assignment Entropy : {np.mean([r['mean_entropy'] for r in all_results]):.4f} \xc2\xb1 "
      f"{np.std([r['mean_entropy'] for r in all_results]):.4f} bits")
print(f"  Log Loss                : {np.mean([r['log_loss'] for r in all_results]):.4f} \xc2\xb1 "
      f"{np.std([r['log_loss'] for r in all_results]):.4f}")
print(f"\n  Uncertainty: MC with Dropout only (BatchNorm frozen) + label alignment ({N_EVAL_PASSES} passes)")


df_all = pd.DataFrame(all_results)
df_all.to_csv("arma_modularity_10run_weighted_summary.csv", index=False, float_format="%.6f")
print(f"\n   Saved 10-run weighted summary -> arma_modularity_10run_weighted_summary.csv")
print(f"{'='*60}\n  ANALYSIS COMPLETE\n{'='*60}")
