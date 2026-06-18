import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix,
                             adjusted_rand_score,
                             normalized_mutual_info_score,
                             adjusted_mutual_info_score,
                             silhouette_score,
                             davies_bouldin_score)
from contextlib import contextmanager
import random
import copy
import pandas as pd


# ─── CONFIG ───────────────────────────────────────────────────────────────────
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
FEATS_DIM  = 180
K          = 2
ACTIV      = "ELU"            # activation for GCN (ReLU, ELU, etc.)
ALPHA      = 0.92              # graph construction threshold
CUT        = 0                 # use cut loss (0) or modularity (1)
TAU        = 0.07              # temperature for contrastive loss
BETA       = 0.5               # balance in contrastive loss
EMA_DECAY  = 0.7               # EMA decay for target encoder
LAMBDA_CON = 4                 # weight for contrastive loss
NUM_EPOCHS = 2500              # training epochs
T_STRUCT   = 2.0               # temperature for structural loss softmax
C_ENTROPY  = 0.005             # entropy regularisation weight
N_EVAL_PASSES = 30             # MC passes for evaluation
NUM_GCN_LAYERS = 3             # number of GCN layers (replaces ARMA layers)
HIDDEN_DIM = 256               # hidden dimension for GCN and MLP

# ─── DATA ─────────────────────────────────────────────────────────────────────
cn_data  = np.load("/home/snu/Downloads/Histogram_CN_FA_20bin_updated.npy",  allow_pickle=True)
mci_data = np.load("/home/snu/Downloads/Histogram_MCI_FA_20bin_updated.npy", allow_pickle=True)

X = np.vstack([cn_data, mci_data])
y = np.hstack([np.zeros(cn_data.shape[0], dtype=np.int64),
               np.ones(mci_data.shape[0],  dtype=np.int64)])

np.random.seed(42)
perm = np.random.permutation(X.shape[0])
X, y = X[perm], y[perm]
print(f"Features: {X.shape}, Labels: {y.shape} (CN: {np.sum(y==0)}, MCI: {np.sum(y==1)})")
print("NOTE: Diagnostic labels are held out and used ONLY for post-hoc external validation.")

features_np = X.astype(np.float32)


# ─── GRAPH UTILITIES ──────────────────────────────────────────────────────────
def create_adj(features, cut, alpha=1.0):
    F_ = features / np.linalg.norm(features, axis=1, keepdims=True)
    W  = np.dot(F_, F_.T)
    if cut == 0:
        W = np.where(W >= alpha, 1, 0).astype(np.float32)
        W = (W / W.max()).astype(np.float32)
    else:
        W = (W * (W >= alpha)).astype(np.float32)
    return W

def edge_index_from_dense(W):
    r, c = np.nonzero(W > 0)
    return np.vstack([r, c]).astype(np.int64), W[r, c].astype(np.float32)

def aug_edge(ei, drop=0.2, seed=None):
    rng = np.random.default_rng(seed)
    return ei[:, rng.random(ei.shape[1]) >= drop]

def to_data(feats, ei, device):
    return Data(x=torch.from_numpy(feats).float().to(device),
                edge_index=torch.from_numpy(ei.astype(np.int64)).long().to(device))


W0               = create_adj(features_np, CUT, ALPHA)
A1               = torch.from_numpy(W0).float().to(DEVICE)
edge_index_np, _ = edge_index_from_dense(W0)
data0            = to_data(features_np, edge_index_np, DEVICE)
print("Graph:", data0)


# ─── LOSSES (unchanged from ARMA version) ─────────────────────────────────────
def jsd_loss(p, q, tau=0.07, eps=1e-8):
    p_ = F.softmax(p / tau, dim=-1) + eps
    q_ = F.softmax(q / tau, dim=-1) + eps
    m  = 0.5 * (p_ + q_)
    kl = lambda a, b: (a * (a / b).log()).sum(dim=-1)
    return (0.5 * (kl(p_, m) + kl(q_, m)) / np.log(2)).mean()

def contrastive_loss(h1, h2, z1, z2, beta=0.5, tau=0.07):
    l1 = beta * jsd_loss(h1, h2, tau) + (1 - beta) * jsd_loss(h1, z2, tau)
    l2 = beta * jsd_loss(h2, h1, tau) + (1 - beta) * jsd_loss(h2, z1, tau)
    return l1, l2


# ─── MODEL: GCN ENCODER (replaces ARMAEncoder) ────────────────────────────────
ACTIVATIONS = {"RELU": F.relu, "ELU": F.elu, "SELU": F.selu, "GELU": F.gelu}

class GCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, activ="ELU", num_layers=3, dropout=0.3):
        super().__init__()
        self.act = ACTIVATIONS.get(activ, F.relu)
        self.drop = nn.Dropout(dropout)

        layers = []
        bn_layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            layers.append(GCNConv(in_dim, hidden_dim))
            bn_layers.append(nn.BatchNorm1d(hidden_dim))
        self.convs = nn.ModuleList(layers)
        self.bns = nn.ModuleList(bn_layers)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, data, return_intermediates=False):
        x, ei = data.x, data.edge_index
        intermediates = []
        for conv, bn in zip(self.convs, self.bns):
            x = self.drop(self.act(bn(conv(x, ei))))
            if return_intermediates:
                intermediates.append(x.detach().cpu().numpy())
        out = self.proj(x)
        return (out, intermediates) if return_intermediates else out


class MLP(nn.Module):
    def __init__(self, inp, out, hid):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, hid), nn.BatchNorm1d(hid), nn.PReLU(),
            nn.Dropout(0.4),
            nn.Linear(hid, out)
        )
    def forward(self, x): return self.net(x)


class EMA:
    def __init__(self, beta): self.beta = beta
    def update(self, old, new):
        return new if old is None else old * self.beta + (1 - self.beta) * new

def update_ema(ema, target, online):
    for tp, op in zip(target.parameters(), online.parameters()):
        tp.data = ema.update(tp.data, op.data)


class GCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_clusters, device, activ="ELU",
                 ema_decay=0.7, cut=True, beta=0.5, tau=0.07,
                 num_gcn_layers=3, T_struct=2.0):
        super().__init__()
        self.device = device
        self.num_clusters = num_clusters
        self.cut = cut
        self.beta, self.tau, self.T_struct = beta, tau, T_struct

        self.online_encoder = GCNEncoder(input_dim, hidden_dim, activ,
                                         num_layers=num_gcn_layers)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.online_predictor = MLP(hidden_dim, num_clusters, hidden_dim)
        self.ema = EMA(ema_decay)

    def update_ma(self):
        update_ema(self.ema, self.target_encoder, self.online_encoder)

    def forward(self, d1, d2):
        h1, h2 = self.online_encoder(d1), self.online_encoder(d2)
        lg1, lg2 = self.online_predictor(h1), self.online_predictor(h2)
        with torch.no_grad():
            z1 = self.target_encoder(d1).detach()
            z2 = self.target_encoder(d2).detach()
        l1, l2 = contrastive_loss(h1, h2, z1, z2, self.beta, self.tau)
        return lg1, lg2, l1, l2

    def struct_loss(self, A, S):
        if self.cut:
            return self._cut_loss(A, S)
        return self._modularity_loss(A, S)

    def _cut_loss(self, A, S):
        S  = F.softmax(S / self.T_struct, dim=1)
        Ap = (A @ S).t() @ S
        D  = torch.diag(A.sum(dim=-1))
        Dp = (D @ S).t() @ S
        mc = -(Ap.trace() / Dp.trace())
        SS = S.t() @ S
        I  = torch.eye(self.num_clusters, device=self.device)
        oc = torch.norm(SS / SS.norm() - I / I.norm())
        return mc + oc

    def _modularity_loss(self, A, S):
        C = F.softmax(S, dim=1)
        d = A.sum(dim=1); m = A.sum()
        B = A - torch.ger(d, d) / (2 * m)
        k = torch.tensor(self.num_clusters, device=self.device, dtype=torch.float32)
        mod  = (-1 / (2 * m)) * torch.trace(C.t() @ B @ C)
        coll = (k.sqrt() / S.shape[0]) * torch.norm(C.sum(dim=0), p='fro') - 1
        return mod + coll


# ─── CLUSTERING METRICS (unchanged) ───────────────────────────────────────────
def compute_clustering_metrics(embeddings: np.ndarray,
                               pred_labels: np.ndarray,
                               true_labels: np.ndarray,
                               space_name: str = "") -> dict:
    unique_preds = np.unique(pred_labels)
    n_valid = sum((pred_labels == c).sum() >= 2 for c in unique_preds)
    can_geom = (len(unique_preds) >= 2) and (n_valid == len(unique_preds))

    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels, average_method='arithmetic')
    ami = adjusted_mutual_info_score(true_labels, pred_labels, average_method='arithmetic')

    if can_geom:
        sil = silhouette_score(embeddings, pred_labels, metric='euclidean')
        db  = davies_bouldin_score(embeddings, pred_labels)
    else:
        sil = float('nan')
        db  = float('nan')
    return dict(space=space_name, ARI=ari, NMI=nmi, AMI=ami,
                Silhouette=sil, DaviesBouldin=db)


def evaluate_clustering_from_mc(model, feats, ei, y_true, yp, logits_mean,
                                device, prefix=""):
    model.eval()
    d = to_data(feats, ei, device)
    with torch.no_grad():
        hidden = model.online_encoder(d).cpu().numpy()

    logits_emb = logits_mean
    ari_direct  = adjusted_rand_score(y_true, yp)
    ari_flipped = adjusted_rand_score(y_true, 1 - yp)
    if ari_flipped > ari_direct:
        yp = 1 - yp
        logits_emb = logits_emb[:, ::-1]

    results = [
        compute_clustering_metrics(logits_emb, yp, y_true, space_name="logit (MC avg)"),
        compute_clustering_metrics(hidden, yp, y_true, space_name="hidden (single pass)"),
    ]

    sep = "─" * 72
    header = (f"  {'Space':<22} {'ARI':>8} {'NMI':>8} {'AMI':>8} "
              f"{'Silhouette':>12} {'DaviesBouldin':>14}")
    print(f"\n{sep}\n  CLUSTERING METRICS (MC predictions){' '+prefix if prefix else ''}\n{sep}")
    print(header); print(f"  {sep}")
    for r in results:
        sil_str = f"{r['Silhouette']:>12.4f}" if not np.isnan(r['Silhouette']) else "         N/A"
        db_str  = f"{r['DaviesBouldin']:>14.4f}" if not np.isnan(r['DaviesBouldin']) else "           N/A"
        print(f"  {r['space']:<22} {r['ARI']:>8.4f} {r['NMI']:>8.4f} {r['AMI']:>8.4f} "
              f"{sil_str}{db_str}")
    print(f"  {sep}")
    return results


def print_clustering_summary(all_records, depth_label="GCN layers"):
    sep = "─" * 90
    print(f"\n{sep}")
    print(f"  CLUSTERING METRICS SUMMARY  [{depth_label}]  (mean ± std, {len(all_records)} seeds)")
    print(f"{sep}")

    spaces = [("logit (MC avg)",       "Logit space (MC-averaged, K-dim)"),
              ("hidden (single pass)", "Hidden space (256-dim encoder output)")]

    for space_key, space_label in spaces:
        print(f"\n  {space_label}")
        print(f"  {'Metric':<16} {'Mean':>9} {'Std':>9} {'Min':>9} {'Max':>9}")
        print(f"  {'─'*55}")
        for metric, hi_lo in [("ARI", "↑"), ("NMI", "↑"), ("AMI", "↑"),
                               ("Silhouette", "↑"), ("DaviesBouldin", "↓")]:
            key   = f"{metric}_{space_key}"
            vals  = np.array([r.get(key, np.nan) for r in all_records])
            valid = vals[~np.isnan(vals)]
            if len(valid) == 0:
                print(f"  {metric+' '+hi_lo:<16}  {'N/A':>9}")
                continue
            print(f"  {metric+' '+hi_lo:<16}  {valid.mean():>9.4f}  {valid.std():>9.4f}"
                  f"  {valid.min():>9.4f}  {valid.max():>9.4f}")
    print(f"\n{sep}")


# ─── CLUSTER ASSIGNMENT UNCERTAINTY (identical to ARMA version) ───────────────
def softmax_np(logits):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def entropy_bits(p):
    p = np.clip(p, 1e-12, 1.0)
    return -np.sum(p * np.log2(p), axis=1)

def compute_cluster_assignment_uncertainty(logits_stack):
    N, K, P = logits_stack.shape
    probs = np.stack([softmax_np(logits_stack[:, :, p]) for p in range(P)], axis=2)
    p_mean = probs.mean(axis=2)
    H_assign = entropy_bits(p_mean)
    H_aleat = np.stack([entropy_bits(probs[:, :, p]) for p in range(P)], axis=1).mean(axis=1)
    MI = np.clip(H_assign - H_aleat, 0, None)
    return H_assign, H_aleat, MI, p_mean

def print_cluster_uncertainty_report(logits_stack, yp, y_true,
                                     n_passes, sample_ids=None):
    sep = "─" * 72
    H_assign, H_aleat, MI, p_mean = compute_cluster_assignment_uncertainty(logits_stack)

    label_map = {0: "CN", 1: "MCI"}
    if sample_ids is None:
        sample_ids = list(range(len(y_true)))

    df = pd.DataFrame({
        "sample_id":          sample_ids,
        "true_label":         [label_map[int(l)] for l in y_true],
        "cluster_assignment": [label_map[int(l)] for l in yp],
        "correct_ext_valid":  (yp == y_true).astype(int),
        "p_cluster0":         p_mean[:, 0],
        "p_cluster1":         p_mean[:, 1],
        "entropy_assignment": H_assign,
        "entropy_aleatoric":  H_aleat,
        "model_uncertainty":  MI,
    })

    print(f"\n{sep}\n  CLUSTER ASSIGNMENT UNCERTAINTY\n{sep}")
    print(f"  (Entropy of soft cluster assignments — {n_passes} MC passes)\n")
    print(f"  {'Metric':<30} {'Mean':>9} {'Std':>9} {'Min':>9} {'Max':>9}")
    print(f"  {'─'*65}")
    for col, label in [
        ("entropy_assignment", "Assignment entropy H[p̄] (bits)"),
        ("entropy_aleatoric",  "Aleatoric entropy E[H] (bits)"),
        ("model_uncertainty",  "Model uncertainty MI (bits)"),
        ("p_cluster1",        "Soft assignment p(cluster=1)"),
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

    low_unc  = (H_assign < 0.3).sum()
    high_unc = (H_assign > 0.7).sum()
    print(f"\n  Assignment confidence summary:")
    print(f"    Low entropy  (< 0.3 bits, high-confidence): {low_unc}/{len(y_true)} "
          f"({100*low_unc/len(y_true):.1f}%)")
    print(f"    High entropy (> 0.7 bits, ambiguous):        {high_unc}/{len(y_true)} "
          f"({100*high_unc/len(y_true):.1f}%)")
    print(f"\n  Mean assignment entropy: {H_assign.mean():.4f} ± {H_assign.std():.4f} bits")
    print(f"  (Max possible entropy for K=2: {np.log2(2):.4f} bits)")
    print(f"{sep}")
    return df


# ─── MC EVALUATION (feature masking + edge dropping) ─────────────────────────
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

def evaluate_model(model, feats, ei, y, device, n_passes=30, seed_base=9999):
    all_logits = []
    with mc_dropout_mode(model):
        with torch.no_grad():
            for i in range(n_passes):
                rng = np.random.default_rng(seed_base + i)
                fa  = feats * (rng.random(feats.shape) >= 0.2).astype(np.float32)
                dr  = 0.15 + 0.10 * (i % 3)
                ei_ = aug_edge(ei, dr, seed=seed_base + i)
                d   = to_data(fa, ei_, device)
                lg  = model.online_predictor(model.online_encoder(d)).cpu().numpy()
                all_logits.append(lg)

    logits_stack = np.stack(all_logits, axis=2)
    logits_mean  = logits_stack.mean(axis=2)
    yp = np.argmax(logits_mean, axis=1)

    a  = accuracy_score(y, yp)
    ai = accuracy_score(y, 1 - yp)
    if ai > a:
        yp           = 1 - yp
        logits_mean  = logits_mean[:, ::-1].copy()
        logits_stack = logits_stack[:, ::-1, :].copy()

    ypp = F.softmax(torch.from_numpy(logits_mean), dim=1).numpy()
    return yp, ypp, logits_stack, logits_mean


# ─── TRAINING LOOP (one run) ──────────────────────────────────────────────────
def run_once(seed_offset=0, verbose=False, return_probs=False,
             n_eval_passes=30, num_gcn_layers=NUM_GCN_LAYERS):
    np.random.seed(42 + seed_offset)
    random.seed(42 + seed_offset)
    torch.manual_seed(42 + seed_offset)

    model = GCNModel(FEATS_DIM, HIDDEN_DIM, K, DEVICE, ACTIV,
                     ema_decay=EMA_DECAY, cut=CUT, beta=BETA, tau=TAU,
                     num_gcn_layers=num_gcn_layers, T_struct=T_STRUCT).to(DEVICE)

    opt = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sch = CosineAnnealingLR(opt, T_max=NUM_EPOCHS, eta_min=1e-6)

    for ep in range(NUM_EPOCHS):
        rng = np.random.default_rng(ep + seed_offset)

        fa1 = features_np * (rng.random(features_np.shape) >= 0.2).astype(np.float32)
        af2 = features_np.copy()
        n_, d_ = af2.shape
        fi = rng.choice(n_ * d_, size=int(n_ * d_ * 0.2), replace=False)
        af2[fi // d_, fi % d_] = 0.0
        fa2 = af2.astype(np.float32)

        d1 = to_data(fa1, aug_edge(edge_index_np, 0.2, seed=ep + seed_offset), DEVICE)
        d2 = to_data(fa2, aug_edge(edge_index_np, 0.2, seed=ep + seed_offset + 999), DEVICE)

        model.train()
        opt.zero_grad()
        lg1, lg2, l1, l2 = model(d1, d2)

        p1       = F.softmax(lg1, dim=1)
        entropy  = -(p1 * torch.log(p1 + 1e-8)).sum(dim=1).mean()
        cont     = (l1 + l2) / 2.0
        loss     = model.struct_loss(A1, lg1) + LAMBDA_CON * cont - C_ENTROPY * entropy

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        sch.step()
        model.update_ma()

        if verbose and ep % 500 == 0:
            print(f"  Epoch {ep:4d} | Total: {loss.item():.4f} | "
                  f"KL-Cont: {cont.item():.6f} | Entropy: {entropy.item():.4f} | "
                  f"LR: {sch.get_last_lr()[0]:.2e}")

    yp, ypp, logits_stack, logits_mean = evaluate_model(
        model, features_np, edge_index_np, y, DEVICE,
        n_passes=n_eval_passes, seed_base=9999 + seed_offset
    )

    # Return weighted metrics as well
    acc = accuracy_score(y, yp)
    prec_w = precision_score(y, yp, average='weighted', zero_division=0)
    rec_w  = recall_score(y, yp, average='weighted', zero_division=0)
    f1_w   = f1_score(y, yp, average='weighted', zero_division=0)
    metrics_weighted = (acc, prec_w, rec_w, f1_w)

    if return_probs:
        return metrics_weighted, yp, ypp, logits_stack, logits_mean, model
    return metrics_weighted, model


# ─── MAIN: RUN FULL EVALUATION (single run, depth ablation, 10 seeds, etc.) ───
METRIC_NAMES = ["Accuracy", "Weighted Precision", "Weighted Recall", "Weighted F1"]
SEP = "═" * 72

print(f"\n{SEP}")
print(f"  PART A — FIRST RUN (GCN with {NUM_GCN_LAYERS} layers) – CN vs MCI")
print(f"  NOTE: Model trained WITHOUT diagnostic labels.")
print(f"  Labels used ONLY for post-hoc external validation.")
print(f"{SEP}")

metrics0, yp0, ypp0, logits_stack0, logits_mean0, model3 = run_once(
    seed_offset=0, verbose=True, return_probs=True,
    n_eval_passes=N_EVAL_PASSES, num_gcn_layers=NUM_GCN_LAYERS
)

print("\n── Post-hoc External Validation (single run) ────────")
print("  (Diagnostic labels NOT used during training)")
for n, v in zip(METRIC_NAMES, metrics0):
    print(f"  {n:<20}: {v:.4f}")

lm = logits_stack0.mean(axis=2)
diff = lm[:, 1] - lm[:, 0]
print(f"\n── Logit diagnostics ───────────────────────────────")
print(f"  mean={lm.mean():.2f}  std={lm.std():.2f}  min={lm.min():.2f}  max={lm.max():.2f}")
print(f"  Logit diff (MCI-CN): mean={diff.mean():.2f}  std={diff.std():.2f}  "
      f"min={diff.min():.2f}  max={diff.max():.2f}")

# Uncertainty report for first run
df_unc0 = print_cluster_uncertainty_report(
    logits_stack0, yp0, y,
    n_passes=N_EVAL_PASSES,
    sample_ids=list(range(len(y)))
)
df_unc0.to_csv("gcn_cluster_uncertainty_part_a.csv", index=False, float_format="%.6f")
print(f"\n  CSV → gcn_cluster_uncertainty_part_a.csv  ({len(df_unc0)} subjects)")

# Clustering metrics for first run (will use weighted metrics indirectly)
clustering_results_A = evaluate_clustering_from_mc(
    model3, features_np, edge_index_np, y, yp0, logits_mean0, DEVICE,
    prefix="Part A — GCN, seed 0"
)

# ─── DEPTH ABLATION (vary number of GCN layers) ──────────────────────────────
print(f"\n{SEP}\n  PART B — DEPTH ABLATION (1 / 2 / 3 GCN layers)\n{SEP}")
ablation_results = {}
trained_models = {}
ablation_clustering = {1: [], 2: [], 3: []}

for n_layers in [1, 2, 3]:
    print(f"\n  ── Depth = {n_layers} GCN layer(s) ──")
    seed_records = []
    for seed in range(3):
        print(f"    Seed {seed} ... ", end="", flush=True)
        (acc_w, prec_w, rec_w, f1_w), model_i = run_once(
            seed_offset=seed, num_gcn_layers=n_layers,
            n_eval_passes=N_EVAL_PASSES, return_probs=False
        )
        yp_s, ypp_s, logits_stack_s, logits_mean_s = evaluate_model(
            model_i, features_np, edge_index_np, y, DEVICE,
            n_passes=N_EVAL_PASSES, seed_base=9999 + seed
        )

        cl_res = evaluate_clustering_from_mc(
            model_i, features_np, edge_index_np, y, yp_s, logits_mean_s, DEVICE,
            prefix=f"{n_layers}L seed {seed} (quiet)"
        )
        cl_flat = {}
        for r in cl_res:
            sp = r['space']
            for metric in ['ARI', 'NMI', 'AMI', 'Silhouette', 'DaviesBouldin']:
                cl_flat[f"{metric}_{sp}"] = r[metric]
        ablation_clustering[n_layers].append(cl_flat)

        H_assign, _, _, _ = compute_cluster_assignment_uncertainty(logits_stack_s)
        seed_records.append(dict(
            depth=n_layers, seed=seed,
            acc=acc_w, prec=prec_w, rec=rec_w, f1=f1_w,
            mean_entropy=H_assign.mean(),
            std_entropy=H_assign.std(),
            mean_conf=ypp_s.max(axis=1).mean(),
        ))
        trained_models[n_layers] = model_i
        print(f"Acc={acc_w:.3f}  MeanEntropy={H_assign.mean():.4f}")
    ablation_results[n_layers] = seed_records

# ─── 10-SEED EVALUATION (fixed 3 GCN layers) ─────────────────────────────────
print(f"\n{SEP}\n  PART C — 10-SEED EVALUATION (3 GCN layers, CN vs MCI)\n{SEP}")
print("  Diagnostic labels used ONLY for post-hoc external validation.\n")

col_w = "─" * 110
print(f"  {'Seed':>4}  {'Acc':>7}  {'Prec_w':>8}  {'Rec_w':>8}  {'F1_w':>8}  "
      f"{'MeanEntropy':>13}  {'StdEntropy':>11}  {'TN':>5}  {'FP':>5}  {'FN':>5}  {'TP':>5}")
print(f"  {col_w}")

calibration_rows = []
seed_clustering_records = []
weighted_metrics_arrays = []  # to collect weighted metrics for final table

for i in range(10):
    m_tup, yp_i, ypp_i, logits_stack_i, logits_mean_i, model_i = run_once(
        seed_offset=i, return_probs=True,
        n_eval_passes=N_EVAL_PASSES, num_gcn_layers=3
    )
    acc_i, prec_w_i, rec_w_i, f1_w_i = m_tup
    H_assign_i, _, _, _ = compute_cluster_assignment_uncertainty(logits_stack_i)
    mean_ent_i = float(H_assign_i.mean())
    std_ent_i  = float(H_assign_i.std())
    tn_i, fp_i, fn_i, tp_i = confusion_matrix(y, yp_i).ravel()

    calibration_rows.append(dict(
        seed=i, acc=acc_i, prec=prec_w_i, rec=rec_w_i, f1=f1_w_i,
        mean_entropy=mean_ent_i, std_entropy=std_ent_i,
        tn=int(tn_i), fp=int(fp_i), fn=int(fn_i), tp=int(tp_i),
    ))

    # Also collect ARI, NMI, AMI, Silhouette, DB from clustering evaluation
    cl_res_i = evaluate_clustering_from_mc(
        model_i, features_np, edge_index_np, y, yp_i, logits_mean_i, DEVICE,
        prefix=f"Part C seed {i} (quiet)"
    )
    cl_flat_i = {}
    for r in cl_res_i:
        sp = r['space']
        for metric in ['ARI', 'NMI', 'AMI', 'Silhouette', 'DaviesBouldin']:
            cl_flat_i[f"{metric}_{sp}"] = r[metric]
    seed_clustering_records.append(cl_flat_i)

    weighted_metrics_arrays.append([acc_i, prec_w_i, rec_w_i, f1_w_i])

    print(f"  {i:>4}  {acc_i:>7.4f}  {prec_w_i:>8.4f}  {rec_w_i:>8.4f}  {f1_w_i:>8.4f}  "
          f"{mean_ent_i:>13.4f}  {std_ent_i:>11.4f}  "
          f"{tn_i:>5}  {fp_i:>5}  {fn_i:>5}  {tp_i:>5}")

# ──────────────────────────────────────────────────────────────────────────────
#  HORIZONTAL TABLE WITH MEAN ± STD (weighted metrics over 10 runs)
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  HORIZONTAL TABLE FOR MANUSCRIPT (mean ± std over 10 runs)")
print("  Metrics: Accuracy, Weighted Precision, Weighted Recall, Weighted F1, NMI, ARI, AMI, Silhouette, Davies‑Bouldin")
print(SEP)

# Extract arrays for the nine metrics using the 10-seed evaluation
acc_vals = np.array([r['acc'] for r in calibration_rows])
prec_w_vals = np.array([r['prec'] for r in calibration_rows])
rec_w_vals = np.array([r['rec'] for r in calibration_rows])
f1_w_vals = np.array([r['f1'] for r in calibration_rows])

# Extract ARI, NMI, AMI, Silhouette, DaviesBouldin from seed_clustering_records
# (using "logit (MC avg)" space as the primary clustering result)
ari_vals = np.array([r.get('ARI_logit (MC avg)', np.nan) for r in seed_clustering_records])
nmi_vals = np.array([r.get('NMI_logit (MC avg)', np.nan) for r in seed_clustering_records])
ami_vals = np.array([r.get('AMI_logit (MC avg)', np.nan) for r in seed_clustering_records])
sil_vals = np.array([r.get('Silhouette_logit (MC avg)', np.nan) for r in seed_clustering_records])
db_vals  = np.array([r.get('DaviesBouldin_logit (MC avg)', np.nan) for r in seed_clustering_records])

metrics_names = [
    "Accuracy", "Prec (weighted)", "Recall (weighted)", "F1 (weighted)",
    "NMI", "ARI", "AMI", "Silhouette", "Davies‑Bouldin"
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

# Print tab‑separated line (easy to copy)
print("\nMethod\t" + "\t".join(metrics_names))
row = "GCN"
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
latex_row = "GCN"
for m, s in zip(means, stds):
    if np.isnan(m):
        latex_row += " & N/A±N/A"
    else:
        latex_row += f" & ${m:.4f}\\pm{s:.4f}$"
latex_row += " \\\\"
print(latex_row)

# ─── 10-RUN SUMMARY (mean ± std) ──────────────────────────────────────────────
results_np = np.array([[r['acc'], r['prec'], r['rec'], r['f1'],
                        r['mean_entropy'], r['std_entropy']] for r in calibration_rows])
counts_np = np.array([[r['tn'], r['fp'], r['fn'], r['tp']] for r in calibration_rows], dtype=float)

sep100 = "─" * 100
print(f"\n{sep100}")
print(f"  10-RUN SUMMARY  (mean ± std)                         [GCN, 3 layers, CN vs MCI]")
print(f"  NOTE: Diagnostic labels used only for post-hoc external validation.")
print(f"{sep100}")

ext_val_labels = [
    ("Accuracy",      "Post-hoc external validation (weighted)"),
    ("Weighted Precision", "Post-hoc external validation"),
    ("Weighted Recall",    "Post-hoc external validation"),
    ("Weighted F1",        "Post-hoc external validation"),
    ("Mean Entropy",  "Mean cluster assignment entropy (bits)"),
    ("Std Entropy",   "Std of assignment entropy across subjects"),
]

print(f"\n  A.  EXTERNAL VALIDATION & ASSIGNMENT UNCERTAINTY")
print(f"  {'Metric':<18}  {'Mean':>9}  {'Std':>9}  {'Min':>9}  {'Max':>9}  Note")
print(f"  {sep100}")
for col_idx, (label, note) in enumerate(ext_val_labels):
    vals = results_np[:, col_idx]
    print(f"  {label:<18}  {vals.mean():>9.4f}  {vals.std():>9.4f}  "
          f"{vals.min():>9.4f}  {vals.max():>9.4f}  {note}")

count_labels = [
    ("TN  (CN→C0)",   "CN assigned to Cluster 0 — post-hoc"),
    ("FP  (CN→C1)",   "CN assigned to Cluster 1 — post-hoc"),
    ("FN  (MCI→C0)",  "MCI assigned to Cluster 0 — post-hoc"),
    ("TP  (MCI→C1)",  "MCI assigned to Cluster 1 — post-hoc"),
]
print(f"\n  B.  CONFUSION MATRIX COUNTS (post-hoc external validation only)")
print(f"  {'Group':<15}  {'Mean':>7}  {'Std':>7}  {'Min':>5}  {'Max':>5}  Note")
print(f"  {sep100}")
for col_idx, (label, note) in enumerate(count_labels):
    vals = counts_np[:, col_idx]
    print(f"  {label:<15}  {vals.mean():>7.1f}  {vals.std():>7.2f}  "
          f"{vals.min():>5.0f}  {vals.max():>5.0f}  {note}")

print(f"\n  C.  CLUSTER STABILITY ACROSS 10 SEEDS")
print(f"  {'Metric':<20}  {'Mean':>9}  {'Std':>9}  Note")
print(f"  {sep100}")
print(f"  {'Weighted F1':<20}  {results_np[:,3].mean():>9.4f}  {results_np[:,3].std():>9.4f}  "
      f"Consistency of weighted F1")
print(f"  {'Assignment Entropy':<20}  {results_np[:,4].mean():>9.4f}  {results_np[:,4].std():>9.4f}  "
      f"Low std = stable assignment confidence")
print(f"  {sep100}\n")

# ─── CLUSTERING METRICS SUMMARY ──────────────────────────────────────────────
print(f"\n{SEP}\n  PART D — CLUSTERING METRICS (ARI / NMI / AMI / Silhouette / Davies-Bouldin)\n{SEP}")
print(f"\n  ── 10-SEED EVALUATION (3 GCN layers) ──")
print_clustering_summary(seed_clustering_records, depth_label="3 GCN layers — 10 seeds")

print(f"\n  ── DEPTH ABLATION (3 seeds per depth) ──")
for n_layers in [1, 2, 3]:
    print_clustering_summary(
        ablation_clustering[n_layers],
        depth_label=f"{n_layers} GCN layer{'s' if n_layers>1 else ''} — 3 seeds"
    )

# ─── FINAL SUMMARY ───────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  GCN — Final Summary (10 runs, 3 layers, weighted metrics)")
print(f"{'='*60}")
print(f"  ACC (unweighted)        : {acc_vals.mean():.4f} ± {acc_vals.std():.4f}")
print(f"  Weighted Precision      : {prec_w_vals.mean():.4f} ± {prec_w_vals.std():.4f}")
print(f"  Weighted Recall         : {rec_w_vals.mean():.4f} ± {rec_w_vals.std():.4f}")
print(f"  Weighted F1             : {f1_w_vals.mean():.4f} ± {f1_w_vals.std():.4f}")
print(f"  NMI                     : {nmi_vals.mean():.4f} ± {nmi_vals.std():.4f}")
print(f"  ARI                     : {ari_vals.mean():.4f} ± {ari_vals.std():.4f}")
print(f"  Mean Assignment Entropy : {np.mean(results_np[:,4]):.4f} ± {np.std(results_np[:,4]):.4f} bits")
print(f"  (Max possible for K=2   : {np.log2(2):.4f} bits)")
print(f"\n  Uncertainty: MC dropout + feature/edge perturbations ({N_EVAL_PASSES} passes)")
print(f"\n  ✓ Saved cluster uncertainty CSV → gcn_cluster_uncertainty_part_a.csv")
print(f"\n{'='*60}\n  ANALYSIS COMPLETE\n{'='*60}")

import umap

# __PART H — t-SNE VISUALIZATION OF CLUSTERS ─────────────────────────────────
print(f"\n{SEP}\n  PART H — t-SNE VISUALIZATION OF DISCOVERED CLUSTERS\n{SEP}")
print("  Visualizing the hidden space embeddings to assess cluster separation.\n")

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist

# Set publication-quality style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300

# Get hidden embeddings from your trained model
model3.eval()
d = to_data(features_np, edge_index_np, DEVICE)
with torch.no_grad():
    hidden_embeddings = model3.online_encoder(d).cpu().numpy()  # (N, 256)

print(f"  Hidden embedding shape: {hidden_embeddings.shape}")

# Get uncertainty values
H_assign, H_aleat, MI, p_mean = compute_cluster_assignment_uncertainty(logits_stack0)
high_uncertainty_mask = H_assign > 0.7
low_uncertainty_mask = H_assign <= 0.3

# Define cluster assignments and masks
cluster_assignments = yp0
cluster0_mask = (cluster_assignments == 0)
cluster1_mask = (cluster_assignments == 1)

# Compute t-SNE once with best perplexity
best_perp = 50
print(f"  Computing t-SNE with perplexity={best_perp}...", end=" ", flush=True)
tsne = TSNE(n_components=2, random_state=42, perplexity=best_perp, init='pca', max_iter=1000)
tsne_results = tsne.fit_transform(hidden_embeddings)
print("done")

# ============================================================================
# FIGURE 1: t-SNE colored by cluster assignment
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Plot clusters
ax.scatter(tsne_results[cluster0_mask, 0], tsne_results[cluster0_mask, 1],
           c='#2E86AB', label='Cluster 0 (Healthier phenotype)',
           alpha=0.7, s=60, edgecolors='white', linewidth=0.5, zorder=2)
ax.scatter(tsne_results[cluster1_mask, 0], tsne_results[cluster1_mask, 1],
           c='#A23B72', label='Cluster 1 (Worse phenotype)',
           alpha=0.7, s=60, edgecolors='white', linewidth=0.5, zorder=2)

# Add cluster centers
cluster0_center = tsne_results[cluster0_mask].mean(axis=0)
cluster1_center = tsne_results[cluster1_mask].mean(axis=0)
ax.scatter(cluster0_center[0], cluster0_center[1], c='darkblue', s=200,
           edgecolors='black', linewidth=2, marker='*', label='Cluster 0 center', zorder=3)
ax.scatter(cluster1_center[0], cluster1_center[1], c='darkred', s=200,
           edgecolors='black', linewidth=2, marker='*', label='Cluster 1 center', zorder=3)

ax.set_title('t-SNE Visualization of GCN Embeddings\nColored by Cluster Assignment',
             fontsize=14, fontweight='bold')
ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
ax.legend(loc='best', framealpha=0.9)
ax.grid(True, alpha=0.2, linestyle='--')
ax.set_facecolor('#f8f9fa')

plt.tight_layout()
plt.savefig('tsne_by_cluster_assignment.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("  ✓ Saved: tsne_by_cluster_assignment.png")

# ============================================================================
# FIGURE 2: t-SNE colored by true diagnosis (reference only)
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

cn_mask = (y == 0)
mci_mask = (y == 1)

ax.scatter(tsne_results[cn_mask, 0], tsne_results[cn_mask, 1],
           c='#2E86AB', label='CN (True diagnosis)',
           alpha=0.7, s=60, edgecolors='white', linewidth=0.5)
ax.scatter(tsne_results[mci_mask, 0], tsne_results[mci_mask, 1],
           c='#F18F01', label='MCI (True diagnosis)',
           alpha=0.7, s=60, edgecolors='white', linewidth=0.5)

ax.set_title('t-SNE Visualization of GCN Embeddings\nColored by True Diagnosis (Reference Only)',
             fontsize=14, fontweight='bold')
ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
ax.legend(loc='best', framealpha=0.9)
ax.grid(True, alpha=0.2, linestyle='--')
ax.set_facecolor('#f8f9fa')

plt.tight_layout()
plt.savefig('tsne_by_true_diagnosis.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("  ✓ Saved: tsne_by_true_diagnosis.png")

# ============================================================================
# FIGURE 3: t-SNE with uncertainty heatmap (most informative)
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# Create colormap from blue (low entropy) to red (high entropy)
scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1],
                     c=H_assign, cmap='RdYlBu_r',
                     alpha=0.8, s=70, edgecolors='black', linewidth=0.5,
                     vmin=0, vmax=1.0)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=30)
cbar.set_label('Assignment Entropy (bits)', fontsize=12, fontweight='bold')
cbar.ax.tick_params(labelsize=10)

# Mark the ambiguous subject (if any)
if high_uncertainty_mask.sum() > 0:
    ax.scatter(tsne_results[high_uncertainty_mask, 0],
               tsne_results[high_uncertainty_mask, 1],
               c='red', s=200, edgecolors='black', linewidth=2,
               marker='X', label=f'Ambiguous (n={high_uncertainty_mask.sum()}, entropy > 0.7 bits)',
               zorder=5)

# Add cluster centers
ax.scatter(cluster0_center[0], cluster0_center[1], c='darkblue', s=250,
           edgecolors='white', linewidth=2, marker='o', label='Cluster 0 center', zorder=4)
ax.scatter(cluster1_center[0], cluster1_center[1], c='darkred', s=250,
           edgecolors='white', linewidth=2, marker='o', label='Cluster 1 center', zorder=4)

# Add text box with statistics
stats_text = f'Mean Entropy: {H_assign.mean():.4f} \u00b1 {H_assign.std():.4f} bits\n'
stats_text += f'Confident (entropy \u2264 0.3): {low_uncertainty_mask.sum()}/{len(H_assign)} ({100*low_uncertainty_mask.sum()/len(H_assign):.1f}%)\n'
stats_text += f'Ambiguous (entropy > 0.7): {high_uncertainty_mask.sum()}/{len(H_assign)} ({100*high_uncertainty_mask.sum()/len(H_assign):.1f}%)\n'
# Separation ratio will be calculated later, so exclude for now or define a placeholder
# stats_text += f'Separation Ratio: {separation_ratio:.2f}'

ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.set_title('t-SNE Visualization of GCN Embeddings\nColored by Cluster Assignment Uncertainty',
             fontsize=14, fontweight='bold')
ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
ax.legend(loc='lower right', framealpha=0.9)
ax.grid(True, alpha=0.2, linestyle='--')
ax.set_facecolor('#f8f9fa')

plt.tight_layout()
plt.savefig('tsne_uncertainty_heatmap.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("  ✓ Saved: tsne_uncertainty_heatmap.png")

# ============================================================================
# FIGURE 4: Side-by-side comparison (Cluster Assignment vs True Diagnosis)
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: By cluster assignment
ax = axes[0]
ax.scatter(tsne_results[cluster0_mask, 0], tsne_results[cluster0_mask, 1],
           c='#2E86AB', label='Cluster 0', alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
ax.scatter(tsne_results[cluster1_mask, 0], tsne_results[cluster1_mask, 1],
           c='#A23B72', label='Cluster 1', alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
ax.scatter(cluster0_center[0], cluster0_center[1], c='darkblue', s=150,
           edgecolors='black', linewidth=2, marker='*', zorder=3)
ax.scatter(cluster1_center[0], cluster1_center[1], c='darkred', s=150,
           edgecolors='black', linewidth=2, marker='*', zorder=3)
ax.set_title('Colored by Cluster Assignment', fontsize=12, fontweight='bold')
ax.set_xlabel('t-SNE Dimension 1')
ax.set_ylabel('t-SNE Dimension 2')
ax.legend(loc='best')
ax.grid(True, alpha=0.2)

# Right: By true diagnosis
ax = axes[1]
ax.scatter(tsne_results[cn_mask, 0], tsne_results[cn_mask, 1],
           c='#2E86AB', label='CN (True)', alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
ax.scatter(tsne_results[mci_mask, 0], tsne_results[mci_mask, 1],
           c='#F18F01', label='MCI (True)', alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
ax.set_title('Colored by True Diagnosis (Reference Only)', fontsize=12, fontweight='bold')
ax.set_xlabel('t-SNE Dimension 1')
ax.set_ylabel('t-SNE Dimension 2')
ax.legend(loc='best')
ax.grid(True, alpha=0.2)

plt.suptitle(f't-SNE Visualization (perplexity={best_perp})', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('tsne_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("  ✓ Saved: tsne_comparison.png")

# ============================================================================
# FIGURE 5: Entropy distribution histogram
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

n, bins, patches = ax.hist(H_assign, bins=30, edgecolor='black', alpha=0.7, color='steelblue')

# Color bins based on entropy ranges
for i, (patch, bin_edge) in enumerate(zip(patches, bins[:-1])):
    if bin_edge < 0.3:
        patch.set_facecolor('#2E86AB')  # Low entropy - confident
    elif bin_edge > 0.7:
        patch.set_facecolor('#F18F01')  # High entropy - ambiguous
    else:
        patch.set_facecolor('#A23B72')  # Medium entropy

ax.axvline(H_assign.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {H_assign.mean():.4f}')
ax.axvline(0.3, color='green', linestyle=':', linewidth=1.5, label='Confidence threshold (0.3 bits)')
ax.axvline(0.7, color='orange', linestyle=':', linewidth=1.5, label='Ambiguity threshold (0.7 bits)')

ax.set_xlabel('Assignment Entropy (bits)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Cluster Assignment Entropy', fontsize=14, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.2, axis='y')

# Add text annotation
ax.text(0.98, 0.98, f'n = {len(H_assign)}\nMean = {H_assign.mean():.4f}\nStd = {H_assign.std():.4f}',
        transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('entropy_distribution.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("  ✓ Saved: entropy_distribution.png")

# ============================================================================
# UMAP alternative (optional)
# ============================================================================
try:
    # import umap # Already imported above

    print("\n  Computing UMAP for comparison...", end=" ", flush=True)
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3)
    umap_embeddings = reducer.fit_transform(hidden_embeddings)
    print("done")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: By cluster assignment
    ax = axes[0]
    ax.scatter(umap_embeddings[cluster0_mask, 0], umap_embeddings[cluster0_mask, 1],
               c='#2E86AB', label='Cluster 0', alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
    ax.scatter(umap_embeddings[cluster1_mask, 0], umap_embeddings[cluster1_mask, 1],
               c='#A23B72', label='Cluster 1', alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
    ax.set_title('UMAP - By Cluster Assignment', fontsize=12, fontweight='bold')
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    ax.legend()
    ax.grid(True, alpha=0.2)

    # Right: By true diagnosis
    ax = axes[1]
    ax.scatter(umap_embeddings[cn_mask, 0], umap_embeddings[cn_mask, 1],
               c='#2E86AB', label='CN (true)', alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
    ax.scatter(umap_embeddings[mci_mask, 0], umap_embeddings[mci_mask, 1],
               c='#F18F01', label='MCI (true)', alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
    ax.set_title('UMAP - By True Diagnosis (reference)', fontsize=12, fontweight='bold')
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    ax.legend()
    ax.grid(True, alpha=0.2)

    plt.suptitle('UMAP Visualization of Hidden Embeddings', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('umap_visualization.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print("  ✓ Saved: umap_visualization.png")

except ImportError:
    print("\n  ⚠ UMAP not installed. Install with: pip install umap-learn")
except Exception as e:
    print(f"\n  ⚠ UMAP error: {e}")

# ============================================================================
# Print quantitative cluster separation metrics
# ============================================================================
print(f"\n{SEP}")
print("  CLUSTER SEPARATION METRICS (from hidden embeddings)")
print(f"{SEP}")

# Compute inter-cluster and intra-cluster distances
cluster0_emb = hidden_embeddings[cluster0_mask]
cluster1_emb = hidden_embeddings[cluster1_mask]

# Intra-cluster distances
intra0 = pairwise_distances(cluster0_emb).mean()
intra1 = pairwise_distances(cluster1_emb).mean()
intra_mean = (intra0 + intra1) / 2

# Inter-cluster distances
inter = pairwise_distances(cluster0_emb, cluster1_emb).mean()

# Separation ratio
separation_ratio = inter / intra_mean

print(f"\n  Intra-cluster distance (Cluster 0): {intra0:.4f}")
print(f"  Intra-cluster distance (Cluster 1): {intra1:.4f}")
print(f"  Inter-cluster distance: {inter:.4f}")
print(f"  Separation ratio (inter/intra): {separation_ratio:.4f}")
if separation_ratio > 1.5:
    print(f"  ✓ Excellent separation")
elif separation_ratio > 1.0:
    print(f"  ⚠ Moderate separation")
else:
    print(f"  ✗ Poor separation")

# Cluster diameters
cluster0_diameter = cdist(cluster0_emb, cluster0_emb).max()
cluster1_diameter = cdist(cluster1_emb, cluster1_emb).max()

print(f"\n  Cluster 0 diameter: {cluster0_diameter:.4f}")
print(f"  Cluster 1 diameter: {cluster1_diameter:.4f}")
print(f"  Compactness (1/diameter): Cluster 0 = {1/cluster0_diameter:.4f}, Cluster 1 = {1/cluster1_diameter:.4f}")

print(f"\n{SEP}")
print("  t-SNE/UMAP visualization complete")
print(f"{SEP}\n")


fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Create scatter with colormap based on entropy
scatter = ax.scatter(tsne_results[best_perp][:, 0],
                     tsne_results[best_perp][:, 1],
                     c=H_assign, cmap='RdYlBu_r',
                     alpha=0.7, s=50, edgecolors='white', linewidth=0.5)

# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Assignment Entropy (bits)', fontsize=10)

# Mark the ambiguous subject differently
if high_uncertainty_mask.sum() > 0:
    ax.scatter(tsne_results[best_perp][high_uncertainty_mask, 0],
               tsne_results[best_perp][high_uncertainty_mask, 1],
               c='red', s=150, edgecolors='black', linewidth=2,
               marker='X', label=f'Ambiguous (n={high_uncertainty_mask.sum()})', zorder=5)

# Add cluster centers
cluster0_center = tsne_results[best_perp][cluster0_mask].mean(axis=0)
cluster1_center = tsne_results[best_perp][cluster1_mask].mean(axis=0)
ax.scatter(cluster0_center[0], cluster0_center[1], c='darkblue', s=200,
           edgecolors='black', linewidth=2, marker='o', label='Cluster 0 center')
ax.scatter(cluster1_center[0], cluster1_center[1], c='darkred', s=200,
           edgecolors='black', linewidth=2, marker='o', label='Cluster 1 center')
