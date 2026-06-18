import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, log_loss,
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
ACTIV      = "ELU"
ALPHA      = 0.83
CUT        = 0
TAU        = 0.5
BETA       = 0.5
EMA_DECAY  = 0.7
LAMBDA_CON = 4
NUM_EPOCHS = 2000
T_STRUCT   = 2.0
C_ENTROPY  = 0.05
N_EVAL_PASSES = 30

# GCN specific
NUM_GCN_LAYERS = 1
HIDDEN_DIM = 256


# ─── DATA ─────────────────────────────────────────────────────────────────────
cn_data  = np.load("/home/snu/Downloads/Histogram_CN_FA_20bin_updated.npy",  allow_pickle=True)
ad_data = np.load("/home/snu/Downloads/Histogram_AD_FA_20bin_updated.npy", allow_pickle=True)

X = np.vstack([cn_data, ad_data])
y = np.hstack([np.zeros(cn_data.shape[0], dtype=np.int64),
               np.ones(ad_data.shape[0],  dtype=np.int64)])

np.random.seed(42)
perm = np.random.permutation(X.shape[0])
X, y = X[perm], y[perm]
print(f"Features: {X.shape}, Labels: {y.shape} (CN: {np.sum(y==0)}, AD: {np.sum(y==1)})")
print("NOTE: Diagnostic labels are held out and used ONLY for post-hoc external validation.\n")

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


# ─── LOSS ─────────────────────────────────────────────────────────────────────
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


# ─── MODEL: GCN ENCODER ──────────────────────────────────────────────────────
ACTIVATIONS = {"SELU": F.selu, "SiLU": F.silu, "GELU": F.gelu,
               "ELU": F.elu, "RELU": F.relu}

class GCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, activ="ELU",
                 num_layers=1, dropout=0.3):
        super().__init__()
        self.act = ACTIVATIONS.get(activ, F.elu)
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
    def __init__(self, input_dim, hidden_dim, num_clusters, device, activ,
                 ema_decay=0.7, cut=True, beta=0.5, tau=0.07,
                 num_gcn_layers=3, T_struct=2.0):
        super().__init__()
        self.device        = device
        self.num_clusters  = num_clusters
        self.cut           = cut
        self.beta, self.tau, self.T_struct = beta, tau, T_struct

        self.online_encoder   = GCNEncoder(input_dim, hidden_dim, activ,
                                           num_layers=num_gcn_layers)
        self.target_encoder   = copy.deepcopy(self.online_encoder)
        self.online_predictor = MLP(hidden_dim, num_clusters, hidden_dim)
        self.ema              = EMA(ema_decay)

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


# ─── CLUSTERING METRICS ───────────────────────────────────────────────────────
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


def get_embeddings_for_clustering(model, feats, ei, device):
    model.eval()
    d = to_data(feats, ei, device)
    with torch.no_grad():
        hidden  = model.online_encoder(d)
        logits  = model.online_predictor(hidden)
    return logits.cpu().numpy(), hidden.cpu().numpy()


def evaluate_clustering_from_mc(model, feats, ei, y_true, yp, logits_mean,
                                device, prefix=""):
    model.eval()
    d = to_data(feats, ei, device)
    with torch.no_grad():
        hidden = model.online_encoder(d).cpu().numpy()

    logits_emb = logits_mean
    ari_direct = adjusted_rand_score(y_true, yp)
    ari_flipped = adjusted_rand_score(y_true, 1 - yp)
    if ari_flipped > ari_direct:
        yp = 1 - yp
        logits_emb = logits_emb[:, ::-1]

    results = [
        compute_clustering_metrics(logits_emb, yp, y_true, space_name="logit (MC avg)"),
        compute_clustering_metrics(hidden, yp, y_true, space_name="hidden (single pass)"),
    ]

    sep = "─" * 72
    header = f"  {'Space':<22} {'ARI':>8} {'NMI':>8} {'AMI':>8} {'Silhouette':>12} {'DaviesBouldin':>14}"
    print(f"\n{sep}\n  CLUSTERING METRICS (using MC predictions){' '+prefix if prefix else ''}\n{sep}")
    print(header)
    print(f"  {sep}")
    for r in results:
        sil_str = f"{r['Silhouette']:>12.4f}" if not np.isnan(r['Silhouette']) else "         N/A"
        db_str  = f"{r['DaviesBouldin']:>14.4f}" if not np.isnan(r['DaviesBouldin']) else "           N/A"
        print(f"  {r['space']:<22} {r['ARI']:>8.4f} {r['NMI']:>8.4f} {r['AMI']:>8.4f} "
              f"{sil_str}{db_str}")
    print(f"  {sep}")
    return results


def print_clustering_summary(all_records, depth_label="3 GCN layers"):
    sep = "─" * 90
    print(f"\n{sep}")
    print(f"  CLUSTERING METRICS SUMMARY  [{depth_label}]  (mean ± std, {len(all_records)} seeds)")
    print(f"{sep}")

    spaces = [("logit (MC avg)",  "Logit space (MC-averaged, K-dim)"),
              ("hidden (single pass)", "Hidden space (256-dim encoder output)")]

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


# ─── EVALUATION ───────────────────────────────────────────────────────────────
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

def compute_ece(probs, y_true, n_bins=10):
    conf  = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    corr  = (preds == y_true).astype(float)
    ece, n = 0.0, len(y_true)
    edges  = np.linspace(0, 1, n_bins + 1)
    bin_data = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (conf >= lo) & (conf <= hi if i == n_bins - 1 else conf < hi)
        if not mask.sum(): continue
        ba, bc, bn = corr[mask].mean(), conf[mask].mean(), mask.sum()
        ece += (bn / n) * abs(ba - bc)
        bin_data.append((0.5 * (lo + hi), ba, bc, bn))
    return ece, bin_data

def find_temperature(logits, y_true, T_range=(1.0, 10.0), steps=200):
    N   = len(y_true)
    idx = np.random.default_rng(0).choice(N, size=max(1, N // 5), replace=False)
    lv, yv = logits[idx], y_true[idx]
    best_T, best_ece = 2.0, float('inf')
    for T in np.linspace(T_range[0], T_range[1], steps):
        p = F.softmax(torch.from_numpy(lv) / T, dim=1).numpy()
        e, _ = compute_ece(p, yv)
        if e < best_ece:
            best_ece, best_T = e, T
    return best_T

def evaluate_model(model, feats, ei, y, device, n_passes=30, seed_base=9999, temperature=None):
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
        a = ai

    T = temperature if temperature is not None else find_temperature(logits_mean, y)
    T = max(1.0, T)
    if T > 5.0:
        print(f"  ⚠ T={T:.1f} > 5 — overconfident logits "
              f"(max |diff| ±{np.abs(logits_mean[:,1]-logits_mean[:,0]).max():.1f})")

    ypp = F.softmax(torch.from_numpy(logits_mean) / T, dim=1).numpy()
    return yp, ypp, logits_stack, logits_mean, a, T


# ─── UNCERTAINTY ──────────────────────────────────────────────────────────────
def softmax_np(logits):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def entropy_bits(p):
    p = np.clip(p, 1e-12, 1.0)
    return -np.sum(p * np.log2(p), axis=1)

def compute_uncertainty_from_logits(logits_stack):
    N, K, P    = logits_stack.shape
    probs      = np.stack([softmax_np(logits_stack[:, :, p]) for p in range(P)], axis=2)
    H_aleat    = np.stack([entropy_bits(probs[:, :, p]) for p in range(P)], axis=1).mean(axis=1)
    p_mean     = probs.mean(axis=2)
    H_pred     = entropy_bits(p_mean)
    MI         = np.clip(H_pred - H_aleat, 0, None)
    pred_var   = probs[:, 1, :].var(axis=1)
    return MI, H_pred, H_aleat, pred_var, p_mean

def compute_uncertainty(yp, ypp, logits_stack, y_true, T_used, sample_ids=None):
    MI, H_pred, H_aleat, pred_var, _ = compute_uncertainty_from_logits(logits_stack)
    P = logits_stack.shape[2]
    pass_preds     = np.stack([np.argmax(logits_stack[:, :, p], axis=1) for p in range(P)], axis=1)
    pass_disagree  = (pass_preds != yp[:, None]).mean(axis=1)
    ypp_c          = np.clip(ypp, 1e-12, 1)
    label_map      = {0: "CN", 1: "AD"}
    if sample_ids is None:
        sample_ids = list(range(len(y_true)))
    return pd.DataFrame({
        "sample_id":         sample_ids,
        "true_label":        [label_map[int(l)] for l in y_true],
        "predicted_label":   [label_map[int(l)] for l in yp],
        "correct":           (yp == y_true).astype(int),
        "p_CN":              ypp[:, 0],
        "p_AD":              ypp[:, 1],
        "entropy_total":     H_pred,
        "uncertainty_epist": MI,
        "uncertainty_aleat": H_aleat,
        "pred_variance":     pred_var,
        "pass_disagree_pct": pass_disagree,
        "confidence":        ypp_c.max(axis=1),
        "margin":            np.abs(ypp[:, 0] - ypp[:, 1]),
    })

def print_uncertainty_report(df, ypp, y_true, yp, T_used, n_passes,
                              low_conf_thr=0.65, disagree_thr=0.20,
                              csv_path="uncertainty_per_subject_CN_AD_GCN.csv"):
    sep  = "─" * 72
    cols = ["sample_id", "true_label", "predicted_label", "correct",
            "p_CN", "p_AD", "entropy_total", "uncertainty_epist",
            "uncertainty_aleat", "pred_variance", "pass_disagree_pct", "confidence"]
    cols_amb = ["sample_id", "true_label", "predicted_label", "p_CN", "p_AD",
                "entropy_total", "uncertainty_epist", "pred_variance",
                "pass_disagree_pct", "confidence"]
    df_s = df.sort_values("entropy_total", ascending=False).reset_index(drop=True)
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.width", 170)
    pd.set_option("display.max_columns", 16)

    print(f"\n{sep}\n  A.  TOP-10 MOST UNCERTAIN SUBJECTS\n{sep}")
    print(df_s.head(10)[cols].to_string(index=True))

    print(f"\n{sep}\n  B.  UNCERTAINTY SUMMARY STATISTICS\n{sep}")
    for col, label in [
        ("entropy_total",     "Total entropy H[p̄]     (bits)"),
        ("uncertainty_epist", "Epistemic MI            (bits)"),
        ("uncertainty_aleat", "Aleatoric E[H[p]]      (bits)"),
        ("pred_variance",     "Predictive variance p_AD    "),
        ("pass_disagree_pct", "Pass-level disagreement %    "),
        ("confidence",        "Confidence max(p̄) [T-scaled] "),
        ("margin",            "Margin |p_CN - p_AD|         "),
    ]:
        print(f"  {label}: mean={df[col].mean():.4f}  std={df[col].std():.4f}  "
              f"min={df[col].min():.4f}  max={df[col].max():.4f}")

    ece, bin_data = compute_ece(ypp, y_true)
    print(f"\n{sep}\n  C.  CALIBRATION\n{sep}")
    print(f"  MC passes: {n_passes}  |  Temperature T: {T_used:.3f}")
    print(f"  ECE: {ece:.4f}  |  LogLoss: {log_loss(y_true, ypp):.4f}")
    print(f"\n  Reliability diagram [bin_mid | acc | conf | n]:")
    for mid, acc, conf, cnt in bin_data:
        flag = "▲ under" if acc > conf else "▼ over"
        print(f"    [{mid:.2f}]  acc={acc:.3f}  conf={conf:.3f}  {flag}-confident  n={cnt}")

    n_low  = (df["confidence"] < low_conf_thr).sum()
    low_df = df[df["confidence"] < low_conf_thr]
    print(f"\n{sep}\n  D.  LOW-CONFIDENCE SUBJECTS  (confidence < {low_conf_thr})\n{sep}")
    print(f"  Count: {n_low}/{len(df)}  ({100*n_low/len(df):.1f}%)")
    for label in ["CN", "AD"]:
        cnt   = (low_df["true_label"] == label).sum()
        wrong = ((low_df["true_label"] == label) & (low_df["correct"] == 0)).sum()
        print(f"    True {label}: {cnt} subjects  ({wrong} misclassified)")

    ambiguous = df[(df["confidence"] < low_conf_thr) & (df["correct"] == 0)].sort_values(
        "entropy_total", ascending=False)
    print(f"\n{sep}\n  E.  UNCERTAIN + MISCLASSIFIED (clinically ambiguous)\n{sep}")
    if len(ambiguous) == 0:
        print(f"  None at confidence threshold {low_conf_thr}.")
        wrong_df = df[df["correct"] == 0].sort_values("entropy_total", ascending=False)
        if len(wrong_df):
            print("  All misclassified (sorted by entropy):")
            print(wrong_df[cols_amb].to_string(index=False))
    else:
        print(ambiguous[cols_amb].to_string(index=False))
    print(f"\n  Ambiguous (low-conf + wrong): {len(ambiguous)}")

    high_dis = df[df["pass_disagree_pct"] >= disagree_thr].sort_values(
        "pass_disagree_pct", ascending=False)
    print(f"\n{sep}\n  F.  HIGH-DISAGREEMENT SUBJECTS  (pass disagreement >= {disagree_thr:.0%})\n{sep}")
    if len(high_dis) == 0:
        print(f"  None at threshold {disagree_thr:.0%}.")
    else:
        print(high_dis[cols_amb].to_string(index=False))
    print(f"  Total: {len(high_dis)}")

    confidence = ypp.max(axis=1)
    FP = (y_true == 0) & (yp == 1)
    FN = (y_true == 1) & (yp == 0)
    tn, fp, fn, tp = confusion_matrix(y_true, yp).ravel()
    print(f"\n{sep}\n  G.  HIGH-CONFIDENCE ERROR ANALYSIS\n{sep}")
    print(f"  Confusion Matrix: TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    for thr in [0.70, 0.80, 0.90]:
        fp_h = int(np.sum(FP & (confidence >= thr)))
        fn_h = int(np.sum(FN & (confidence >= thr)))
        print(f"\n  Confidence >= {thr:.2f}:")
        print(f"    High-conf FP: {fp_h}/{fp}" + (f"  ({100*fp_h/fp:.0f}%)" if fp > 0 else ""))
        print(f"    High-conf FN: {fn_h}/{fn}" + (f"  ({100*fn_h/fn:.0f}%)" if fn > 0 else ""))

    df_s.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\n{sep}\n  CSV → {csv_path}  ({len(df_s)} subjects)\n{sep}")


# ─── MAD METRICS (for oversmoothing analysis) ─────────────────────────────────
def cosine_distance_matrix(H):
    n = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-12)
    return 1.0 - np.clip(n @ n.T, -1.0, 1.0)

def compute_mad_metrics(H, adj, labels, name=""):
    N    = H.shape[0]
    D    = cosine_distance_matrix(H)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    conn = adj > 0
    same = labels[:, None] == labels[None, :]
    def mad(m): return D[m].mean() if m.sum() > 0 else float('nan')
    return dict(
        name        = name,
        MAD_all     = mad(mask),
        MAD_local   = mad(mask & conn),
        MAD_remote  = mad(mask & ~conn),
        MADGap      = mad(mask & ~conn) - mad(mask & conn),
        MAD_within  = mad(mask & same),
        MAD_between = mad(mask & ~same),
        Class_Sep   = mad(mask & ~same) - mad(mask & same),
        Mean_Sim    = 1.0 - mad(mask),
    )

def print_mad_table(results):
    sep    = "─" * 90
    header = (f"{'Layer':<22} {'MAD_all':>9} {'MAD_local':>10} {'MAD_remote':>11} "
              f"{'MADGap':>9} {'MAD_within':>11} {'MAD_btwn':>9} {'ClassSep':>9}")
    print(f"\n{sep}\n{header}\n{sep}")
    for r in results:
        print(f"  {r['name']:<20} {r['MAD_all']:>9.4f} {r['MAD_local']:>10.4f} "
              f"{r['MAD_remote']:>11.4f} {r['MADGap']:>9.4f} "
              f"{r['MAD_within']:>11.4f} {r['MAD_between']:>9.4f} {r['Class_Sep']:>9.4f}")
    print(sep)

def extract_embeddings(model, feats, ei, device):
    model.eval()
    d = to_data(feats, ei, device)
    with torch.no_grad():
        final, inters = model.online_encoder(d, return_intermediates=True)
    return inters, final.cpu().numpy()

def layer_similarity_analysis(model, feats, ei, y, W0, prefix=""):
    sep = "─" * 72
    print(f"\n{sep}\n  PER-LAYER REPRESENTATION SIMILARITY  [{prefix}]\n{sep}")
    results = [compute_mad_metrics(feats, W0, y, name="Input features")]
    inters, final_np = extract_embeddings(model, feats, ei, DEVICE)
    for i, emb in enumerate(inters):
        results.append(compute_mad_metrics(emb, W0, y, name=f"GCN layer {i+1}"))
    results.append(compute_mad_metrics(final_np, W0, y, name="Final proj (output)"))
    print_mad_table(results)

    D = cosine_distance_matrix(final_np)
    N = len(y)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    same = (y[:, None] == y[None, :]) & mask
    diff = (y[:, None] != y[None, :]) & mask
    sw = 1 - D[same]; bw = 1 - D[diff]
    print(f"\n  Final-layer cosine similarity:")
    print(f"    Within-class  : mean={sw.mean():.4f}  std={sw.std():.4f}  "
          f"min={sw.min():.4f}  max={sw.max():.4f}")
    print(f"    Between-class : mean={bw.mean():.4f}  std={bw.std():.4f}  "
          f"min={bw.min():.4f}  max={bw.max():.4f}")

    r_in = results[0]; r_fin = results[-1]
    over_smooth = r_fin['MADGap'] < r_in['MADGap'] * 0.5
    print(f"\n  MADGap: Input={r_in['MADGap']:.4f} → Final={r_fin['MADGap']:.4f}  "
          f"{'⚠ OVER-SMOOTHING' if over_smooth else '✓ gap preserved'}")
    return results

def high_conf_errors(y_true, yp, ypp, thr=0.70):
    conf = ypp.max(axis=1)
    FP   = (y_true == 0) & (yp == 1)
    FN   = (y_true == 1) & (yp == 0)
    return (int(np.sum(FP & (conf >= thr))), int(np.sum(FN & (conf >= thr))),
            int(FP.sum()), int(FN.sum()))

def quick_mad(model):
    _, final_np = extract_embeddings(model, features_np, edge_index_np, DEVICE)
    r_in  = compute_mad_metrics(features_np, W0, y, name="Input")
    r_out = compute_mad_metrics(final_np, W0, y, name="Output")
    return r_in['MADGap'], r_out['MADGap'], r_in['Class_Sep'], r_out['Class_Sep']


# ─── TRAINING (returns weighted metrics) ──────────────────────────────────────
def run_once(seed_offset=0, verbose=False, return_probs=False,
             n_eval_passes=30, num_gcn_layers=NUM_GCN_LAYERS):
    np.random.seed(42 + seed_offset)
    random.seed(42 + seed_offset)
    torch.manual_seed(42 + seed_offset)

    model = GCNModel(FEATS_DIM, HIDDEN_DIM, K, DEVICE, ACTIV,
                     ema_decay=EMA_DECAY, cut=CUT, beta=BETA, tau=TAU,
                     num_gcn_layers=num_gcn_layers,
                     T_struct=T_STRUCT).to(DEVICE)

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

    yp, ypp, logits_stack, logits_mean, a, T = evaluate_model(
        model, features_np, edge_index_np, y, DEVICE,
        n_passes=n_eval_passes, seed_base=9999 + seed_offset
    )
    # --- WEIGHTED METRICS (over both classes) ---
    acc_w = accuracy_score(y, yp)
    prec_w = precision_score(y, yp, average='weighted', zero_division=0)
    rec_w  = recall_score(y, yp, average='weighted', zero_division=0)
    f1_w   = f1_score(y, yp, average='weighted', zero_division=0)
    logloss = log_loss(y, ypp)
    metrics_weighted = (acc_w, prec_w, rec_w, f1_w, logloss)

    if return_probs:
        return metrics_weighted, yp, ypp, logits_stack, logits_mean, T, model
    return metrics_weighted, model


# ─── MAIN ─────────────────────────────────────────────────────────────────────
METRIC_NAMES = ["Accuracy", "Weighted Precision", "Weighted Recall", "Weighted F1", "LogLoss"]
SEP = "═" * 72

print(f"\n{SEP}\n  PART A — FIRST RUN (1 GCN layer) – CN vs AD\n{SEP}")
metrics0, yp0, ypp0, logits_stack0, logits_mean0, T0, model3 = run_once(
    seed_offset=0, verbose=True, return_probs=True,
    n_eval_passes=N_EVAL_PASSES, num_gcn_layers=NUM_GCN_LAYERS
)

print("\n── Single-run Classification Results (weighted) ───────────────")
for n, v in zip(METRIC_NAMES, metrics0):
    print(f"  {n:<20}: {v:.4f}")
print(f"  Temperature           : {T0:.3f}")

lm = logits_stack0.mean(axis=2)
diff = lm[:, 1] - lm[:, 0]
print(f"\n── Logit diagnostics ───────────────────────────────")
print(f"  mean={lm.mean():.2f}  std={lm.std():.2f}  min={lm.min():.2f}  max={lm.max():.2f}")
print(f"  Logit diff (AD-CN): mean={diff.mean():.2f}  std={diff.std():.2f}  "
      f"min={diff.min():.2f}  max={diff.max():.2f}")

df_unc = compute_uncertainty(yp0, ypp0, logits_stack0, y_true=y, T_used=T0)
print_uncertainty_report(df_unc, ypp0, y, yp0, T0, N_EVAL_PASSES,
                         csv_path="uncertainty_per_subject_CN_AD_GCN.csv")

# ── Clustering metrics for Part A ─────────────────────────────────────
clustering_results_A = evaluate_clustering_from_mc(
    model3, features_np, edge_index_np, y, yp0, logits_mean0, DEVICE,
    prefix="Part A — 1-layer GCN, seed 0"
)

print(f"\n{SEP}\n  PART B — PER-LAYER OVER-SMOOTHING ANALYSIS\n{SEP}")
layer_similarity_analysis(model3, features_np, edge_index_np, y, W0,
                          prefix="1-layer GCN (CN vs AD)")

print(f"\n{SEP}\n  PART C — DEPTH ABLATION  (1 / 2 / 3 GCN layers)\n{SEP}")
ablation_results = {}
trained_models   = {}
ablation_clustering = {1: [], 2: [], 3: []}

for n_layers in [1, 2, 3]:
    print(f"\n  ── Depth = {n_layers} GCN layer(s) ──")
    seed_records = []
    for seed in range(3):
        print(f"    Seed {seed} ... ", end="", flush=True)
        (acc_s, prec_s, rec_s, f1_s, ll_s), model_i = run_once(
            seed_offset=seed, num_gcn_layers=n_layers,
            n_eval_passes=N_EVAL_PASSES, return_probs=False
        )
        yp_s, ypp_s, logits_stack_s, logits_mean_s, a_s, T_s = evaluate_model(
            model_i, features_np, edge_index_np, y, DEVICE,
            n_passes=N_EVAL_PASSES, seed_base=9999 + seed
        )
        ece_s, _ = compute_ece(ypp_s, y)
        fp70, fn70, fp_t, fn_t = high_conf_errors(y, yp_s, ypp_s)

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

        seed_records.append(dict(
            depth=n_layers, seed=seed, acc=acc_s,
            prec=prec_s, rec=rec_s, f1=f1_s,
            logloss=ll_s, ece=ece_s, T=T_s,
            mean_conf=ypp_s.max(axis=1).mean(),
            fp70=fp70, fn70=fn70, fp_tot=fp_t, fn_tot=fn_t
        ))
        trained_models[n_layers] = model_i
        print(f"Acc={acc_s:.3f}  T={T_s:.1f}  ECE={ece_s:.4f}  FP70={fp70}/{fp_t}  FN70={fn70}/{fn_t}")
    ablation_results[n_layers] = seed_records

print("\n  ── MADGap at each depth ──")
mad_depth = {}
for n_layers in [1, 2, 3]:
    g_in, g_out, cs_in, cs_out = quick_mad(trained_models[n_layers])
    mad_depth[n_layers] = (g_in, g_out, cs_in, cs_out)
    print(f"    {n_layers} layer(s):  MADGap_input={g_in:.4f}  MADGap_final={g_out:.4f}  "
          f"ClassSep_input={cs_in:.4f}  ClassSep_final={cs_out:.4f}")

sep = "─" * 100
print(f"\n{sep}\n  DEPTH ABLATION SUMMARY  (mean ± std over 3 seeds)\n{sep}")
print(f"  {'Depth':<7} {'Acc':>7} {'Prec':>7} {'Rec':>6} {'F1':>7} "
      f"{'LogLoss':>8} {'ECE':>7} {'T':>7} {'MeanConf':>9} "
      f"{'FP@70':>7} {'FN@70':>7} {'MADGap↑':>9} {'ClassSep↑':>10}")
print(sep)
for n_layers in [1, 2, 3]:
    recs = ablation_results[n_layers]
    def ms(k): return np.mean([r[k] for r in recs]), np.std([r[k] for r in recs])
    a_m,a_s=ms("acc"); p_m,p_s=ms("prec"); r_m,r_s=ms("rec"); f_m,f_s=ms("f1")
    l_m,l_s=ms("logloss"); e_m,e_s=ms("ece"); t_m,t_s=ms("T")
    c_m,c_s=ms("mean_conf"); fp_m,fp_s=ms("fp70"); fn_m,fn_s=ms("fn70")
    _, g_out, _, cs_out = mad_depth[n_layers]
    print(f"  {n_layers} layer{'s' if n_layers>1 else ' ':<5}  "
          f"{a_m:.3f}±{a_s:.3f}  {p_m:.3f}±{p_s:.3f}  {r_m:.3f}±{r_s:.3f}  "
          f"{f_m:.3f}±{f_s:.3f}  {l_m:.4f}±{l_s:.4f}  {e_m:.4f}±{e_s:.4f}  "
          f"{t_m:.1f}±{t_s:.1f}  {c_m:.4f}±{c_s:.4f}  {fp_m:.1f}±{fp_s:.1f}  "
          f"{fn_m:.1f}±{fn_s:.1f}  {g_out:.4f}  {cs_out:.4f}")
print(sep)

print(f"\n{SEP}\n  PART D — DIAGNOSIS (CN vs AD)\n{SEP}")
def _mean(key, d): return np.mean([r[key] for r in ablation_results[d]])
for depth in [1, 2, 3]:
    g_in, g_out, _, cs_out = mad_depth[depth]
    T_d   = _mean("T",   depth)
    ece_d = _mean("ece", depth)
    conf_d= _mean("mean_conf", depth)
    acc_d = _mean("acc", depth)
    flag = "↓ collapsed" if g_out < g_in * 0.5 else "→ preserved"
    print(f"  {depth} layer(s): MADGap {g_in:.4f}→{g_out:.4f} {flag} | "
          f"ClassSep={cs_out:.4f} | T={T_d:.2f} | ECE={ece_d:.4f} | "
          f"Conf={conf_d:.4f} | Acc={acc_d:.4f}")

fp70_3 = _mean("fp70", 3); fn70_3 = _mean("fn70", 3)
print(f"\n  Overconfidence check: T ≈ {T_d:.2f}")
print(f"  High-conf errors (3 layers, thr=0.70): FP={fp70_3:.0f}  FN={fn70_3:.0f}")
print(f"\n  Over-smoothing:")
print(f"    MADGap_3layers={mad_depth[3][1]:.4f}  vs  MADGap_input={mad_depth[3][0]:.4f}")
if mad_depth[3][1] < mad_depth[3][0] * 0.5:
    print("    ⚠ OVER-SMOOTHING detected")
else:
    print("    ✓ Gap preserved – GCN not overly smoothed")


# ─── PART E — 10-SEED EVALUATION WITH WEIGHTED METRICS ────────────────
print(f"\n{SEP}\n  PART E — 10-SEED EVALUATION (1 GCN layer, CN vs AD)\n{SEP}")

def conf_group(mask, conf):
    return float(conf[mask].mean()) if mask.sum() > 0 else float('nan')

col_w = "─" * 130
print(f"  {'Seed':>4}  {'Acc':>7}  {'Prec_w':>8}  {'Rec_w':>8}  {'F1_w':>8}  "
      f"{'LogLoss':>8}  {'ECE':>7}  {'MeanConf':>9}  {'T':>6}  "
      f"{'TN':>5}  {'FP':>5}  {'FN':>5}  {'TP':>5}  "
      f"{'Conf_TN':>8}  {'Conf_FP':>8}  {'Conf_FN':>8}  {'Conf_TP':>8}")
print(f"  {col_w}")

calibration_rows = []      # will store weighted metrics
seed_clustering_records = []
high_conf_counts = {thr: {'fp': [], 'fn': []} for thr in [0.70, 0.80, 0.90]}

for i in range(10):
    m_w, yp_i, ypp_i, logits_stack_i, logits_mean_i, T_i, model_i = run_once(
        seed_offset=i,
        return_probs=True,
        n_eval_passes=N_EVAL_PASSES,
        num_gcn_layers=NUM_GCN_LAYERS
    )
    acc_i, prec_w, rec_w, f1_w, ll_i = m_w   # weighted metrics

    ece_i, _    = compute_ece(ypp_i, y)
    mean_conf_i = float(ypp_i.max(axis=1).mean())
    conf_all    = ypp_i.max(axis=1)

    tn_i, fp_i, fn_i, tp_i = confusion_matrix(y, yp_i).ravel()

    mask_tn = (y == 0) & (yp_i == 0)
    mask_fp = (y == 0) & (yp_i == 1)
    mask_fn = (y == 1) & (yp_i == 0)
    mask_tp = (y == 1) & (yp_i == 1)

    conf_tn = conf_group(mask_tn, conf_all)
    conf_fp = conf_group(mask_fp, conf_all)
    conf_fn = conf_group(mask_fn, conf_all)
    conf_tp = conf_group(mask_tp, conf_all)

    calibration_rows.append(dict(
        seed=i, acc=acc_i, prec_weighted=prec_w, rec_weighted=rec_w, f1_weighted=f1_w,
        logloss=ll_i, ece=ece_i, mean_conf=mean_conf_i, T=T_i,
        tn=int(tn_i), fp=int(fp_i), fn=int(fn_i), tp=int(tp_i),
        conf_tn=conf_tn, conf_fp=conf_fp, conf_fn=conf_fn, conf_tp=conf_tp,
    ))

    for thr in [0.70, 0.80, 0.90]:
        fp_h, fn_h, _, _ = high_conf_errors(y, yp_i, ypp_i, thr=thr)
        high_conf_counts[thr]['fp'].append(fp_h)
        high_conf_counts[thr]['fn'].append(fn_h)

    # Clustering metrics for this seed
    cl_res_i = evaluate_clustering_from_mc(
        model_i, features_np, edge_index_np, y, yp_i, logits_mean_i, DEVICE,
        prefix=f"Part E seed {i} (quiet)"
    )
    cl_flat_i = {}
    for r in cl_res_i:
        sp = r['space']
        for metric in ['ARI', 'NMI', 'AMI', 'Silhouette', 'DaviesBouldin']:
            cl_flat_i[f"{metric}_{sp}"] = r[metric]
    seed_clustering_records.append(cl_flat_i)

    def fmt_conf(v): return f"{v:8.4f}" if not np.isnan(v) else "     N/A"
    print(f"  {i:>4}  {acc_i:>7.4f}  {prec_w:>8.4f}  {rec_w:>8.4f}  {f1_w:>8.4f}  "
          f"{ll_i:>8.4f}  {ece_i:>7.4f}  {mean_conf_i:>9.4f}  {T_i:>6.3f}  "
          f"{tn_i:>5}  {fp_i:>5}  {fn_i:>5}  {tp_i:>5}  "
          f"{fmt_conf(conf_tn)}  {fmt_conf(conf_fp)}  {fmt_conf(conf_fn)}  {fmt_conf(conf_tp)}")

# ── 10-Run Summary (weighted) ──────────────────────────────────────────
results_np = np.array([[r['acc'], r['prec_weighted'], r['rec_weighted'], r['f1_weighted'], r['logloss'],
                        r['ece'], r['mean_conf'], r['T']]
                       for r in calibration_rows])

counts_np = np.array([[r['tn'], r['fp'], r['fn'], r['tp']]
                      for r in calibration_rows], dtype=float)

conf_np = np.array([[r['conf_tn'], r['conf_fp'], r['conf_fn'], r['conf_tp']]
                    for r in calibration_rows])

sep100 = "─" * 100
print(f"\n{sep100}")
print(f"  10-RUN SUMMARY  (mean ± std)                              [1 GCN layer, CN vs AD]")
print(f"{sep100}")

metric_labels = [
    ("Accuracy (weighted)", "Higher is better"),
    ("Weighted Precision",  "Higher is better"),
    ("Weighted Recall",     "Higher is better"),
    ("Weighted F1",         "Higher is better"),
    ("Log Loss",            "Lower is better"),
    ("ECE",                 "Lower is better"),
    ("Mean Conf",           "Closer to Accuracy = well-calibrated"),
    ("Temperature T",       "1.0 = already calibrated, >1 = needed rescaling"),
]

print(f"\n  A.  CLASSIFICATION & CALIBRATION METRICS (weighted)")
print(f"  {'Metric':<20}  {'Mean':>9}  {'Std':>9}  {'Min':>9}  {'Max':>9}  Note")
print(f"  {sep100}")
for col_idx, (label, note) in enumerate(metric_labels):
    vals = results_np[:, col_idx]
    print(f"  {label:<20}  {vals.mean():>9.4f}  {vals.std():>9.4f}  "
          f"{vals.min():>9.4f}  {vals.max():>9.4f}  {note}")

count_labels = [
    ("TN  (CN→CN)",  "True Negatives  — CN correctly classified as CN"),
    ("FP  (CN→AD)",  "False Positives — CN misclassified as AD"),
    ("FN  (AD→CN)",  "False Negatives — AD misclassified as CN  ← clinical risk"),
    ("TP  (AD→AD)",  "True Positives  — AD correctly classified as AD"),
]

print(f"\n  B.  CONFUSION MATRIX COUNTS")
print(f"  {'Group':<14}  {'Mean':>7}  {'Std':>7}  {'Min':>5}  {'Max':>5}  Note")
print(f"  {sep100}")
for col_idx, (label, note) in enumerate(count_labels):
    vals = counts_np[:, col_idx]
    print(f"  {label:<14}  {vals.mean():>7.1f}  {vals.std():>7.2f}  "
          f"{vals.min():>5.0f}  {vals.max():>5.0f}  {note}")

conf_group_labels = [
    ("Conf_TN",  "Avg confidence on correctly predicted CN — expect high"),
    ("Conf_FP",  "Avg confidence on CN misclassified as AD — high = dangerous"),
    ("Conf_FN",  "Avg confidence on AD misclassified as CN — high = dangerous"),
    ("Conf_TP",  "Avg confidence on correctly predicted AD — expect high"),
]

print(f"\n  C.  MEAN PREDICTION CONFIDENCE PER CONFUSION GROUP")
print(f"  {'Group':<12}  {'Mean':>9}  {'Std':>9}  {'Min':>9}  {'Max':>9}  Note")
print(f"  {sep100}")
for col_idx, (label, note) in enumerate(conf_group_labels):
    vals = conf_np[:, col_idx]
    mean_v = np.nanmean(vals)
    std_v  = np.nanstd(vals)
    min_v  = np.nanmin(vals) if not np.all(np.isnan(vals)) else float('nan')
    max_v  = np.nanmax(vals) if not np.all(np.isnan(vals)) else float('nan')
    def fv(v): return f"{v:9.4f}" if not np.isnan(v) else "      N/A"
    print(f"  {label:<12}  {fv(mean_v)}  {fv(std_v)}  {fv(min_v)}  {fv(max_v)}  {note}")

print(f"\n  D.  HIGH‑CONFIDENCE FALSE POSITIVES & FALSE NEGATIVES")
print(f"  {'Threshold':>10}  {'FP (mean ± std)':>24}  {'FN (mean ± std)':>24}")
print(f"  {sep100}")
for thr in [0.70, 0.80, 0.90]:
    fp_arr = np.array(high_conf_counts[thr]['fp'])
    fn_arr = np.array(high_conf_counts[thr]['fn'])
    print(f"  ≥{thr:.2f}       {fp_arr.mean():>6.1f} ± {fp_arr.std():<6.2f}      "
          f"{fn_arr.mean():>6.1f} ± {fn_arr.std():<6.2f}")

print(f"  {sep100}\n")

# ─── PART F — CLUSTERING METRICS SUMMARY ─────────────────────────────────────
print(f"\n{SEP}\n  PART F — CLUSTERING METRICS (ARI / NMI / AMI / Silhouette / Davies-Bouldin)\n{SEP}")

print(f"\n  ── F.1  10-SEED EVALUATION (1 GCN layer) ──")
print_clustering_summary(seed_clustering_records, depth_label="1 GCN layer — 10 seeds")

print(f"\n  ── F.2  DEPTH ABLATION (3 seeds per depth) ──")
for n_layers in [1, 2, 3]:
    print_clustering_summary(
        ablation_clustering[n_layers],
        depth_label=f"{n_layers} GCN layer{'s' if n_layers > 1 else ' '} — 3 seeds"
    )

print(f"\n  ── F.3  CROSS-DEPTH COMPARISON (hidden space, mean over seeds) ──")
sep72 = "─" * 72
print(f"\n  {'Depth':<12} {'ARI↑':>8} {'NMI↑':>8} {'AMI↑':>8} "
      f"{'Silhouette↑':>13} {'DaviesBouldin↓':>16}")
print(f"  {sep72}")
for n_layers in [1, 2, 3]:
    recs = ablation_clustering[n_layers]
    def mn(key):
        vals = np.array([r[key] for r in recs])
        valid = vals[~np.isnan(vals)]
        return valid.mean() if len(valid) > 0 else float('nan')
    ari = mn("ARI_hidden (single pass)")
    nmi = mn("NMI_hidden (single pass)")
    ami = mn("AMI_hidden (single pass)")
    sil = mn("Silhouette_hidden (single pass)")
    db  = mn("DaviesBouldin_hidden (single pass)")
    sil_s = f"{sil:>13.4f}" if not np.isnan(sil) else "          N/A"
    db_s  = f"{db:>16.4f}"  if not np.isnan(db)  else "             N/A"
    print(f"  {n_layers} layer{'s' if n_layers>1 else ' ':<7}  "
          f"{ari:>8.4f}  {nmi:>8.4f}  {ami:>8.4f}{sil_s}{db_s}")
print(f"  {sep72}")
print(f"\n  Interpretation for cross-depth table (hidden space):")
print(f"    Rising ARI/NMI/AMI with depth → deeper stacks recover more label structure.")
print(f"    Rising Silhouette with depth   → embeddings become more geometrically clustered.")
print(f"    Falling Davies-Bouldin         → clusters tighten relative to centroid distances.")
print(f"    If any metric degrades at depth 3, combine with MADGap to diagnose")
print(f"    over-smoothing as the cause.\n")

# ============================================================================
#  FINAL HORIZONTAL TABLE WITH WEIGHTED METRICS (mean ± std over 10 runs)
# ============================================================================
print("\n" + "=" * 72)
print("  HORIZONTAL TABLE FOR MANUSCRIPT (mean ± std over 10 runs)")
print("  Metrics: Accuracy, Weighted Precision, Weighted Recall, Weighted F1, NMI, ARI, AMI, Silhouette, Davies‑Bouldin")
print("=" * 72)

# Combine classification (weighted) and clustering metrics from 10 seeds
all_weighted_results = []
for i in range(len(calibration_rows)):
    row_calib = calibration_rows[i]
    row_cluster = seed_clustering_records[i]

    acc = row_calib['acc']
    prec = row_calib['prec_weighted']
    rec = row_calib['rec_weighted']
    f1 = row_calib['f1_weighted']
    nmi = row_cluster.get('NMI_hidden (single pass)', np.nan)
    ari = row_cluster.get('ARI_hidden (single pass)', np.nan)
    ami = row_cluster.get('AMI_hidden (single pass)', np.nan)
    sil = row_cluster.get('Silhouette_hidden (single pass)', np.nan)
    db  = row_cluster.get('DaviesBouldin_hidden (single pass)', np.nan)

    all_weighted_results.append({
        'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1,
        'nmi': nmi, 'ari': ari, 'ami': ami,
        'silhouette': sil, 'db': db
    })

acc_vals = np.array([r['acc'] for r in all_weighted_results])
prec_vals = np.array([r['prec'] for r in all_weighted_results])
rec_vals = np.array([r['rec'] for r in all_weighted_results])
f1_vals = np.array([r['f1'] for r in all_weighted_results])
nmi_vals = np.array([r['nmi'] for r in all_weighted_results])
ari_vals = np.array([r['ari'] for r in all_weighted_results])
ami_vals = np.array([r['ami'] for r in all_weighted_results])
sil_vals = np.array([r['silhouette'] for r in all_weighted_results])
db_vals = np.array([r['db'] for r in all_weighted_results])

metrics_names = [
    "Accuracy", "Precision", "Recall", "F1",
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
row = "GCN"
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
latex_row = "GCN"
for m, s in zip(means, stds):
    if np.isnan(m):
        latex_row += " & N/A±N/A"
    else:
        latex_row += f" & ${m:.4f}\\pm{s:.4f}$"
latex_row += " \\\\"
print(latex_row)

# Optional: save all results to CSV
df_all = pd.DataFrame(all_weighted_results)
df_all.to_csv("gcn_cn_vs_ad_10run_weighted_summary.csv", index=False, float_format="%.6f")
print(f"\n  ✓ Saved weighted summary → gcn_cn_vs_ad_10run_weighted_summary.csv")
print(f"\n{'='*60}\n  ANALYSIS COMPLETE\n{'='*60}")

import numpy as np
import pandas as pd # Ensure pandas is imported if not already in this cell

# Create all_results by combining calibration_rows and seed_clustering_records
all_results = []
for i in range(len(calibration_rows)):
    row_calib = calibration_rows[i]
    row_cluster = seed_clustering_records[i]

    # Extract classification metrics (assuming 'prec', 'rec', 'f1' are from binary average)
    acc = row_calib['acc']
    prec = row_calib['prec']
    rec = row_calib['rec']
    f1 = row_calib['f1']

    # Extract clustering metrics from 'hidden (single pass)' space
    nmi = row_cluster.get('NMI_hidden (single pass)', np.nan)
    ari = row_cluster.get('ARI_hidden (single pass)', np.nan)
    ami = row_cluster.get('AMI_hidden (single pass)', np.nan)
    silhouette = row_cluster.get('Silhouette_hidden (single pass)', np.nan)
    db = row_cluster.get('DaviesBouldin_hidden (single pass)', np.nan)

    all_results.append({
        'acc': acc,
        'prec': prec, # Not weighted, default for binary classification
        'rec': rec,   # Not weighted, default for binary classification
        'f1': f1,     # Not weighted, default for binary classification
        'nmi': nmi,
        'ari': ari,
        'ami': ami,
        'silhouette': silhouette,
        'db': db
    })


# ============================================================================
#  HORIZONTAL TABLE WITH MEAN ± STD (weighted metrics)
# ============================================================================
print("\n" + "=" * 72)
print("  HORIZONTAL TABLE FOR MANUSCRIPT (mean ± std over 10 runs)")
print("  Metrics: Accuracy, Precision, Recall, F1, NMI, ARI, AMI, Silhouette, Davies‑Bouldin")
print("=" * 72)

# Extract arrays for the nine metrics (now using the created all_results)
acc_vals = np.array([r['acc'] for r in all_results])
prec_vals = np.array([r['prec'] for r in all_results]) # Changed from prec_w_vals
rec_vals = np.array([r['rec'] for r in all_results])   # Changed from rec_w_vals
f1_vals = np.array([r['f1'] for r in all_results])     # Changed from f1_w_vals
nmi_vals = np.array([r['nmi'] for r in all_results])
ari_vals = np.array([r['ari'] for r in all_results])
ami_vals = np.array([r['ami'] for r in all_results])
sil_vals = np.array([r['silhouette'] for r in all_results])
db_vals = np.array([r['db'] for r in all_results])

metrics_names = [
    "Accuracy", "Precision", "Recall", "F1", # Removed "(weighted)"
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

# Tab‑separated line (ready to copy)
print("\nMethod\t" + "\t".join(metrics_names))
row = "ARMA"
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
latex_row = "ARMA"
for m, s in zip(means, stds):
    if np.isnan(m):
        latex_row += " & N/A±N/A"
    else:
        latex_row += f" & ${m:.4f}\\pm{s:.4f}$"
latex_row += " \\\\"
print(latex_row)
