#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import ARMAConv
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
ACTIV      = "SELU"
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


# ─── MODEL ────────────────────────────────────────────────────────────────────
ACTIVATIONS = {"SELU": F.selu, "SiLU": F.silu, "GELU": F.gelu,
               "ELU": F.elu, "RELU": F.relu}

class MLP(nn.Module):
    def __init__(self, inp, out, hid):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, hid), nn.BatchNorm1d(hid), nn.PReLU(),
            nn.Dropout(0.4),
            nn.Linear(hid, out)
        )
    def forward(self, x): return self.net(x)


class ARMAEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, device, activ="SELU",
                 num_stacks=1, num_layers=1, num_arma_layers=3):
        super().__init__()
        self.device = device
        self.act    = ACTIVATIONS.get(activ, F.elu)

        def _arma(i, o):
            return ARMAConv(i, o, num_stacks=num_stacks, num_layers=num_layers,
                            act=self.act, shared_weights=True, dropout=0.25)

        self.arma_layers = nn.ModuleList(
            [_arma(input_dim if i == 0 else hidden_dim, hidden_dim)
             for i in range(num_arma_layers)]
        )
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(hidden_dim)
                                        for _ in range(num_arma_layers)])
        self.drop = nn.Dropout(0.3)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, data, return_intermediates=False):
        x, ei       = data.x, data.edge_index
        intermediates = []
        for arma, bn in zip(self.arma_layers, self.bn_layers):
            x = self.drop(self.act(bn(arma(x, ei))))
            if return_intermediates:
                intermediates.append(x.detach().cpu().numpy())
        out = self.proj(x)
        return (out, intermediates) if return_intermediates else out


class EMA:
    def __init__(self, beta): self.beta = beta
    def update(self, old, new):
        return new if old is None else old * self.beta + (1 - self.beta) * new

def update_ema(ema, target, online):
    for tp, op in zip(target.parameters(), online.parameters()):
        tp.data = ema.update(tp.data, op.data)


class ARMAModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_clusters, device, activ,
                 ema_decay=0.7, cut=True, beta=0.5, tau=0.07,
                 num_arma_layers=3, T_struct=2.0):
        super().__init__()
        self.device        = device
        self.num_clusters  = num_clusters
        self.cut           = cut
        self.beta, self.tau, self.T_struct = beta, tau, T_struct

        self.online_encoder   = ARMAEncoder(input_dim, hidden_dim, device, activ,
                                            num_arma_layers=num_arma_layers)
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
    """Return logit-space and hidden-space embeddings (numpy arrays) via single forward pass."""
    model.eval()
    d = to_data(feats, ei, device)
    with torch.no_grad():
        hidden  = model.online_encoder(d)           # (N, 256)
        logits  = model.online_predictor(hidden)    # (N, K)
    return logits.cpu().numpy(), hidden.cpu().numpy()


def evaluate_clustering_from_mc(model, feats, ei, y_true, yp, logits_mean,
                                device, prefix=""):
    """
    Compute clustering metrics using the same MC-averaged predictions (yp)
    and the hidden embeddings from a single forward pass.
    The logits_mean is used only for the logit-space embeddings.
    """
    model.eval()
    d = to_data(feats, ei, device)
    with torch.no_grad():
        hidden = model.online_encoder(d).cpu().numpy()   # (N, 256)

    # For logit space, we use the MC-averaged probabilities (temperature scaled)
    logits_emb = logits_mean   # shape (N, K)

    # Ensure same label orientation as yp
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


def print_clustering_summary(all_records, depth_label="3 ARMA layers"):
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
                              csv_path="uncertainty_per_subject_CN_AD.csv"):
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


# ─── MAD METRICS ──────────────────────────────────────────────────────────────
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
        results.append(compute_mad_metrics(emb, W0, y, name=f"ARMAConv layer {i+1}"))
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
             n_eval_passes=30, num_arma_layers=3):
    np.random.seed(42 + seed_offset)
    random.seed(42 + seed_offset)
    torch.manual_seed(42 + seed_offset)

    model = ARMAModel(FEATS_DIM, 256, K, DEVICE, ACTIV,
                      ema_decay=EMA_DECAY, cut=CUT, beta=BETA, tau=TAU,
                      num_arma_layers=num_arma_layers,
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
    # --- Unweighted metrics (for compatibility with existing code) ---
    metrics_unw = (a,
                   precision_score(y, yp, zero_division=0),
                   recall_score(y, yp, zero_division=0),
                   f1_score(y, yp, zero_division=0),
                   log_loss(y, ypp))
    # --- Weighted metrics (for final table) ---
    metrics_w = (a,
                 precision_score(y, yp, average='weighted', zero_division=0),
                 recall_score(y, yp, average='weighted', zero_division=0),
                 f1_score(y, yp, average='weighted', zero_division=0),
                 log_loss(y, ypp))

    if return_probs:
        return metrics_unw, metrics_w, yp, ypp, logits_stack, logits_mean, T, model
    return metrics_unw, model


# ─── MAIN ─────────────────────────────────────────────────────────────────────
METRIC_NAMES = ["Accuracy", "Precision", "Recall", "F1", "LogLoss"]
SEP = "═" * 72

print(f"\n{SEP}\n  PART A — FIRST RUN (3 ARMA layers) – CN vs AD\n{SEP}")
metrics0_unw, metrics0_w, yp0, ypp0, logits_stack0, logits_mean0, T0, model3 = run_once(
    seed_offset=0, verbose=True, return_probs=True,
    n_eval_passes=N_EVAL_PASSES, num_arma_layers=3
)

print("\n── Single-run Classification Results ───────────────")
print("  Unweighted:")
for n, v in zip(METRIC_NAMES, metrics0_unw):
    print(f"    {n:<12}: {v:.4f}")
print("  Weighted (for final table):")
for n, v in zip(["Accuracy", "Weighted Prec", "Weighted Rec", "Weighted F1", "LogLoss"], metrics0_w):
    print(f"    {n:<12}: {v:.4f}")
print(f"  Temperature  : {T0:.3f}")

lm = logits_stack0.mean(axis=2)
diff = lm[:, 1] - lm[:, 0]
print(f"\n── Logit diagnostics ───────────────────────────────")
print(f"  mean={lm.mean():.2f}  std={lm.std():.2f}  min={lm.min():.2f}  max={lm.max():.2f}")
print(f"  Logit diff (AD-CN): mean={diff.mean():.2f}  std={diff.std():.2f}  "
      f"min={diff.min():.2f}  max={diff.max():.2f}")

df_unc = compute_uncertainty(yp0, ypp0, logits_stack0, y_true=y, T_used=T0)
print_uncertainty_report(df_unc, ypp0, y, yp0, T0, N_EVAL_PASSES,
                         csv_path="uncertainty_per_subject_CN_AD.csv")

# ── Clustering metrics for Part A (using MC predictions) ─────────────────────
clustering_results_A = evaluate_clustering_from_mc(
    model3, features_np, edge_index_np, y, yp0, logits_mean0, DEVICE,
    prefix="Part A — 3-layer ARMA, seed 0"
)

print(f"\n{SEP}\n  PART B — PER-LAYER OVER-SMOOTHING ANALYSIS\n{SEP}")
layer_similarity_analysis(model3, features_np, edge_index_np, y, W0,
                          prefix="3-layer ARMA (CN vs AD)")

print(f"\n{SEP}\n  PART C — DEPTH ABLATION  (1 / 2 / 3 ARMA layers)\n{SEP}")
ablation_results = {}
trained_models   = {}

# Storage for clustering metrics across ablation seeds
ablation_clustering = {1: [], 2: [], 3: []}

for n_layers in [1, 2, 3]:
    print(f"\n  ── Depth = {n_layers} ARMA layer(s) ──")
    seed_records = []
    for seed in range(3):
        print(f"    Seed {seed} ... ", end="", flush=True)
        (acc_s, prec_s, rec_s, f1_s, ll_s), model_i = run_once(
            seed_offset=seed, num_arma_layers=n_layers,
            n_eval_passes=N_EVAL_PASSES, return_probs=False
        )
        # Re‑evaluate to get predictions for clustering
        yp_s, ypp_s, logits_stack_s, logits_mean_s, a_s, T_s = evaluate_model(
            model_i, features_np, edge_index_np, y, DEVICE,
            n_passes=N_EVAL_PASSES, seed_base=9999 + seed
        )
        ece_s, _ = compute_ece(ypp_s, y)
        fp70, fn70, fp_t, fn_t = high_conf_errors(y, yp_s, ypp_s)

        # Collect clustering metrics for this (depth, seed) pair
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
    print("    ✓ Gap preserved – ARMA band‑pass filter intact")


# ─── PART E — 10-SEED EVALUATION WITH WEIGHTED METRICS ────────────────
print(f"\n{SEP}\n  PART E — 10-SEED EVALUATION (3 ARMA layers, CN vs AD)\n{SEP}")

def conf_group(mask, conf):
    return float(conf[mask].mean()) if mask.sum() > 0 else float('nan')

col_w = "─" * 130
print(f"  {'Seed':>4}  {'Acc':>7}  {'Prec_w':>8}  {'Rec_w':>8}  {'F1_w':>8}  "
      f"{'LogLoss':>8}  {'ECE':>7}  {'MeanConf':>9}  {'T':>6}  "
      f"{'TN':>5}  {'FP':>5}  {'FN':>5}  {'TP':>5}  "
      f"{'Conf_TN':>8}  {'Conf_FP':>8}  {'Conf_FN':>8}  {'Conf_TP':>8}")
print(f"  {col_w}")

# Storage for weighted metrics (for final table)
calibration_rows = []   # will contain weighted metrics
seed_clustering_records = []
high_conf_counts = {thr: {'fp': [], 'fn': []} for thr in [0.70, 0.80, 0.90]}

for i in range(10):
    m_unw, m_w, yp_i, ypp_i, logits_stack_i, logits_mean_i, T_i, model_i = run_once(
        seed_offset=i,
        return_probs=True,
        n_eval_passes=N_EVAL_PASSES,
        num_arma_layers=3
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

    # Compute clustering metrics for this seed using MC predictions
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

# ── 10-Run Summary (using weighted metrics) ──────────────────────────
results_np = np.array([[r['acc'], r['prec_weighted'], r['rec_weighted'], r['f1_weighted'], r['logloss'],
                        r['ece'], r['mean_conf'], r['T']]
                       for r in calibration_rows])

counts_np = np.array([[r['tn'], r['fp'], r['fn'], r['tp']]
                      for r in calibration_rows], dtype=float)

conf_np = np.array([[r['conf_tn'], r['conf_fp'], r['conf_fn'], r['conf_tp']]
                    for r in calibration_rows])

sep100 = "─" * 100
print(f"\n{sep100}")
print(f"  10-RUN SUMMARY  (mean ± std)                              [3 ARMA layers, CN vs AD]")
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

print(f"\n  ── F.1  10-SEED EVALUATION (3 ARMA layers) ──")
print_clustering_summary(seed_clustering_records, depth_label="3 ARMA layers — 10 seeds")

print(f"\n  ── F.2  DEPTH ABLATION (3 seeds per depth) ──")
for n_layers in [1, 2, 3]:
    print_clustering_summary(
        ablation_clustering[n_layers],
        depth_label=f"{n_layers} ARMA layer{'s' if n_layers > 1 else ' '} — 3 seeds"
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

# Optional: save all results to CSV
df_all = pd.DataFrame(all_weighted_results)
df_all.to_csv("arma_cn_vs_ad_10run_weighted_summary.csv", index=False, float_format="%.6f")
print(f"\n  ✓ Saved weighted summary → arma_cn_vs_ad_10run_weighted_summary.csv")
print(f"\n{'='*60}\n  ANALYSIS COMPLETE\n{'='*60}")


# In[8]:


# ─── PART G — BIOLOGICAL INTERPRETATION (CN vs AD) ──────────────────────────
print(f"\n{SEP}\n  PART G — BIOLOGICAL INTERPRETATION OF DISCOVERED CLUSTERS (CN vs AD)\n{SEP}")
print("  Using variables the model NEVER saw during training:")
print("    • Age")
print("    • Sex")
print("    • FA histogram patterns (direct from data)")
print("\n  This tests whether clusters reflect latent biological structure,")
print("  not just diagnostic label approximation.\n")

# ─── LOAD METADATA (CN + AD) ──────────────────────────────────────────────
def load_metadata(cn_path, ad_path, perm_indices):
    cn_meta  = pd.read_csv(cn_path)
    ad_meta = pd.read_csv(ad_path)
    cn_meta['diagnosis']  = 'CN'
    ad_meta['diagnosis'] = 'AD'
    all_meta = pd.concat([cn_meta, ad_meta], axis=0, ignore_index=True)
    all_meta = all_meta.iloc[perm_indices].reset_index(drop=True)
    return all_meta

cn_meta_path  = "/home/snu/Downloads/CN_Metadata.csv"
ad_meta_path = "/home/snu/Downloads/AD_Metadata.csv"   # <-- change path as needed

metadata = None
age_col  = None
sex_col  = None

try:
    metadata = load_metadata(cn_meta_path, ad_meta_path, perm)
    print(f"✓ Loaded metadata: {len(metadata)} subjects")
    print(f"  Columns: {list(metadata.columns)}")
    for col in metadata.columns:
        if 'age' in col.lower():
            age_col = col
        if 'sex' in col.lower() or 'gender' in col.lower():
            sex_col = col
    if age_col:  print(f"  → Age column found: '{age_col}'")
    else:        print(f"  ⚠ No age column found. Available: {list(metadata.columns)}")
    if sex_col:  print(f"  → Sex column found: '{sex_col}'")
    else:        print(f"  ⚠ No sex column found.")
except Exception as e:
    print(f"✗ Error loading metadata: {e}")
    print("  Continuing without metadata...")

cluster_assignments = yp0   # from your evaluation
cluster0_mask = (cluster_assignments == 0)
cluster1_mask = (cluster_assignments == 1)

print(f"\n  Cluster sizes: Cluster 0 = {cluster0_mask.sum()} subjects, "
      f"Cluster 1 = {cluster1_mask.sum()} subjects")

# ─── 1. AGE ANALYSIS ──────────────────────────────────────────────────────────
age_significant = False
if metadata is not None and age_col:
    print(f"\n{sep72}\n  1. AGE ANALYSIS\n{sep72}")
    from scipy import stats
    age_values = pd.to_numeric(metadata[age_col], errors='coerce')
    valid_age  = ~age_values.isna()
    if valid_age.sum() > 0:
        ages_cluster0 = age_values[cluster0_mask & valid_age]
        ages_cluster1 = age_values[cluster1_mask & valid_age]
        t_stat, p_val_age = stats.ttest_ind(ages_cluster0, ages_cluster1, equal_var=False)
        print(f"\n  Cluster 0 (n={len(ages_cluster0)}):  mean age = {ages_cluster0.mean():.1f} ± {ages_cluster0.std():.1f} years")
        print(f"  Cluster 1 (n={len(ages_cluster1)}):  mean age = {ages_cluster1.mean():.1f} ± {ages_cluster1.std():.1f} years")
        print(f"\n  Statistical test (Welch's t-test):\n    t = {t_stat:.3f}, p = {p_val_age:.4f}")
        age_significant = p_val_age < 0.05
        if age_significant:
            print(f"    ✓ SIGNIFICANT age difference between clusters")
            age_direction = (
                f"Cluster 1 is OLDER by {ages_cluster1.mean() - ages_cluster0.mean():.1f} years"
                if ages_cluster1.mean() > ages_cluster0.mean()
                else f"Cluster 0 is OLDER by {ages_cluster0.mean() - ages_cluster1.mean():.1f} years"
            )
            print(f"    → {age_direction}")
        else:
            print(f"    → No significant age difference (p > 0.05)")
        pooled_std = np.sqrt((ages_cluster0.std()**2 + ages_cluster1.std()**2) / 2)
        cohen_d    = abs(ages_cluster0.mean() - ages_cluster1.mean()) / pooled_std
        print(f"    Cohen's d = {cohen_d:.3f} ({'large' if cohen_d > 0.8 else 'medium' if cohen_d > 0.5 else 'small' if cohen_d > 0.2 else 'negligible'} effect)")
        print(f"\n  Age distribution by cluster:")
        print(f"    Cluster 0:  Q1={ages_cluster0.quantile(0.25):.0f}  "
              f"Median={ages_cluster0.median():.0f}  Q3={ages_cluster0.quantile(0.75):.0f}")
        print(f"    Cluster 1:  Q1={ages_cluster1.quantile(0.25):.0f}  "
              f"Median={ages_cluster1.median():.0f}  Q3={ages_cluster1.quantile(0.75):.0f}")
    else:
        print("  ⚠ No valid age data found")

# ─── 2. SEX ANALYSIS ──────────────────────────────────────────────────────────
sex_significant = False
sex_binary = None
if metadata is not None and sex_col:
    print(f"\n{sep72}\n  2. SEX ANALYSIS\n{sep72}")
    from scipy import stats
    sex_raw    = metadata[sex_col]
    unique_sex = sex_raw.unique()
    print(f"  Unique sex values: {unique_sex}")
    if len(unique_sex) == 2:
        lower_vals = [str(s).lower() for s in unique_sex]
        if 'female' in lower_vals or 'f' in lower_vals:
            female_idx  = lower_vals.index('female') if 'female' in lower_vals else lower_vals.index('f')
            female_val  = unique_sex[female_idx]
            male_val    = unique_sex[1 - female_idx]
            sex_binary  = (sex_raw == male_val).astype(int)
        elif 'male' in lower_vals or 'm' in lower_vals:
            male_idx    = lower_vals.index('male') if 'male' in lower_vals else lower_vals.index('m')
            male_val    = unique_sex[male_idx]
            female_val  = unique_sex[1 - male_idx]
            sex_binary  = (sex_raw == male_val).astype(int)
    if sex_binary is not None:
        valid_sex    = ~sex_binary.isna()
        sex_cluster0 = sex_binary[cluster0_mask & valid_sex]
        sex_cluster1 = sex_binary[cluster1_mask & valid_sex]
        pct_male0    = sex_cluster0.mean() * 100
        pct_male1    = sex_cluster1.mean() * 100
        contingency  = np.array([
            [(sex_cluster0 == 0).sum(), (sex_cluster0 == 1).sum()],
            [(sex_cluster1 == 0).sum(), (sex_cluster1 == 1).sum()]
        ])
        chi2, p_val_sex, dof, expected = stats.chi2_contingency(contingency)
        print(f"\n  Cluster 0: {pct_male0:.1f}% male / {100-pct_male0:.1f}% female (n={len(sex_cluster0)})")
        print(f"  Cluster 1: {pct_male1:.1f}% male / {100-pct_male1:.1f}% female (n={len(sex_cluster1)})")
        print(f"\n  Chi-square test: χ² = {chi2:.3f}, p = {p_val_sex:.4f}")
        sex_significant = p_val_sex < 0.05
        if sex_significant:
            print(f"  ✓ SIGNIFICANT sex imbalance between clusters")
            if pct_male1 > pct_male0:
                print(f"    → Cluster 1 has more males (+{pct_male1 - pct_male0:.1f}%)")
            else:
                print(f"    → Cluster 0 has more males (+{pct_male0 - pct_male1:.1f}%)")
        else:
            print(f"  → No significant sex difference (p > 0.05)")
    else:
        print("  ⚠ Could not map sex to binary")

# ─── 3. FA HISTOGRAM DIFFERENCES ──────────────────────────────────────────────
print(f"\n{sep72}\n  3. WHITE MATTER PHENOTYPE (FA histogram differences)\n{sep72}")
print("  This is the MOST BIOLOGICALLY RELEVANT interpretation.")
print("  The model was trained on these histograms but never saw diagnosis.")
print("  Cluster differences directly reflect white matter microstructure.\n")

hist_features = features_np
n_bins        = 20
n_directions  = hist_features.shape[1] // n_bins
hist_reshaped = (hist_features.reshape(-1, n_directions, n_bins)
                 if n_directions > 1
                 else hist_features.reshape(-1, 1, n_bins))
if n_directions <= 1:
    n_directions = 1
else:
    print(f"  Detected {n_directions} directions × {n_bins} bins = {hist_features.shape[1]} features")

mean_hist_cluster0 = hist_reshaped[cluster0_mask].mean(axis=0)
mean_hist_cluster1 = hist_reshaped[cluster1_mask].mean(axis=0)
hist_diff = mean_hist_cluster1 - mean_hist_cluster0

from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import fdrcorrection

print(f"\n  Per-bin statistical comparison (Mann-Whitney U test):")
print(f"  {'Bin':<6} {'Dir':<6} {'Cluster0':>10} {'Cluster1':>10} {'Diff':>10} {'p-value':>10}")
print(f"  {sep72}")

significant_bins = []
all_pvals        = []

for d in range(n_directions):
    for b in range(n_bins):
        vals0 = hist_reshaped[cluster0_mask, d, b]
        vals1 = hist_reshaped[cluster1_mask, d, b]
        if vals0.max() == 0 and vals1.max() == 0:
            continue
        stat, p_val = mannwhitneyu(vals0, vals1, alternative='two-sided')
        all_pvals.append(p_val)
        if p_val < 0.05:
            significant_bins.append((d, b, p_val))
        if b < 5 or b >= n_bins - 2:
            sig_marker = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"  {b:>3}   {d:>3}    {vals0.mean():>8.4f}  {vals1.mean():>8.4f}  "
                  f"{hist_diff[d,b]:>+8.4f}  {p_val:>8.4f} {sig_marker}")
    if d < n_directions - 1:
        print(f"  {'...':<6} {'...':<6} {'...':>10} {'...':>10} {'...':>10} {'...':>10}")

rejected, pvals_fdr = fdrcorrection(all_pvals, alpha=0.05)
n_sig_fdr = rejected.sum()

print(f"\n  Summary:")
print(f"    Total significant bins (uncorrected p < 0.05): {len(significant_bins)} / {len(all_pvals)}")
print(f"    Significant bins (FDR-corrected q < 0.05): {n_sig_fdr} / {len(all_pvals)}")
print(f"    Bins with Cluster1 > Cluster0: {(hist_diff > 0).sum()}")
print(f"    Bins with Cluster1 < Cluster0: {(hist_diff < 0).sum()}")

abs_diff  = np.abs(hist_diff.flatten())
top_indices = np.argsort(abs_diff)[-10:][::-1]
print(f"\n  Top 10 discriminating bins (largest absolute difference):")
print(f"  {'Bin':<6} {'Dir':<6} {'Diff (C1-C0)':>15} {'Interpretation'}")
for idx in top_indices[:10]:
    d_idx = idx // n_bins
    b_idx = idx % n_bins
    diff_val = hist_diff.flatten()[idx]
    interp = (f"Cluster1 has MORE mass in bin {b_idx}" if diff_val > 0
              else f"Cluster1 has LESS mass in bin {b_idx}")
    print(f"  {b_idx:>3}   {d_idx:>3}       {diff_val:>+12.6f}    {interp}")

print(f"\n  BIOLOGICAL INTERPRETATION:")
print(f"    Low FA bins (first 1/3)    →  Reduced integrity / free water")
print(f"    Mid FA bins (middle 1/3)   →  Normal white matter")
print(f"    High FA bins (last 1/3)    →  Highly coherent tracts")

low_bins      = slice(0, n_bins // 3)
mid_bins      = slice(n_bins // 3, 2 * n_bins // 3)
high_bins     = slice(2 * n_bins // 3, n_bins)
low_diff      = hist_diff[:, low_bins].mean()
mid_diff      = hist_diff[:, mid_bins].mean()
high_diff_val = hist_diff[:, high_bins].mean()

print(f"\n  Direction-averaged FA bin differences (Cluster1 - Cluster0):")
print(f"    Low FA bins :  {low_diff:+.4f}  {'(↑ MORE in Cluster1)' if low_diff > 0 else '(↓ LESS in Cluster1)'}")
print(f"    Mid FA bins :  {mid_diff:+.4f}  {'(↑ MORE in Cluster1)' if mid_diff > 0 else '(↓ LESS in Cluster1)'}")
print(f"    High FA bins:  {high_diff_val:+.4f}  {'(↑ MORE in Cluster1)' if high_diff_val > 0 else '(↓ LESS in Cluster1)'}")

if high_diff_val < 0 and low_diff > 0:
    fa_pattern     = "WORSE white matter integrity"
    better_cluster = "Cluster 0"
    print(f"\n  ✓ PATTERN: Cluster1 shows LOWER high-FA and HIGHER low-FA")
    print(f"    → Suggests WORSE white matter integrity in Cluster1")
    print(f"    → Cluster0 has BETTER white matter integrity")
elif high_diff_val > 0 and low_diff < 0:
    fa_pattern     = "BETTER white matter integrity"
    better_cluster = "Cluster 1"
    print(f"\n  ✓ PATTERN: Cluster1 shows HIGHER high-FA and LOWER low-FA")
    print(f"    → Suggests BETTER white matter integrity in Cluster1")
else:
    fa_pattern     = "Mixed pattern"
    better_cluster = "N/A"
    print(f"\n  → Mixed pattern - not a simple integrity difference")

# ─── 4. REFERENCE: CLUSTER vs DIAGNOSIS (CN vs AD) ──────────────────────────
print(f"\n{sep72}\n  4. REFERENCE: Cluster vs Diagnosis (CN vs AD)\n{sep72}")
print("  Note: Diagnosis was NOT used for clustering, only for post-hoc external validation.\n")

diagnosis_binary  = y   # 0=CN, 1=AD
cm_diag           = confusion_matrix(diagnosis_binary, cluster_assignments)

print(f"  Confusion Matrix (rows=true diagnosis, cols=cluster):")
print(f"                     Cluster 0    Cluster 1")
print(f"    CN (n={(diagnosis_binary==0).sum()})          {cm_diag[0,0]:>6}       {cm_diag[0,1]:>6}")
print(f"    AD (n={(diagnosis_binary==1).sum()})          {cm_diag[1,0]:>6}       {cm_diag[1,1]:>6}")

pct_cn_in_cluster0  = (diagnosis_binary[cluster0_mask] == 0).mean() * 100
pct_ad_in_cluster1 = (diagnosis_binary[cluster1_mask] == 1).mean() * 100   # renamed

print(f"\n  Cluster composition:")
print(f"    Cluster 0: {pct_cn_in_cluster0:.1f}% CN, {100-pct_cn_in_cluster0:.1f}% AD")
print(f"    Cluster 1: {100-pct_ad_in_cluster1:.1f}% CN, {pct_ad_in_cluster1:.1f}% AD")
agreement = (cluster_assignments == diagnosis_binary).mean() * 100
print(f"\n  Overall agreement: {agreement:.1f}%")

# ─── 5. SUMMARY TABLE ─────────────────────────────────────────────────────────
print(f"\n{sep72}\n  5. SUMMARY: BIOLOGICAL CHARACTERIZATION OF CLUSTERS (CN vs AD)\n{sep72}")
print(f"\n  {'Feature':<35} {'Cluster 0':>22} {'Cluster 1':>22} {'Insight':>20}")
print(f"  {sep72}")

if metadata is not None and age_col and 'ages_cluster0' in locals():
    age0_str   = f"{ages_cluster0.mean():.1f} ± {ages_cluster0.std():.1f}"
    age1_str   = f"{ages_cluster1.mean():.1f} ± {ages_cluster1.std():.1f}"
    age_insight = f"p={p_val_age:.3f} {'SIG' if age_significant else 'n.s.'}"
    print(f"  {'Age (years)':<35} {age0_str:>22} {age1_str:>22} {age_insight:>20}")

if metadata is not None and sex_col and sex_binary is not None:
    sex0_str    = f"{pct_male0:.1f}% male"
    sex1_str    = f"{pct_male1:.1f}% male"
    sex_insight = f"p={p_val_sex:.3f} {'SIG' if sex_significant else 'n.s.'}"
    print(f"  {'Sex (% male)':<35} {sex0_str:>22} {sex1_str:>22} {sex_insight:>20}")

print(f"  {'FA white matter integrity':<35} {'':>22} {'':>22} {fa_pattern:>20}")
if better_cluster != "N/A":
    print(f"  {'  → Better integrity':<35} {'':>22} {'':>22} {f'{better_cluster}':>20}")

print(f"  {'Diagnosis (% CN)':<35} {f'{pct_cn_in_cluster0:.1f}%':>22} "
      f"{f'{100-pct_ad_in_cluster1:.1f}%':>22} {'(reference only)':>20}")
print(f"  {'Sample size (n)':<35} {cluster0_mask.sum():>22} {cluster1_mask.sum():>22}")

# ─── 6. CONCLUSION (CN vs AD) ───────────────────────────────────────────────────
print(f"\n{sep72}\n  6. CONCLUSION\n{sep72}")
conclusions = []
if fa_pattern == "WORSE white matter integrity":
    conclusions += [
        "  ✓ PRIMARY FINDING: Clusters reflect biologically meaningful",
        "    differences in white matter microstructure.",
        f"    → Cluster 0 shows BETTER white matter integrity",
        f"    → Cluster 1 shows WORSE white matter integrity",
    ]
elif fa_pattern == "BETTER white matter integrity":
    conclusions += [
        "  ✓ PRIMARY FINDING: Clusters reflect biologically meaningful",
        "    differences in white matter microstructure.",
        f"    → Cluster {better_cluster} shows BETTER white matter integrity",
    ]
else:
    conclusions.append("  → FA patterns show mixed differences across the spectrum")

if age_significant:
    conclusions.append(f"  → Age also differs significantly (p={p_val_age:.3f})")
else:
    conclusions.append("  → No significant age difference between clusters")

if sex_significant:
    conclusions.append(f"  → Sex also differs significantly (p={p_val_sex:.3f})")

conclusions += [
    "\n  Interpretation:",
    "    The model discovered a latent biological axis related to",
    "    white matter health, independent of diagnostic labels (CN vs AD).",
    "",
    "    Diagnostic labels were not used during training and were",
    "    employed only for post-hoc external validation of the",
    "    discovered clusters.",
    "",
    "  Limitations:",
    "    • Single-cohort study — no external validation dataset.",
    "    • Diagnostic labels used only for post-hoc external validation.",
    "    • Generalization to unseen subjects has not been assessed.",
]
for line in conclusions:
    print(line)

print(f"\n{SEP}\n")

# ─── 7. EXPORT RESULTS (CN vs AD) ─────────────────────────────────────────────
print(f"\n{sep72}\n  7. EXPORT RESULTS FOR MANUSCRIPT (CN vs AD)\n{sep72}")

paper_results = {
    "Metric": [
        "Cluster 0 size (n)",
        "Cluster 1 size (n)",
        "Age (years) - Cluster 0",
        "Age (years) - Cluster 1",
        "Age p-value",
        "Male % - Cluster 0",
        "Male % - Cluster 1",
        "Sex p-value",
        "% CN in Cluster 0",
        "% AD in Cluster 1",
        "Diagnosis agreement (%)",
        "FA pattern",
        "Better integrity cluster",
        "Sig. FA bins (FDR q<0.05)",
        "Mean assignment entropy (bits)",
    ],
    "Value": [
        f"{cluster0_mask.sum()}",
        f"{cluster1_mask.sum()}",
        f"{ages_cluster0.mean():.1f} ± {ages_cluster0.std():.1f}" if 'ages_cluster0' in locals() else "N/A",
        f"{ages_cluster1.mean():.1f} ± {ages_cluster1.std():.1f}" if 'ages_cluster1' in locals() else "N/A",
        f"{p_val_age:.4f} ({'n.s.' if not age_significant else 'significant'})" if 'p_val_age' in locals() else "N/A",
        f"{pct_male0:.1f}%" if 'pct_male0' in locals() else "N/A",
        f"{pct_male1:.1f}%" if 'pct_male1' in locals() else "N/A",
        f"{p_val_sex:.4f} ({'n.s.' if not sex_significant else 'significant'})" if 'p_val_sex' in locals() else "N/A",
        f"{pct_cn_in_cluster0:.1f}%",
        f"{pct_ad_in_cluster1:.1f}%",
        f"{agreement:.1f}%",
        fa_pattern,
        better_cluster if better_cluster != "N/A" else "N/A",
        f"{n_sig_fdr} / {len(all_pvals)}",
        f"{df_unc0['entropy_assignment'].mean():.4f} ± {df_unc0['entropy_assignment'].std():.4f}",
    ]
}

df_paper = pd.DataFrame(paper_results)
print("\n  Summary table for manuscript:")
print(df_paper.to_string(index=False))

df_paper.to_csv("cluster_biological_characterization_AD.csv", index=False)
print(f"\n  ✓ Saved to: cluster_biological_characterization_AD.csv")

if len(significant_bins) > 0:
    bin_results = []
    for d_b, b_b, p_b in significant_bins[:50]:
        vals0 = hist_reshaped[cluster0_mask, d_b, b_b]
        vals1 = hist_reshaped[cluster1_mask, d_b, b_b]
        bin_results.append({
            "direction":       d_b,
            "bin":             b_b,
            "cluster0_mean":   vals0.mean(),
            "cluster1_mean":   vals1.mean(),
            "difference":      vals1.mean() - vals0.mean(),
            "p_value":         p_b,
            "significant_fdr": bool(fdrcorrection([p_b])[0][0]),
        })
    pd.DataFrame(bin_results).to_csv("significant_fa_bins_AD.csv", index=False)
    print(f"  ✓ Saved significant bin details to: significant_fa_bins_AD.csv")

print(f"\n{SEP}\n")
print("  BIOLOGICAL INTERPRETATION (CN vs AD) COMPLETE")
print(f"\n{SEP}")


# In[5]:


print(f"\n{SEP}\n  PART H — t-SNE: Input Features vs. ModCon-ARMA Embeddings (CN vs AD)\n{SEP}")

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# Set publication‑quality style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 300

# ----- 1. Get hidden embeddings from the trained model -----
model3.eval()
d = to_data(features_np, edge_index_np, DEVICE)
with torch.no_grad():
    hidden_embeddings = model3.online_encoder(d).cpu().numpy()  # (N, 256)

print(f"  Input features shape: {features_np.shape}")
print(f"  Hidden embeddings shape: {hidden_embeddings.shape}")

# ----- 2. Compute t‑SNE for input features and hidden embeddings -----
best_perp = 40
print(f"  Computing t‑SNE with perplexity={best_perp}...", end=" ", flush=True)

tsne_input = TSNE(n_components=2, random_state=42, perplexity=best_perp,
                  init='pca', max_iter=1000)
tsne_input_results = tsne_input.fit_transform(features_np)

tsne_hidden = TSNE(n_components=2, random_state=42, perplexity=best_perp,
                   init='pca', max_iter=1000)
tsne_hidden_results = tsne_hidden.fit_transform(hidden_embeddings)
print("done")

# ----- 3. Prepare colour masks -----
# True labels: y (0=CN, 1=AD)
cn_mask = (y == 0)
ad_mask = (y == 1)

# Cluster assignments: yp0 (from the first run)
cluster0_mask = (yp0 == 0)
cluster1_mask = (yp0 == 1)

# Uncertainty (H_assign from earlier)
H_assign = df_unc["entropy_total"].values # Use the precomputed entropy_total from df_unc
high_uncertainty_mask = H_assign > 0.7
low_uncertainty_mask = H_assign <= 0.3

# Misclassified samples (optional)
misclassified_mask = (yp0 != y)

# ============================================================================
# FIGURE 1: MAIN — Input vs. Learned (Side‑by‑Side)
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ---- Left: Input features (coloured by true diagnosis) ----
ax1.scatter(tsne_input_results[cn_mask, 0],
            tsne_input_results[cn_mask, 1],
            c='#2E86AB', label='CN (True)', alpha=0.7, s=40,
            edgecolors='white', linewidth=0.5)
ax1.scatter(tsne_input_results[ad_mask, 0],
            tsne_input_results[ad_mask, 1],
            c='#D62728', label='AD (True)', alpha=0.7, s=40,
            edgecolors='white', linewidth=0.5)
ax1.set_title('(a) Input Feature Space\n(Coloured by True Diagnosis)', fontsize=12, fontweight='bold')
ax1.set_xlabel('t-SNE Dimension 1')
ax1.set_ylabel('t-SNE Dimension 2')
ax1.legend(loc='best', framealpha=0.9)
ax1.grid(True, alpha=0.2)
ax1.set_facecolor('#f8f9fa')

# ---- Right: Learned embeddings (coloured by cluster assignment) ----
ax2.scatter(tsne_hidden_results[cluster0_mask, 0],
            tsne_hidden_results[cluster0_mask, 1],
            c='#2E86AB', label='Cluster 0 (CN-dominant)', alpha=0.7, s=40,
            edgecolors='white', linewidth=0.5)
ax2.scatter(tsne_hidden_results[cluster1_mask, 0],
            tsne_hidden_results[cluster1_mask, 1],
            c='#D62728', label='Cluster 1 (AD-dominant)', alpha=0.7, s=40,
            edgecolors='white', linewidth=0.5)

# Cluster centres
cluster0_center = tsne_hidden_results[cluster0_mask].mean(axis=0)
cluster1_center = tsne_hidden_results[cluster1_mask].mean(axis=0)
ax2.scatter(cluster0_center[0], cluster0_center[1], c='darkblue', s=150,
            edgecolors='black', linewidth=2, marker='*', label='Cluster 0 centre', zorder=3)
ax2.scatter(cluster1_center[0], cluster1_center[1], c='darkred', s=150,
            edgecolors='black', linewidth=2, marker='*', label='Cluster 1 centre', zorder=3)

ax2.set_title('(b) ModCon-ARMA Embedding Space\n(Coloured by Cluster Assignment)', fontsize=12, fontweight='bold')
ax2.set_xlabel('t-SNE Dimension 1')
ax2.set_ylabel('t-SNE Dimension 2')
ax2.legend(loc='best', framealpha=0.9)
ax2.grid(True, alpha=0.2)
ax2.set_facecolor('#f8f9fa')

plt.suptitle(f't-SNE Comparison (CN vs AD, perplexity={best_perp})', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('tsne_fig1_main_comparison_CN_AD.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("  ✓ Saved: tsne_fig1_main_comparison_CN_AD.png")

# ============================================================================
# FIGURE 2: Learned Embeddings + True Labels (Validation)
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

ax.scatter(tsne_hidden_results[cn_mask, 0],
           tsne_hidden_results[cn_mask, 1],
           c='#2E86AB', label='CN (True label)', alpha=0.7, s=50,
           edgecolors='white', linewidth=0.5)
ax.scatter(tsne_hidden_results[ad_mask, 0],
           tsne_hidden_results[ad_mask, 1],
           c='#D62728', label='AD (True label)', alpha=0.7, s=50,
           edgecolors='white', linewidth=0.5)

ax.set_title('ModCon-ARMA Embedding Space\n(Coloured by True Diagnosis, Reference)', fontsize=12, fontweight='bold')
ax.set_xlabel('t-SNE Dimension 1')
ax.set_ylabel('t-SNE Dimension 2')
ax.legend(loc='best', framealpha=0.9)
ax.grid(True, alpha=0.2)
ax.set_facecolor('#f8f9fa')

plt.tight_layout()
plt.savefig('tsne_fig2_true_labels_validation_CN_AD.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("  ✓ Saved: tsne_fig2_true_labels_validation_CN_AD.png")

# ============================================================================
# FIGURE 3: Uncertainty Heatmap (Optional)
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(9, 7))

scatter = ax.scatter(tsne_hidden_results[:, 0], tsne_hidden_results[:, 1],
                     c=H_assign, cmap='RdYlBu_r', alpha=0.8, s=60,
                     edgecolors='black', linewidth=0.5, vmin=0, vmax=1.0)

cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=25)
cbar.set_label('Assignment Entropy (bits)', fontsize=10, fontweight='bold')
cbar.ax.tick_params(labelsize=9)

if high_uncertainty_mask.sum() > 0:
    ax.scatter(tsne_hidden_results[high_uncertainty_mask, 0],
               tsne_hidden_results[high_uncertainty_mask, 1],
               c='red', s=120, edgecolors='black', linewidth=1.5,
               marker='X', label=f'Ambiguous (n={high_uncertainty_mask.sum()}, entropy > 0.7)', zorder=5)

ax.set_title('ModCon-ARMA Embedding Space\n(Coloured by Assignment Uncertainty)', fontsize=12, fontweight='bold')
ax.set_xlabel('t-SNE Dimension 1')
ax.set_ylabel('t-SNE Dimension 2')
ax.legend(loc='best', framealpha=0.9)
ax.grid(True, alpha=0.2)
ax.set_facecolor('#f8f9fa')

plt.tight_layout()
plt.savefig('tsne_fig3_uncertainty_heatmap_CN_AD.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("  ✓ Saved: tsne_fig3_uncertainty_heatmap_CN_AD.png")

# ============================================================================
# FIGURE 4: Misclassified Samples (Optional)
# ============================================================================
if misclassified_mask.sum() > 0:
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    correct_mask = ~misclassified_mask
    ax.scatter(tsne_hidden_results[correct_mask, 0],
               tsne_hidden_results[correct_mask, 1],
               c='gray', label='Correctly assigned', alpha=0.5, s=30,
               edgecolors='none')

    ax.scatter(tsne_hidden_results[misclassified_mask, 0],
               tsne_hidden_results[misclassified_mask, 1],
               c='red', label=f'Misclassified (n={misclassified_mask.sum()})',
               alpha=0.9, s=80, edgecolors='black', linewidth=1.5, marker='X')

    ax.set_title('ModCon-ARMA Embedding Space\n(Misclassified Samples Highlighted)', fontsize=12, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.2)
    ax.set_facecolor('#f8f9fa')

    plt.tight_layout()
    plt.savefig('tsne_fig4_misclassified_CN_AD.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print("  ✓ Saved: tsne_fig4_misclassified_CN_AD.png")

print("\n  ✅ All t-SNE figures saved for CN vs AD.")


# In[7]:


# ─── PART H — t-SNE VISUALIZATION OF CLUSTERS (CN vs AD) ───────────────────────
print(f"\n{SEP}\n  PART H — t-SNE VISUALIZATION OF DISCOVERED CLUSTERS (CN vs AD)\n{SEP}")
print("  Visualizing the hidden space embeddings to assess cluster separation.\n")

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist

# Set publication-quality style (slightly reduced for CN vs AD)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 300

# Get hidden embeddings from your trained model
model3.eval()
d = to_data(features_np, edge_index_np, DEVICE)
with torch.no_grad():
    hidden_embeddings = model3.online_encoder(d).cpu().numpy()  # (N, 256)

print(f"  Hidden embedding shape: {hidden_embeddings.shape}")

# Get uncertainty values (from your CN vs AD run)
# Note: These should be from your CN vs AD logits_stack0
# The compute_uncertainty_from_logits function returns: MI, H_pred (total entropy), H_aleat, pred_var, p_mean
# We need H_assign to be H_pred (total entropy) and H_aleat to be H_aleat.
_, H_assign, H_aleat, _, _ = compute_uncertainty_from_logits(logits_stack0)
high_uncertainty_mask = H_assign > 0.7
low_uncertainty_mask = H_assign <= 0.3

# Compute t-SNE once with best perplexity (smaller for CN vs AD n=223)
best_perp = 40  # Reduced from 50 for smaller dataset
print(f"  Computing t-SNE with perplexity={best_perp}...", end=" ", flush=True)
tsne = TSNE(n_components=2, random_state=42, perplexity=best_perp, init='pca', max_iter=1000)
tsne_results = tsne.fit_transform(hidden_embeddings)
print("done")

# Get cluster masks for CN vs AD
cluster0_mask = (yp0 == 0)
cluster1_mask = (yp0 == 1)
cn_mask = (y == 0)
ad_mask = (y == 1)

# Cluster centers
cluster0_center = tsne_results[cluster0_mask].mean(axis=0)
cluster1_center = tsne_results[cluster1_mask].mean(axis=0)

# Separation ratio (compute if not already)
from sklearn.metrics import pairwise_distances
cluster0_emb = hidden_embeddings[cluster0_mask]
cluster1_emb = hidden_embeddings[cluster1_mask]
intra0 = pairwise_distances(cluster0_emb).mean() if len(cluster0_emb) > 1 else 0
intra1 = pairwise_distances(cluster1_emb).mean() if len(cluster1_emb) > 1 else 0
intra_mean = (intra0 + intra1) / 2
inter = pairwise_distances(cluster0_emb, cluster1_emb).mean() if len(cluster0_emb) > 0 and len(cluster1_emb) > 0 else 0
separation_ratio = inter / intra_mean if intra_mean > 0 else 0

# ============================================================================
# FIGURE 1: t-SNE colored by cluster assignment (reduced size)
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

ax.scatter(tsne_results[cluster0_mask, 0], tsne_results[cluster0_mask, 1],
           c='#2E86AB', label='Cluster 0 (Healthier phenotype)',
           alpha=0.7, s=50, edgecolors='white', linewidth=0.5, zorder=2)
ax.scatter(tsne_results[cluster1_mask, 0], tsne_results[cluster1_mask, 1],
           c='#A23B72', label='Cluster 1 (Worse phenotype)',
           alpha=0.7, s=50, edgecolors='white', linewidth=0.5, zorder=2)

ax.scatter(cluster0_center[0], cluster0_center[1], c='darkblue', s=150,
           edgecolors='black', linewidth=2, marker='*', label='Cluster 0 center', zorder=3)
ax.scatter(cluster1_center[0], cluster1_center[1], c='darkred', s=150,
           edgecolors='black', linewidth=2, marker='*', label='Cluster 1 center', zorder=3)

ax.set_title('t-SNE: ARMA Embeddings (CN vs AD)\nColored by Cluster Assignment',
             fontsize=12, fontweight='bold')
ax.set_xlabel('t-SNE Dimension 1', fontsize=10)
ax.set_ylabel('t-SNE Dimension 2', fontsize=10)
ax.legend(loc='best', framealpha=0.9, fontsize=8)
ax.grid(True, alpha=0.2, linestyle='--')
ax.set_facecolor('#f8f9fa')

plt.tight_layout()
plt.savefig('tsne_by_cluster_assignment_CN_vs_AD.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("  ✓ Saved: tsne_by_cluster_assignment_CN_vs_AD.png")

# ============================================================================
# FIGURE 2: t-SNE colored by true diagnosis (reduced size)
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

ax.scatter(tsne_results[cn_mask, 0], tsne_results[cn_mask, 1],
           c='#2E86AB', label='CN (True diagnosis)',
           alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
ax.scatter(tsne_results[ad_mask, 0], tsne_results[ad_mask, 1],
           c='#F18F01', label='AD (True diagnosis)',
           alpha=0.7, s=50, edgecolors='white', linewidth=0.5)

ax.set_title('t-SNE: ARMA Embeddings (CN vs AD)\nColored by True Diagnosis (Reference)',
             fontsize=12, fontweight='bold')
ax.set_xlabel('t-SNE Dimension 1', fontsize=10)
ax.set_ylabel('t-SNE Dimension 2', fontsize=10)
ax.legend(loc='best', framealpha=0.9, fontsize=8)
ax.grid(True, alpha=0.2, linestyle='--')
ax.set_facecolor('#f8f9fa')

plt.tight_layout()
plt.savefig('tsne_by_true_diagnosis_CN_vs_AD.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("  ✓ Saved: tsne_by_true_diagnosis_CN_vs_AD.png")

# ============================================================================
# FIGURE 3: t-SNE with uncertainty heatmap (reduced size)
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(9, 7))

scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1],
                     c=H_assign, cmap='RdYlBu_r',
                     alpha=0.8, s=60, edgecolors='black', linewidth=0.5,
                     vmin=0, vmax=1.0)

cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=25)
cbar.set_label('Assignment Entropy (bits)', fontsize=10, fontweight='bold')
cbar.ax.tick_params(labelsize=9)

if high_uncertainty_mask.sum() > 0:
    ax.scatter(tsne_results[high_uncertainty_mask, 0],
               tsne_results[high_uncertainty_mask, 1],
               c='red', s=120, edgecolors='black', linewidth=1.5,
               marker='X', label=f'Ambiguous (n={high_uncertainty_mask.sum()}, entropy > 0.7 bits)',
               zorder=5)

ax.scatter(cluster0_center[0], cluster0_center[1], c='darkblue', s=180,
           edgecolors='white', linewidth=2, marker='o', label='Cluster 0 center', zorder=4)
ax.scatter(cluster1_center[0], cluster1_center[1], c='darkred', s=180,
           edgecolors='white', linewidth=2, marker='o', label='Cluster 1 center', zorder=4)

stats_text = f'Mean Entropy: {H_assign.mean():.4f} \u00B1 {H_assign.std():.4f} bits\n'
stats_text += f'Confident (entropy \u2264 0.3): {low_uncertainty_mask.sum()}/{len(H_assign)} ({100*low_uncertainty_mask.sum()/len(H_assign):.1f}%)\n'
stats_text += f'Ambiguous (entropy > 0.7): {high_uncertainty_mask.sum()}/{len(H_assign)} ({100*high_uncertainty_mask.sum()/len(H_assign):.1f}%)\n'
stats_text += f'Separation Ratio: {separation_ratio:.2f}'

ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.set_title('t-SNE: ARMA Embeddings (CN vs AD)\nColored by Assignment Uncertainty',
             fontsize=12, fontweight='bold')
ax.set_xlabel('t-SNE Dimension 1', fontsize=10)
ax.set_ylabel('t-SNE Dimension 2', fontsize=10)
ax.legend(loc='lower right', framealpha=0.9, fontsize=8)
ax.grid(True, alpha=0.2, linestyle='--')
ax.set_facecolor('#f8f9fa')

plt.tight_layout()
plt.savefig('tsne_uncertainty_heatmap_CN_vs_AD.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("  ✓ Saved: tsne_uncertainty_heatmap_CN_vs_AD.png")

# ============================================================================
# FIGURE 4: Side-by-side comparison (reduced size)
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

ax = axes[0]
ax.scatter(tsne_results[cluster0_mask, 0], tsne_results[cluster0_mask, 1],
           c='#2E86AB', label='Cluster 0', alpha=0.7, s=40, edgecolors='white', linewidth=0.5)
ax.scatter(tsne_results[cluster1_mask, 0], tsne_results[cluster1_mask, 1],
           c='#A23B72', label='Cluster 1', alpha=0.7, s=40, edgecolors='white', linewidth=0.5)
ax.scatter(cluster0_center[0], cluster0_center[1], c='darkblue', s=120,
           edgecolors='black', linewidth=2, marker='*', zorder=3)
ax.scatter(cluster1_center[0], cluster1_center[1], c='darkred', s=120,
           edgecolors='black', linewidth=2, marker='*', zorder=3)
ax.set_title('By Cluster Assignment', fontsize=11, fontweight='bold')
ax.set_xlabel('t-SNE Dimension 1')
ax.set_ylabel('t-SNE Dimension 2')
ax.legend(loc='best', fontsize=8)
ax.grid(True, alpha=0.2)

ax = axes[1]
ax.scatter(tsne_results[cn_mask, 0], tsne_results[cn_mask, 1],
           c='#2E86AB', label='CN', alpha=0.7, s=40, edgecolors='white', linewidth=0.5)
ax.scatter(tsne_results[ad_mask, 0], tsne_results[ad_mask, 1],
           c='#F18F01', label='AD', alpha=0.7, s=40, edgecolors='white', linewidth=0.5)
ax.set_title('By True Diagnosis (Reference)', fontsize=11, fontweight='bold')
ax.set_xlabel('t-SNE Dimension 1')
ax.set_ylabel('t-SNE Dimension 2')
ax.legend(loc='best', fontsize=8)
ax.grid(True, alpha=0.2)

plt.suptitle(f't-SNE Visualization (CN vs AD, perplexity={best_perp})', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('tsne_comparison_CN_vs_AD.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("  ✓ Saved: tsne_comparison_CN_vs_AD.png")

# ============================================================================
# FIGURE 5: Entropy distribution histogram (reduced size)
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

n, bins, patches = ax.hist(H_assign, bins=25, edgecolor='black', alpha=0.7, color='steelblue')

for i, (patch, bin_edge) in enumerate(zip(patches, bins[:-1])):
    if bin_edge < 0.3:
        patch.set_facecolor('#2E86AB')
    elif bin_edge > 0.7:
        patch.set_facecolor('#F18F01')
    else:
        patch.set_facecolor('#A23B72')

ax.axvline(H_assign.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {H_assign.mean():.4f}')
ax.axvline(0.3, color='green', linestyle=':', linewidth=1.5, label='Confidence threshold (0.3)')
ax.axvline(0.7, color='orange', linestyle=':', linewidth=1.5, label='Ambiguity threshold (0.7)')

ax.set_xlabel('Assignment Entropy (bits)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Distribution of Cluster Assignment Entropy (CN vs AD)', fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.2, axis='y')

ax.text(0.98, 0.98, f'n = {len(H_assign)}\nMean = {H_assign.mean():.4f}\nStd = {H_assign.std():.4f}',
        transform=ax.transAxes, fontsize=9, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('entropy_distribution_CN_vs_AD.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("  ✓ Saved: entropy_distribution_CN_vs_AD.png")

# ============================================================================
# Print quantitative cluster separation metrics
# ============================================================================
print(f"\n{sep72}")
print("  CLUSTER SEPARATION METRICS (CN vs AD - from hidden embeddings)")
print(f"{sep72}")

print(f"\n  Intra-cluster distance (Cluster 0): {intra0:.4f}")
print(f"  Intra-cluster distance (Cluster 1): {intra1:.4f}")
print(f"  Inter-cluster distance: {inter:.4f}")
print(f"  Separation ratio (inter/intra): {separation_ratio:.4f}")
if separation_ratio > 1.5:
    print(f"  \u2713 Good separation")
elif separation_ratio > 1.0:
    print(f"  \u26A0 Moderate separation")
else:
    print(f"  \u2717 Poor separation")

# Cluster diameters
if len(cluster0_emb) > 1:
    cluster0_diameter = cdist(cluster0_emb, cluster0_emb).max()
else:
    cluster0_diameter = 0
if len(cluster1_emb) > 1:
    cluster1_diameter = cdist(cluster1_emb, cluster1_emb).max()
else:
    cluster1_diameter = 0

print(f"\n  Cluster 0 diameter: {cluster0_diameter:.4f}")
print(f"  Cluster 1 diameter: {cluster1_diameter:.4f}")
if cluster0_diameter > 0:
    print(f"  Compactness (1/diameter): Cluster 0 = {1/cluster0_diameter:.4f}, Cluster 1 = {1/cluster1_diameter:.4f}")

print(f"\n{sep72}")
print("  t-SNE visualization complete (CN vs AD)")
print(f"{sep72}\n")


# In[ ]:


"""
ARMA Model – Ablation Study for CN vs AD
Systematically varies hyperparameters and reports clustering performance.
Run this script after ensuring all dependencies are installed.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import ARMAConv
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
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import fdrcorrection

# =============================================================================
# CONFIGURATION (default values for CN vs AD – matches your main script)
# =============================================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FEATS_DIM = 180
K = 2
ACTIV = "SELU"               # default activation
ALPHA = 0.83                 # graph threshold (changed from 0.92 to 0.83 for AD)
CUT = 0                      # 0 = cut loss, 1 = modularity loss
TAU = 0.5                    # changed from 0.07 to 0.5 (matches main config)
BETA = 0.5
EMA_DECAY = 0.7
LAMBDA_CON = 4
NUM_EPOCHS = 2500            # you can reduce for faster ablations
T_STRUCT = 2.0
C_ENTROPY = 0.05             # changed from 0.005 to 0.05 (matches main config)
N_EVAL_PASSES = 30
NUM_ARMA_LAYERS = 3          # number of ARMAConv layers
NUM_STACKS = 1
NUM_LAYERS = 1
GAT_DROPOUT = 0.25
DROPOUT = 0.3
MLP_DROPOUT = 0.4

# Ablation settings
NUM_SEEDS = 10               # number of random seeds per configuration
EPOCHS_REDUCED = 2500        # use full epochs (2500) for reliable results
N_EVAL_PASSES_REDUCED = 30   # use full MC passes

# Parameter grids (each as list of values)
ABLATION_GRIDS = {
    "ACTIV": ["SELU", "ELU", "RELU", "GELU"],
    "NUM_ARMA_LAYERS": [1, 2, 3],
    "LAMBDA_CON": [0, 1, 4, 8],
    "C_ENTROPY": [0, 0.05, 0.1],          # adjusted from 0,0.005,0.01 to match main config
    "T_STRUCT": [1.0, 2.0, 3.0],
    "EMA_DECAY": [0.5, 0.7, 0.9],
    "TAU": [0.3, 0.5, 0.7],               # adjusted to surround 0.5
    "BETA": [0.3, 0.5, 0.7],
    "ALPHA": [0.80, 0.83, 0.86],          # values around 0.83
    "CUT": [0, 1],                        # cut loss vs modularity loss
}

# =============================================================================
# DATA LOADING – CN vs AD (fixed permutation, same as main script)
# =============================================================================
cn_data  = np.load("/home/snu/Downloads/Histogram_CN_FA_20bin_updated.npy",  allow_pickle=True)
ad_data  = np.load("/home/snu/Downloads/Histogram_AD_FA_20bin_updated.npy", allow_pickle=True)

X = np.vstack([cn_data, ad_data]).astype(np.float32)
y = np.hstack([np.zeros(cn_data.shape[0], dtype=np.int64),
               np.ones(ad_data.shape[0],  dtype=np.int64)])

np.random.seed(42)
perm = np.random.permutation(X.shape[0])
X, y = X[perm], y[perm]
features_np = X.astype(np.float32)
print(f"Data loaded: {X.shape[0]} subjects, {X.shape[1]} features.")
print(f"CN: {np.sum(y==0)}, AD: {np.sum(y==1)}")

# =============================================================================
# GRAPH CONSTRUCTION (will be recomputed when ALPHA changes)
# =============================================================================
def create_adj(features, alpha=1.0):
    F_ = features / np.linalg.norm(features, axis=1, keepdims=True)
    W  = np.dot(F_, F_.T)
    W = np.where(W >= alpha, 1, 0).astype(np.float32)
    W = (W / W.max()).astype(np.float32)
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

# =============================================================================
# LOSSES (same as original)
# =============================================================================
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

# =============================================================================
# MODEL COMPONENTS (same as original)
# =============================================================================
ACTIVATIONS = {"SELU": F.selu, "SiLU": F.silu, "GELU": F.gelu,
               "ELU": F.elu, "RELU": F.relu}

class MLP(nn.Module):
    def __init__(self, inp, out, hid):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, hid), nn.BatchNorm1d(hid), nn.PReLU(),
            nn.Dropout(MLP_DROPOUT),
            nn.Linear(hid, out)
        )
    def forward(self, x): return self.net(x)

class ARMAEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, device, activ="SELU",
                 num_stacks=1, num_layers=1, num_arma_layers=3):
        super().__init__()
        self.device = device
        self.act    = ACTIVATIONS.get(activ, F.elu)
        def _arma(i, o):
            return ARMAConv(i, o, num_stacks=num_stacks, num_layers=num_layers,
                            act=self.act, shared_weights=True, dropout=GAT_DROPOUT)
        self.arma_layers = nn.ModuleList(
            [_arma(input_dim if i == 0 else hidden_dim, hidden_dim)
             for i in range(num_arma_layers)]
        )
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_arma_layers)])
        self.drop = nn.Dropout(DROPOUT)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, data, return_intermediates=False):
        x, ei = data.x, data.edge_index
        intermediates = []
        for arma, bn in zip(self.arma_layers, self.bn_layers):
            x = self.drop(self.act(bn(arma(x, ei))))
            if return_intermediates:
                intermediates.append(x.detach().cpu().numpy())
        out = self.proj(x)
        return (out, intermediates) if return_intermediates else out

class EMA:
    def __init__(self, beta): self.beta = beta
    def update(self, old, new):
        return new if old is None else old * self.beta + (1 - self.beta) * new

def update_ema(ema, target, online):
    for tp, op in zip(target.parameters(), online.parameters()):
        tp.data = ema.update(tp.data, op.data)

class ARMAModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_clusters, device, activ,
                 ema_decay, cut, beta, tau, num_arma_layers, T_struct):
        super().__init__()
        self.device = device
        self.num_clusters = num_clusters
        self.cut = cut
        self.beta, self.tau, self.T_struct = beta, tau, T_struct
        self.online_encoder = ARMAEncoder(input_dim, hidden_dim, device, activ,
                                          num_arma_layers=num_arma_layers)
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

# =============================================================================
# EVALUATION FUNCTIONS (simplified for speed, but with weighted metrics)
# =============================================================================
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

def evaluate_model(model, feats, ei, y, device, n_passes=10, seed_base=9999):
    all_logits = []
    with mc_dropout_mode(model):
        with torch.no_grad():
            for i in range(n_passes):
                rng = np.random.default_rng(seed_base + i)
                fa = feats * (rng.random(feats.shape) >= 0.2).astype(np.float32)
                dr = 0.15 + 0.10 * (i % 3)
                ei_ = aug_edge(ei, dr, seed=seed_base + i)
                d = to_data(fa, ei_, device)
                lg = model.online_predictor(model.online_encoder(d)).cpu().numpy()
                all_logits.append(lg)
    logits_stack = np.stack(all_logits, axis=2)
    logits_mean = logits_stack.mean(axis=2)
    yp = np.argmax(logits_mean, axis=1)
    a = accuracy_score(y, yp)
    ai = accuracy_score(y, 1 - yp)
    if ai > a:
        yp = 1 - yp
        logits_mean = logits_mean[:, ::-1].copy()
    ypp = F.softmax(torch.from_numpy(logits_mean), dim=1).numpy()
    return yp, ypp, logits_stack, logits_mean

def compute_entropy(probs):
    p = np.clip(probs, 1e-12, 1.0)
    return -np.sum(p * np.log2(p), axis=1)

def compute_metrics(y_true, y_pred, y_prob, features):
    # Weighted classification metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    # Clustering metrics
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic')
    ami = adjusted_mutual_info_score(y_true, y_pred, average_method='arithmetic')
    # Geometric metrics (silhouette may be NaN)
    if len(np.unique(y_pred)) >= 2:
        sil = silhouette_score(features, y_pred, metric='euclidean')
        db = davies_bouldin_score(features, y_pred)
    else:
        sil, db = np.nan, np.nan
    # Entropy of soft assignments
    entropy = compute_entropy(y_prob)
    mean_ent = entropy.mean()
    std_ent = entropy.std()
    return {
        'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1,
        'nmi': nmi, 'ari': ari, 'ami': ami,
        'silhouette': sil, 'db': db,
        'mean_entropy': mean_ent, 'std_entropy': std_ent
    }

def train_and_evaluate(config, seeds=NUM_SEEDS, epochs=EPOCHS_REDUCED, n_eval=N_EVAL_PASSES_REDUCED):
    """Train model with given config (dict of hyperparameters) and return mean/std over seeds."""
    results = []
    for seed in range(seeds):
        np.random.seed(42 + seed)
        random.seed(42 + seed)
        torch.manual_seed(42 + seed)
        # Build graph with current ALPHA
        W0 = create_adj(features_np, config['ALPHA'])
        A1 = torch.from_numpy(W0).float().to(DEVICE)
        ei_np, _ = edge_index_from_dense(W0)
        # Model
        model = ARMAModel(FEATS_DIM, 256, K, DEVICE, config['ACTIV'],
                          ema_decay=config['EMA_DECAY'], cut=config['CUT'],
                          beta=config['BETA'], tau=config['TAU'],
                          num_arma_layers=config['NUM_ARMA_LAYERS'],
                          T_struct=config['T_STRUCT']).to(DEVICE)
        opt = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        sch = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
        for ep in range(epochs):
            rng = np.random.default_rng(ep + seed)
            fa1 = features_np * (rng.random(features_np.shape) >= 0.2).astype(np.float32)
            af2 = features_np.copy()
            n_, d_ = af2.shape
            fi = rng.choice(n_ * d_, size=int(n_ * d_ * 0.2), replace=False)
            af2[fi // d_, fi % d_] = 0.0
            fa2 = af2.astype(np.float32)
            d1 = to_data(fa1, aug_edge(ei_np, 0.2, seed=ep + seed), DEVICE)
            d2 = to_data(fa2, aug_edge(ei_np, 0.2, seed=ep + seed + 999), DEVICE)
            model.train()
            opt.zero_grad()
            lg1, lg2, l1, l2 = model(d1, d2)
            p1 = F.softmax(lg1, dim=1)
            entropy = -(p1 * torch.log(p1 + 1e-8)).sum(dim=1).mean()
            cont = (l1 + l2) / 2.0
            loss = model.struct_loss(A1, lg1) + config['LAMBDA_CON'] * cont - config['C_ENTROPY'] * entropy
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            sch.step()
            model.update_ma()
        # Evaluate
        yp, ypp, _, _ = evaluate_model(model, features_np, ei_np, y, DEVICE, n_passes=n_eval)
        metrics = compute_metrics(y, yp, ypp, features_np)
        results.append(metrics)
    # Aggregate over seeds
    agg = {}
    for key in results[0].keys():
        vals = np.array([r[key] for r in results])
        agg[key] = (np.nanmean(vals), np.nanstd(vals))
    return agg

# =============================================================================
# MAIN ABLATION LOOP
# =============================================================================
def run_ablation():
    all_rows = []
    # First, run baseline with default parameters
    print("Running baseline configuration...")
    default_config = {
        'ACTIV': ACTIV, 'NUM_ARMA_LAYERS': NUM_ARMA_LAYERS, 'LAMBDA_CON': LAMBDA_CON,
        'C_ENTROPY': C_ENTROPY, 'T_STRUCT': T_STRUCT, 'EMA_DECAY': EMA_DECAY,
        'TAU': TAU, 'BETA': BETA, 'ALPHA': ALPHA, 'CUT': CUT
    }
    baseline = train_and_evaluate(default_config, seeds=NUM_SEEDS, epochs=EPOCHS_REDUCED)
    row = {'Parameter': 'Baseline', 'Value': 'default'}
    for k, (m, s) in baseline.items():
        row[f'{k}_mean'] = m
        row[f'{k}_std'] = s
    all_rows.append(row)
    # Now loop over each hyperparameter
    for param, values in ABLATION_GRIDS.items():
        for val in values:
            if val == default_config.get(param):
                continue   # skip duplicate of baseline
            print(f"\nAblating {param} = {val}")
            config = default_config.copy()
            config[param] = val
            # Special handling for ALPHA: need to recompute graph each time – already done in train_and_evaluate
            res = train_and_evaluate(config, seeds=NUM_SEEDS, epochs=EPOCHS_REDUCED)
            row = {'Parameter': param, 'Value': str(val)}
            for k, (m, s) in res.items():
                row[f'{k}_mean'] = m
                row[f'{k}_std'] = s
            all_rows.append(row)
    # Convert to DataFrame and save
    df = pd.DataFrame(all_rows)
    csv_path = "arma_ablation_results_cn_vs_ad.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nAblation completed. Results saved to {csv_path}")
    # Print summary table (optional)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print(df.round(4).to_string())
    return df

if __name__ == "__main__":
    run_ablation()

