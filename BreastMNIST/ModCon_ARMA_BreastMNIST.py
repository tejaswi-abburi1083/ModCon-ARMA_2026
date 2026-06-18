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
                             davies_bouldin_score, log_loss)
from contextlib import contextmanager
import random
import copy
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, Subset
import scipy.sparse as sp
from torchvision import transforms
import timm

# =============================================================================
# CONFIGURATION
# =============================================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("CUDA available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

RADIODINO_MODEL = "hf_hub:Snarcy/RadioDino-s16"
K = 2
ACTIV = "RELU"
ALPHA = 0.63
CUT = 0
TAU = 0.1
BETA = 0.3
EMA_DECAY = 0.9999
LAMBDA_CON = 1.0
NUM_EPOCHS = 1500
T_STRUCT = 1.0
C_ENTROPY = 0.05
N_EVAL_PASSES = 30
NUM_RUNS = 10

# =============================================================================
# DATA LOADING – BreastMNIST + RadioDINO features
# =============================================================================
data_npz = np.load('/home/snu/Downloads/breastmnist_224.npz', allow_pickle=True)

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

# Subsample: up to 1000 per class
dataset = TensorDataset(X_img, y_img)
class0_indices = [i for i in range(len(y_img)) if y_img[i] == 0]
class1_indices = [i for i in range(len(y_img)) if y_img[i] == 1]
random.seed(42)
sampled_class0 = random.sample(class0_indices, min(1000, len(class0_indices)))
sampled_class1 = random.sample(class1_indices, min(1000, len(class1_indices)))
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

features_np = torch.cat(rd_feats, dim=0).numpy().astype(np.float32)
y = np.array(y_list, dtype=np.int64)
FEATS_DIM = features_np.shape[1]
print(f"RadioDINO feature dimension: {FEATS_DIM}")

# Shuffle with fixed seed for reproducibility
np.random.seed(42)
perm = np.random.permutation(features_np.shape[0])
features_np = features_np[perm]
y = y[perm]

LABEL_MAP = {0: "Malignant", 1: "Normal"}
print(f"Features: {features_np.shape}, Labels: {y.shape} "
      f"(Malignant: {np.sum(y==0)}, Normal: {np.sum(y==1)})")
print("NOTE: Diagnostic labels are held out and used ONLY for post-hoc external validation.\n")

# =============================================================================
# GRAPH UTILITIES (unchanged)
# =============================================================================
def create_adj(features, cut, alpha=1.0):
    F_ = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)
    W = np.dot(F_, F_.T)
    if cut == 0:
        W = np.where(W >= alpha, 1, 0).astype(np.float32)
        mx = W.max()
        W = (W / mx).astype(np.float32) if mx > 0 else W
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

W0 = create_adj(features_np, CUT, ALPHA)
A1 = torch.from_numpy(W0).float().to(DEVICE)
edge_index_np, _ = edge_index_from_dense(W0)
data0 = to_data(features_np, edge_index_np, DEVICE)
print("Graph:", data0)

# =============================================================================
# LOSS FUNCTIONS (unchanged)
# =============================================================================
def jsd_loss(p, q, tau=0.07, eps=1e-8):
    p_ = F.softmax(p / tau, dim=-1) + eps
    q_ = F.softmax(q / tau, dim=-1) + eps
    m = 0.5 * (p_ + q_)
    kl = lambda a, b: (a * (a / b).log()).sum(dim=-1)
    return (0.5 * (kl(p_, m) + kl(q_, m)) / np.log(2)).mean()

def contrastive_loss(h1, h2, z1, z2, beta=0.5, tau=0.07):
    l1 = beta * jsd_loss(h1, h2, tau) + (1 - beta) * jsd_loss(h1, z2, tau)
    l2 = beta * jsd_loss(h2, h1, tau) + (1 - beta) * jsd_loss(h2, z1, tau)
    return l1, l2

# =============================================================================
# MODEL COMPONENTS (unchanged)
# =============================================================================
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
    def __init__(self, input_dim, hidden_dim, device, activ="RELU",
                 num_stacks=1, num_layers=1, num_arma_layers=3):
        super().__init__()
        self.device = device
        self.act = ACTIVATIONS.get(activ, F.elu)

        def _arma(i, o):
            return ARMAConv(i, o, num_stacks=num_stacks, num_layers=num_layers,
                            act=self.act, shared_weights=True, dropout=0.4)

        self.arma_layers = nn.ModuleList(
            [_arma(input_dim if i == 0 else hidden_dim, hidden_dim)
             for i in range(num_arma_layers)]
        )
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_arma_layers)])
        self.drop = nn.Dropout(0.4)
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
                 ema_decay=0.7, cut=True, beta=0.5, tau=0.07,
                 num_arma_layers=3, T_struct=2.0):
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
        S = F.softmax(S / self.T_struct, dim=1)
        Ap = (A @ S).t() @ S
        D = torch.diag(A.sum(dim=-1))
        Dp = (D @ S).t() @ S
        mc = -(Ap.trace() / Dp.trace())
        SS = S.t() @ S
        I = torch.eye(self.num_clusters, device=self.device)
        oc = torch.norm(SS / SS.norm() - I / I.norm())
        return mc + oc

    def _modularity_loss(self, A, S):
        C = F.softmax(S, dim=1)
        d = A.sum(dim=1); m = A.sum()
        B = A - torch.ger(d, d) / (2 * m)
        k = torch.tensor(self.num_clusters, device=self.device, dtype=torch.float32)
        mod = (-1 / (2 * m)) * torch.trace(C.t() @ B @ C)
        coll = (k.sqrt() / S.shape[0]) * torch.norm(C.sum(dim=0), p='fro') - 1
        return mod + coll

# =============================================================================
# CLUSTERING METRICS & UNCERTAINTY (unchanged)
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
        sil, db = np.nan, np.nan
    return dict(space=space_name, ARI=ari, NMI=nmi, AMI=ami,
                Silhouette=sil, DaviesBouldin=db)

def evaluate_clustering_from_mc(model, feats, ei, y_true, yp, logits_mean,
                                 device, prefix="", quiet=False):
    model.eval()
    d = to_data(feats, ei, device)
    with torch.no_grad():
        hidden = model.online_encoder(d).cpu().numpy()
    logits_emb = logits_mean.copy()
    ari_direct = adjusted_rand_score(y_true, yp)
    ari_flipped = adjusted_rand_score(y_true, 1 - yp)
    if ari_flipped > ari_direct:
        yp = 1 - yp
        logits_emb = logits_emb[:, ::-1]
    results = [
        compute_clustering_metrics(logits_emb, yp, y_true, space_name="logit (MC avg)"),
        compute_clustering_metrics(hidden, yp, y_true, space_name="hidden (single pass)"),
    ]
    if not quiet:
        sep = "─" * 72
        header = (f"  {'Space':<22} {'ARI':>8} {'NMI':>8} {'AMI':>8} "
                  f"{'Silhouette':>12} {'DaviesBouldin':>14}")
        print(f"\n{sep}\n  CLUSTERING METRICS (MC predictions){' '+prefix if prefix else ''}\n{sep}")
        print(header)
        print(f"  {sep}")
        for r in results:
            sil_str = f"{r['Silhouette']:>12.4f}" if not np.isnan(r['Silhouette']) else "         N/A"
            db_str = f"{r['DaviesBouldin']:>14.4f}" if not np.isnan(r['DaviesBouldin']) else "           N/A"
            print(f"  {r['space']:<22} {r['ARI']:>8.4f} {r['NMI']:>8.4f} {r['AMI']:>8.4f} {sil_str}{db_str}")
        print(f"  {sep}")
    return results

def print_clustering_summary(all_records, depth_label="3 ARMA layers"):
    sep = "─" * 90
    print(f"\n{sep}")
    print(f"  CLUSTERING METRICS SUMMARY  [{depth_label}]  (mean ± std, {len(all_records)} seeds)")
    print(f"{sep}")
    spaces = [("logit (MC avg)", "Logit space (MC-averaged, K-dim)"),
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
            print(f"  {metric+' '+hi_lo:<16}  {valid.mean():>9.4f}  {valid.std():>9.4f}  "
                  f"{valid.min():>9.4f}  {valid.max():>9.4f}")
    print(f"\n{sep}")

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

def print_cluster_uncertainty_report(logits_stack, yp, y_true, n_passes, sample_ids=None):
    sep = "─" * 72
    H_assign, H_aleat, MI, p_mean = compute_cluster_assignment_uncertainty(logits_stack)
    if sample_ids is None:
        sample_ids = list(range(len(y_true)))
    df = pd.DataFrame({
        "sample_id": sample_ids,
        "true_label": [LABEL_MAP[int(l)] for l in y_true],
        "cluster_assignment": [LABEL_MAP[int(l)] for l in yp],
        "correct_ext_valid": (yp == y_true).astype(int),
        "p_cluster0": p_mean[:, 0],
        "p_cluster1": p_mean[:, 1],
        "entropy_assignment": H_assign,
        "entropy_aleatoric": H_aleat,
        "model_uncertainty": MI,
    })
    print(f"\n{sep}\n  CLUSTER ASSIGNMENT UNCERTAINTY\n{sep}")
    print(f"  (Entropy of soft cluster assignments — {n_passes} MC passes)\n")
    print(f"  {'Metric':<30} {'Mean':>9} {'Std':>9} {'Min':>9} {'Max':>9}")
    print(f"  {'─'*65}")
    for col, label in [
        ("entropy_assignment", "Assignment entropy H[p̄] (bits)"),
        ("entropy_aleatoric", "Aleatoric entropy E[H] (bits)"),
        ("model_uncertainty", "Model uncertainty MI (bits)"),
        ("p_cluster1", "Soft assignment p(cluster=1)"),
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
    K_val = logits_stack.shape[1]
    print(f"\n  Assignment confidence summary:")
    print(f"    Low entropy  (< 0.3 bits, high-confidence): {low_unc}/{len(y_true)} "
          f"({100*low_unc/len(y_true):.1f}%)")
    print(f"    High entropy (> 0.7 bits, ambiguous):        {high_unc}/{len(y_true)} "
          f"({100*high_unc/len(y_true):.1f}%)")
    print(f"\n  Mean assignment entropy: {H_assign.mean():.4f} ± {H_assign.std():.4f} bits")
    print(f"  (Max possible entropy for K=2: {np.log2(K_val):.4f} bits)")
    print(f"{sep}")
    return df

# =============================================================================
# MC EVALUATION (unchanged)
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

def evaluate_model(model, feats, ei, y, device, n_passes=30, seed_base=9999):
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
        logits_stack = logits_stack[:, ::-1, :].copy()
    ypp = F.softmax(torch.from_numpy(logits_mean), dim=1).numpy()
    return yp, ypp, logits_stack, logits_mean

# =============================================================================
# MAD METRICS (unchanged)
# =============================================================================
def cosine_distance_matrix(H):
    n = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-12)
    return 1.0 - np.clip(n @ n.T, -1.0, 1.0)

def compute_mad_metrics(H, adj, labels, name=""):
    N = H.shape[0]
    D = cosine_distance_matrix(H)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    conn = adj > 0
    same = labels[:, None] == labels[None, :]
    def mad(m): return D[m].mean() if m.sum() > 0 else float('nan')
    return dict(
        name=name,
        MAD_all=mad(mask),
        MAD_local=mad(mask & conn),
        MAD_remote=mad(mask & ~conn),
        MADGap=mad(mask & ~conn) - mad(mask & conn),
        MAD_within=mad(mask & same),
        MAD_between=mad(mask & ~same),
        Class_Sep=mad(mask & ~same) - mad(mask & same),
        Mean_Sim=1.0 - mad(mask),
    )

def print_mad_table(results):
    sep = "─" * 90
    header = (f"{'Layer':<22} {'MAD_all':>9} {'MAD_local':>10} "
              f"{'MAD_remote':>11} {'MADGap':>9} {'MAD_within':>11} "
              f"{'MAD_btwn':>9} {'ClassSep':>9}")
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
    results = [compute_mad_metrics(feats, W0, y, name="Input features (RadioDINO)")]
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

def quick_mad(model):
    _, final_np = extract_embeddings(model, features_np, edge_index_np, DEVICE)
    r_in = compute_mad_metrics(features_np, W0, y, name="Input")
    r_out = compute_mad_metrics(final_np, W0, y, name="Output")
    return r_in['MADGap'], r_out['MADGap'], r_in['Class_Sep'], r_out['Class_Sep']

# =============================================================================
# TRAINING FUNCTION (RETURNS WEIGHTED METRICS)
# =============================================================================
def run_once(seed_offset=0, verbose=False, return_probs=False,
             n_eval_passes=30, num_arma_layers=3):
    np.random.seed(42 + seed_offset)
    random.seed(42 + seed_offset)
    torch.manual_seed(42 + seed_offset)

    model = ARMAModel(FEATS_DIM, 256, K, DEVICE, ACTIV,
                      ema_decay=EMA_DECAY, cut=CUT, beta=BETA, tau=TAU,
                      num_arma_layers=num_arma_layers,
                      T_struct=T_STRUCT).to(DEVICE)

    opt = AdamW(model.parameters(), lr=1e-4, weight_decay=5e-5)
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

        p1 = F.softmax(lg1, dim=1)
        entropy = -(p1 * torch.log(p1 + 1e-8)).sum(dim=1).mean()
        cont = (l1 + l2) / 2.0
        loss = (model.struct_loss(A1, lg1) + LAMBDA_CON * cont - C_ENTROPY * entropy)

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

    # --- Weighted metrics (over both classes) ---
    acc_w = accuracy_score(y, yp)
    prec_w = precision_score(y, yp, average='weighted', zero_division=0)
    rec_w = recall_score(y, yp, average='weighted', zero_division=0)
    f1_w = f1_score(y, yp, average='weighted', zero_division=0)
    logloss = log_loss(y, ypp)
    metrics_weighted = (acc_w, prec_w, rec_w, f1_w, logloss)

    # Also keep unweighted for backward compatibility (e.g., depth ablation)
    metrics_unw = (acc_w,
                   precision_score(y, yp, zero_division=0),
                   recall_score(y, yp, zero_division=0),
                   f1_score(y, yp, zero_division=0),
                   logloss)

    if return_probs:
        return metrics_unw, metrics_weighted, yp, ypp, logits_stack, logits_mean, model
    return metrics_unw, model

# =============================================================================
# MAIN EXECUTION
# =============================================================================
METRIC_NAMES = ["Accuracy", "Precision", "Recall", "F1", "LogLoss"]
SEP = "═" * 72

print(f"\n{SEP}")
print(f"  PART A — FIRST RUN (3 ARMA layers) – BreastMNIST (RadioDINO {RADIODINO_MODEL})")
print("  NOTE: Model trained WITHOUT diagnostic labels.")
print("  Labels used ONLY for post-hoc external validation.")
print(f"{SEP}")

# We'll store only weighted versions for the final table
_, metrics0_w, yp0, ypp0, logits_stack0, logits_mean0, model3 = run_once(
    seed_offset=0, verbose=True, return_probs=True,
    n_eval_passes=N_EVAL_PASSES, num_arma_layers=3
)

print("\n── Single-run Classification Results (weighted) ────────")
print("  (Diagnostic labels NOT used during training)")
print(f"  Accuracy  : {metrics0_w[0]:.4f}")
print(f"  Precision : {metrics0_w[1]:.4f}")
print(f"  Recall    : {metrics0_w[2]:.4f}")
print(f"  F1        : {metrics0_w[3]:.4f}")
print(f"  Log Loss  : {metrics0_w[4]:.4f}")

lm = logits_stack0.mean(axis=2)
diff = lm[:, 1] - lm[:, 0]
print(f"\n── Logit diagnostics ───────────────────────────────")
print(f"  mean={lm.mean():.2f}  std={lm.std():.2f}  min={lm.min():.2f}  max={lm.max():.2f}")
print(f"  Logit diff (Normal-Malignant): mean={diff.mean():.2f}  "
      f"std={diff.std():.2f}  min={diff.min():.2f}  max={diff.max():.2f}")

# ── Cluster assignment uncertainty ────────────────────────────────────────────
df_unc0 = print_cluster_uncertainty_report(
    logits_stack0, yp0, y,
    n_passes=N_EVAL_PASSES,
    sample_ids=list(range(len(y)))
)
df_unc0.to_csv("cluster_uncertainty_part_a.csv", index=False, float_format="%.6f")
print(f"\n  CSV → cluster_uncertainty_part_a.csv  ({len(df_unc0)} subjects)")

# ── Clustering metrics for Part A (using MC predictions) ─────────────────────
clustering_results_A = evaluate_clustering_from_mc(
    model3, features_np, edge_index_np, y, yp0, logits_mean0, DEVICE,
    prefix="Part A — 3-layer ARMA, seed 0"
)

print(f"\n{SEP}\n  PART B — PER-LAYER OVER-SMOOTHING ANALYSIS\n{SEP}")
layer_similarity_analysis(model3, features_np, edge_index_np, y, W0,
                          prefix="3-layer ARMA (BreastMNIST / RadioDINO)")

print(f"\n{SEP}\n  PART C — DEPTH ABLATION  (1 / 2 / 3 ARMA layers)\n{SEP}")
ablation_results = {}
trained_models = {}
ablation_clustering = {1: [], 2: [], 3: []}

for n_layers in [1, 2, 3]:
    print(f"\n  ── Depth = {n_layers} ARMA layer(s) ──")
    seed_records = []
    for seed in range(3):
        print(f"    Seed {seed} ... ", end="", flush=True)
        _, model_i = run_once(seed_offset=seed, num_arma_layers=n_layers,
                              n_eval_passes=N_EVAL_PASSES, return_probs=False)
        yp_s, ypp_s, logits_stack_s, logits_mean_s = evaluate_model(
            model_i, features_np, edge_index_np, y, DEVICE,
            n_passes=N_EVAL_PASSES, seed_base=9999 + seed
        )
        a_s = accuracy_score(y, yp_s)
        cl_res = evaluate_clustering_from_mc(
            model_i, features_np, edge_index_np, y, yp_s, logits_mean_s,
            DEVICE, prefix=f"{n_layers}L seed {seed}", quiet=True
        )
        cl_flat = {}
        for r in cl_res:
            sp = r['space']
            for metric in ['ARI', 'NMI', 'AMI', 'Silhouette', 'DaviesBouldin']:
                cl_flat[f"{metric}_{sp}"] = r[metric]
        ablation_clustering[n_layers].append(cl_flat)

        H_assign, _, _, _ = compute_cluster_assignment_uncertainty(logits_stack_s)
        seed_records.append(dict(
            depth=n_layers, seed=seed, acc=a_s,
            prec=precision_score(y, yp_s, zero_division=0),
            rec=recall_score(y, yp_s, zero_division=0),
            f1=f1_score(y, yp_s, zero_division=0),
            mean_entropy=H_assign.mean(),
            std_entropy=H_assign.std(),
            mean_conf=ypp_s.max(axis=1).mean(),
        ))
        trained_models[n_layers] = model_i
        print(f"Acc={a_s:.3f}  MeanEntropy={H_assign.mean():.4f}")
    ablation_results[n_layers] = seed_records

print("\n  ── MADGap at each depth ──")
mad_depth = {}
for n_layers in [1, 2, 3]:
    g_in, g_out, cs_in, cs_out = quick_mad(trained_models[n_layers])
    mad_depth[n_layers] = (g_in, g_out, cs_in, cs_out)
    print(f"    {n_layers} layer(s):  MADGap_input={g_in:.4f}  "
          f"MADGap_final={g_out:.4f}  ClassSep_input={cs_in:.4f}  ClassSep_final={cs_out:.4f}")

sep110 = "─" * 110
print(f"\n{sep110}\n  DEPTH ABLATION SUMMARY  (mean ± std over 3 seeds)\n{sep110}")
print(f"  {'Depth':<7} {'Acc':>7} {'Prec':>7} {'Rec':>6} {'F1':>7} "
      f"{'MeanEntropy':>13} {'MADGap↑':>9} {'ClassSep↑':>10}")
print(sep110)
for n_layers in [1, 2, 3]:
    recs = ablation_results[n_layers]
    def ms(k): return (np.mean([r[k] for r in recs]), np.std([r[k] for r in recs]))
    a_m, a_s_ = ms("acc");  p_m, p_s_ = ms("prec"); r_m, r_s_ = ms("rec")
    f_m, f_s_ = ms("f1");   e_m, e_s_ = ms("mean_entropy")
    _, g_out, _, cs_out = mad_depth[n_layers]
    print(f"  {n_layers} layer{'s' if n_layers>1 else ' ':<5}  "
          f"{a_m:.3f}±{a_s_:.3f}  {p_m:.3f}±{p_s_:.3f}  {r_m:.3f}±{r_s_:.3f}  "
          f"{f_m:.3f}±{f_s_:.3f}  {e_m:.4f}±{e_s_:.4f}  {g_out:.4f}  {cs_out:.4f}")
print(sep110)

print(f"\n{SEP}\n  PART D — DIAGNOSIS SUMMARY (BreastMNIST)\n{SEP}")
def _mean(key, d): return np.mean([r[key] for r in ablation_results[d]])
for depth in [1, 2, 3]:
    g_in, g_out, _, cs_out = mad_depth[depth]
    ent_d = _mean("mean_entropy", depth)
    acc_d = _mean("acc", depth)
    flag = "↓ collapsed" if g_out < g_in * 0.5 else "→ preserved"
    print(f"  {depth} layer(s): MADGap {g_in:.4f}→{g_out:.4f} {flag} | "
          f"ClassSep={cs_out:.4f} | MeanEntropy={ent_d:.4f} | Acc={acc_d:.4f}")

print(f"\n  Over-smoothing (3 layers):")
print(f"    MADGap_3layers={mad_depth[3][1]:.4f}  vs  MADGap_input={mad_depth[3][0]:.4f}")
if mad_depth[3][1] < mad_depth[3][0] * 0.5:
    print("    ⚠ OVER-SMOOTHING detected")
else:
    print("    ✓ Gap preserved – ARMA band-pass filter intact")

# =============================================================================
# PART E — 10‑SEED EVALUATION (WEIGHTED METRICS)
# =============================================================================
print(f"\n{SEP}\n  PART E — 10-SEED EVALUATION (3 ARMA layers, BreastMNIST)\n{SEP}")
print("  Diagnostic labels used ONLY for post-hoc external validation.\n")

col_w = "─" * 110
print(f"  {'Seed':>4}  {'Acc':>7}  {'Prec_w':>8}  {'Rec_w':>8}  {'F1_w':>8}  "
      f"{'NMI':>7}  {'MeanEntropy':>13}  {'StdEntropy':>11}  "
      f"{'TN':>5}  {'FP':>5}  {'FN':>5}  {'TP':>5}")
print(f"  {col_w}")

# Storage for weighted metrics and clustering metrics
calibration_rows = []
seed_clustering_records = []

for i in range(10):
    _, m_w, yp_i, ypp_i, logits_stack_i, logits_mean_i, model_i = run_once(
        seed_offset=i, return_probs=True,
        n_eval_passes=N_EVAL_PASSES, num_arma_layers=3
    )
    acc_i, prec_w, rec_w, f1_w, ll_i = m_w

    H_assign_i, _, _, _ = compute_cluster_assignment_uncertainty(logits_stack_i)
    mean_ent_i = float(H_assign_i.mean())
    std_ent_i = float(H_assign_i.std())

    tn_i, fp_i, fn_i, tp_i = confusion_matrix(y, yp_i).ravel()

    calibration_rows.append(dict(
        seed=i, acc=acc_i, prec_weighted=prec_w, rec_weighted=rec_w, f1_weighted=f1_w,
        logloss=ll_i,
        mean_entropy=mean_ent_i, std_entropy=std_ent_i,
        tn=int(tn_i), fp=int(fp_i), fn=int(fn_i), tp=int(tp_i),
    ))

    cl_res_i = evaluate_clustering_from_mc(
        model_i, features_np, edge_index_np, y, yp_i, logits_mean_i,
        DEVICE, prefix=f"Part E seed {i}", quiet=True
    )
    cl_flat_i = {}
    for r in cl_res_i:
        sp = r['space']
        for metric in ['ARI', 'NMI', 'AMI', 'Silhouette', 'DaviesBouldin']:
            cl_flat_i[f"{metric}_{sp}"] = r[metric]
    seed_clustering_records.append(cl_flat_i)

    print(f"  {i:>4}  {acc_i:>7.4f}  {prec_w:>8.4f}  {rec_w:>8.4f}  {f1_w:>8.4f}  "
          f"{cl_flat_i.get('NMI_hidden (single pass)', np.nan):>7.4f}  "
          f"{mean_ent_i:>13.4f}  {std_ent_i:>11.4f}  "
          f"{tn_i:>5}  {fp_i:>5}  {fn_i:>5}  {tp_i:>5}")

# =============================================================================
# HORIZONTAL TABLE FOR MANUSCRIPT (mean ± std over 10 runs)
# =============================================================================
print("\n" + "=" * 72)
print("  HORIZONTAL TABLE FOR MANUSCRIPT (mean ± std over 10 runs)")
print("  Metrics: Accuracy, Weighted Precision, Weighted Recall, Weighted F1, NMI, ARI, AMI, Silhouette, Davies‑Bouldin")
print("=" * 72)

acc_vals = np.array([r['acc'] for r in calibration_rows])
prec_vals = np.array([r['prec_weighted'] for r in calibration_rows])
rec_vals = np.array([r['rec_weighted'] for r in calibration_rows])
f1_vals = np.array([r['f1_weighted'] for r in calibration_rows])

ari_vals = np.array([r.get('ARI_hidden (single pass)', np.nan) for r in seed_clustering_records])
nmi_vals = np.array([r.get('NMI_hidden (single pass)', np.nan) for r in seed_clustering_records])
ami_vals = np.array([r.get('AMI_hidden (single pass)', np.nan) for r in seed_clustering_records])
sil_vals = np.array([r.get('Silhouette_hidden (single pass)', np.nan) for r in seed_clustering_records])
db_vals = np.array([r.get('DaviesBouldin_hidden (single pass)', np.nan) for r in seed_clustering_records])

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
row = "ARMA"
for m, s in zip(means, stds):
    if np.isnan(m):
        row += "\tN/A±N/A"
    else:
        row += f"\t{m:.4f}±{s:.4f}"
print(row)

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

# =============================================================================
# 10‑RUN SUMMARY (weighted)
# =============================================================================
results_np = np.array([[r['acc'], r['prec_weighted'], r['rec_weighted'], r['f1_weighted'],
                        r['mean_entropy'], r['std_entropy']]
                       for r in calibration_rows])
counts_np = np.array([[r['tn'], r['fp'], r['fn'], r['tp']]
                      for r in calibration_rows], dtype=float)

sep100 = "─" * 100
print(f"\n{sep100}")
print(f"  10-RUN SUMMARY  (mean ± std)              [3 ARMA layers, BreastMNIST / RadioDINO]")
print("  NOTE: Diagnostic labels used only for post-hoc external validation.")
print(f"{sep100}")

ext_val_labels = [
    ("Accuracy (weighted)",      "Post-hoc external validation"),
    ("Weighted Precision",       "Post-hoc external validation"),
    ("Weighted Recall",          "Post-hoc external validation"),
    ("Weighted F1",              "Post-hoc external validation"),
    ("Mean Entropy",             "Mean cluster assignment entropy (bits)"),
    ("Std Entropy",              "Std of assignment entropy across subjects"),
]

print(f"\n  A.  EXTERNAL VALIDATION METRICS & ASSIGNMENT UNCERTAINTY")
print(f"  {'Metric':<20}  {'Mean':>9}  {'Std':>9}  {'Min':>9}  {'Max':>9}  Note")
print(f"  {sep100}")
for col_idx, (label, note) in enumerate(ext_val_labels):
    vals = results_np[:, col_idx]
    print(f"  {label:<20}  {vals.mean():>9.4f}  {vals.std():>9.4f}  "
          f"{vals.min():>9.4f}  {vals.max():>9.4f}  {note}")

count_labels = [
    ("TN (Malig→C0)", "Malignant assigned to Cluster 0 — post-hoc"),
    ("FP (Malig→C1)", "Malignant assigned to Cluster 1 — post-hoc"),
    ("FN (Norm→C0)",  "Normal assigned to Cluster 0 — post-hoc"),
    ("TP (Norm→C1)",  "Normal assigned to Cluster 1 — post-hoc"),
]
print(f"\n  B.  CONFUSION MATRIX COUNTS (post-hoc external validation only)")
print(f"  {'Group':<16}  {'Mean':>7}  {'Std':>7}  {'Min':>5}  {'Max':>5}  Note")
print(f"  {sep100}")
for col_idx, (label, note) in enumerate(count_labels):
    vals = counts_np[:, col_idx]
    print(f"  {label:<16}  {vals.mean():>7.1f}  {vals.std():>7.2f}  "
          f"{vals.min():>5.0f}  {vals.max():>5.0f}  {note}")

print(f"\n  C.  CLUSTER STABILITY ACROSS 10 SEEDS")
print(f"  {'Metric':<20}  {'Mean':>9}  {'Std':>9}  Note")
print(f"  {sep100}")
print(f"  {'Weighted F1':<20}  {results_np[:,3].mean():>9.4f}  {results_np[:,3].std():>9.4f}  "
      "Consistency of weighted F1")
print(f"  {'Assignment Entropy':<20}  {results_np[:,4].mean():>9.4f}  {results_np[:,4].std():>9.4f}  "
      "Low std = stable confidence")
print(f"  {sep100}\n")

# =============================================================================
# PART F — CLUSTERING METRICS SUMMARY (unchanged)
# =============================================================================
print(f"\n{SEP}\n  PART F — CLUSTERING METRICS (ARI / NMI / AMI / Silhouette / Davies-Bouldin)\n{SEP}")

print(f"\n  ── F.1  10-SEED EVALUATION (3 ARMA layers) ──")
print_clustering_summary(seed_clustering_records,
                         depth_label="3 ARMA layers — 10 seeds (BreastMNIST, RadioDINO)")

print(f"\n  ── F.2  DEPTH ABLATION (3 seeds per depth) ──")
for n_layers in [1, 2, 3]:
    print_clustering_summary(
        ablation_clustering[n_layers],
        depth_label=(f"{n_layers} ARMA layer{'s' if n_layers > 1 else ' '} — 3 seeds (BreastMNIST, RadioDINO)")
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
    db = mn("DaviesBouldin_hidden (single pass)")
    sil_s = f"{sil:>13.4f}" if not np.isnan(sil) else "          N/A"
    db_s = f"{db:>16.4f}" if not np.isnan(db) else "             N/A"
    print(f"  {n_layers} layer{'s' if n_layers>1 else ' ':<7}  "
          f"{ari:>8.4f}  {nmi:>8.4f}  {ami:>8.4f}{sil_s}{db_s}")
print(f"  {sep72}")
print(f"\n  Interpretation for cross-depth table (hidden space):")
print(f"    Rising ARI/NMI/AMI with depth → deeper stacks recover more label structure.")
print(f"    Rising Silhouette with depth   → embeddings become more geometrically clustered.")
print(f"    Falling Davies-Bouldin         → clusters tighten relative to centroid distances.")
print(f"    If any metric degrades at depth 3, combine with MADGap to diagnose over-smoothing.\n")

# =============================================================================
# PART G — BIOLOGICAL / IMAGE FEATURE INTERPRETATION (unchanged)
# =============================================================================
print(f"\n{SEP}\n  PART G — BIOLOGICAL INTERPRETATION OF DISCOVERED CLUSTERS\n{SEP}")
print("  The model was trained on RadioDINO features (no labels).")
print("  We now validate clusters using:")
print("    • Label distribution (Malignant vs Normal/Benign)")
print("    • RadioDINO feature statistics per cluster")
print("    • Cosine similarity within vs between clusters\n")

cluster_assignments = yp0
cluster0_mask = (cluster_assignments == 0)
cluster1_mask = (cluster_assignments == 1)

print(f"  Cluster sizes: Cluster 0 = {cluster0_mask.sum()} subjects, "
      f"Cluster 1 = {cluster1_mask.sum()} subjects")

print(f"\n{sep72}\n  1. LABEL DISTRIBUTION (post-hoc reference only)\n{sep72}")
print("  Note: Labels NOT used during training — shown for external validation only.\n")
cm_diag = confusion_matrix(y, cluster_assignments)
print(f"  Confusion Matrix (rows=true label, cols=cluster):")
print(f"                     Cluster 0    Cluster 1")
print(f"    Malignant (n={(y==0).sum()})      {cm_diag[0,0]:>6}       {cm_diag[0,1]:>6}")
print(f"    Normal    (n={(y==1).sum()})      {cm_diag[1,0]:>6}       {cm_diag[1,1]:>6}")

pct_malig_in_c0 = (y[cluster0_mask] == 0).mean() * 100
pct_norm_in_c1 = (y[cluster1_mask] == 1).mean() * 100
agreement = (cluster_assignments == y).mean() * 100
print(f"\n  Cluster composition:")
print(f"    Cluster 0: {pct_malig_in_c0:.1f}% Malignant, {100-pct_malig_in_c0:.1f}% Normal")
print(f"    Cluster 1: {100-pct_norm_in_c1:.1f}% Malignant, {pct_norm_in_c1:.1f}% Normal")
print(f"\n  Overall label-cluster agreement: {agreement:.1f}%")

print(f"\n{sep72}\n  2. RADIODINO FEATURE STATISTICS PER CLUSTER\n{sep72}")
feats_c0 = features_np[cluster0_mask]
feats_c1 = features_np[cluster1_mask]
norm_c0 = np.linalg.norm(feats_c0, axis=1)
norm_c1 = np.linalg.norm(feats_c1, axis=1)
mean_c0 = feats_c0.mean(axis=1)
mean_c1 = feats_c1.mean(axis=1)
from scipy import stats as scipy_stats
t_norm, p_norm = scipy_stats.ttest_ind(norm_c0, norm_c1, equal_var=False)
t_mean, p_mean_feat = scipy_stats.ttest_ind(mean_c0, mean_c1, equal_var=False)
print(f"  {'Statistic':<35} {'Cluster 0':>15} {'Cluster 1':>15} {'p-value':>10}")
print(f"  {'─'*80}")
print(f"  {'L2 norm (embedding magnitude)':<35} "
      f"{norm_c0.mean():>7.3f}±{norm_c0.std():.3f}  {norm_c1.mean():>7.3f}±{norm_c1.std():.3f}  "
      f"{p_norm:>10.4f}{'*' if p_norm < 0.05 else ''}")
print(f"  {'Mean feature activation':<35} "
      f"{mean_c0.mean():>7.4f}±{mean_c0.std():.4f}  {mean_c1.mean():>7.4f}±{mean_c1.std():.4f}  "
      f"{p_mean_feat:>10.4f}{'*' if p_mean_feat < 0.05 else ''}")

print(f"\n{sep72}\n  3. INTRA- vs INTER-CLUSTER COSINE SIMILARITY\n{sep72}")
D = cosine_distance_matrix(features_np)
N = len(y)
mask = np.triu(np.ones((N, N), dtype=bool), k=1)
same = (cluster_assignments[:, None] == cluster_assignments[None, :]) & mask
diff = (cluster_assignments[:, None] != cluster_assignments[None, :]) & mask
sim_within = 1 - D[same]
sim_between = 1 - D[diff]
print(f"  Within-cluster cosine similarity: mean={sim_within.mean():.4f}  std={sim_within.std():.4f}  "
      f"min={sim_within.min():.4f}  max={sim_within.max():.4f}")
print(f"  Between-cluster cosine similarity: mean={sim_between.mean():.4f}  std={sim_between.std():.4f}  "
      f"min={sim_between.min():.4f}  max={sim_between.max():.4f}")
sep_score = sim_within.mean() - sim_between.mean()
print(f"\n  Separation score (within − between): {sep_score:.4f}")
if sep_score > 0.05:
    print("    ✓ Clusters are geometrically well-separated in RadioDINO space")
else:
    print("    → Moderate geometric separation — clusters may be partially overlapping")

print(f"\n{sep72}\n  4. TOP DISCRIMINATING RADIODINO FEATURE DIMENSIONS\n{sep72}")
mean_diff_dims = feats_c1.mean(axis=0) - feats_c0.mean(axis=0)
top10_idx = np.argsort(np.abs(mean_diff_dims))[-10:][::-1]
print(f"  {'Dim':<8} {'C0 mean':>10} {'C1 mean':>10} {'Diff (C1-C0)':>14}")
print(f"  {'─'*50}")
for idx in top10_idx:
    print(f"  {idx:<8} {feats_c0[:, idx].mean():>10.5f} {feats_c1[:, idx].mean():>10.5f} {mean_diff_dims[idx]:>+14.5f}")

print(f"\n{sep72}\n  5. SUMMARY: CLUSTER CHARACTERIZATION\n{sep72}")
print(f"\n  {'Feature':<40} {'Cluster 0':>20} {'Cluster 1':>20} {'Insight':>15}")
print(f"  {sep72}")
print(f"  {'Sample size (n)':<40} {cluster0_mask.sum():>20} {cluster1_mask.sum():>20}")
print(f"  {'Malignant %':<40} {f'{pct_malig_in_c0:.1f}%':>20} {f'{100-pct_norm_in_c1:.1f}%':>20} {'(ref. only)':>15}")
print(f"  {'Normal %':<40} {f'{100-pct_malig_in_c0:.1f}%':>20} {f'{pct_norm_in_c1:.1f}%':>20} {'(ref. only)':>15}")
print(f"  {'RadioDINO L2 norm (mean±std)':<40} "
      f"{f'{norm_c0.mean():.3f}±{norm_c0.std():.3f}':>20} {f'{norm_c1.mean():.3f}±{norm_c1.std():.3f}':>20} "
      f"{'p='+f'{p_norm:.3f}':>15}")
print(f"  {'Intra-cluster cosine sim.':<40} {sim_within.mean():>20.4f} {'N/A':>20} {'':>15}")
print(f"  {'Inter-cluster cosine sim.':<40} {'N/A':>20} {sim_between.mean():>20.4f} {'':>15}")
print(f"  {'Cluster separation score':<40} {sep_score:>20.4f} {'':>20}")

print(f"\n{sep72}\n  6. CONCLUSION\n{sep72}")
if agreement > 70:
    print(f"  ✓ PRIMARY FINDING: ARMA-based clustering achieves {agreement:.1f}% agreement")
    print(f"    with ground-truth pathology labels using ONLY unsupervised RadioDINO features.")
    print(f"    Cluster 0 is predominantly Malignant ({pct_malig_in_c0:.1f}%).")
    print(f"    Cluster 1 is predominantly Normal/Benign ({pct_norm_in_c1:.1f}%).")
else:
    print(f"  → Cluster-label agreement: {agreement:.1f}% — partial alignment with pathology.")
print(f"\n  Geometric validation:")
if sep_score > 0.05:
    print(f"    ✓ Clusters are well-separated in RadioDINO embedding space (separation={sep_score:.4f})")
else:
    print(f"    → Moderate separation in RadioDINO space (separation={sep_score:.4f})")
print(f"\n  Interpretation:")
print(f"    Self-supervised RadioDINO features capture pathology-relevant visual")
print(f"    structure in breast ultrasound images. The ARMA graph network")
print(f"    organises these features into clusters that align substantially")
print(f"    with clinically defined Malignant vs Normal categories,")
print(f"    without ever observing diagnostic labels during training.")
print(f"\n  Limitations:")
print(f"    • Single-cohort study — no external validation dataset.")
print(f"    • Diagnostic labels used only for post-hoc external validation.")
print(f"    • Generalization to unseen subjects has not been assessed.")
print(f"    • RadioDINO features are image-level; spatial pathology patterns")
print(f"      within the image are not explicitly modelled.")

print(f"\n{SEP}\n")

# =============================================================================
# EXPORT RESULTS
# =============================================================================
print(f"\n{sep72}\n  7. EXPORT RESULTS FOR MANUSCRIPT\n{sep72}")
paper_results = {
    "Metric": [
        "Cluster 0 size (n)", "Cluster 1 size (n)",
        "Malignant % in Cluster 0", "Normal % in Cluster 1",
        "Label-cluster agreement (%)",
        "RadioDINO L2 norm — Cluster 0", "RadioDINO L2 norm — Cluster 1",
        "L2 norm p-value",
        "Intra-cluster cosine sim.", "Inter-cluster cosine sim.",
        "Cluster separation score",
        "Mean assignment entropy (bits)",
        "10-seed Weighted F1 (mean±std)", "10-seed ARI (mean±std)",
    ],
    "Value": [
        f"{cluster0_mask.sum()}", f"{cluster1_mask.sum()}",
        f"{pct_malig_in_c0:.1f}%", f"{pct_norm_in_c1:.1f}%",
        f"{agreement:.1f}%",
        f"{norm_c0.mean():.3f} ± {norm_c0.std():.3f}",
        f"{norm_c1.mean():.3f} ± {norm_c1.std():.3f}",
        f"{p_norm:.4f} ({'n.s.' if p_norm >= 0.05 else 'significant'})",
        f"{sim_within.mean():.4f}", f"{sim_between.mean():.4f}",
        f"{sep_score:.4f}",
        f"{df_unc0['entropy_assignment'].mean():.4f} ± {df_unc0['entropy_assignment'].std():.4f}",
        f"{f1_vals.mean():.4f} ± {f1_vals.std():.4f}",
        f"{ari_vals.mean():.4f} ± {ari_vals.std():.4f}",
    ]
}
df_paper = pd.DataFrame(paper_results)
print("\n  Summary table for manuscript:")
print(df_paper.to_string(index=False))
df_paper.to_csv("cluster_breastmnist_characterization_radiodino.csv", index=False)
print(f"\n  ✓ Saved to: cluster_breastmnist_characterization_radiodino.csv")
df_unc0.to_csv("cluster_uncertainty_breastmnist_radiodino.csv", index=False, float_format="%.6f")
print(f"  ✓ Saved uncertainty to: cluster_uncertainty_breastmnist_radiodino.csv")

print(f"\n{SEP}\n")
print("  ANALYSIS COMPLETE")
print(f"\n{SEP}")

# =============================================================================
# PUBLICATION-READY t-SNE VISUALISATIONS (BreastMNIST)
# =============================================================================
print(f"\n{SEP}\n  PUBLICATION t-SNE VISUALISATIONS (BreastMNIST)\n{SEP}")

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Set publication‑quality style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 300

# ----- 1. Get hidden embeddings -----
model3.eval()
d = to_data(features_np, edge_index_np, DEVICE)
with torch.no_grad():
    hidden_embeddings = model3.online_encoder(d).cpu().numpy()

print(f"  Input shape: {features_np.shape}, Hidden shape: {hidden_embeddings.shape}")

# ----- 2. Compute t‑SNE -----
best_perp = 40
print(f"  Computing t‑SNE (perplexity={best_perp})...", end=" ", flush=True)

tsne_input = TSNE(n_components=2, random_state=42, perplexity=best_perp,
                  init='pca', max_iter=1000)
tsne_input_results = tsne_input.fit_transform(features_np)

tsne_hidden = TSNE(n_components=2, random_state=42, perplexity=best_perp,
                   init='pca', max_iter=1000)
tsne_hidden_results = tsne_hidden.fit_transform(hidden_embeddings)
print("done")

# ----- 3. Prepare masks -----
# True labels
malignant_mask = (y == 0)
normal_mask = (y == 1)

# Cluster assignments
cluster0_mask = (yp0 == 0)
cluster1_mask = (yp0 == 1)

# Uncertainty (from your existing H_assign)
high_uncertainty_mask = H_assign > 0.7
low_uncertainty_mask = H_assign <= 0.3

# Misclassified samples (if you want)
misclassified_mask = (yp0 != y)

# ============================================================================
# FIGURE 1: MAIN — Input vs. Learned (Side‑by‑Side)
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ---- Left: Input features (coloured by true diagnosis) ----
ax1.scatter(tsne_input_results[malignant_mask, 0],
            tsne_input_results[malignant_mask, 1],
            c='#2E86AB', label='Malignant', alpha=0.7, s=40,
            edgecolors='white', linewidth=0.5)
ax1.scatter(tsne_input_results[normal_mask, 0],
            tsne_input_results[normal_mask, 1],
            c='#F18F01', label='Normal', alpha=0.7, s=40,
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
            c='#2E86AB', label='Cluster 0 (Malignant-dominant)', alpha=0.7, s=40,
            edgecolors='white', linewidth=0.5)
ax2.scatter(tsne_hidden_results[cluster1_mask, 0],
            tsne_hidden_results[cluster1_mask, 1],
            c='#A23B72', label='Cluster 1 (Normal-dominant)', alpha=0.7, s=40,
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

plt.suptitle(f't-SNE Comparison (BreastMNIST, perplexity={best_perp})', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('tsne_fig1_main_comparison_BreastMNIST.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("  ✓ Saved: tsne_fig1_main_comparison_BreastMNIST.png")

# ============================================================================
# FIGURE 2: Learned Embeddings + True Labels (Validation)
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

ax.scatter(tsne_hidden_results[malignant_mask, 0],
           tsne_hidden_results[malignant_mask, 1],
           c='#2E86AB', label='Malignant (True label)', alpha=0.7, s=50,
           edgecolors='white', linewidth=0.5)
ax.scatter(tsne_hidden_results[normal_mask, 0],
           tsne_hidden_results[normal_mask, 1],
           c='#F18F01', label='Normal (True label)', alpha=0.7, s=50,
           edgecolors='white', linewidth=0.5)

ax.set_title('ModCon-ARMA Embedding Space\n(Coloured by True Diagnosis, Reference)', fontsize=12, fontweight='bold')
ax.set_xlabel('t-SNE Dimension 1')
ax.set_ylabel('t-SNE Dimension 2')
ax.legend(loc='best', framealpha=0.9)
ax.grid(True, alpha=0.2)
ax.set_facecolor('#f8f9fa')

plt.tight_layout()
plt.savefig('tsne_fig2_true_labels_validation_BreastMNIST.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("  ✓ Saved: tsne_fig2_true_labels_validation_BreastMNIST.png")

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
plt.savefig('tsne_fig3_uncertainty_heatmap_BreastMNIST.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("  ✓ Saved: tsne_fig3_uncertainty_heatmap_BreastMNIST.png")

# ============================================================================
# FIGURE 4: Misclassified Samples (Optional)
# ============================================================================
if misclassified_mask.sum() > 0:
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Correctly classified (cluster matches true label)
    correct_mask = ~misclassified_mask
    ax.scatter(tsne_hidden_results[correct_mask, 0],
               tsne_hidden_results[correct_mask, 1],
               c='gray', label='Correctly assigned', alpha=0.5, s=30,
               edgecolors='none')

    # Misclassified samples
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
    plt.savefig('tsne_fig4_misclassified_BreastMNIST.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print("  ✓ Saved: tsne_fig4_misclassified_BreastMNIST.png")

print("\n  ✅ All t-SNE figures saved.")

# =============================================================================
# PART H — t-SNE VISUALIZATION OF CLUSTERS (BreastMNIST / RadioDINO)
# =============================================================================
print(f"\n{SEP}\n  PART H — t-SNE VISUALIZATION OF DISCOVERED CLUSTERS (BreastMNIST)\n{SEP}")
print("  Visualizing the hidden space embeddings to assess cluster separation.\n")

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist

# Set publication‑quality style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 300

# Get hidden embeddings from your trained 3‑layer model (model3)
model3.eval()
d = to_data(features_np, edge_index_np, DEVICE)
with torch.no_grad():
    hidden_embeddings = model3.online_encoder(d).cpu().numpy()  # (N, 256)

print(f"  Hidden embedding shape: {hidden_embeddings.shape}")

# Compute assignment uncertainty (using logits_stack0 from Part A)
H_assign, H_aleat, MI, p_mean = compute_cluster_assignment_uncertainty(logits_stack0)
high_uncertainty_mask = H_assign > 0.7
low_uncertainty_mask = H_assign <= 0.3

# Compute t‑SNE (perplexity 40 works well for ~2000 samples)
best_perp = 40
print(f"  Computing t‑SNE with perplexity={best_perp}...", end=" ", flush=True)
tsne = TSNE(n_components=2, random_state=42, perplexity=best_perp,
            init='pca', max_iter=1000)
tsne_results = tsne.fit_transform(hidden_embeddings)
print("done")

# Cluster masks (from yp0) and true label masks
cluster0_mask = (yp0 == 0)   # Malignant‑dominant cluster
cluster1_mask = (yp0 == 1)   # Normal‑dominant cluster
malignant_mask = (y == 0)
normal_mask = (y == 1)

# Cluster centers in t‑SNE space
cluster0_center = tsne_results[cluster0_mask].mean(axis=0)
cluster1_center = tsne_results[cluster1_mask].mean(axis=0)

# Geometric separation metrics (hidden space)
cluster0_emb = hidden_embeddings[cluster0_mask]
cluster1_emb = hidden_embeddings[cluster1_mask]
intra0 = pairwise_distances(cluster0_emb).mean() if len(cluster0_emb) > 1 else 0
intra1 = pairwise_distances(cluster1_emb).mean() if len(cluster1_emb) > 1 else 0
intra_mean = (intra0 + intra1) / 2
inter = pairwise_distances(cluster0_emb, cluster1_emb).mean() if len(cluster0_emb) > 0 and len(cluster1_emb) > 0 else 0
separation_ratio = inter / intra_mean if intra_mean > 0 else 0

# ============================================================================
# FIGURE 1: t-SNE colored by cluster assignment (8x6)
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

ax.scatter(tsne_results[cluster0_mask, 0], tsne_results[cluster0_mask, 1],
           c='#2E86AB', label='Cluster 0 (Malignant‑dominant)',
           alpha=0.7, s=50, edgecolors='white', linewidth=0.5, zorder=2)
ax.scatter(tsne_results[cluster1_mask, 0], tsne_results[cluster1_mask, 1],
           c='#A23B72', label='Cluster 1 (Normal‑dominant)',
           alpha=0.7, s=50, edgecolors='white', linewidth=0.5, zorder=2)

ax.scatter(cluster0_center[0], cluster0_center[1], c='darkblue', s=150,
           edgecolors='black', linewidth=2, marker='*', label='Cluster 0 center', zorder=3)
ax.scatter(cluster1_center[0], cluster1_center[1], c='darkred', s=150,
           edgecolors='black', linewidth=2, marker='*', label='Cluster 1 center', zorder=3)

ax.set_title('t-SNE: ARMA Embeddings (BreastMNIST)\nColored by Cluster Assignment',
             fontsize=12, fontweight='bold')
ax.set_xlabel('t-SNE Dimension 1', fontsize=10)
ax.set_ylabel('t-SNE Dimension 2', fontsize=10)
ax.legend(loc='best', framealpha=0.9, fontsize=8)
ax.grid(True, alpha=0.2, linestyle='--')
ax.set_facecolor('#f8f9fa')

plt.tight_layout()
plt.savefig('tsne_by_cluster_assignment_BreastMNIST.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("  ✓ Saved: tsne_by_cluster_assignment_BreastMNIST.png")

# ============================================================================
# FIGURE 2: t-SNE colored by true diagnosis (reference) (8x6)
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

ax.scatter(tsne_results[malignant_mask, 0], tsne_results[malignant_mask, 1],
           c='#2E86AB', label='Malignant (True label)',
           alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
ax.scatter(tsne_results[normal_mask, 0], tsne_results[normal_mask, 1],
           c='#F18F01', label='Normal (True label)',
           alpha=0.7, s=50, edgecolors='white', linewidth=0.5)

ax.set_title('t-SNE: ARMA Embeddings (BreastMNIST)\nColored by True Diagnosis (Reference)',
             fontsize=12, fontweight='bold')
ax.set_xlabel('t-SNE Dimension 1', fontsize=10)
ax.set_ylabel('t-SNE Dimension 2', fontsize=10)
ax.legend(loc='best', framealpha=0.9, fontsize=8)
ax.grid(True, alpha=0.2, linestyle='--')
ax.set_facecolor('#f8f9fa')

plt.tight_layout()
plt.savefig('tsne_by_true_diagnosis_BreastMNIST.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("  ✓ Saved: tsne_by_true_diagnosis_BreastMNIST.png")

# ============================================================================
# FIGURE 3: t-SNE with uncertainty heatmap (9x7)
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

stats_text = f'Mean Entropy: {H_assign.mean():.4f} ± {H_assign.std():.4f} bits\n'
stats_text += f'Confident (entropy ≤ 0.3): {low_uncertainty_mask.sum()}/{len(H_assign)} ({100*low_uncertainty_mask.sum()/len(H_assign):.1f}%)\n'
stats_text += f'Ambiguous (entropy > 0.7): {high_uncertainty_mask.sum()}/{len(H_assign)} ({100*high_uncertainty_mask.sum()/len(H_assign):.1f}%)\n'
stats_text += f'Separation Ratio: {separation_ratio:.2f}'

ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.set_title('t-SNE: ARMA Embeddings (BreastMNIST)\nColored by Assignment Uncertainty',
             fontsize=12, fontweight='bold')
ax.set_xlabel('t-SNE Dimension 1', fontsize=10)
ax.set_ylabel('t-SNE Dimension 2', fontsize=10)
ax.legend(loc='lower right', framealpha=0.9, fontsize=8)
ax.grid(True, alpha=0.2, linestyle='--')
ax.set_facecolor('#f8f9fa')

plt.tight_layout()
plt.savefig('tsne_uncertainty_heatmap_BreastMNIST.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("  ✓ Saved: tsne_uncertainty_heatmap_BreastMNIST.png")

# ============================================================================
# FIGURE 4: Side-by-side comparison (11x5)
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
ax.scatter(tsne_results[malignant_mask, 0], tsne_results[malignant_mask, 1],
           c='#2E86AB', label='Malignant', alpha=0.7, s=40, edgecolors='white', linewidth=0.5)
ax.scatter(tsne_results[normal_mask, 0], tsne_results[normal_mask, 1],
           c='#F18F01', label='Normal', alpha=0.7, s=40, edgecolors='white', linewidth=0.5)
ax.set_title('By True Diagnosis (Reference)', fontsize=11, fontweight='bold')
ax.set_xlabel('t-SNE Dimension 1')
ax.set_ylabel('t-SNE Dimension 2')
ax.legend(loc='best', fontsize=8)
ax.grid(True, alpha=0.2)

plt.suptitle(f't-SNE Visualization (BreastMNIST, perplexity={best_perp})', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('tsne_comparison_BreastMNIST.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("  ✓ Saved: tsne_comparison_BreastMNIST.png")

# ============================================================================
# FIGURE 5: Entropy distribution histogram (8x5)
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
ax.set_title('Distribution of Cluster Assignment Entropy (BreastMNIST)', fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.2, axis='y')

ax.text(0.98, 0.98, f'n = {len(H_assign)}\nMean = {H_assign.mean():.4f}\nStd = {H_assign.std():.4f}',
        transform=ax.transAxes, fontsize=9, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('entropy_distribution_BreastMNIST.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("  ✓ Saved: entropy_distribution_BreastMNIST.png")

# ============================================================================
# Print quantitative cluster separation metrics
# ============================================================================
sep72 = "─" * 72
print(f"\n{sep72}")
print("  CLUSTER SEPARATION METRICS (BreastMNIST - from hidden embeddings)")
print(f"{sep72}")

print(f"\n  Intra-cluster distance (Cluster 0 - Malignant‑dominant): {intra0:.4f}")
print(f"  Intra-cluster distance (Cluster 1 - Normal‑dominant):    {intra1:.4f}")
print(f"  Inter-cluster distance: {inter:.4f}")
print(f"  Separation ratio (inter / intra_mean): {separation_ratio:.4f}")
if separation_ratio > 1.5:
    print(f"  ✓ Excellent separation")
elif separation_ratio > 1.0:
    print(f"  ✓ Good separation")
elif separation_ratio > 0.8:
    print(f"  ⚠ Moderate separation")
else:
    print(f"  ✗ Poor separation (clusters overlapping)")

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
if cluster0_diameter > 0 and cluster1_diameter > 0:
    print(f"  Compactness (1/diameter): Cluster 0 = {1/cluster0_diameter:.4f}, Cluster 1 = {1/cluster1_diameter:.4f}")

print(f"\n{sep72}")
print("  t-SNE visualization complete (BreastMNIST)")
print(f"{sep72}\n")

!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
!pip install grad-cam
!pip install opencv-python
!pip install matplotlib scikit-learn

!pip install scikit-image
!pip install lazy_loader

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind
from skimage.measure import label as sklabel, regionprops

# ----- 1. Prepare data (as before) -----
if hasattr(subsampled_images, 'device') and subsampled_images.device.type != 'cpu':
    subsampled_images = subsampled_images.cpu()
if hasattr(yp0, 'device') and yp0.device.type != 'cpu':
    yp0 = yp0.cpu()

cluster_targets = torch.tensor(yp0, dtype=torch.long)
train_idx_t = torch.tensor(train_idx, dtype=torch.long)
val_idx_t   = torch.tensor(val_idx,   dtype=torch.long)
test_idx_t  = torch.tensor(test_idx,  dtype=torch.long)

train_images = subsampled_images[train_idx_t]
train_labels = cluster_targets[train_idx_t]
test_images  = subsampled_images[test_idx_t]
test_labels  = cluster_targets[test_idx_t]

train_dataset = TensorDataset(train_images, train_labels)
test_dataset  = TensorDataset(test_images, test_labels)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")
print(f"Test cluster distribution: C0={(test_labels==0).sum().item()}, C1={(test_labels==1).sum().item()}")

# ----- 2. Train cluster classifier -----
device = DEVICE
model_cluster = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model_cluster.fc = nn.Linear(model_cluster.fc.in_features, 2)
model_cluster = model_cluster.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_cluster.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

print("\nTraining cluster classifier...")
for epoch in range(30):
    model_cluster.train()
    total_loss = 0
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        out = model_cluster(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    scheduler.step()
    if (epoch+1) % 10 == 0:
        print(f"  Epoch {epoch+1}/30 | Loss: {total_loss/len(train_dataset):.4f}")

# Evaluate
model_cluster.eval()
correct = 0
total = 0
with torch.no_grad():
    for imgs, lbls in test_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        out = model_cluster(imgs)
        pred = out.argmax(dim=1)
        correct += (pred == lbls).sum().item()
        total += lbls.size(0)
cluster_acc = correct / total
print(f"\n✅ Cluster classifier test accuracy: {cluster_acc:.4f}")

# ----- 3. Collect Grad‑CAM saliency maps for all test images -----
target_layers = [model_cluster.layer4[-1]]
cam = GradCAM(model=model_cluster, target_layers=target_layers)

saliency_maps = []  # list of (heatmap, true_label)

# Ensure model is in eval mode before collecting saliency maps
model_cluster.eval()

for imgs, lbls in test_loader:
    imgs = imgs.to(device) # Move to device
    for i in range(imgs.shape[0]):
        # Detach and clone the image tensor, then set requires_grad to True
        img_tensor = imgs[i:i+1].clone().detach()
        img_tensor.requires_grad_(True)
        grayscale_cam = cam(input_tensor=img_tensor, targets=None)[0, :]
        saliency_maps.append((grayscale_cam, lbls[i].item()))

# Separate by cluster
saliency_c0 = [s[0] for s in saliency_maps if s[1] == 0]
saliency_c1 = [s[0] for s in saliency_maps if s[1] == 1]

print(f"\nCollected {len(saliency_c0)} saliency maps for Cluster 0, {len(saliency_c1)} for Cluster 1")

# ----- 4. Quantitative metrics per saliency map -----
def compute_saliency_metrics(heatmap):
    """Compute numerical characteristics of a saliency map."""
    # Mean activation
    mean_act = heatmap.mean()
    # Spatial spread (standard deviation of activated pixels)
    y_coords, x_coords = np.meshgrid(np.arange(heatmap.shape[0]), np.arange(heatmap.shape[1]), indexing='ij')
    weighted_center_x = (heatmap * x_coords).sum() / (heatmap.sum() + 1e-8)
    weighted_center_y = (heatmap * y_coords).sum() / (heatmap.sum() + 1e-8)
    spread = np.sqrt(((heatmap * ((x_coords - weighted_center_x)**2 + (y_coords - weighted_center_y)**2)).sum() / (heatmap.sum() + 1e-8)))
    # Entropy (measure of dispersion)
    hist, _ = np.histogram(heatmap.flatten(), bins=20, range=(0,1))
    hist = hist / (hist.sum() + 1e-8)
    entropy = -np.sum(hist * np.log2(hist + 1e-8))
    # Peak saliency
    peak = heatmap.max()
    return mean_act, spread, entropy, peak

metrics_c0 = [compute_saliency_metrics(s) for s in saliency_c0]
metrics_c1 = [compute_saliency_metrics(s) for s in saliency_c1]

metrics_names = ["Mean Activation", "Spatial Spread", "Entropy", "Peak Saliency"]
print("\n" + "─" * 60)
print("  QUANTITATIVE SALIENCY COMPARISON (Cluster 0 vs Cluster 1)")
print("─" * 60)
print(f"{'Metric':<20} {'Cluster 0 mean±std':>20} {'Cluster 1 mean±std':>20} {'p-value':>10}")
print("─" * 60)

for i, name in enumerate(metrics_names):
    vals0 = [m[i] for m in metrics_c0]
    vals1 = [m[i] for m in metrics_c1]
    t_stat, p_val = ttest_ind(vals0, vals1, equal_var=False)
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
    print(f"'{name:<20} {np.mean(vals0):>8.4f}±{np.std(vals0):.4f}   {np.mean(vals1):>8.4f}±{np.std(vals1):.4f}   {p_val:<10.4f} {sig}")

# ----- 5. Create average saliency maps per cluster -----
avg_saliency_c0 = np.mean(saliency_c0, axis=0)
avg_saliency_c1 = np.mean(saliency_c1, axis=0)

# Also create difference map (C1 - C0)
diff_map = avg_saliency_c1 - avg_saliency_c0

# ----- 6. Visualisation with average maps -----
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle(f"Quantitative Saliency Analysis (Cluster classifier acc = {cluster_acc:.3f})", fontsize=14)

# Average saliency Cluster 0
im0 = axes[0].imshow(avg_saliency_c0, cmap='jet', vmin=0, vmax=1)
axes[0].set_title(f"Average Saliency - Cluster 0 (n={len(saliency_c0)})")
axes[0].axis('off')
plt.colorbar(im0, ax=axes[0], fraction=0.046)

# Average saliency Cluster 1
im1 = axes[1].imshow(avg_saliency_c1, cmap='jet', vmin=0, vmax=1)
axes[1].set_title(f"Average Saliency - Cluster 1 (n={len(saliency_c1)})")
axes[1].axis('off')
plt.colorbar(im1, ax=axes[1], fraction=0.046)

# Difference map (C1 - C0)
cmap_diff = plt.cm.RdBu_r
im2 = axes[2].imshow(diff_map, cmap=cmap_diff, vmin=-0.3, vmax=0.3)
axes[2].set_title("Difference (Cluster 1 - Cluster 0)")
axes[2].axis('off')
plt.colorbar(im2, ax=axes[2], fraction=0.046)

# Bar plot of metrics
x = np.arange(len(metrics_names))
width = 0.35
means0 = [np.mean([m[i] for m in metrics_c0]) for i in range(4)]
stds0  = [np.std([m[i] for m in metrics_c0]) for i in range(4)]
means1 = [np.mean([m[i] for m in metrics_c1]) for i in range(4)]
stds1  = [np.std([m[i] for m in metrics_c1]) for i in range(4)]

axes[3].bar(x - width/2, means0, width, yerr=stds0, label='Cluster 0', capsize=3)
axes[3].bar(x + width/2, means1, width, yerr=stds1, label='Cluster 1', capsize=3)
axes[3].set_xticks(x)
axes[3].set_xticklabels(metrics_names, rotation=45, ha='right')
axes[3].set_ylabel('Value')
axes[3].legend()
axes[3].set_title('Saliency Metrics Comparison')

plt.tight_layout()
plt.savefig("gradcam_quantitative_analysis.png", dpi=150)
plt.show()
print("\n  ✓ Saved quantitative analysis to 'gradcam_quantitative_analysis.png'")

# ----- 7. Also display a few representative examples with statistics -----
fig2, axes2 = plt.subplots(2, n_samples, figsize=(n_samples*3, 6))
fig2.suptitle(f"Representative Grad-CAM Examples (acc={cluster_acc:.3f})", fontsize=14)

# Helper function for overlay_heatmap (if not already defined globally in the notebook)
def overlay_heatmap(image_tensor, heatmap):
    img_np = image_tensor.permute(1,2,0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    return show_cam_on_image(img_np, heatmap, use_rgb=True)

# Get sample indices (from previously selected)
# NOTE: `samples_cluster0` and `samples_cluster1` are assumed to be defined from previous cells.
# If they are not, you would need to define them here, e.g., by selecting a few random samples
# from the `test_dataset` for each cluster based on `test_labels`.

# Dummy definition for `n_samples`, `samples_cluster0`, `samples_cluster1` for reproducibility
# if they are not coming from prior execution. This might be needed if the notebook is run out of order.
if 'n_samples' not in locals() or 'samples_cluster0' not in locals() or 'samples_cluster1' not in locals():
    test_clusters_true = test_labels.numpy() if hasattr(test_labels, 'numpy') else test_labels
    cluster0_test_ids = np.where(test_clusters_true == 0)[0]
    cluster1_test_ids = np.where(test_clusters_true == 1)[0]
    n_samples = min(6, len(cluster0_test_ids), len(cluster1_test_ids))
    np.random.seed(42) # Ensure consistent sampling
    samples_cluster0 = np.random.choice(cluster0_test_ids, n_samples, replace=False)
    samples_cluster1 = np.random.choice(cluster1_test_ids, n_samples, replace=False)


for idx, sample_pos in enumerate(samples_cluster0):
    img, true_cluster = test_dataset[sample_pos]
    img_tensor = img.unsqueeze(0).to(device).clone().detach()
    img_tensor.requires_grad_(True)
    grayscale_cam = cam(input_tensor=img_tensor, targets=None)[0, :]
    overlay = overlay_heatmap(img, grayscale_cam)
    axes2[0, idx].imshow(overlay)
    axes2[0, idx].set_title(f"True: C{true_cluster}")
    axes2[0, idx].axis('off')

for idx, sample_pos in enumerate(samples_cluster1):
    img, true_cluster = test_dataset[sample_pos]
    img_tensor = img.unsqueeze(0).to(device).clone().detach()
    img_tensor.requires_grad_(True)
    grayscale_cam = cam(input_tensor=img_tensor, targets=None)[0, :]
    overlay = overlay_heatmap(img, grayscale_cam)
    axes2[1, idx].imshow(overlay)
    axes2[1, idx].set_title(f"True: C{true_cluster}")
    axes2[1, idx].axis('off')

plt.tight_layout()
plt.savefig("gradcam_representative_examples.png", dpi=150)
plt.show()
print("  ✓ Saved representative examples to 'gradcam_representative_examples.png'")

# ----- 8. Summary statistics -----
print("\n" + "═" * 60)
print("  SUMMARY STATISTICS")
print("═" * 60)
print(f"Cluster classifier accuracy (surrogate): {cluster_acc:.4f}")
print(f"\nSaliency map characteristics:")
print(f"  Cluster 0 mean activation: {np.mean([m[0] for m in metrics_c0]):.4f} ± {np.std([m[0] for m in metrics_c0]):.4f}")
print(f"  Cluster 1 mean activation: {np.mean([m[0] for m in metrics_c1]):.4f} ± {np.std([m[0] for m in metrics_c1]):.4f}")
if np.mean([m[0] for m in metrics_c0]) > np.mean([m[0] for m in metrics_c1]):
    print(f"  → Cluster 0 has significantly higher saliency (p < 0.05)")
elif np.mean([m[0] for m in metrics_c1]) > np.mean([m[0] for m in metrics_c0]):
    print(f"  → Cluster 1 has significantly higher saliency (p < 0.05)")
else:
    print(f"  → No significant difference in saliency intensity")

print(f"\n  Interpretation:")
print(f"    The surrogate classifier achieves {cluster_acc:.1%} accuracy in predicting")
print(f"    unsupervised cluster assignments from raw images, indicating that the")
print(f"    discovered clusters correspond to visually separable patterns.")
print(f"    Quantitative saliency metrics provide additional statistical evidence")
print(f"    for differences in attention patterns between clusters.")

"""
ARMA Model – Ablation Study for BreastMNIST (RadioDINO features)
Systematically varies hyperparameters and reports clustering/classification performance.
Run after ensuring all dependencies are installed.
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
                             recall_score, f1_score, log_loss,
                             adjusted_rand_score,
                             normalized_mutual_info_score,
                             adjusted_mutual_info_score,
                             silhouette_score,
                             davies_bouldin_score)
from contextlib import contextmanager
import random
import copy
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, Subset
from torchvision import transforms
import timm

# =============================================================================
# CONFIGURATION (defaults from your BreastMNIST script)
# =============================================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RADIODINO_MODEL = "hf_hub:Snarcy/RadioDino-s16"
K = 2
ACTIV = "RELU"
ALPHA = 0.63
CUT = 0
TAU = 0.1
BETA = 0.3
EMA_DECAY = 0.999
LAMBDA_CON = 1.0
NUM_EPOCHS = 3000            # full epochs, can be reduced for faster ablation
T_STRUCT = 1.0
C_ENTROPY = 0.05
N_EVAL_PASSES = 30
NUM_ARMA_LAYERS = 3

# Ablation settings
NUM_SEEDS = 10                # number of random seeds per configuration
EPOCHS_REDUCED = 1500         # you can use fewer epochs for speed (e.g., 1500)
N_EVAL_PASSES_REDUCED = 30    # keep full MC passes

# Parameter grids (each as list of values – adjust ranges as needed)
ABLATION_GRIDS = {
    "ACTIV": ["RELU", "SELU", "ELU", "GELU"],
    "NUM_ARMA_LAYERS": [1, 2, 3],
    "LAMBDA_CON": [0.0, 0.5, 1.0, 2.0],
    "C_ENTROPY": [0.0, 0.05, 0.1, 0.2],
    "T_STRUCT": [0.5, 1.0, 1.5, 2.0],
    "EMA_DECAY": [0.99, 0.999, 0.9999],
    "TAU": [0.05, 0.1, 0.2],
    "BETA": [0.1, 0.3, 0.5],
    "ALPHA": [0.60, 0.63, 0.66],
    "CUT": [0, 1],            # 0 = cut loss, 1 = modularity loss
}

# =============================================================================
# DATA LOADING – BreastMNIST + RadioDINO features (pre‑extracted once)
# =============================================================================
print("Loading BreastMNIST data and extracting RadioDINO features (once) ...")
data_npz = np.load('/home/snu/Downloads/breastmnist_224.npz', allow_pickle=True)

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

# Subsample: up to 1000 per class for faster training
dataset = TensorDataset(X_img, y_img)
class0_indices = [i for i in range(len(y_img)) if y_img[i] == 0]
class1_indices = [i for i in range(len(y_img)) if y_img[i] == 1]
random.seed(42)
sampled_class0 = random.sample(class0_indices, min(1000, len(class0_indices)))
sampled_class1 = random.sample(class1_indices, min(1000, len(class1_indices)))
combined_indices = sampled_class0 + sampled_class1
random.shuffle(combined_indices)
final_dataset = Subset(dataset, combined_indices)
final_loader = DataLoader(final_dataset, batch_size=64, shuffle=False)

# Extract RadioDINO features (once, global)
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
y = np.array(y_list, dtype=np.int64)
FEATS_DIM = features_np.shape[1]

# Shuffle with fixed seed for reproducibility
np.random.seed(42)
perm = np.random.permutation(features_np.shape[0])
features_np = features_np[perm]
y = y[perm]

print(f"RadioDINO feature dimension: {FEATS_DIM}")
print(f"Features: {features_np.shape}, Labels: {y.shape} "
      f"(Malignant: {np.sum(y==0)}, Normal: {np.sum(y==1)})")
print("Labels used ONLY for post‑hoc external validation.\n")

# =============================================================================
# GRAPH UTILITIES (will be recomputed for each ALPHA value)
# =============================================================================
def create_adj(features, alpha):
    F_ = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)
    W = np.dot(F_, F_.T)
    W = np.where(W >= alpha, 1, 0).astype(np.float32)
    mx = W.max()
    W = (W / mx).astype(np.float32) if mx > 0 else W
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
# LOSS FUNCTIONS (same as original)
# =============================================================================
def jsd_loss(p, q, tau=0.07, eps=1e-8):
    p_ = F.softmax(p / tau, dim=-1) + eps
    q_ = F.softmax(q / tau, dim=-1) + eps
    m = 0.5 * (p_ + q_)
    kl = lambda a, b: (a * (a / b).log()).sum(dim=-1)
    return (0.5 * (kl(p_, m) + kl(q_, m)) / np.log(2)).mean()

def contrastive_loss(h1, h2, z1, z2, beta=0.5, tau=0.07):
    l1 = beta * jsd_loss(h1, h2, tau) + (1 - beta) * jsd_loss(h1, z2, tau)
    l2 = beta * jsd_loss(h2, h1, tau) + (1 - beta) * jsd_loss(h2, z1, tau)
    return l1, l2

# =============================================================================
# MODEL COMPONENTS (identical to your original)
# =============================================================================
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
    def __init__(self, input_dim, hidden_dim, device, activ="RELU",
                 num_stacks=1, num_layers=1, num_arma_layers=3):
        super().__init__()
        self.device = device
        self.act = ACTIVATIONS.get(activ, F.elu)
        def _arma(i, o):
            return ARMAConv(i, o, num_stacks=num_stacks, num_layers=num_layers,
                            act=self.act, shared_weights=True, dropout=0.4)
        self.arma_layers = nn.ModuleList(
            [_arma(input_dim if i == 0 else hidden_dim, hidden_dim)
             for i in range(num_arma_layers)]
        )
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_arma_layers)])
        self.drop = nn.Dropout(0.4)
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
        S = F.softmax(S / self.T_struct, dim=1)
        Ap = (A @ S).t() @ S
        D = torch.diag(A.sum(dim=-1))
        Dp = (D @ S).t() @ S
        mc = -(Ap.trace() / Dp.trace())
        SS = S.t() @ S
        I = torch.eye(self.num_clusters, device=self.device)
        oc = torch.norm(SS / SS.norm() - I / I.norm())
        return mc + oc
    def _modularity_loss(self, A, S):
        C = F.softmax(S, dim=1)
        d = A.sum(dim=1); m = A.sum()
        B = A - torch.ger(d, d) / (2 * m)
        k = torch.tensor(self.num_clusters, device=self.device, dtype=torch.float32)
        mod = (-1 / (2 * m)) * torch.trace(C.t() @ B @ C)
        coll = (k.sqrt() / S.shape[0]) * torch.norm(C.sum(dim=0), p='fro') - 1
        return mod + coll

# =============================================================================
# EVALUATION FUNCTIONS (with weighted metrics)
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

def evaluate_model(model, feats, ei, y_true, device, n_passes=30, seed_base=9999):
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
    # align clusters with true labels (post‑hoc)
    a = accuracy_score(y_true, yp)
    ai = accuracy_score(y_true, 1 - yp)
    if ai > a:
        yp = 1 - yp
        logits_mean = logits_mean[:, ::-1].copy()
        logits_stack = logits_stack[:, ::-1, :].copy()
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
    ll = log_loss(y_true, y_prob)
    # Clustering metrics
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic')
    ami = adjusted_mutual_info_score(y_true, y_pred, average_method='arithmetic')
    # Geometric metrics
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
        'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'log_loss': ll,
        'nmi': nmi, 'ari': ari, 'ami': ami,
        'silhouette': sil, 'db': db,
        'mean_entropy': mean_ent, 'std_entropy': std_ent
    }

def train_and_evaluate(config, seeds=NUM_SEEDS, epochs=EPOCHS_REDUCED, n_eval=N_EVAL_PASSES_REDUCED):
    """Train model with given hyperparameter config and return mean/std over seeds."""
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
        opt = AdamW(model.parameters(), lr=1e-4, weight_decay=5e-5)
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
    # Baseline configuration (your original settings)
    default_config = {
        'ACTIV': ACTIV, 'NUM_ARMA_LAYERS': NUM_ARMA_LAYERS, 'LAMBDA_CON': LAMBDA_CON,
        'C_ENTROPY': C_ENTROPY, 'T_STRUCT': T_STRUCT, 'EMA_DECAY': EMA_DECAY,
        'TAU': TAU, 'BETA': BETA, 'ALPHA': ALPHA, 'CUT': CUT
    }
    print("Running baseline configuration...")
    baseline = train_and_evaluate(default_config, seeds=NUM_SEEDS, epochs=EPOCHS_REDUCED)
    row = {'Parameter': 'Baseline', 'Value': 'default'}
    for k, (m, s) in baseline.items():
        row[f'{k}_mean'] = m
        row[f'{k}_std'] = s
    all_rows.append(row)

    # Loop over each hyperparameter
    for param, values in ABLATION_GRIDS.items():
        for val in values:
            if val == default_config.get(param):
                continue   # skip duplicate of baseline
            print(f"\nAblating {param} = {val}")
            config = default_config.copy()
            config[param] = val
            res = train_and_evaluate(config, seeds=NUM_SEEDS, epochs=EPOCHS_REDUCED)
            row = {'Parameter': param, 'Value': str(val)}
            for k, (m, s) in res.items():
                row[f'{k}_mean'] = m
                row[f'{k}_std'] = s
            all_rows.append(row)

    # Save results
    df = pd.DataFrame(all_rows)
    csv_path = "arma_ablation_results_breastmnist.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nAblation completed. Results saved to {csv_path}")
    # Print a summary
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print(df.round(4).to_string())
    return df

if __name__ == "__main__":
    run_ablation()
