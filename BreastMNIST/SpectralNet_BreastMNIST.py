import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, Subset
from torchvision import transforms
import timm
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, normalized_mutual_info_score,
    adjusted_rand_score, adjusted_mutual_info_score,
    silhouette_score, davies_bouldin_score, log_loss
)
from munkres import Munkres
import random
import pandas as pd

# ─── REPRODUCIBILITY ──────────────────────────────────────────────────────────
def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("CUDA available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
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
RADIODINO_MODEL = "hf_hub:Snarcy/RadioDino-s16"

# ─── DATA: BREASTMNIST + RADIODINO FEATURES ───────────────────────────────────
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
radiodino.eval().to(device)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

rd_feats, y_list = [], []
with torch.no_grad():
    for imgs, lbls in final_loader:
        imgs = imgs.to(device)
        imgs_norm = normalize(imgs)
        feats = radiodino(imgs_norm)
        rd_feats.append(feats.cpu())
        y_list.extend(lbls.cpu().tolist())

X = torch.cat(rd_feats, dim=0).numpy().astype(np.float32)
y = np.array(y_list, dtype=np.int64)

# Shuffle once for reproducibility
np.random.seed(42)
perm = np.random.permutation(X.shape[0])
X, y = X[perm], y[perm]

N, F_DIM = X.shape
print(f"Nodes: {N}   Features: {F_DIM}   Malignant: {(y==0).sum()}   Normal: {(y==1).sum()}")
print("NOTE: Diagnostic labels are held out and used ONLY for post-hoc external validation.\n")

# ─── DATASET (pair generation for Siamese) ────────────────────────────────────
class PairDataset(Dataset):
    def __init__(self, pairs, labels):
        self.pairs  = pairs
        self.labels = labels
    def __len__(self):  return len(self.pairs)
    def __getitem__(self, idx):
        return self.pairs[idx][0], self.pairs[idx][1], self.labels[idx]

# ─── MODELS (unchanged) ───────────────────────────────────────────────────────
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

# ─── LOSSES (unchanged) ───────────────────────────────────────────────────────
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

# ─── GRAPH UTILITIES ──────────────────────────────────────────────────────────
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

# ─── TRAINING FUNCTIONS ───────────────────────────────────────────────────────
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

# ─── TEMPERATURE CALIBRATION ──────────────────────────────────────────────────
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

# ─── MC PERTURBATION EVALUATION ───────────────────────────────────────────────
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
    logits_stack = np.stack(all_log_probs, axis=2)   # (N, K, P)
    logits_mean  = logits_stack.mean(axis=2)          # (N, K)
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

# ─── UNCERTAINTY FUNCTIONS ────────────────────────────────────────────────────
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
    label_map = {0: "Malignant", 1: "Normal"}
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
        print(f"  {label:<35}  {vals.mean():>9.4f}  {vals.std():>9.4f}  "
              f"{vals.min():>9.4f}  {vals.max():>9.4f}")
    # Entropy distribution bands
    print(f"\n  Entropy distribution across subjects:")
    print(f"  {'Band (bits)':<20} {'Count':>7} {'%':>7}")
    print(f"  {'─'*38}")
    for lo, hi, lbl in [(0.00,0.10,"Very confident  < 0.1"),
                        (0.10,0.30,"High conf  0.1–0.3"),
                        (0.30,0.50,"Moderate   0.3–0.5"),
                        (0.50,0.70,"Uncertain  0.5–0.7"),
                        (0.70,0.90,"High unc   0.7–0.9"),
                        (0.90,1.01,"Max unc    0.9–1.0")]:
        n = ((H_assign >= lo) & (H_assign < hi)).sum()
        print(f"  {lbl:<20}  {n:>7}  {100*n/len(y_true):>6.1f}%")
    df_sorted = df.sort_values("entropy_assignment", ascending=False).reset_index(drop=True)
    print(f"\n  Top-10 most uncertain cluster assignments:")
    cols_show = ["sample_id", "true_label", "cluster_assignment",
                 "p_cluster0", "p_cluster1", "entropy_assignment", "model_uncertainty"]
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.width", 160)
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

# ─── CLUSTERING METRICS (unchanged) ───────────────────────────────────────────
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
    header = (f"  {'Space':<22} {'ARI':>8} {'NMI':>8} {'AMI':>8} "
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
            print(f"  {metric+' '+hl:<16}  {valid.mean():>9.4f}  {valid.std():>9.4f}"
                  f"  {valid.min():>9.4f}  {valid.max():>9.4f}")
    print(f"\n{sep}")

# ──────────────────────────────────────────────────────────────────────────────
#  MAIN LOOP (10 runs with weighted metrics collection)
# ──────────────────────────────────────────────────────────────────────────────
SEP    = "═" * 72
sep100 = "─" * 100

print(f"\n{SEP}")
print(f"  SpectralNet (Siamese + SpectralNet + KMeans) – BreastMNIST")
print(f"  RadioDINO features (dim={F_DIM}), {N_RUNS} runs, {N_MC_PASSES} MC passes")
print(f"  τ = {TAU_FACTOR} × inter-centroid distance (calibrated per run)")
print(f"  Labels used ONLY for post-hoc external validation.")
print(f"{SEP}\n")

col_w = "─" * 110
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
all_tau = []
calibration_rows = []          # will store weighted metrics for final summary
seed_clustering_records = []

last_logits_stack = None
last_yp = None
last_tau = None

for run in range(N_RUNS):
    setup_seed(42 + run)
    print(f"\n  ── Run {run+1}/{N_RUNS} ──────────────────────────────────────")

    # --- Build positive/negative pairs for Siamese ---
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

    # --- Train Siamese ---
    siamese = SiameseNet(F_DIM, HIDDEN_DIM)
    siamese = train_siamese(siamese, dataloader, epochs=SIAMESE_EPOCHS)
    siamese.eval()
    with torch.no_grad():
        X_embed = siamese.forward_once(torch.tensor(X, dtype=torch.float32)).numpy()

    # --- Build affinity graph ---
    scale_val = compute_scale(X_embed, N_NEIGHBORS)
    W_np = compute_affinity(X_embed, scale_val, N_NEIGHBORS)
    print(f"  Affinity edges: {np.count_nonzero(W_np) // 2}")

    # --- Train SpectralNet ---
    spectral = SpectralNet(F_DIM, N_CLUSTERS, HIDDEN_DIM)
    spectral = train_spectral(spectral, X, W_np, epochs=SPECTRAL_EPOCHS)

    # --- Deterministic embeddings + KMeans ---
    spectral.eval()
    with torch.no_grad():
        Y_det = spectral(torch.tensor(X, dtype=torch.float32)).numpy()
    kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=20, random_state=run)
    kmeans.fit(Y_det)
    centers = kmeans.cluster_centers_

    # --- MC perturbation evaluation ---
    yp_i, ypp_i, logits_stack_i, logits_mean_i, tau_i = mc_uncertainty(
        spectral, X, centers,
        n_passes=N_MC_PASSES, seed_base=9999 + run,
        verbose_calibration=(run == 0)
    )

    # --- Weighted metrics (average over both classes) ---
    acc_i = accuracy_score(y, yp_i)
    prec_w = precision_score(y, yp_i, average='weighted', zero_division=0)
    rec_w  = recall_score(y, yp_i, average='weighted', zero_division=0)
    f1_w   = f1_score(y, yp_i, average='weighted', zero_division=0)
    nmi_i  = normalized_mutual_info_score(y, yp_i, average_method='arithmetic')

    # --- Uncertainty ---
    H_assign_i, _, _, _ = compute_cluster_assignment_uncertainty(logits_stack_i)
    mean_ent_i = float(H_assign_i.mean())
    std_ent_i  = float(H_assign_i.std())

    tn_i, fp_i, fn_i, tp_i = confusion_matrix(y, yp_i).ravel()

    # --- Clustering metrics for this run ---
    cl_res_i = evaluate_clustering(Y_det, yp_i, logits_mean_i, y,
                                   prefix=f"Run {run+1}")
    cl_flat = {}
    for r in cl_res_i:
        sp = r['space']
        for m in ['ARI','NMI','AMI','Silhouette','DaviesBouldin']:
            cl_flat[f"{m}_{sp}"] = r[m]
    seed_clustering_records.append(cl_flat)

    # Store weighted results
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

    print(f"  {run+1:>3}  {acc_i:>7.4f}  {prec_w:>8.4f}  {rec_w:>8.4f}  {f1_w:>8.4f}  "
          f"{nmi_i:>7.4f}  {mean_ent_i:>13.4f}  {std_ent_i:>11.4f}  "
          f"{tn_i:>5}  {fp_i:>5}  {fn_i:>5}  {tp_i:>5}")

    if run == 0:
        last_logits_stack = logits_stack_i
        last_yp = yp_i
        last_tau = tau_i

# --- Detailed uncertainty report (run 1) ---
print(f"\n{SEP}\n  DETAILED UNCERTAINTY REPORT — Run 1\n{SEP}")
df_unc = print_cluster_uncertainty_report(
    last_logits_stack, last_yp, y,
    n_passes=N_MC_PASSES, tau=last_tau,
    sample_ids=list(range(N))
)
df_unc.to_csv("spectralnet_breastmnist_uncertainty_run1.csv", index=False, float_format="%.6f")
print(f"\n  CSV → spectralnet_breastmnist_uncertainty_run1.csv  ({len(df_unc)} subjects)")

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
row = "SpectralNet"
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
latex_row = "SpectralNet"
for m, s in zip(means, stds):
    if np.isnan(m):
        latex_row += " & N/A±N/A"
    else:
        latex_row += f" & ${m:.4f}\\pm{s:.4f}$"
latex_row += " \\\\"
print(latex_row)

# ─── 10‑run summary (using weighted metrics) ─────────────────────────────────
print(f"\n{sep100}")
print(f"  10-RUN SUMMARY  (mean ± std)                [SpectralNet, BreastMNIST]")
print(f"  NOTE: Labels used only for post-hoc external validation.")
print(f"{sep100}")

ext_labels = [
    ("Accuracy",          "Weighted accuracy (unweighted also same here)"),
    ("Weighted Precision", "Post-hoc external validation"),
    ("Weighted Recall",    "Post-hoc external validation"),
    ("Weighted F1",        "Post-hoc external validation"),
    ("NMI",                "Post-hoc external validation"),
    ("Mean Entropy",       "Mean cluster assignment entropy (bits)"),
    ("Std Entropy",        "Std of assignment entropy across subjects"),
]
print(f"\n  A.  EXTERNAL VALIDATION & ASSIGNMENT UNCERTAINTY")
print(f"  {'Metric':<20}  {'Mean':>9}  {'Std':>9}  {'Min':>9}  {'Max':>9}  Note")
print(f"  {sep100}")
ext_means = [means[0], means[1], means[2], means[3], means[4]]
ext_stds = [stds[0], stds[1], stds[2], stds[3], stds[4]]
ext_min = [np.min(acc_vals), np.min(prec_vals), np.min(rec_vals), np.min(f1_vals), np.min(nmi_vals)]
ext_max = [np.max(acc_vals), np.max(prec_vals), np.max(rec_vals), np.max(f1_vals), np.max(nmi_vals)]
for idx, (label, note) in enumerate(ext_labels[:5]):
    print(f"  {label:<20}  {ext_means[idx]:>9.4f}  {ext_stds[idx]:>9.4f}  "
          f"{ext_min[idx]:>9.4f}  {ext_max[idx]:>9.4f}  {note}")

print(f"\n  τ across runs: mean={np.mean(all_tau):.6f}  "
      f"std={np.std(all_tau):.6f}  min={np.min(all_tau):.6f}  max={np.max(all_tau):.6f}")

counts_np = np.array([[r['tn'], r['fp'], r['fn'], r['tp']] for r in calibration_rows], dtype=float)
print(f"\n  B.  CONFUSION MATRIX COUNTS (post-hoc external validation only)")
print(f"  {'Group':<16}  {'Mean':>7}  {'Std':>7}  {'Min':>5}  {'Max':>5}  Note")
print(f"  {sep100}")
count_labels = [
    ("TN  (Malig→C0)", "Malignant assigned to Cluster 0"),
    ("FP  (Malig→C1)", "Malignant assigned to Cluster 1"),
    ("FN  (Norm→C0)",  "Normal assigned to Cluster 0"),
    ("TP  (Norm→C1)",  "Normal assigned to Cluster 1"),
]
for idx, (label, note) in enumerate(count_labels):
    v = counts_np[:, idx]
    print(f"  {label:<16}  {v.mean():>7.1f}  {v.std():>7.2f}  "
          f"{v.min():>5.0f}  {v.max():>5.0f}  {note}")

print(f"\n  C.  CLUSTER STABILITY ACROSS {N_RUNS} RUNS")
print(f"  {'Metric':<22}  {'Mean':>9}  {'Std':>9}  Note")
print(f"  {sep100}")
print(f"  {'Weighted F1':<22}  {np.mean(all_f1_weighted):>9.4f}  {np.std(all_f1_weighted):>9.4f}"
      f"  Consistency of weighted F1")
print(f"  {'Assignment Entropy':<22}  {np.mean(all_ent):>9.4f}  {np.std(all_ent):>9.4f}"
      f"  Low std = stable confidence")
print(f"  {sep100}\n")

# ─── Clustering metrics summary ─────────────────────────────────────────────
print(f"\n{SEP}\n  CLUSTERING METRICS SUMMARY ({N_RUNS} runs)\n{SEP}")
print_clustering_summary(seed_clustering_records, label="SpectralNet — BreastMNIST (RadioDINO)")

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
df_summary.to_csv("spectralnet_breastmnist_10run_summary.csv", index=False, float_format="%.6f")
print(f"\n  ✓ Saved → spectralnet_breastmnist_10run_summary.csv")
print(f"\n{'='*60}\n  ANALYSIS COMPLETE\n{'='*60}")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, Subset
from torchvision import transforms
import timm
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, normalized_mutual_info_score,
    adjusted_rand_score, adjusted_mutual_info_score,
    silhouette_score, davies_bouldin_score, log_loss
)
from munkres import Munkres
import random
import pandas as pd

# -- REPRODUCIBILITY (Copied from above for self-contained execution if needed)
def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# NOTE: `setup_seed` and `device` are globally defined in a previous cell.
# This cell re-imports modules and redefines functions, but assumes global `device`
# and `spectral` are accessible from the main training loop execution.

# Global variable definitions are assumed from prior execution
# N_CLUSTERS, HIDDEN_DIM, BATCH_SIZE, N_NEIGHBORS, SIAMESE_EPOCHS, SPECTRAL_EPOCHS,
# N_RUNS, N_MC_PASSES, FEAT_DROP, FEAT_NOISE_SD, TAU_FACTOR, RADIODINO_MODEL
# X, y, F_DIM, spectral, yp_i, logits_stack_i, SEP (from the main loop)

# Re-define models and losses if this cell is to be run independently
# class SiameseNet(nn.Module): ...
# class OrthoLinear(nn.Module): ...
# class SpectralNet(nn.Module): ...
# def contrastive_loss(dist, labels, margin=1.0): ...
# def spectral_loss(Y, W): ...
# def compute_cluster_assignment_uncertainty(logits_stack): ...

# =============================================================================
# PART H — t-SNE VISUALIZATION OF CLUSTERS (SpectralNet / BreastMNIST)
# =============================================================================
print(f"\n{SEP}\n  PART H — t-SNE VISUALIZATION OF DISCOVERED CLUSTERS (SpectralNet)\n{SEP}")
print("  Visualizing the hidden embedding space (before orthogonal projection).\n")

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances

# Set publication-quality style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 300

# -----------------------------------------------------------------------------
# Extract hidden embeddings (before ortho layer)
# -----------------------------------------------------------------------------
def get_spectral_hidden(model, X_np, device):
    """Return activations of the second linear layer (hidden_dim = 256)."""
    original_model_device = next(model.parameters()).device # Store current device of the model
    model.to(device) # Move model to the specified device for computation
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_np, dtype=torch.float32, device=device)
        x = F.relu(model.fc1(X_t))
        x = F.relu(model.fc2(x))          # shape (N, hidden_dim)
        hidden = x.cpu().numpy()
    model.to(original_model_device) # Move model back to its original device
    return hidden

# Use the trained spectral model from the first run (run 0)
hidden_embeddings = get_spectral_hidden(spectral, X, device)
print(f"  Hidden embedding shape: {hidden_embeddings.shape}")

# Cluster assignments and true labels
cluster_assignments = yp_i   # from first run
true_labels = y

# Assignment uncertainty (using logits_stack_i from run 0)
H_assign, H_aleat, MI, p_mean = compute_cluster_assignment_uncertainty(logits_stack_i)
high_uncertainty_mask = H_assign > 0.7
low_uncertainty_mask = H_assign <= 0.3

# t-SNE (perplexity=40 works well for ~2000 samples)
best_perp = 40
print(f"  Computing t-SNE with perplexity={best_perp}...", end=" ", flush=True)
tsne = TSNE(n_components=2, random_state=42, perplexity=best_perp,
            init='pca', max_iter=1000)
tsne_results = tsne.fit_transform(hidden_embeddings)
print("done")

# Cluster masks
cluster0_mask = (cluster_assignments == 0)
cluster1_mask = (cluster_assignments == 1)
malignant_mask = (true_labels == 0)
normal_mask = (true_labels == 1)

# Cluster centres in t-SNE space
cluster0_center = tsne_results[cluster0_mask].mean(axis=0)
cluster1_center = tsne_results[cluster1_mask].mean(axis=0)

# Separation metrics (in hidden space)
cluster0_hid = hidden_embeddings[cluster0_mask]
cluster1_hid = hidden_embeddings[cluster1_mask]
intra0 = pairwise_distances(cluster0_hid).mean() if len(cluster0_hid) > 1 else 0
intra1 = pairwise_distances(cluster1_hid).mean() if len(cluster1_hid) > 1 else 0
intra_mean = (intra0 + intra1) / 2
inter = pairwise_distances(cluster0_hid, cluster1_hid).mean() if len(cluster0_hid) > 0 and len(cluster1_hid) > 0 else 0
separation_ratio = inter / intra_mean if intra_mean > 0 else 0

# ============================================================================
# FIGURE 1: t-SNE colored by cluster assignment (8x6)
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

ax.scatter(tsne_results[cluster0_mask, 0], tsne_results[cluster0_mask, 1],
           c='#2E86AB', label='Cluster 0 (Malignant-dominant)',
           alpha=0.7, s=50, edgecolors='white', linewidth=0.5, zorder=2)
ax.scatter(tsne_results[cluster1_mask, 0], tsne_results[cluster1_mask, 1],
           c='#A23B72', label='Cluster 1 (Normal-dominant)',
           alpha=0.7, s=50, edgecolors='white', linewidth=0.5, zorder=2)

ax.scatter(cluster0_center[0], cluster0_center[1], c='darkblue', s=150,
           edgecolors='black', linewidth=2, marker='*', label='Cluster 0 center', zorder=3)
ax.scatter(cluster1_center[0], cluster1_center[1], c='darkred', s=150,
           edgecolors='black', linewidth=2, marker='*', label='Cluster 1 center', zorder=3)

ax.set_title('t-SNE: SpectralNet Embeddings (BreastMNIST)\nColored by Cluster Assignment',
             fontsize=12, fontweight='bold')
ax.set_xlabel('t-SNE Dimension 1', fontsize=10)
ax.set_ylabel('t-SNE Dimension 2', fontsize=10)
ax.legend(loc='best', framealpha=0.9, fontsize=8)
ax.grid(True, alpha=0.2, linestyle='--')
ax.set_facecolor('#f8f9fa')
plt.tight_layout()
plt.savefig('tsne_by_cluster_assignment_SpectralNet.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("  ✓ Saved: tsne_by_cluster_assignment_SpectralNet.png")

# ============================================================================
# FIGURE 2: t-SNE colored by true diagnosis (8x6)
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

ax.scatter(tsne_results[malignant_mask, 0], tsne_results[malignant_mask, 1],
           c='#2E86AB', label='Malignant (True label)',
           alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
ax.scatter(tsne_results[normal_mask, 0], tsne_results[normal_mask, 1],
           c='#F18F01', label='Normal (True label)',
           alpha=0.7, s=50, edgecolors='white', linewidth=0.5)

ax.set_title('t-SNE: SpectralNet Embeddings (BreastMNIST)\nColored by True Diagnosis (Reference)',
             fontsize=12, fontweight='bold')
ax.set_xlabel('t-SNE Dimension 1', fontsize=10)
ax.set_ylabel('t-SNE Dimension 2', fontsize=10)
ax.legend(loc='best', framealpha=0.9, fontsize=8)
ax.grid(True, alpha=0.2, linestyle='--')
ax.set_facecolor('#f8f9fa')
plt.tight_layout()
plt.savefig('tsne_by_true_diagnosis_SpectralNet.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("  ✓ Saved: tsne_by_true_diagnosis_SpectralNet.png")

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
stats_text += f'Confident (entropy \u2264 0.3): {low_uncertainty_mask.sum()}/{len(H_assign)} ({100*low_uncertainty_mask.sum()/len(H_assign):.1f}%)\n'
stats_text += f'Ambiguous (entropy > 0.7): {high_uncertainty_mask.sum()}/{len(H_assign)} ({100*high_uncertainty_mask.sum()/len(H_assign):.1f}%)\n'
stats_text += f'Separation Ratio: {separation_ratio:.2f}'

ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.set_title('t-SNE: SpectralNet Embeddings (BreastMNIST)\nColored by Assignment Uncertainty',
             fontsize=12, fontweight='bold')
ax.set_xlabel('t-SNE Dimension 1', fontsize=10)
ax.set_ylabel('t-SNE Dimension 2', fontsize=10)
ax.legend(loc='lower right', framealpha=0.9, fontsize=8)
ax.grid(True, alpha=0.2, linestyle='--')
ax.set_facecolor('#f8f9fa')
plt.tight_layout()
plt.savefig('tsne_uncertainty_heatmap_SpectralNet.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("  ✓ Saved: tsne_uncertainty_heatmap_SpectralNet.png")

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

plt.suptitle(f't-SNE Visualization (SpectralNet, perplexity={best_perp})', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('tsne_comparison_SpectralNet.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("  ✓ Saved: tsne_comparison_SpectralNet.png")

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
ax.set_title('Distribution of Cluster Assignment Entropy (SpectralNet)', fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.2, axis='y')

ax.text(0.98, 0.98, f'n = {len(H_assign)}\nMean = {H_assign.mean():.4f}\nStd = {H_assign.std():.4f}',
        transform=ax.transAxes, fontsize=9, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('entropy_distribution_SpectralNet.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("  ✓ Saved: entropy_distribution_SpectralNet.png")

# ============================================================================
# Print separation metrics
# ============================================================================
sep72 = "─" * 72
print(f"\n{sep72}")
print("  CLUSTER SEPARATION METRICS (SpectralNet hidden space)")
print(f"{sep72}")
print(f"\n  Intra-cluster distance (Cluster 0 - Malignant-dominant): {intra0:.4f}")
print(f"  Intra-cluster distance (Cluster 1 - Normal-dominant):    {intra1:.4f}")
print(f"  Inter-cluster distance: {inter:.4f}")
print(f"  Separation ratio (inter / intra_mean): {separation_ratio:.4f}")
if separation_ratio > 1.5:
    print("  \u2713 Excellent separation")
elif separation_ratio > 1.0:
    print("  \u2713 Good separation")
elif separation_ratio > 0.8:
    print("  \u26A0 Moderate separation")
else:
    print("  \u2717 Poor separation (clusters overlapping)")

# Cluster diameters
if len(cluster0_hid) > 1:
    cluster0_diameter = cdist(cluster0_hid, cluster0_hid).max()
else:
    cluster0_diameter = 0
if len(cluster1_hid) > 1:
    cluster1_diameter = cdist(cluster1_hid, cluster1_hid).max()
else:
    cluster1_diameter = 0

print(f"\n  Cluster 0 diameter: {cluster0_diameter:.4f}")
print(f"  Cluster 1 diameter: {cluster1_diameter:.4f}")
if cluster0_diameter > 0 and cluster1_diameter > 0:
    print(f"  Compactness (1/diameter): C0 = {1/cluster0_diameter:.4f}, C1 = {1/cluster1_diameter:.4f}")

print(f"\n{sep72}")
print("  t-SNE visualization complete (SpectralNet)")
print(f"{sep72}\n")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    adjusted_rand_score, normalized_mutual_info_score,
    adjusted_mutual_info_score, silhouette_score, davies_bouldin_score
)
from sklearn.cluster import KMeans
from contextlib import contextmanager
import random
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import fdrcorrection
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

FEATS_DIM = 180
HIDDEN_DIM = 256          # GCN hidden dimension
PROJ_DIM = 128            # Projection dimension (default in SIGNA)
NUM_CLUSTERS = 2
NUM_EPOCHS = 1000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
EMBED_DROPOUT_PROB = 0.4
ACTIVATION = 'prelu'
NUM_LAYERS = 2
USE_LAYER_NORM = True
N_EVAL_PASSES = 30
SEED_BASE = 42

# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING (same as your original pipeline)
# ═══════════════════════════════════════════════════════════════════════════
cn_data = np.load("/home/snu/Downloads/Histogram_CN_FA_20bin_updated.npy", allow_pickle=True)
mci_data = np.load("/home/snu/Downloads/Histogram_MCI_FA_20bin_updated.npy", allow_pickle=True)

X = np.vstack([cn_data, mci_data])
y = np.hstack([np.zeros(cn_data.shape[0], dtype=np.int64),
               np.ones(mci_data.shape[0], dtype=np.int64)])

np.random.seed(SEED_BASE)
perm = np.random.permutation(X.shape[0])
X, y = X[perm], y[perm]
features_np = X.astype(np.float32)

print(f"Features: {X.shape}, Labels: {y.shape} (CN: {np.sum(y==0)}, MCI: {np.sum(y==1)})")
print("NOTE: Diagnostic labels are held out and used ONLY for post-hoc external validation.\n")

# ═══════════════════════════════════════════════════════════════════════════
# GRAPH CONSTRUCTION (same as your method)
# ═══════════════════════════════════════════════════════════════════════════
def create_adj(features, cut, alpha=0.92):
    F_ = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)
    W = np.dot(F_, F_.T)
    if cut == 0:
        W = np.where(W >= alpha, 1, 0).astype(np.float32)
    else:
        W = (W * (W >= alpha)).astype(np.float32)
    return W

def edge_index_from_dense(W):
    r, c = np.nonzero(W > 0)
    return np.vstack([r, c]).astype(np.int64), W[r, c].astype(np.float32)

W0 = create_adj(features_np, cut=0, alpha=0.92)
edge_index_np, _ = edge_index_from_dense(W0)
adj_dense = torch.from_numpy(W0).float().to(DEVICE)
data = Data(x=torch.from_numpy(features_np).float(),
            edge_index=torch.from_numpy(edge_index_np).long()).to(DEVICE)

print(f"Graph: nodes={features_np.shape[0]}, edges={edge_index_np.shape[1]}\n")

# ═══════════════════════════════════════════════════════════════════════════
# SIGNA MODEL (official architecture)
# ═══════════════════════════════════════════════════════════════════════════
class GCNEncoder(nn.Module):
    """2-layer GCN with dropout before each layer and optional LayerNorm."""
    def __init__(self, in_dim, hidden_dim, num_layers=2, dropout_prob=0.4,
                 activation='prelu', use_ln=True):
        super().__init__()
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.use_ln = use_ln

        if activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            self.activation = nn.PReLU()

        self.convs = nn.ModuleList()
        self.lns = nn.ModuleList() if use_ln else None

        # First layer
        self.convs.append(GCNConv(in_dim, hidden_dim))
        if use_ln:
            self.lns.append(nn.LayerNorm(hidden_dim))

        # Second layer (if num_layers >= 2)
        if num_layers == 2:
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            if use_ln:
                self.lns.append(nn.LayerNorm(hidden_dim))

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            if self.dropout_prob > 0:
                x = F.dropout(x, p=self.dropout_prob, training=self.training)
            x = self.convs[i](x, edge_index)
            if self.use_ln:
                x = self.lns[i](x)
            x = self.activation(x)
        return x

class SIGNAModel(nn.Module):
    """Complete SIGNA model with projection head and official loss."""
    def __init__(self, encoder, hidden_dim, proj_dim):
        super().__init__()
        self.encoder = encoder
        self.fc1 = nn.Linear(hidden_dim, proj_dim)
        self.fc2 = nn.Linear(proj_dim, hidden_dim)
        self.activation = nn.PReLU()
        self.eps = torch.tensor([1e-6], device=DEVICE)

    def forward(self, x, edge_index):
        """Return node embeddings (before projection)."""
        return self.encoder(x, edge_index)

    def project(self, z):
        """Project to lower-dimensional space (used for contrast)."""
        h = self.activation(self.fc1(z))
        return self.fc2(h)

    def loss(self, z, adj):
        """
        Official SIGNA loss (Eq. 12) using full adjacency with self-loops.
        Args:
            z: node embeddings [N, hidden_dim]
            adj: dense adjacency matrix with self-loops [N, N] (0/1)
        """
        h = self.project(z)                     # [N, hidden_dim]
        h = F.normalize(h, p=2, dim=1)         # ℓ2-normalize
        rec = (torch.mm(h, h.t()) + 1.0) / 2.0  # similarity in [0,1]

        # Positive pairs: neighbours (including self)
        pos_mask = adj > 0
        neg_mask = ~pos_mask

        # Binary cross‑entropy (averaged per node)
        pos_loss = -(pos_mask.float() * torch.log(rec + self.eps)).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1e-6)
        neg_loss = -(neg_mask.float() * torch.log(1 - rec + self.eps)).sum(dim=1) / neg_mask.sum(dim=1).clamp(min=1e-6)

        return (pos_loss + neg_loss).mean()

# ═══════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS FOR EVALUATION & UNCERTAINTY
# ═══════════════════════════════════════════════════════════════════════════
def entropy_bits(p):
    p = np.clip(p, 1e-12, 1.0)
    return -np.sum(p * np.log2(p), axis=1)

def softmax_np(logits):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

@contextmanager
def mc_dropout_mode(model):
    """Enable dropout during evaluation for MC uncertainty."""
    model.train()
    try:
        yield model
    finally:
        model.eval()

def evaluate_clustering(embeddings, pred_labels, true_labels, space_name=""):
    """Compute ARI, NMI, AMI, Silhouette, Davies-Bouldin."""
    unique = np.unique(pred_labels)
    n_valid = sum((pred_labels == c).sum() >= 2 for c in unique)
    can_geom = (len(unique) >= 2) and (n_valid == len(unique))

    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels, average_method='arithmetic')
    ami = adjusted_mutual_info_score(true_labels, pred_labels, average_method='arithmetic')

    if can_geom:
        sil = silhouette_score(embeddings, pred_labels, metric='euclidean')
        db = davies_bouldin_score(embeddings, pred_labels)
    else:
        sil, db = np.nan, np.nan

    return {'space': space_name, 'ARI': ari, 'NMI': nmi, 'AMI': ami,
            'Silhouette': sil, 'DaviesBouldin': db}

def compute_cluster_uncertainty(logits_stack):
    """Compute assignment entropy and aleatoric/epistemic decomposition."""
    N, K, P = logits_stack.shape
    p_mean = logits_stack.mean(axis=2)                     # (N, K)
    H_assign = entropy_bits(p_mean)
    H_aleat = np.stack([entropy_bits(logits_stack[:, :, p]) for p in range(P)], axis=1).mean(axis=1)
    MI = np.clip(H_assign - H_aleat, 0, None)
    return H_assign, H_aleat, MI, p_mean

def print_uncertainty_report(logits_stack, yp, y_true, n_passes, sample_ids=None):
    H_assign, H_aleat, MI, p_mean = compute_cluster_uncertainty(logits_stack)
    label_map = {0: "CN", 1: "MCI"}
    if sample_ids is None:
        sample_ids = list(range(len(y_true)))

    df = pd.DataFrame({
        "sample_id": sample_ids,
        "true_label": [label_map[int(l)] for l in y_true],
        "cluster": [label_map[int(l)] for l in yp],
        "correct": (yp == y_true).astype(int),
        "p_cluster0": p_mean[:, 0],
        "p_cluster1": p_mean[:, 1],
        "entropy_assign": H_assign,
        "entropy_aleat": H_aleat,
        "model_unc": MI,
    })

    sep = "─" * 72
    print(f"\n{sep}\n  CLUSTER ASSIGNMENT UNCERTAINTY (SIGNA)\n{sep}")
    print(f"  ({n_passes} MC passes with edge dropout)\n")
    print(f"  {'Metric':<30} {'Mean':>9} {'Std':>9} {'Min':>9} {'Max':>9}")
    print("  " + "─" * 65)
    for col, label in [
        ("entropy_assign", "Assignment entropy H[p̄] (bits)"),
        ("entropy_aleat",  "Aleatoric entropy E[H] (bits)"),
        ("model_unc",      "Model uncertainty MI (bits)"),
        ("p_cluster1",     "Soft assignment p(cluster=1)"),
    ]:
        vals = df[col].values
        print(f"  {label:<30}  {vals.mean():>9.4f}  {vals.std():>9.4f}  "
              f"{vals.min():>9.4f}  {vals.max():>9.4f}")

    df_sorted = df.sort_values("entropy_assign", ascending=False).reset_index(drop=True)
    print("\n  Top‑10 most uncertain assignments:")
    cols = ["sample_id", "true_label", "cluster", "p_cluster0", "p_cluster1", "entropy_assign", "model_unc"]
    print(df_sorted.head(10)[cols].to_string(index=True))

    low = (H_assign < 0.3).sum()
    high = (H_assign > 0.7).sum()
    print(f"\n  Confidence summary:")
    print(f"    Low entropy (<0.3 bits, high confidence): {low}/{len(y_true)} ({100*low/len(y_true):.1f}%)")
    print(f"    High entropy (>0.7 bits, ambiguous):     {high}/{len(y_true)} ({100*high/len(y_true):.1f}%)")
    print(f"\n  Mean assignment entropy: {H_assign.mean():.4f} ± {H_assign.std():.4f} bits")
    print(f"  (Max possible for K=2: {np.log2(2):.4f} bits)")
    print(sep)
    return df

# ═══════════════════════════════════════════════════════════════════════════
# MC EVALUATION (cluster assignments + uncertainty)
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_model_mc(model, edge_index, y, n_passes=30, seed_base=9999):
    """Return hard cluster assignments (K‑means), soft probs, and MC logits stack."""
    model.eval()
    all_projections = []   # (N, proj_dim) per pass

    with torch.no_grad():
        for i in range(n_passes):
            # Apply edge dropout for each MC pass (stochastic neighbor masking during inference)
            mask = torch.bernoulli(torch.full((edge_index.shape[1],), 0.6, device=edge_index.device)).bool()
            ei_masked = edge_index[:, mask]
            z = model(data.x, ei_masked)
            h = model.project(z).cpu().numpy()
            all_projections.append(h)

    proj_stack = np.stack(all_projections, axis=2)   # (N, proj_dim, P)
    proj_mean = proj_stack.mean(axis=2)             # (N, proj_dim)

    # K‑means on mean projections
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=0, n_init=20)
    yp = kmeans.fit_predict(proj_mean)
    # Align to majority diagnosis
    if accuracy_score(y, yp) < accuracy_score(y, 1 - yp):
        yp = 1 - yp

    # Soft assignments using distance to cluster centres (temperature τ)
    centers = kmeans.cluster_centers_
    dists = np.linalg.norm(proj_mean[:, None, :] - centers[None, :, :], axis=2)
    tau = max(0.05 * np.std(dists), 1e-8)
    logits_soft = -(dists ** 2) / tau
    soft_assign = softmax_np(logits_soft)

    # Build MC soft assignment stack for uncertainty
    mc_soft = []
    for p in range(n_passes):
        proj_p = proj_stack[:, :, p]
        dists_p = np.linalg.norm(proj_p[:, None, :] - centers[None, :, :], axis=2)
        soft_p = softmax_np(-(dists_p ** 2) / tau)
        mc_soft.append(soft_p)
    soft_stack = np.stack(mc_soft, axis=2)   # (N, K, P)

    return yp, soft_assign, soft_stack, proj_mean

# ═══════════════════════════════════════════════════════════════════════════
# SINGLE RUN (training + evaluation)
# ═══════════════════════════════════════════════════════════════════════════
def run_once(seed_offset=0, verbose=False, return_probs=False, n_eval_passes=30):
    torch.manual_seed(SEED_BASE + seed_offset)
    np.random.seed(SEED_BASE + seed_offset)
    random.seed(SEED_BASE + seed_offset)

    # Build adjacency with self‑loops (required for SIGNA loss)
    adj_with_self = adj_dense + torch.eye(adj_dense.shape[0], device=DEVICE)
    adj_with_self = (adj_with_self > 0).float()

    encoder = GCNEncoder(FEATS_DIM, HIDDEN_DIM, num_layers=NUM_LAYERS,
                         dropout_prob=EMBED_DROPOUT_PROB,
                         activation=ACTIVATION, use_ln=USE_LAYER_NORM)
    model = SIGNAModel(encoder, HIDDEN_DIM, PROJ_DIM).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    for epoch in range(NUM_EPOCHS):
        model.train()
        optimizer.zero_grad()
        z = model(data.x, data.edge_index)
        loss = model.loss(z, adj_with_self)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        if verbose and epoch % 200 == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:4d} | Loss: {loss.item():.6f} | LR: {lr:.2e}")

    # Evaluation (clustering)
    yp, soft_assign, soft_stack, proj_mean = evaluate_model_mc(
        model, data.edge_index, y, n_passes=n_eval_passes
    )
    acc = accuracy_score(y, yp)
    prec_w = precision_score(y, yp, average='weighted', zero_division=0)
    rec_w = recall_score(y, yp, average='weighted', zero_division=0)
    f1_w = f1_score(y, yp, average='weighted', zero_division=0)
    weighted_metrics = (acc, prec_w, rec_w, f1_w)

    if return_probs:
        return weighted_metrics, yp, soft_assign, soft_stack, proj_mean, model
    return weighted_metrics, model

# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════
SEP = "═" * 72
print(SEP)
print("  SIGNA (strict official implementation) – CN vs MCI")
print("  Single‑view graph contrastive learning with normalized JSD")
print("  Labels used ONLY for post‑hoc external validation.\n")

print("--- Training SIGNA (seed 0) ---")
metrics0, yp0, soft0, stack0, proj0, model0 = run_once(
    seed_offset=0, verbose=True, return_probs=True, n_eval_passes=N_EVAL_PASSES
)
print(f"\nPost‑hoc external validation: Acc={metrics0[0]:.4f}, Prec_w={metrics0[1]:.4f}, "
      f"Rec_w={metrics0[2]:.4f}, F1_w={metrics0[3]:.4f}")

# Uncertainty report
df_unc = print_uncertainty_report(stack0, yp0, y, N_EVAL_PASSES, sample_ids=list(range(len(y))))
df_unc.to_csv("signa_cluster_uncertainty.csv", index=False, float_format="%.6f")
print("\n  ✓ Saved uncertainty to signa_cluster_uncertainty.csv")

# Clustering metrics (single run)
res = evaluate_clustering(proj0, yp0, y, space_name="SIGNA embeddings (seed 0)")
print(f"\nClustering (seed 0): ARI={res['ARI']:.4f}, NMI={res['NMI']:.4f}, "
      f"Silhouette={res['Silhouette']:.4f}, DB={res['DaviesBouldin']:.4f}")

# 10‑seed evaluation
print(f"\n{SEP}")
print("  10‑SEED EVALUATION")
print(SEP)
print(f"  {'Seed':>4}  {'Acc':>7}  {'Prec_w':>8}  {'Rec_w':>8}  {'F1_w':>8}")
print("  " + "─" * 45)

all_metrics = []
seed_records = []   # for clustering metrics

for seed in range(10):
    metrics_w, _ = run_once(seed_offset=seed, return_probs=False)
    all_metrics.append(metrics_w)

    # Also get clustering result for this seed
    _, yp_s, _, _, proj_s, _ = run_once(seed_offset=seed, return_probs=True)
    rec = evaluate_clustering(proj_s, yp_s, y, space_name="")
    seed_records.append(rec)

    print(f"  {seed:>4}  {metrics_w[0]:>7.4f}  {metrics_w[1]:>8.4f}  {metrics_w[2]:>8.4f}  {metrics_w[3]:>8.4f}")

# Aggregate
acc_vals = np.array([m[0] for m in all_metrics])
prec_vals = np.array([m[1] for m in all_metrics])
rec_vals = np.array([m[2] for m in all_metrics])
f1_vals = np.array([m[3] for m in all_metrics])

nmi_vals = np.array([r['NMI'] for r in seed_records])
ari_vals = np.array([r['ARI'] for r in seed_records])
ami_vals = np.array([r['AMI'] for r in seed_records])
sil_vals = np.array([r['Silhouette'] for r in seed_records])
db_vals = np.array([r['DaviesBouldin'] for r in seed_records])

# Horizontal table for manuscript
print(f"\n{SEP}")
print("  HORIZONTAL TABLE FOR MANUSCRIPT (mean ± std over 10 runs)")
print("  Metrics: Accuracy, Weighted Precision, Weighted Recall, Weighted F1, NMI, ARI, AMI, Silhouette, Davies‑Bouldin")
print(SEP)

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
row = "SIGNA"
for m, s in zip(means, stds):
    if np.isnan(m):
        row += "\tN/A±N/A"
    else:
        row += f"\t{m:.4f}±{s:.4f}"
print(row)

# LaTeX version
print("\n" + "-" * 72)
print("  LaTeX code for the horizontal table:")
print("-" * 72)
latex_row = "SIGNA"
for m, s in zip(means, stds):
    if np.isnan(m):
        latex_row += " & N/A\\pm N/A"
    else:
        latex_row += f" & ${m:.4f}\\pm{s:.4f}$"
latex_row += " \\\\"
print(latex_row)

# Summary table
print(f"\n{SEP}")
print("  10‑RUN SUMMARY (mean ± std)                         [SIGNA, CN vs MCI]")
print(SEP)
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
]:
    print(f"  {name:<20}  {vals.mean():>9.4f}  {vals.std():>9.4f}  "
          f"{vals.min():>9.4f}  {vals.max():>9.4f}")

print(f"\n{SEP}")
print("  ANALYSIS COMPLETE")
print(SEP)
