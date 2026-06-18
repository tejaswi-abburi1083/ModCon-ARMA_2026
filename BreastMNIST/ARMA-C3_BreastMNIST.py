import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import Data
from torch_geometric.nn import ARMAConv
from torch.utils.data import TensorDataset, DataLoader, Subset
from torchvision import transforms
import timm
import random
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, confusion_matrix, normalized_mutual_info_score,
    adjusted_rand_score, adjusted_mutual_info_score,
    silhouette_score, davies_bouldin_score
)
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
from contextlib import contextmanager
import copy
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("CUDA available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

RADIODINO_MODEL = "hf_hub:Snarcy/RadioDino-s16"
K = 2
ACTIV = "ELU"
ALPHA = 0.63                     # graph threshold
CUT = 0                          # use cut loss (0) or modularity (1)
TAU_SIM = 0.2                    # temperature for similarity
BETA = 0.6                       # trade‑off in contrastive loss
EMA_DECAY = 0.5                  # moving average decay
LAMBDA_CON = 0.3                 # weight for contrastive loss
NUM_EPOCHS = 5000
N_EVAL_PASSES = 30               # MC passes for uncertainty
NUM_RUNS = 10                    # number of independent runs

# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADING – BREASTMNIST + RADIODINO FEATURES
# ═══════════════════════════════════════════════════════════════════════════
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

# Subsample: up to 1000 per class (same as other baselines)
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

X = torch.cat(rd_feats, dim=0).numpy().astype(np.float32)
y = np.array(y_list, dtype=np.int64)

# Fixed permutation (seed 42) for reproducibility across runs
np.random.seed(42)
perm = np.random.permutation(X.shape[0])
X, y = X[perm], y[perm]

FEATS_DIM = X.shape[1]
print(f"Features: {X.shape}, Labels: {y.shape} (Malignant: {np.sum(y==0)}, Normal: {np.sum(y==1)})")
print("NOTE: Diagnostic labels are held out and used ONLY for post-hoc external validation.\n")

# ═══════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS (graph, augmentations, hungarian mapping)
# ═══════════════════════════════════════════════════════════════════════════
def create_adj(features, cut, alpha=1.0):
    F_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)
    W = np.dot(F_norm, F_norm.T)
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

def aug_random_edge_edge_index(edge_index_np, drop_percent=0.2, seed=None):
    rng = np.random.default_rng(seed)
    keep = rng.random(edge_index_np.shape[1]) >= drop_percent
    return edge_index_np[:, keep]

def load_data_from_edge_index(node_feats_np, edge_index_np, device):
    node_feats = torch.from_numpy(node_feats_np).float()
    edge_index = torch.from_numpy(edge_index_np.astype(np.int64)).long()
    return node_feats.to(device), edge_index.to(device)

def to_data(node_feats_np, edge_index_np, device):
    x, ei = load_data_from_edge_index(node_feats_np, edge_index_np, device)
    return Data(x=x, edge_index=ei)

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

# ═══════════════════════════════════════════════════════════════════════════
#  MODEL DEFINITIONS (original ARMA with contrastive loss)
# ═══════════════════════════════════════════════════════════════════════════
def sim(h1, h2, tau=TAU_SIM):
    z1 = F.normalize(h1, dim=-1, p=2)
    z2 = F.normalize(h2, dim=-1, p=2)
    return torch.mm(z1, z2.t()) / tau

def contrastive_loss_wo_cross_network(h1, h2, z):
    f = lambda x: torch.exp(x)
    intra_sim = f(sim(h1, h1))
    inter_sim = f(sim(h1, h2))
    return -torch.log(inter_sim.diag() / (intra_sim.sum(dim=-1) + inter_sim.sum(dim=-1) - intra_sim.diag()))

def contrastive_loss_wo_cross_view(h1, h2, z):
    f = lambda x: torch.exp(x)
    cross_sim = f(sim(h1, z))
    return -torch.log(cross_sim.diag() / cross_sim.sum(dim=-1))

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
    def __init__(self, input_dim, hidden_dim, device, activ="ELU", num_stacks=1, num_layers=1):
        super(ARMAEncoder, self).__init__()
        self.device = device
        activations = {
            "SELU": F.selu, "SiLU": F.silu, "GELU": F.gelu,
            "ELU": F.elu, "RELU": F.relu
        }
        self.act = activations.get(activ, F.elu)
        self.arma = ARMAConv(
            in_channels=input_dim, out_channels=hidden_dim,
            num_stacks=num_stacks, num_layers=num_layers,
            act=self.act, shared_weights=True, dropout=0.25
        )
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

class EMA:
    def __init__(self, beta):
        self.beta = beta
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

class ARMA(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_clusters, device, activ, moving_average_decay=0.5, cut=True):
        super(ARMA, self).__init__()
        self.device = device
        self.num_clusters = num_clusters
        self.cut = cut
        self.beta = 0.6

        self.online_encoder = ARMAEncoder(input_dim, hidden_dim, device, activ)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.online_predictor = MLP(hidden_dim, num_clusters, hidden_dim)

        self.target_ema_updater = EMA(moving_average_decay)
        self.loss = self.cut_loss if cut else self.modularity_loss

    def update_ma(self):
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(self, data1, data2):
        x1 = self.online_encoder(data1)
        logits1 = self.online_predictor(x1)
        x2 = self.online_encoder(data2)
        logits2 = self.online_predictor(x2)

        with torch.no_grad():
            target_proj_one = self.target_encoder(data1).detach()
            target_proj_two = self.target_encoder(data2).detach()

        l1 = self.beta * contrastive_loss_wo_cross_network(x1, x2, target_proj_two) + \
             (1.0 - self.beta) * contrastive_loss_wo_cross_view(x1, x2, target_proj_two)

        l2 = self.beta * contrastive_loss_wo_cross_network(x2, x1, target_proj_one) + \
             (1.0 - self.beta) * contrastive_loss_wo_cross_view(x2, x1, target_proj_one)

        return logits1, logits2, l1, l2

    def modularity_loss(self, A, S):
        C = F.softmax(S, dim=1)
        d = A.sum(dim=1)
        m = A.sum()
        B = A - torch.ger(d, d) / (2 * m)
        k = torch.tensor(self.num_clusters, device=self.device, dtype=torch.float32)
        n = S.shape[0]
        modularity_term = (-1 / (2 * m)) * torch.trace(C.T @ B @ C)
        collapse_reg_term = (torch.sqrt(k) / n) * torch.norm(C.sum(dim=0), p='fro') - 1
        return modularity_term + collapse_reg_term

    def cut_loss(self, A, S):
        S = F.softmax(S, dim=1)
        A_pool = (A @ S).T @ S
        num = torch.trace(A_pool)
        D = torch.diag(A.sum(dim=1))
        D_pooled = (D @ S).T @ S
        den = torch.trace(D_pooled)
        mincut_loss = -(num / den)
        St_S = S.T @ S
        I = torch.eye(self.num_clusters, device=self.device)
        ortho_loss = torch.norm(St_S / torch.norm(St_S) - I / torch.norm(I))
        return mincut_loss + ortho_loss

# ═══════════════════════════════════════════════════════════════════════════
#  MC DROPOUT CONTEXT MANAGER (keeps BatchNorm frozen, only dropout stochastic)
# ═══════════════════════════════════════════════════════════════════════════
@contextmanager
def mc_dropout_mode(model):
    """Set model to eval mode, then set all Dropout layers to train mode."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    try:
        yield model
    finally:
        model.eval()

# ═══════════════════════════════════════════════════════════════════════════
#  TRAINING FUNCTION (one run)
# ═══════════════════════════════════════════════════════════════════════════
def train_once(seed, verbose=True):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Build base graph (full)
    W0 = create_adj(X, CUT, ALPHA)
    edge_index_np, _ = edge_index_from_dense(W0)
    A1 = torch.from_numpy(W0).float().to(DEVICE)

    model = ARMA(FEATS_DIM, 256, K, DEVICE, ACTIV, moving_average_decay=EMA_DECAY, cut=CUT).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=200, gamma=0.5)

    for epoch in range(NUM_EPOCHS):
        # Augmentations
        rng = np.random.default_rng(epoch + seed)

        # View 1: feature masking + edge drop
        mask = rng.random(X.shape) >= 0.2
        features_aug1 = (X * mask.astype(np.float32))
        edge_idx1 = aug_random_edge_edge_index(edge_index_np, drop_percent=0.2, seed=epoch + seed)

        # View 2: feature cell dropout + edge drop
        features_aug2 = X.copy()
        n, d = features_aug2.shape
        drop_cells = int(n * d * 0.2)
        flat_idx = rng.choice(n * d, size=drop_cells, replace=False)
        rows = flat_idx // d
        cols = flat_idx % d
        features_aug2[rows, cols] = 0.0
        edge_idx2 = aug_random_edge_edge_index(edge_index_np, drop_percent=0.2, seed=epoch + seed + 999)

        data1 = to_data(features_aug1, edge_idx1, DEVICE)
        data2 = to_data(features_aug2, edge_idx2, DEVICE)

        model.train()
        optimizer.zero_grad()
        logits1, logits2, l1, l2 = model(data1, data2)
        unsup_loss = model.loss(A1, logits1)
        cont_loss = ((l1 + l2) / 2).mean()
        total_loss = unsup_loss + LAMBDA_CON * cont_loss
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        model.update_ma()

        if verbose and epoch % 500 == 0:
            print(f"  Epoch {epoch:4d} | Total: {total_loss.item():.4f} | "
                  f"Unsup: {unsup_loss.item():.4f} | Cont: {cont_loss.item():.4f}")

    full_data = to_data(X, edge_index_np, DEVICE)
    return model, full_data, edge_index_np

# ═══════════════════════════════════════════════════════════════════════════
#  MC EVALUATION WITH DROPOUT-ONLY STOCHASTICITY + LABEL ALIGNMENT
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_mc(model, full_data, edge_index_np, n_passes=30, seed_base=42):
    all_logits = []
    ref_labels = None

    with mc_dropout_mode(model):
        with torch.no_grad():
            for i in range(n_passes):
                rng = np.random.default_rng(seed_base + i)

                # feature masking (same as training)
                mask = rng.random(X.shape) >= 0.2
                feats_mc = X * mask.astype(np.float32)

                # edge dropping
                ei_mc = aug_random_edge_edge_index(edge_index_np, drop_percent=0.2, seed=seed_base + i)

                data_mc = to_data(feats_mc, ei_mc, DEVICE)

                # forward pass (dropout active)
                logits = model.online_predictor(model.online_encoder(data_mc)).cpu().numpy()  # (N, K)

                # Get hard labels for this pass
                labels_mc = np.argmax(logits, axis=1)

                if i == 0:
                    ref_labels = labels_mc.copy()
                else:
                    labels_aligned = hungarian_map(ref_labels, labels_mc)
                    if (labels_aligned != labels_mc).any():
                        logits = logits[:, ::-1].copy()
                all_logits.append(logits)

    logits_stack = np.stack(all_logits, axis=2)        # (N, K, P)
    logits_mean = logits_stack.mean(axis=2)            # (N, K)
    yp = np.argmax(logits_mean, axis=1)

    # Align cluster labels to majority diagnosis (post-hoc)
    acc = accuracy_score(y, yp)
    acc_inv = accuracy_score(y, 1 - yp)
    if acc_inv > acc:
        yp = 1 - yp
        logits_mean = logits_mean[:, ::-1].copy()
        logits_stack = logits_stack[:, ::-1, :].copy()

    ypp = F.softmax(torch.from_numpy(logits_mean), dim=1).numpy()
    return yp, ypp, logits_stack, logits_mean

# ═══════════════════════════════════════════════════════════════════════════
#  UNCERTAINTY & METRICS FUNCTIONS (UPDATED TO USE WEIGHTED METRICS)
# ═══════════════════════════════════════════════════════════════════════════
def entropy_bits(p):
    p = np.clip(p, 1e-12, 1.0)
    return -np.sum(p * np.log2(p), axis=1)

def compute_cluster_assignment_uncertainty(logits_stack):
    N, K, P = logits_stack.shape
    probs = np.stack([F.softmax(torch.from_numpy(logits_stack[:, :, p]), dim=1).numpy()
                      for p in range(P)], axis=2)
    p_mean = probs.mean(axis=2)
    H_assign = entropy_bits(p_mean)
    H_aleat = np.stack([entropy_bits(probs[:, :, p]) for p in range(P)], axis=1).mean(axis=1)
    MI = np.clip(H_assign - H_aleat, 0, None)
    return H_assign, H_aleat, MI, p_mean

def compute_all_metrics(yp, ypp, logits_stack, X, y):
    # Weighted classification metrics (average over both classes)
    acc = accuracy_score(y, yp)
    prec_weighted = precision_score(y, yp, average='weighted', zero_division=0)
    rec_weighted  = recall_score(y, yp, average='weighted', zero_division=0)
    f1_weighted   = f1_score(y, yp, average='weighted', zero_division=0)
    ll = log_loss(y, ypp[:, 1])

    # Clustering external validation
    nmi = normalized_mutual_info_score(y, yp, average_method='arithmetic')
    ari = adjusted_rand_score(y, yp)
    ami = adjusted_mutual_info_score(y, yp, average_method='arithmetic')

    # Geometric metrics on original features (standardized)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    unique_preds = np.unique(yp)
    can_geom = (len(unique_preds) >= 2) and all((yp == c).sum() >= 2 for c in unique_preds)
    if can_geom:
        sil = silhouette_score(X_scaled, yp, metric='euclidean')
        db = davies_bouldin_score(X_scaled, yp)
    else:
        sil, db = np.nan, np.nan

    # Uncertainty metrics
    H_assign, H_aleat, MI, _ = compute_cluster_assignment_uncertainty(logits_stack)
    mean_ent = H_assign.mean()
    std_ent = H_assign.std()

    return {
        'acc': acc,
        'prec_weighted': prec_weighted,
        'rec_weighted': rec_weighted,
        'f1_weighted': f1_weighted,
        'nmi': nmi, 'ari': ari, 'ami': ami,
        'silhouette': sil, 'db': db,
        'log_loss': ll, 'mean_entropy': mean_ent, 'std_entropy': std_ent,
        'H_assign': H_assign, 'yp': yp
    }

# ═══════════════════════════════════════════════════════════════════════════
#  RUN MULTIPLE SEEDS AND COLLECT RESULTS
# ═══════════════════════════════════════════════════════════════════════════
all_results = []
last_yp = None
last_H_assign = None
last_logits_stack = None

print("═" * 72)
print("  ARMA (original design with contrastive loss) – BreastMNIST (RadioDINO)")
print(f"  Graph threshold α = {ALPHA}")
print(f"  {NUM_RUNS} runs × {N_EVAL_PASSES} MC passes (Dropout only, BatchNorm frozen)")
print("  NOTE: Diagnostic labels used ONLY for post-hoc external validation.")
print("═" * 72)
print(f"\n  {'Run':>4}  {'Acc':>7}  {'Prec_w':>8}  {'Rec_w':>8}  {'F1_w':>8}  "
      f"{'NMI':>7}  {'MeanEntropy':>13}  {'LogLoss':>9}")
print("  " + "─" * 95)

for run in range(NUM_RUNS):
    print(f"  {run:>4}  ...", end="", flush=True)
    model, full_data, edge_index_np = train_once(seed=42 + run, verbose=False)
    yp, ypp, logits_stack, logits_mean = evaluate_mc(
        model, full_data, edge_index_np,
        n_passes=N_EVAL_PASSES, seed_base=9999 + run
    )
    metrics = compute_all_metrics(yp, ypp, logits_stack, X, y)

    all_results.append(metrics)
    print(f" {metrics['acc']:7.4f}  {metrics['prec_weighted']:8.4f}  {metrics['rec_weighted']:8.4f}  "
          f"{metrics['f1_weighted']:8.4f}  {metrics['nmi']:7.4f}  {metrics['mean_entropy']:13.4f}  "
          f"{metrics['log_loss']:9.6f}")

    if run == NUM_RUNS - 1:
        last_yp = metrics['yp']
        last_H_assign = metrics['H_assign']
        last_logits_stack = logits_stack

# =============================================================================
#  HORIZONTAL TABLE FOR MANUSCRIPT (mean ± std over 10 runs)
# =============================================================================
print("\n" + "=" * 72)
print("  HORIZONTAL TABLE FOR MANUSCRIPT (mean ± std over 10 runs)")
print("  Metrics: Accuracy, Weighted Precision, Weighted Recall, Weighted F1, NMI, ARI, AMI, Silhouette, Davies‑Bouldin")
print("=" * 72)

# Extract weighted classification metrics
acc_vals = np.array([r['acc'] for r in all_results])
prec_vals = np.array([r['prec_weighted'] for r in all_results])
rec_vals = np.array([r['rec_weighted'] for r in all_results])
f1_vals = np.array([r['f1_weighted'] for r in all_results])
nmi_vals = np.array([r['nmi'] for r in all_results])
ari_vals = np.array([r['ari'] for r in all_results])
ami_vals = np.array([r['ami'] for r in all_results])
sil_vals = np.array([r['silhouette'] for r in all_results])
db_vals = np.array([r['db'] for r in all_results])

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

# =============================================================================
#  SUMMARY TABLES (updated to use weighted metrics)
# =============================================================================
results_np = np.array([[r['acc'], r['prec_weighted'], r['rec_weighted'], r['f1_weighted'],
                        r['nmi'], r['ari'], r['ami'],
                        r['mean_entropy'], r['std_entropy'],
                        r['log_loss'], r['silhouette'], r['db']] for r in all_results])

print("\n" + "═" * 72)
print("  10-RUN SUMMARY (mean ± std)                         [ARMA (original), BreastMNIST]")
print("  NOTE: Diagnostic labels used only for post-hoc external validation.")
print("═" * 72)

print("\n  A.  EXTERNAL VALIDATION & ASSIGNMENT UNCERTAINTY (weighted)")
print(f"  {'Metric':<20}  {'Mean':>9}  {'Std':>9}  {'Min':>9}  {'Max':>9}")
print("  " + "─" * 60)
weighted_metrics = ['acc', 'prec_weighted', 'rec_weighted', 'f1_weighted', 'nmi', 'ari', 'ami',
                    'mean_entropy', 'std_entropy', 'log_loss']
metric_names = ['Accuracy', 'Weighted Precision', 'Weighted Recall', 'Weighted F1',
                'NMI', 'ARI', 'AMI', 'Mean Entropy', 'Std Entropy', 'Log Loss']
for idx, name in enumerate(weighted_metrics):
    vals = results_np[:, idx]
    print(f"  {metric_names[idx]:<20}  {vals.mean():>9.4f}  {vals.std():>9.4f}  "
          f"{vals.min():>9.4f}  {vals.max():>9.4f}")

print("\n  B.  GEOMETRIC CLUSTERING METRICS (on original feature space)")
print(f"  {'Metric':<20}  {'Mean':>9}  {'Std':>9}  {'Min':>9}  {'Max':>9}  {'Valid runs':>10}")
print("  " + "─" * 70)
geo_metrics = [('silhouette', 'Silhouette'), ('db', 'Davies‑Bouldin')]
for col, name in zip([10, 11], geo_metrics):
    vals = results_np[:, col]
    valid = vals[~np.isnan(vals)]
    if len(valid) > 0:
        print(f"  {name[1]:<20}  {valid.mean():>9.4f}  {valid.std():>9.4f}  "
              f"{valid.min():>9.4f}  {valid.max():>9.4f}  {len(valid):>10}")
    else:
        print(f"  {name[1]:<20}  {'N/A':>9}  {'N/A':>9}  {'N/A':>9}  {'N/A':>9}  {0:>10}")

print("\n  C.  CLUSTER STABILITY ACROSS RUNS")
print(f"  {'Metric':<22}  {'Mean':>9}  {'Std':>9}  Note")
print("  " + "─" * 55)
print(f"  {'Weighted F1':<22}  {f1_vals.mean():>9.4f}  {f1_vals.std():>9.4f}  "
      "Consistency of weighted F1")
print(f"  {'Mean Entropy':<22}  {np.mean(results_np[:,7]):>9.4f}  {np.std(results_np[:,7]):>9.4f}  "
      "Low std = stable assignment confidence")
print("  " + "─" * 55)

# ═══════════════════════════════════════════════════════════════════════════
#  DETAILED UNCERTAINTY REPORT (last run)
# ═══════════════════════════════════════════════════════════════════════════
if last_logits_stack is not None:
    sep = "─" * 72
    H_assign, H_aleat, MI, p_mean = compute_cluster_assignment_uncertainty(last_logits_stack)
    label_map = {0: "Malignant", 1: "Normal"}
    df = pd.DataFrame({
        "sample_id": range(len(y)),
        "true_label": [label_map[int(l)] for l in y],
        "cluster_assignment": [label_map[int(l)] for l in last_yp],
        "correct_ext_valid": (last_yp == y).astype(int),
        "p_cluster0": p_mean[:, 0],
        "p_cluster1": p_mean[:, 1],
        "entropy_assignment": H_assign,
        "entropy_aleatoric": H_aleat,
        "model_uncertainty": MI,
    })
    print(f"\n{sep}\n  DETAILED UNCERTAINTY REPORT (last run)\n{sep}")
    print(f"  (Based on {N_EVAL_PASSES} MC passes, Dropout only + label alignment)\n")
    print(f"  {'Metric':<30} {'Mean':>9} {'Std':>9} {'Min':>9} {'Max':>9}")
    print(f"  {'─'*65}")
    for col, label in [
        ("entropy_assignment", "Assignment entropy H[p̄] (bits)"),
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

    df.to_csv("arma_original_breastmnist_uncertainty_last_run.csv", index=False, float_format="%.6f")
    print(f"\n  ✓ Saved per-subject uncertainty to arma_original_breastmnist_uncertainty_last_run.csv")

# ═══════════════════════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  ARMA (original design) — Final Summary ({NUM_RUNS} runs, weighted metrics)")
print(f"{'='*60}")
print(f"  ACC (unweighted) : {acc_vals.mean():.4f} ± {acc_vals.std():.4f}")
print(f"  Weighted PREC    : {prec_vals.mean():.4f} ± {prec_vals.std():.4f}")
print(f"  Weighted REC     : {rec_vals.mean():.4f} ± {rec_vals.std():.4f}")
print(f"  Weighted F1      : {f1_vals.mean():.4f} ± {f1_vals.std():.4f}")
print(f"  NMI              : {nmi_vals.mean():.4f} ± {nmi_vals.std():.4f}")
print(f"  ARI              : {ari_vals.mean():.4f} ± {ari_vals.std():.4f}")
print(f"  Mean Assignment Entropy: {np.mean(results_np[:,7]):.4f} ± {np.std(results_np[:,7]):.4f} bits")
print(f"  Log Loss         : {np.mean(results_np[:,9]):.4f} ± {np.std(results_np[:,9]):.4f}")
print(f"\n  Uncertainty: MC with Dropout only (BatchNorm frozen) + label alignment ({N_EVAL_PASSES} passes)")
print(f"{'='*60}\n  ANALYSIS COMPLETE\n{'='*60}")

# Export all results to CSV
df_all = pd.DataFrame(all_results)
df_all.to_csv("arma_original_breastmnist_10run_weighted_summary.csv", index=False, float_format="%.6f")
print(f"\n  ✓ Saved 10‑run weighted summary → arma_original_breastmnist_10run_weighted_summary.csv")
