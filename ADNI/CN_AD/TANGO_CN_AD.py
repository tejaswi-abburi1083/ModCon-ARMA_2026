import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import lil_matrix
from multiprocessing import Pool, cpu_count
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    normalized_mutual_info_score, adjusted_rand_score,
    adjusted_mutual_info_score, silhouette_score,
    davies_bouldin_score
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def hungarian_map(y_true, y_pred):
    """Map y_pred labels to best match y_true using Hungarian algorithm."""
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

def compute_similarity_for_points(points, data, neighbors, max_dis):
    n = data.shape[0]
    local_similarity_matrix = lil_matrix((n, n), dtype=np.float32)
    neighbor_sets = {i: set(neighbors[i]) for i in points}
    for i in points:
        i_neighbors = neighbor_sets[i]
        point_i = data[i]
        for j in neighbors[i]:
            if j in neighbor_sets:
                j_neighbors = neighbor_sets[j]
            else:
                j_neighbors = set(neighbors[j])
            shared_neighbors = i_neighbors & j_neighbors
            if shared_neighbors:
                shared_idx = list(shared_neighbors)
                shared_points = data[shared_idx]
                point_j = data[j]
                d_i = np.linalg.norm(shared_points - point_i[np.newaxis, :], axis=1) / (max_dis + 1e-12)
                d_j = np.linalg.norm(shared_points - point_j[np.newaxis, :], axis=1) / (max_dis + 1e-12)
                d = 0.5 * (d_i + d_j)
                similarity = np.sum(np.exp(-d * d))
                if similarity > 0:
                    local_similarity_matrix[i, j] = similarity
    return local_similarity_matrix

def compute_similarity(data, k):
    n = data.shape[0]
    nn_model = NearestNeighbors(n_neighbors=k, algorithm='auto')
    nn_model.fit(data)
    distances, neighbors = nn_model.kneighbors(data)
    max_dis = np.max(distances) if distances.size else 1.0
    num_processes = max(1, cpu_count() - 1)
    points_split = np.array_split(range(n), num_processes)
    args = [(points, data, neighbors, max_dis) for points in points_split]
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(compute_similarity_for_points, args)
    similarity_matrix = results[0]
    for mat in results[1:]:
        similarity_matrix = similarity_matrix + mat
    similarity_matrix = similarity_matrix.maximum(similarity_matrix.transpose())
    if similarity_matrix.data.size > 0:
        similarity_matrix.data = similarity_matrix.data / similarity_matrix.max()
    similarity_matrix.setdiag(1.0 + 1e-15)
    return similarity_matrix.tocsr()

def compute_threshold_matrix(data, k):
    n = data.shape[0]
    similarity_matrix = compute_similarity(data, k)
    density = np.zeros(n, dtype=np.float32)
    top_k_indices = []
    for i in range(n):
        start, end = similarity_matrix.indptr[i], similarity_matrix.indptr[i+1]
        row_data = similarity_matrix.data[start:end]
        row_idx = similarity_matrix.indices[start:end]
        if row_data.size == 0:
            top_k_indices.append(np.array([], dtype=int))
            continue
        order = np.argsort(-row_data)
        sorted_idx = row_idx[order]
        sorted_data = row_data[order]
        k_eff = min(k, sorted_data.size)
        density[i] = np.sum(sorted_data[:k_eff])
        top_k_indices.append(sorted_idx)
    if density.max() > 0:
        density = density / density.max()
    nearest_neighbor_ranks = np.full(n, -1, dtype=int)
    for i in range(n):
        cur = density[i]
        for rank, nb in enumerate(top_k_indices[i]):
            if density[nb] > cur:
                nearest_neighbor_ranks[i] = rank
                break
    leader_points = np.full(n, -1, dtype=int)
    degree = density.copy()
    max_rank = int(nearest_neighbor_ranks.max()) if nearest_neighbor_ranks.max() >= 0 else 1
    sorted_by_density_indices = np.argsort(density)
    for i in sorted_by_density_indices:
        if nearest_neighbor_ranks[i] != -1:
            neighbor_idx = top_k_indices[i][nearest_neighbor_ranks[i]]
            contribution = degree[i] * np.exp(- (float(nearest_neighbor_ranks[i]) / float(max_rank))**2)
            degree[neighbor_idx] += contribution
    for i in range(n):
        if nearest_neighbor_ranks[i] != -1:
            neighbor_idx = top_k_indices[i][nearest_neighbor_ranks[i]]
            if degree[i] < degree[neighbor_idx]:
                leader_points[i] = neighbor_idx
    core_points = np.where(leader_points == -1)[0]
    core_idx_mapping = np.full(n, -1, dtype=int)
    core_idx_mapping[core_points] = np.arange(core_points.shape[0], dtype=int)
    visited = np.zeros(n, dtype=bool)
    for i in range(n):
        if visited[i]:
            continue
        if leader_points[i] == -1:
            leader_points[i] = i
            visited[i] = True
            continue
        cur = i
        stack = []
        while leader_points[cur] != -1 and leader_points[cur] != cur:
            stack.append(cur)
            visited[cur] = True
            cur = leader_points[cur]
        if leader_points[cur] == -1:
            leader_points[cur] = cur
        visited[cur] = True
        core = cur
        while stack:
            node = stack.pop()
            leader_points[node] = core
    S_coo = similarity_matrix.tocoo()
    rows, cols, vals = S_coo.row, S_coo.col, S_coo.data
    mask = rows < cols
    rows, cols, vals = rows[mask], cols[mask], vals[mask]
    weights = vals * density[rows] * density[cols]
    core_i = leader_points[rows]
    core_j = leader_points[cols]
    inter_mask = core_i != core_j
    core_i = core_i[inter_mask]
    core_j = core_j[inter_mask]
    weights = weights[inter_mask]
    core_i_mapped = core_idx_mapping[core_i]
    core_j_mapped = core_idx_mapping[core_j]
    valid_mask = (core_i_mapped >= 0) & (core_j_mapped >= 0)
    core_i_mapped = core_i_mapped[valid_mask].astype(int)
    core_j_mapped = core_j_mapped[valid_mask].astype(int)
    weights = weights[valid_mask]
    edges = list(zip(weights.tolist(), core_i_mapped.tolist(), core_j_mapped.tolist()))
    edges.sort(reverse=True, key=lambda x: x[0])
    m = core_points.shape[0]
    if m == 0:
        return np.zeros((0, 0), dtype=np.float32), leader_points, core_idx_mapping
    threshold_matrix = np.zeros((m, m), dtype=np.float32)
    core_labels = np.arange(m, dtype=int)
    for sim, i, j in edges:
        if core_labels[i] != core_labels[j]:
            label_i = core_labels[i]
            label_j = core_labels[j]
            comp_i = (core_labels == label_i)
            comp_j = (core_labels == label_j)
            threshold_matrix[np.ix_(comp_i, comp_j)] = sim
            core_labels[comp_i] = label_j
    threshold_matrix = np.maximum(threshold_matrix, threshold_matrix.T)
    np.fill_diagonal(threshold_matrix, 1.0 + 1e-15)
    return threshold_matrix, leader_points, core_idx_mapping

def tango(data, cluster_num, k, run_seed=None):
    threshold_matrix, leader_points, core_idx_mapping = compute_threshold_matrix(data, k)
    if threshold_matrix.size == 0 or threshold_matrix.shape[0] < cluster_num:
        S_full = compute_similarity(data, k).toarray()
        clustering = SpectralClustering(
            n_clusters=cluster_num,
            affinity='precomputed',
            assign_labels='kmeans',
            random_state=run_seed
        )
        labels_full = clustering.fit_predict(S_full)
        return labels_full
    clustering = SpectralClustering(
        n_clusters=cluster_num,
        affinity='precomputed',
        assign_labels='kmeans',
        random_state=run_seed
    )
    core_labels = clustering.fit_predict(threshold_matrix)
    labels_full = core_labels[core_idx_mapping[leader_points]]
    return labels_full

def entropy_bits(p):
    p = np.clip(p, 1e-12, 1.0)
    return -np.sum(p * np.log2(p), axis=1)

def evaluate_tango_mc(X, y, k_neighbors, n_passes=30, seed_base=42):
    """
    Run TANGO multiple times varying only the random seed.
    Each run's labels are aligned to a fixed reference clustering
    using hungarian_map to avoid label-switching artifacts.
    Returns:
        yp : majority-vote hard labels (aligned to diagnosis)
        soft_assignments : (N, K) average one-hot probabilities over passes
        probs_stack : (N, K, P) for uncertainty decomposition (optional)
    """
    N = X.shape[0]

    ref_labels = tango(X, cluster_num=2, k=k_neighbors, run_seed=42)

    all_probs = []
    all_labels = []
    for p in range(n_passes):
        seed = seed_base + p
        labels_mc = tango(X, cluster_num=2, k=k_neighbors, run_seed=seed)

        labels_mc = hungarian_map(ref_labels, labels_mc)
        all_labels.append(labels_mc)
        prob = np.zeros((N, 2), dtype=np.float32)
        prob[np.arange(N), labels_mc] = 1.0
        all_probs.append(prob)


    probs_stack = np.stack(all_probs, axis=2)
    soft_assignments = probs_stack.mean(axis=2)


    labels_stack = np.stack(all_labels, axis=1)
    yp_majority = np.zeros(N, dtype=np.int64)
    for i in range(N):
        vals, counts = np.unique(labels_stack[i], return_counts=True)
        yp_majority[i] = vals[np.argmax(counts)]


    acc = accuracy_score(y, yp_majority)
    acc_inv = accuracy_score(y, 1 - yp_majority)
    if acc_inv > acc:
        yp_majority = 1 - yp_majority
        soft_assignments = soft_assignments[:, ::-1].copy()
        probs_stack = probs_stack[:, ::-1, :].copy()

    return yp_majority, soft_assignments, probs_stack

def print_cluster_uncertainty_report(soft_assignments, yp, y_true, n_passes, sample_ids=None):
    sep = "-" * 72
    H_assign = entropy_bits(soft_assignments)
    label_map = {0: "CN", 1: "Ad"}
    if sample_ids is None:
        sample_ids = list(range(len(y_true)))

    df = pd.DataFrame({
        "sample_id":          sample_ids,
        "true_label":         [label_map[int(l)] for l in y_true],
        "cluster_assignment": [label_map[int(l)] for l in yp],
        "correct_ext_valid":  (yp == y_true).astype(int),
        "p_cluster0":         soft_assignments[:, 0],
        "p_cluster1":         soft_assignments[:, 1],
        "entropy_assignment": H_assign,
    })

    print(f"\n{sep}\n  CLUSTER ASSIGNMENT UNCERTAINTY (TANGO)\n{sep}")
    print(f"  (Entropy of soft cluster assignments - {n_passes} MC passes, seed variation only)\n")
    print(f"  {'Metric':<30} {'Mean':>9} {'Std':>9} {'Min':>9} {'Max':>9}")
    print(f"  {' -'*32}")
    for col, label in [
        ("entropy_assignment", "Assignment entropy H[p](bits)"),
        ("p_cluster1",        "Soft assignment p(cluster=1)"),
    ]:
        vals = df[col].values
        print(f"  {label:<30}  {vals.mean():>9.4f}  {vals.std():>9.4f}  "
              f"{vals.min():>9.4f}  {vals.max():>9.4f}")

    df_sorted = df.sort_values("entropy_assignment", ascending=False).reset_index(drop=True)
    print(f"\n  Top-10 most uncertain cluster assignments:")
    cols_show = ["sample_id", "true_label", "cluster_assignment",
                 "p_cluster0", "p_cluster1", "entropy_assignment"]
    print(df_sorted.head(10)[cols_show].to_string(index=True))

    low_unc  = (H_assign < 0.3).sum()
    high_unc = (H_assign > 0.7).sum()
    print(f"\n  Assignment confidence summary:")
    print(f"    Low entropy  (< 0.3 bits, high-confidence): {low_unc}/{len(y_true)} "
          f"({100*low_unc/len(y_true):.1f}%)")
    print(f"    High entropy (> 0.7 bits, ambiguous):        {high_unc}/{len(y_true)} "
          f"({100*high_unc/len(y_true):.1f}%)")
    print(f"\n  Mean assignment entropy: {H_assign.mean():.4f} +- {H_assign.std():.4f} bits")
    print(f"  (Max possible entropy for K=2: {np.log2(2):.4f} bits)")
    print(f"{sep}")
    return df

def main():

    cn_path = "/home/snu/Downloads/Histogram_CN_FA_20bin_updated.npy"
    Ad_path = "/home/snu/Downloads/Histogram_AD_FA_20bin_updated.npy"

    CN = np.load(cn_path, allow_pickle=True)
    AD = np.load(Ad_path, allow_pickle=True)
    X_cn = np.array(list(CN))
    X_Ad = np.array(list(AD))
    X = np.vstack([X_cn, X_Ad]).astype(np.float32)
    y = np.hstack([np.zeros(X_cn.shape[0], dtype=np.int64),
                   np.ones(X_Ad.shape[0], dtype=np.int64)])


    np.random.seed(42)
    perm = np.random.permutation(X.shape[0])
    X = X[perm]
    y = y[perm]

    print("-" * 72)
    print("  TANGO BASELINE - CN vs Ad (weighted metrics)")
    print("  NOTE: Diagnostic labels used ONLY for post-hoc external validation.")
    print("-" * 72)
    print(f"\nFeatures: {X.shape}, Labels: {y.shape} (CN: {np.sum(y==0)}, Ad: {np.sum(y==1)})")
    print("NOTE: Diagnostic labels are held out and used ONLY for post-hoc external validation.\n")

    k_neighbors = 50
    n_passes = 30
    n_seeds = 10


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

    print("-" * 72)
    print("  RUNNING 10 SEEDS WITH SEED-VARIATION UNCERTAINTY")
    print("-" * 72)
    print(f"  {'Seed':>4}  {'Acc':>7}  {'Prec_w':>8}  {'Rec_w':>8}  {'F1_w':>8}  "
          f"{'NMI':>7}  {'ARI':>7}  {'MeanEntropy':>13}")
    print("  " + "-" * 85)


    last_seed_yp = None
    last_seed_soft = None


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for seed in range(n_seeds):
        yp, soft_assignments, _ = evaluate_tango_mc(
            X, y, k_neighbors, n_passes=n_passes, seed_base=42 + seed * 1000
        )


        acc = accuracy_score(y, yp)
        prec_w = precision_score(y, yp, average='weighted', zero_division=0)
        rec_w = recall_score(y, yp, average='weighted', zero_division=0)
        f1_w = f1_score(y, yp, average='weighted', zero_division=0)
        nmi = normalized_mutual_info_score(y, yp, average_method='arithmetic')
        ari = adjusted_rand_score(y, yp)
        ami = adjusted_mutual_info_score(y, yp, average_method='arithmetic')


        unique_preds = np.unique(yp)
        n_valid = sum((yp == c).sum() >= 2 for c in unique_preds)
        can_geom = (len(unique_preds) >= 2) and (n_valid == len(unique_preds))
        if can_geom:
            sil = silhouette_score(X_scaled, yp, metric='euclidean')
            db = davies_bouldin_score(X_scaled, yp)
        else:
            sil = np.nan
            db = np.nan

        H_assign = entropy_bits(soft_assignments)
        mean_ent = H_assign.mean()
        std_ent = H_assign.std()

        all_acc.append(acc)
        all_prec_weighted.append(prec_w)
        all_recall_weighted.append(rec_w)
        all_f1_weighted.append(f1_w)
        all_nmi.append(nmi)
        all_ari.append(ari)
        all_ami.append(ami)
        all_silhouette.append(sil)
        all_davies_bouldin.append(db)
        all_mean_entropy.append(mean_ent)
        all_std_entropy.append(std_ent)

        print(f"  {seed:>4}  {acc:>7.4f}  {prec_w:>8.4f}  {rec_w:>8.4f}  {f1_w:>8.4f}  "
              f"{nmi:>7.4f}  {ari:>7.4f}  {mean_ent:>13.4f}")

        if seed == n_seeds - 1:
            last_seed_yp = yp
            last_seed_soft = soft_assignments


    print("\n" + "-" * 72)
    print("  HORIZONTAL TABLE FOR MANUSCRIPT (mean +- std over 10 runs)")
    print("  Metrics: Accuracy, Weighted Precision, Weighted Recall, Weighted F1, NMI, ARI, AMI, Silhouette, Davies-Bouldin")
    print("-" * 72)

    metrics_names = [
        "Accuracy", "Prec (weighted)", "Recall (weighted)", "F1 (weighted)",
        "NMI", "ARI", "AMI", "Silhouette", "Davies-Bouldin"
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


    print("\nMethod\t" + "\t".join(metrics_names))
    row = "TANGO"
    for m, s in zip(means, stds):
        if np.isnan(m):
            row += "\tN/A+-N/A"
        else:
            row += f"\t{m:.4f}+-{s:.4f}"
    print(row)


    print("\n" + "-" * 72)
    print("  LaTeX code for the horizontal table (copy the line below):")
    print("-" * 72)
    latex_row = "TANGO"
    for m, s in zip(means, stds):
        if np.isnan(m):
            latex_row += " & N/A+-N/A"
        else:
            latex_row += f" & ${m:.4f}\pm{s:.4f}$"
    latex_row += " \\"
    print(latex_row)


    print("\n" + "-" * 72)
    print("  DETAILED UNCERTAINTY REPORT (last seed, label-aligned)")
    df_unc = print_cluster_uncertainty_report(
        last_seed_soft, last_seed_yp, y, n_passes=n_passes
    )
    df_unc.to_csv("tango_cluster_uncertainty.csv", index=False, float_format="%.6f")
    print("\n  v Saved uncertainty report to tango_cluster_uncertainty.csv")


    print("\n" + "-" * 72)
    print("  10-RUN SUMMARY (mean +- std)                         [TANGO, CN vs Ad]")
    print("  NOTE: Diagnostic labels used only for post-hoc external validation.")
    print("-" * 72)

    print("\n  A.  EXTERNAL VALIDATION METRICS (weighted)")
    print(f"  {'Metric':<20}  {'Mean':>9}  {'Std':>9}  {'Min':>9}  {'Max':>9}")
    print("  " + "-" * 60)
    for name, vals in zip(
        ["Accuracy", "Weighted Precision", "Weighted Recall", "Weighted F1", "NMI", "ARI", "AMI"],
        [all_acc, all_prec_weighted, all_recall_weighted, all_f1_weighted, all_nmi, all_ari, all_ami]
    ):
        print(f"  {name:<20}  {np.mean(vals):>9.4f}  {np.std(vals):>9.4f}  "
              f"{np.min(vals):>9.4f}  {np.max(vals):>9.4f}")

    print("\n  B.  GEOMETRIC CLUSTERING METRICS (on original feature space)")
    print(f"  {'Metric':<20}  {'Mean':>9}  {'Std':>9}  {'Min':>9}  {'Max':>9}  {'Valid runs':>10}")
    print("  " + "-" * 70)
    for name, vals in zip(["Silhouette", "Davies-Bouldin"], [all_silhouette, all_davies_bouldin]):
        valid = [v for v in vals if not np.isnan(v)]
        if len(valid) > 0:
            print(f"  {name:<20}  {np.mean(valid):>9.4f}  {np.std(valid):>9.4f}  "
                  f"{np.min(valid):>9.4f}  {np.max(valid):>9.4f}  {len(valid):>10}")
        else:
            print(f"  {name:<20}  {'N/A':>9}  {'N/A':>9}  {'N/A':>9}  {'N/A':>9}  {0:>10}")

    print("\n  C.  ASSIGNMENT ENTROPY")
    print(f"  {'Metric':<20}  {'Mean':>9}  {'Std':>9}  {'Min':>9}  {'Max':>9}")
    print("  " + "-" * 60)
    print(f"  {'Mean entropy (bits)':<20}  {np.mean(all_mean_entropy):>9.4f}  {np.std(all_mean_entropy):>9.4f}  "
          f"{np.min(all_mean_entropy):>9.4f}  {np.max(all_mean_entropy):>9.4f}")


    df_summary = pd.DataFrame({
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
    df_summary.to_csv("tango_10run_summary.csv", index=False, float_format="%.6f")
    print("\n  v Saved 10-run summary to tango_10run_summary.csv")

    print("\n" + "-" * 72)
    print("  ANALYSIS COMPLETE")
    print("-" * 72)

if __name__ == "__main__":
    main()
