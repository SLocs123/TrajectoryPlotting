import numpy as np
from dtaidistance import dtw_ndim
import hdbscan

image_size = (3840, 2160)
IMG_DIAG = float(np.hypot(*image_size))

def angle_deg(v1, v2, eps=1e-9):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < eps or n2 < eps:
        return None
    cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))

def gating(traj1, traj2,
           max_angle_deg=45.0,
           max_start_dist=0.30 * IMG_DIAG,
           max_end_dist=0.30 * IMG_DIAG):
    """
    Return True if pair is plausible enough to compare; False if we should block it.
    """
    traj1 = np.asarray(traj1, dtype=float)
    traj2 = np.asarray(traj2, dtype=float)

    if len(traj1) < 2 or len(traj2) < 2:
        return False

    # Start checks
    if np.linalg.norm(traj1[0] - traj2[0]) > max_start_dist:
        return False
    a1 = angle_deg(traj1[1] - traj1[0], traj2[1] - traj2[0])
    if a1 is not None and a1 > max_angle_deg:
        return False

    # End checks
    if np.linalg.norm(traj1[-1] - traj2[-1]) > max_end_dist:
        return False
    a2 = angle_deg(traj1[-1] - traj1[-2], traj2[-1] - traj2[-2])
    if a2 is not None and a2 > max_angle_deg:
        return False

    return True


# -------------------------
# Main pipeline
# -------------------------

trajectories = []  # list of arrays shaped (T,2) (or (T,D) for dtw_ndim)

# Basic sanity: keep only trajectories with >=2 points
valid_idx = [i for i, t in enumerate(trajectories) if len(t) >= 2]
traj_valid = [np.asarray(trajectories[i], dtype=float) for i in valid_idx]

if len(traj_valid) < 2:
    # Not enough to cluster
    labels = np.array([-1] * len(traj_valid), dtype=int)
    probs = np.zeros(len(traj_valid), dtype=float)
    clustered_trajs = []  # or [traj_valid] if you want a single group
else:
    # Pairwise DTW distances
    D = dtw_ndim.distance_matrix(traj_valid)
    D = np.asarray(D, dtype=float)
    np.fill_diagonal(D, 0.0)

    # Compute a robust BIG once (off-diagonal only)
    mask = ~np.eye(len(traj_valid), dtype=bool)
    vals = D[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        BIG = 1e6
    else:
        BIG = np.percentile(vals, 99) * 5.0
        if not np.isfinite(BIG) or BIG <= 0:
            BIG = float(vals.max() * 10.0) if vals.max() > 0 else 1e6

    # Apply gating by inflating distances for blocked pairs
    n = len(traj_valid)
    for i in range(n):
        for j in range(i + 1, n):
            if not gating(traj_valid[i], traj_valid[j]):
                D[i, j] = D[j, i] = BIG

    np.fill_diagonal(D, 0.0)
    if not np.isfinite(D).all():
        raise ValueError("Distance matrix contains inf/nan; HDBSCAN won't handle that.")

    # HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        metric="precomputed",
        min_cluster_size=8,
        min_samples=None,
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(D)
    probs = clusterer.probabilities_

    # Build clustered_trajs:
    # list of clusters, each cluster is a list of trajectories (each trajectory is an array of points)
    clusters = {}
    for t, lab in zip(traj_valid, labels):
        if lab == -1:
            continue  # drop noise; keep it separately if you want
        clusters.setdefault(lab, []).append(t)

    # If you want clusters sorted by label:
    clustered_trajs = [clusters[k] for k in sorted(clusters.keys())]

# clustered_trajs is now:
# [
#   [traj_array, traj_array, ...],   # cluster 0
#   [traj_array, traj_array, ...],   # cluster 1
#   ...
# ]