import numpy as np
from dtaidistance import dtw_ndim
import hdbscan
from tqdm import tqdm

class TrajectoryClusterer():
    def __init__(self, image_size=(3840, 2160), max_angle_deg=45.0, max_start_dist_ratio=0.30, max_end_dist_ratio=0.30):
        self.image_size = image_size
        self.IMG_DIAG = float(np.hypot(*image_size))
        self.max_angle_deg = max_angle_deg
        self.max_start_dist = max_start_dist_ratio * self.IMG_DIAG
        self.max_end_dist = max_end_dist_ratio * self.IMG_DIAG

    def angle_deg(self, v1, v2, eps=1e-9):
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < eps or n2 < eps:
            return None
        cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_theta)))

    def gating(self, traj1, traj2):
        traj1 = np.asarray(traj1, dtype=float)
        traj2 = np.asarray(traj2, dtype=float)

        if len(traj1) < 2 or len(traj2) < 2:
            return False

        # Start checks
        if np.linalg.norm(traj1[0] - traj2[0]) > self.max_start_dist:
            return False
        a1 = self.angle_deg(traj1[1] - traj1[0], traj2[1] - traj2[0])
        if a1 is not None and a1 > self.max_angle_deg:
            return False

        # End checks
        if np.linalg.norm(traj1[-1] - traj2[-1]) > self.max_end_dist:
            return False
        a2 = self.angle_deg(traj1[-1] - traj1[-2], traj2[-1] - traj2[-2])
        if a2 is not None and a2 > self.max_angle_deg:
            return False

        return True

    def cluster_trajectories(self,trajectories):
        # too many trajectories to run dtwdistance on them all, seperate into smaller clusters using basic position gating
        # need tougher gating/similarity
        # need to filter out incomplete tracks



        # Basic sanity: keep only trajectories with >=2 points
        valid_idx = []
        for i, t in enumerate(trajectories):
            if len(t)>=2:
                valid_idx.append(i)
        traj_valid = [np.asarray(trajectories[i], dtype=float) for i in valid_idx]

        if len(traj_valid) >2:
            
            n = len(traj_valid)
            D = np.full((n, n), np.nan, dtype=float)
            np.fill_diagonal(D, 0.0)

            # --- Build candidate list using gating ---
            candidates = []
            for i in range(n):
                for j in range(i+1, n):
                    if self.gating(traj_valid[i], traj_valid[j]):
                        candidates.append((i, j))

            # --- Compute DTW only for candidates ---
            for i, j in tqdm(candidates, desc="Computing DTW distances"):
                D[i, j] = D[j, i] = dtw_ndim.distance_fast(traj_valid[i], traj_valid[j])

            # --- Set BIG for non-candidates ---
            vals = D[np.isfinite(D) & (~np.eye(n, dtype=bool))]
            BIG = np.percentile(vals, 99) * 5 if vals.size else 1e6
            D[~np.isfinite(D)] = BIG
            np.fill_diagonal(D, 0.0)

            n = len(traj_valid)
            for i in range(n):
                for j in range(i + 1, n):
                    if not self.gating(traj_valid[i], traj_valid[j]):
                        D[i, j] = D[j, i] = BIG

            # HDBSCAN clustering
            clusterer = hdbscan.HDBSCAN(
                metric="precomputed",
                min_cluster_size=2,
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
            return clustered_trajs

        # clustered_trajs is now:
        # [
        #   [traj_array, traj_array, ...],   # cluster 0
        #   [traj_array, traj_array, ...],   # cluster 1
        #   ...
        # ]