from .Utils.simple_kf import SimpleKalmanFilterXY, SimpleKalmanFilterWH
from .Utils.multi_kf import MultiKalman
from .Utils.transformations import traj_to_img_domain, img_to_traj_domain, create_traj_map
from .Utils.utils import read_pkl, is_within
import numpy as np

class Trajectory_Filter():
    def __init__(self, traj_dir):
        """
        Initializes the Traj_KF class.

        Args:
            traj_dir (str): The directory path to the trajectory data in pickle format.
            KF_Type (str): The type of Kalman Filter to use. Possible values are 'traj' or 'image'.

        Attributes:
            traj (NoneType): Placeholder for trajectory data.
            polygon_set (dict): A dictionary containing polygon data loaded from the pickle file.
            polygons (list): A list of polygons extracted from the polygon_set.
            assigned (NoneType): Placeholder for assigned data.
            trajectories (NoneType): Placeholder for trajectory information.
            sr (NoneType): Placeholder for spatial reference or related data.
        """
        self.polygon_set = read_pkl(traj_dir)
        self.polygons = self.polygon_set.pop('polygons')
        self.kf_xy = SimpleKalmanFilterXY()
        self.kf_box = SimpleKalmanFilterWH()
        self.define_traj_sr_maps()
        
    
    def initiate(self, track):
        """
        Create track from unassociated measurement. initiate always in the image domain.
        This method initializes the Kalman filter state for a track based on its xywh coordinates.
        """
        xywh = track.xywh
        xy = xywh[:2]
        ah = xywh[2:4]
        
        xymean, xycov = self.kf_xy.initiate(xy)
        boxmean, boxcov = self.kf_box.initiate(ah)
        
        track.mean = self.combine_mean(xymean, boxmean)
        track.cov = [xycov, boxcov]
        
    
    # def multi_initiate(self, track):
    #     """
    #     Create track from unassociated measurements(vectorised).
    #     """

    def predict(self, track):
        """
        Run Kalman filter prediction step. !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! -------------------------------- need to check if track is assigned, handle different domains
        """
        meanxy, meanbox = self.split_mean(track.mean)
        covxy, covbox = track.cov
        if track.assigned:
            maps = self.all_maps[track.assigned]
            predicted_xy, predicted_covxy = self.kf_xy.predict(meanxy, covxy, maps)
            predicted_box, predicted_covbox = self.kf_box.predict(meanbox, covbox, maps)
        else:   
            predicted_xy, predicted_covxy = self.kf_xy.predict(meanxy, covxy)
            predicted_box, predicted_covbox = self.kf_box.predict(meanbox, covbox)
    
        return self.combine_mean(predicted_xy, predicted_box)

    # def multi_predict(self, tracks):
    #     """
    #     Vectorized Kalman filter prediction for multiple tracks.
    #     """
    #     means = np.array([track.mean for track in tracks])
    #     covs = np.array([track.cov for track in tracks])

    #     # Split means and covariances
    #     meanxy = means[:, :2]
    #     meanbox = means[:, 2:4]
    #     covxy = np.array([cov[0] for cov in covs])
    #     covbox = np.array([cov[1] for cov in covs])

    #     # Vectorized prediction for xy and box
    #     predicted_xy, predicted_covxy = self.kf_xy.multi_predict(meanxy, covxy)
    #     predicted_box, predicted_covbox = self.kf_box.multi_predict(meanbox, covbox)

    #     # Combine results
    #     combined_means = np.hstack([predicted_xy, predicted_box])
    #     combined_covs = np.array([[cx, cb] for cx, cb in zip(predicted_covxy, predicted_covbox)])

    #     return combined_means, combined_covs


        
    def update(self, track, xy, wh):
        """
        Update the states of the trajectory tracker, assign and correct tracks and create necassary maps
        """

        if not track.assigned:
            track.assigned = is_within(xy, self.polygons)
            meanxy, meanbox = self.split_mean(track.mean)
            covxy, covbox = track.cov
            updated_xy, updated_covxy = self.kf_xy.update(meanxy, covxy, xy)
            updated_box, updated_covbox = self.kf_box.update(meanbox, covbox, wh)
            
            if not track.assigned:
                return self.combine_mean(updated_xy, updated_box), [updated_covxy, updated_covbox]

            
            dic = self.kf_xy.get_state()
            self.kf_xy = MultiKalman()
            
            return self.combine_mean(updated_xy, updated_box), [updated_covxy, updated_covbox]
        else:
            maps = self.all_maps[track.assigned]
            meanxy, meanbox = self.split_mean(track.mean)
            covxy, covbox = track.cov
            updated_xy, updated_covxy = self.kf_xy.update(meanxy, covxy, xy, maps)
            updated_box, updated_covbox = self.kf_box.update(meanbox, covbox, wh)
            
            return self.combine_mean(updated_xy, updated_box), [updated_covxy, updated_covbox]
    
    def define_traj_sr_maps(self):
        self.all_maps = {}
        for ext_key, internal_dict in self.polygon_set.items():
            trajectories = []
            for trajs in internal_dict.values():
                trajectories.append(np.array(trajs[:, 0]))
            self.all_maps[ext_key] = create_traj_map(trajectories)
        
    def split_mean(self, mean):
        """
        Splits the mean into xy and wh components.
        For mean = [x, y, w, h, vx, vy, vw, vh]:
          xy = [x, y, vx, vy]
          wh = [w, h, vw, vh]
        """
        xy = [mean[0], mean[1], mean[4], mean[5]]
        wh = [mean[2], mean[3], mean[6], mean[7]]
        return xy, wh
    
    def combine_mean(self, xymean, whmean):
        """
        Combines xy and ah components into a single mean vector.
        """
        return (xymean[:2] + whmean[:2], xymean[2:4] + whmean[2:4])
        
        
        
#  functions:
#  update - takes an associated detection and updates the Kalman filter state - during update stage, must check if assigned
#  predict - returns the predicted state of the Kalman filter (using the whole track as input alows saving of states in the track itself, or could use the id to correspond to a local state)
#   takes a track input
#  multi_predict - returns the predicted states of the Kalman filter for multiple trajectories (vectorised)
#    Takes a list of track objects as input
#  reset - resets the Kalman filter state, used to switch domains
# construct_xywh - contructs artificiatl xywh mean for use with standard syntax
 