from .Utils.dual_kf import DualKalman
from .Utils.transformations import traj_to_img_domain, img_to_traj_domain, create_traj_map
from .Utils.utils import read_pkl, is_within
import numpy as np

class Trajectory_Filter():
    def __init__(self, traj_dir, KF_Domain='traj' ) -> None:
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
        self.traj = None
        self.polygon_set = read_pkl(traj_dir)
        self.polygons = self.polygon_set.pop('polygons')
        self.assigned = None
        self.trajectories = None
        self.KF_count = 0
        self.sr = None
        self.maps = None

        #  This will likely have to move to update function, since we need to assign the trajectory first
        if KF_Domain == 'traj':
            self.kf = DualKalman(KF_Domain)
        elif KF_Domain == 'image':
            self.kf = DualKalman(KF_Domain) # change to image domain kf version, currently just placeholder
        else:
            raise ValueError("Invalid Kalman filter domain. Choose 'traj' or 'image'.")
        
    def update_xy(self, xy):
        """
        Update the states of the trajectory tracker, assign and correct tracks and create necassary maps
        """
        if not self.trajectories:
            self.assigned = is_within(xy, self.polygons)
                    
            if not self.assigned:
                return (xy[0] + kf_dx, xy[1] + kf_dy), False # need to find a way to predict motion before traj assignment, likely need aditional kf until assigned
            
            self.sr = []
            self.trajectories = []
            for internal_dict in self.polygon_set[self.assigned].values():
                self.trajectories.append(np.array(internal_dict[:,0]))
                self.sr.append(np.array(internal_dict[:,1]))
            self.map = create_traj_map(self.trajectories)
        
        positions = []
        for map in self.maps:
            pos = traj_to_img_domain(xy, map)
            positions.append(pos)

        self.update_kf()
    

    def update_kf(self, traj_meas, box_meas):
        """
        Update the given kalman filter with the provided position, mean, and covariance.

        Args:
            position (tuple): xy coordinates in image domain.
        """
        
        if self.KF_count == 0:
            self.kf.initiate(traj_measurement=traj_meas, box_measurement=box_meas)
            self.KF_count += 1
        
        self.kf.update(traj_meas, box_meas)
        self.kf.predict()
        
        measurements = self.kf.get_state()
        xy = traj_to_img_domain(measurements['trajectory'][0][:1], self.map)
        ah = measurements['box'][0][:1]
        return xy.extend(ah)