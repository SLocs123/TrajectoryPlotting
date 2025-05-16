from .Utils.dual_kf import DualKalman
from .Utils.multi_kf import MultiKalman
from .Utils.transformations import traj_to_img_domain, img_to_traj_domain, create_traj_map
from .Utils.utils import read_pkl, is_within
import numpy as np

class Trajectory_Filter():
    def __init__(self, traj_dir, position_xyah=[None, None], KF_Domain='traj' ) -> None:
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
        self.assigned = None
        self.domain = KF_Domain
        self.kf = DualKalman()
        self.kf.initiate(position_xyah[0], position_xyah[1]) 

        
    def update_traj_xy(self, xy, ah):
        """
        Update the states of the trajectory tracker, assign and correct tracks and create necassary maps
        """

        if not self.assigned:
            self.assigned = is_within(xy, self.polygons)
            
            if not self.assigned:
                self.kf.update(xy, ah)
                return self.kf.predict() # need to find a way to predict motion before traj assignment, likely need aditional kf until assigned
            self.define_traj_sr_map()

            self.kf.update(xy, ah)
            xyah = self.kf.predict()
            
            dic = self.kf.get_state()
            self.kf = MultiKalman(dic['box'])
            self.kf.initiate(traj_measurements=[img_to_traj_domain(xy, map) for map in self.maps], box_measurement=ah)
            return xyah
        
        positions = []
        for map in self.maps: # type: ignore
            pos = img_to_traj_domain(xy, map)
            positions.append(pos)
        self.kf.update(positions, ah) # need to handle this correctly, multiple pos values
        xyahs = self.kf.predict()

        i, xyah = min(enumerate(xyahs), key=lambda item: item[1][0][1])
        xy = traj_to_img_domain(xyah[0:2], self.maps[i])

        return xy, xyah[2:4]
    
    def define_traj_sr_map(self):
        self.sr = []
        self.trajectories = []
        for internal_dict in self.polygon_set[self.assigned].values():
            self.trajectories.append(np.array(internal_dict[:,0]))
            self.sr.append(np.array(internal_dict[:,1]))
        self.maps = create_traj_map(self.trajectories)