from utils.kf_utils import initialise_kf

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
        self.polygon_set = self.read_pkl(traj_dir)
        self.polygons = self.polygon_set.pop('polygons')
        self.assigned = None
        self.trajectories = None
        self.sr = None

        self.kf = initialise_kf(KF_Domain)


    def update(self, traj_pos):
        traj_pos