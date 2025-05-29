from .simple_kf import SimpleKalmanFilter
import numpy as np
from Utils.transformations import img_to_traj_domain, traj_to_img_domain

class MultiKalman:
    """
    Manages multiple Kalman filters for tracking trajectories in a multi-dimensional space.

    Attributes:
        dim (int): Number of position dimensions (e.g., 2 for x, y or a, h).
        dt (float): Time step for the motion model. This can be dynamically adjusted for better tracking in the future.
        _motion_mat (np.ndarray): State transition matrix for the motion model.
        _update_mat (np.ndarray): Observation matrix for the measurement model.
        _std_weight_position (float): Standard deviation weight for position noise.
        _std_weight_velocity (float): Standard deviation weight for velocity noise.
        kf (SimpleKalmanFilter): Instance of a simple Kalman filter for state estimation.
    """

    def __init__(self, dim=2, std_pos=1. / 20, std_vel=1. / 160):
        """
        Initializes the MultiKalman object with the specified dimensions and noise parameters.

        :param dim: Number of position dimensions (2 → x, y).
        :type dim: int
        :param std_pos: Standard deviation weight for position noise. affects the weight of measurments
        :type std_pos: float
        :param std_vel: Standard deviation weight for velocity noise. affects the weight of the motion model
        :type std_vel: float
        """
        self.kf = SimpleKalmanFilter(dim, std_pos, std_vel)

    def initiate(self, measurements=None, maps=None):
        """
        Initializes trajectory Kalman filters for a set of measurements and corresponding maps.
        This function converts the provided measurements into the trajectory domain for each map,
        then initializes a Kalman filter for each resulting point. The means and covariances of
        the initialized filters are collected and returned.
        Args:
            measurements (optional): The measurement data to be used for initialization.
            maps (optional): A list of map objects corresponding to each measurement.
        Returns:
            tuple: A tuple containing two lists:
                - means: List of mean state vectors for each initialized Kalman filter.
                - covariances: List of covariance matrices for each initialized Kalman filter.
        Note:
            If either `measurements` or `maps` is not provided, the function returns a warning.
        """

        means = []
        covariances = []
        if measurements and maps:
            for map in maps:
                point = img_to_traj_domain(measurements, map)
                mean, cov = self.kf.initiate(point)
                means.append(mean)
                covariances.append(cov)
        else:
            raise ValueError("Measurements and maps must be provided for initialization. Multi_KF cannot be initialized without them due to the unknown number of possible trajectories.")
            
        return means, covariances

    def predict(self, means, covariances, maps):
        """
        Predicts the next state for multiple trajectories using the Kalman filter.

        Args:
            means (list or np.ndarray): List or array of mean state vectors for each trajectory.
            covariances (list or np.ndarray): List or array of covariance matrices corresponding to each mean.
            maps (list): List of map objects or transformation data for each trajectory.

        Returns:
            None

        Description:
            For each trajectory, this method applies the Kalman filter's predict step using the provided mean and covariance.
            The predicted trajectory state is then transformed to the image domain using the corresponding map.
        """
        for i, mean in enumerate(means):
            cov = covariances[i]
            traj = self.kf.predict(mean, cov)
            point = traj_to_img_domain(traj[0], maps[i])


    def update(self, measurements):
        """
        Updates the trajectory Kalman filters with new measurements.
        For each measurement in the input list, this method updates the corresponding
        Kalman filter state (mean and covariance) using the filter's update method.
        The updated means and covariances for all trajectories are returned as lists.
        Args:
            measurements (list or array-like): A list of measurement vectors, one for each trajectory.
        Returns:
            tuple: A tuple containing two lists:
                - updated_means (list): The updated state means for each trajectory.
                - updated_covariances (list): The updated state covariances for each trajectory.
        """
        updated_means = []
        updated_covariances = []
        for i, measurement in enumerate(measurements):
            mean, cov = self.kf.update(self.traj_states[i][0], self.traj_states[i][1], measurement)
            updated_means.append(mean)
            updated_covariances.append(cov)

        return updated_means, updated_covariances

    def get_state(self): # will not work, needs to be adapted to the new structure
        # Get states for all trajectory Kalman filters
        traj_states = [
            {'mean': traj_mean, 'cov': traj_cov}
            for traj_mean, traj_cov in self.traj_states
        ]

        # Return all states
        return {
            'trajectories': traj_states,
            'box': {'mean': self.box_mean, 'cov': self.box_cov}
        }