from .simple_kf import SimpleKalmanFilter

class MultiKalman:
    def __init__(self, box_mean_cov=None):
        self.kf_traj_list = []  # List to hold trajectory Kalman filters
        self.kf_box = SimpleKalmanFilter(dim=2)  # Single Kalman filter for box in image domain
        if box_mean_cov is not None:
            self.box_mean, self.box_cov = box_mean_cov[0], box_mean_cov[1]

    def initiate(self, traj_measurements=None, box_measurement=None):
        # Initialize trajectory Kalman filters
        self.kf_traj_list = []
        self.traj_states = []
        if traj_measurements:
            # kf_traj = SimpleKalmanFilter(dim=2) # THis should be fine for all trajectory kfs
            for traj_measurement in traj_measurements:
                kf_traj = SimpleKalmanFilter(dim=2) # can remove this but need to check if it works as intended
                traj_mean, traj_cov = kf_traj.initiate(traj_measurement)
                self.kf_traj_list.append(kf_traj) # shouldnt need this, should just need a list of mean and cov
                self.traj_states.append((traj_mean, traj_cov))
        else:
            # Default to a single trajectory Kalman filter
            kf_traj = SimpleKalmanFilter(dim=2)
            traj_mean, traj_cov = kf_traj.initiate(None)
            self.kf_traj_list.append(kf_traj)
            self.traj_states.append((traj_mean, traj_cov))

        # Initialize box Kalman filter
        if self.box_mean is None:
            self.box_mean, self.box_cov = self.kf_box.initiate(box_measurement)
        

    def predict(self):
        # Predict for all trajectory Kalman filters
        self.traj_states = [
            kf_traj.predict(traj_mean, traj_cov)
            for kf_traj, (traj_mean, traj_cov) in zip(self.kf_traj_list, self.traj_states)
        ]

        # Predict for box Kalman filter
        self.box_mean, self.box_cov = self.kf_box.predict(self.box_mean, self.box_cov)

        # Return predicted states
        traj_means = [state[0] for state in self.traj_states]
        return traj_means, self.box_mean

    def update(self, traj_measurements, box_measurement):
        # Update all trajectory Kalman filters
        self.traj_states = [
            kf_traj.update(traj_mean, traj_cov, traj_measurement)
            for kf_traj, (traj_mean, traj_cov), traj_measurement in zip(
                self.kf_traj_list, self.traj_states, traj_measurements
            )
        ]

        # Update box Kalman filter
        self.box_mean, self.box_cov = self.kf_box.update(self.box_mean, self.box_cov, box_measurement)

    def get_state(self):
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