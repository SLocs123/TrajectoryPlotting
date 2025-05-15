from .simple_kf import SimpleKalmanFilter
        
class DualKalman:
    def __init__(self):
        self.kf_traj = SimpleKalmanFilter(dim=2)  # for x/y in trajectory domain
        self.kf_box = SimpleKalmanFilter(dim=2)   # for a/h in image domain

    def initiate(self, traj_measurement=None, box_measurement=None):
        self.traj_mean, self.traj_cov = self.kf_traj.initiate(traj_measurement)
        self.box_mean, self.box_cov = self.kf_box.initiate(box_measurement)

    def predict(self):
        self.traj_mean, self.traj_cov = self.kf_traj.predict(self.traj_mean, self.traj_cov)
        self.box_mean, self.box_cov = self.kf_box.predict(self.box_mean, self.box_cov)
        return self.traj_mean, self.box_mean

    def update(self, traj_measurement, box_measurement):
        self.traj_mean, self.traj_cov = self.kf_traj.update(self.traj_mean, self.traj_cov, traj_measurement)
        self.box_mean, self.box_cov = self.kf_box.update(self.box_mean, self.box_cov, box_measurement)

    def get_state(self):
        return {
            'trajectory': (self.traj_mean, self.traj_cov),
            'box': (self.box_mean, self.box_cov)
        }
        