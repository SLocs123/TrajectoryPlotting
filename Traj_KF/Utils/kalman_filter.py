class Traj_KF():
    def __init__(self, kf_domain, position):
        """
        Initializes the Kalman filter with a specified domain.

        Args:
            kf_domain (str): The domain for the Kalman filter, either 'traj' or 'image'.
        """
        self.kf = self.initialise_kf(kf_domain)

    def initialise_kf(self, kf_domain, position):
        """
        Initializes the Kalman filter based on the specified domain.

        Args:
            kf_domain (str): The domain for the Kalman filter, either 'traj' or 'image'.

        Warning:
            Ensure that the position is entered in the chosen domain ('traj' or 'image'),
            as this package does not handle domain conversion, for domain conversion see Utils/traj_utils.-----------------------------------------------------!!!
            
        Returns:
            kf: The initialized Kalman filter object.
        """
        if kf_domain == 'traj':
            kf = KalmanFilter(dim_x=4, dim_z=2)
            kf.x = np.array([[0], [0], [0], [0]])
        elif kf_domain == 'image':
            kf = KalmanFilter(dim_x=6, dim_z=4)
            kf.x = np.array([[0], [0], [0], [0], [0], [0]])
        else:
            raise ValueError("Invalid Kalman filter domain. Choose 'traj' or 'image'.")
    
    def update(self, position, mean, cov):
        """
        Updates the Kalman filter with the given position, mean, and covariance.

        Args:
            position (array): The current position to update the filter with.
            mean (array): The mean of the state estimate.
            cov (array): The covariance of the state estimate.

        Warning:
            Ensure that the position is entered in the chosen domain ('traj' or 'image'),
            as this package does not handle domain conversion, for domain conversion see Utils/traj_utils.-----------------------------------------------------!!!

        Returns:
            updated_position: The updated position after applying the Kalman filter.
        """
        self.kf.predict()
        self.kf.update(position)
        updated_position = self.kf.x
        return updated_position
        
            
