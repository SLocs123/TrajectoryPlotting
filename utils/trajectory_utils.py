import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
import math

def get_all_points(trajectories):
    """
    Flattens all trajectory point lists into a single NumPy array.

    This function takes a dictionary where each value is a list (or array) of points 
    representing individual trajectories, and returns a single NumPy array containing 
    all points across all trajectories.

    Parameters:
        trajectories (dict): A dictionary where each value is a list or array of (x, y) points.

    Returns:
        np.ndarray: A NumPy array of shape (N, 1, 2) or (N, 2), depending on input, 
                    containing all points from all trajectories.
    """
    all_points = []
    for points in trajectories.values():
        all_points.extend(points)
    return np.array(all_points)

def interpolate_trajectory(trajectory, num_points=100, window_length=20, polyorder=2):
    # Separate the coordinates into x and y arrays
    x_original = np.array([coord[0][0] for coord in trajectory])
    y_original = np.array([coord[0][1] for coord in trajectory])
    extra = np.array([coord[1] for coord in trajectory])
    
    # Apply Savitzky-Golay filter to smooth the x and y coordinates
    x_smoothed = savgol_filter(x_original, window_length, polyorder)
    y_smoothed = savgol_filter(y_original, window_length, polyorder)

    # Generate new time steps with exactly 'num_points' points
    time_new = np.linspace(0, len(trajectory) - 1, num_points)

    # Interpolate the coordinates using linear interpolation after smoothing
    x_new = np.interp(time_new, np.arange(len(x_smoothed)), x_smoothed)
    y_new = np.interp(time_new, np.arange(len(y_smoothed)), y_smoothed)

    # Combine the interpolated x and y coordinates back into a list of tuples
    coords_new = np.array(list(zip(x_new, y_new)))
    out = np.array(list(zip(coords_new, extra)))
    return out

def resample_trajectory(trajectory, num_points=100):
    x = np.array([point[0][0] for point in trajectory])
    y = np.array([point[0][1] for point in trajectory])

    # Calculate the arc length
    arc_length = np.cumsum(np.sqrt(np.diff(x, prepend=x[0])**2 + np.diff(y, prepend=y[0])**2))

    # Generate resampled arc length (linearly spaced)
    resampled_arc_length = np.linspace(arc_length[0], arc_length[-1], num_points)

    # Using cubic spline interpolation to smooth the trajectory
    smoothing = 20000
    spline_x = UnivariateSpline(arc_length, x, s=smoothing)
    spline_y = UnivariateSpline(arc_length, y, s=smoothing)

    # Interpolate the x and y coordinates at the new resampled arc lengths
    x_resampled = spline_x(resampled_arc_length)
    y_resampled = spline_y(resampled_arc_length)

    return list(zip(x_resampled, y_resampled))

# Function to sort points in a trajectory
def sort_points(points):
    points = np.array(points)
    sorted_points = [points[0]]
    points = np.delete(points, 0, axis=0)
    while points.size > 0:
        distances = np.linalg.norm(points - sorted_points[-1], axis=1)
        closest_point_index = np.argmin(distances)
        sorted_points.append(points[closest_point_index])
        points = np.delete(points, closest_point_index, axis=0)
    return np.array(sorted_points)

# Function to rotate a rectangle and return its corners ------------------------------return to fix
def rotate_rectangle(point, dir_point, width):
    # length = (math.dist(point, dir_point))*2
    length = (math.dist(point, dir_point))
    delta_x = dir_point[0] - point[0]
    delta_y = dir_point[1] - point[1]
    angle = np.arctan2(delta_y, delta_x)
    perpendicular_angle = angle + np.pi / 2
     # Calculate the coordinates of the rectangle's corners
    half_width = width / 2
    half_height = length / 2

    corners = np.array([
        [-half_width, -half_height],
        [half_width, -half_height],
        [half_width, half_height],
        [-half_width, half_height]
    ])
    rotation_matrix = np.array([
        [np.cos(perpendicular_angle), -np.sin(perpendicular_angle)],
        [np.sin(perpendicular_angle), np.cos(perpendicular_angle)]
    ])
    rotated_corners = corners @ rotation_matrix.T + np.array([point[0], point[1]])

    return rotated_corners