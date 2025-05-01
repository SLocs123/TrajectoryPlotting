import os
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Point, Polygon
import shapely
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from .trajectory_utils import get_mode_list_by_length_kde
from .io_utils import read_labels_from_txt, get_true_labels
from .utils import add_item, get_next_element, is_within
from .trajectory_utils import interpolate_trajectory, resample_trajectory, rotate_rectangle, smooth_trajectory, smooth_density_resample
from .transform_label_json import transform_labelstudio_input, bbox_to_z


def find_linked_polygons(polys, trajs):
    linked_polys = {}
    for id, items in trajs.items():
        items = np.array(items)
        coords = items[:,0]
        detections = [polygon for point in coords for polygon in polys if polygon.contains(Point(point))]
        if not detections:
            continue
        
        start_poly = detections[0]
        end_poly = detections[-1]
        
        reference_poly = Polygon([(824, 526), (826, 609), (879, 606), (876, 516)])
        refernece_end = Polygon([(2334, 495), (2527, 503), (2541, 546), (2331, 531), (2334, 495)])
        if shapely.equals(reference_poly, start_poly) and shapely.equals(refernece_end, end_poly):
            print(f'error id: {id}')

            
        if start_poly != end_poly:
            if start_poly not in linked_polys:
                linked_polys[start_poly] = {}
            if end_poly not in linked_polys[start_poly]:
                linked_polys[start_poly][end_poly] = []
            linked_polys[start_poly][end_poly].append(items)

    
    return linked_polys

# Function to average similar points
def average_similar_points(items_list, poly1, poly2, width, show=True):
    all_points = []
    for item in items_list:
        all_points.extend(item)
    all_points = np.array(all_points)
    longest = max(items_list, key=len)
    resampled_longest = resample_trajectory(longest)

    trajectory = []
    windows = []
    for i, point in enumerate(resampled_longest):
        dir_point = get_next_element(resampled_longest, i)
        rotated_corners = rotate_rectangle(point, dir_point, width)
        win = Polygon(rotated_corners)
        windows.append(win)
        start = [point, [int(all_points[0][1][0]),int(all_points[0][1][1])]]
        points_within = [np.array(start)]
        for item in all_points:
            loc = item[0]
            within, _ = is_within(loc, [win])
            if within:
                points_within.append(np.array(item))
        points_within = np.array(points_within)
        trajectory.append([np.average(points_within[:,0], axis=0), np.average(points_within[:,1], axis = 0)])
    trajectory = np.array(trajectory)
    inted_trajectory = interpolate_trajectory(trajectory)


    if show:
            # Extract the coordinates from the points
            all_points_xy, all_points_sr = zip(*all_points)
            all_points_xy = np.array(all_points_xy)
            all_points_x, all_points_y = all_points_xy[:, 0], all_points_xy[:, 1]
            
            inted_longest_xy = np.array(resampled_longest)
            inted_longest_x, inted_longest_y = inted_longest_xy[:, 0], inted_longest_xy[:, 1]

            trajectory_xy, trajectory_sr = zip(*inted_trajectory)
            trajectory_xy = np.array(trajectory_xy)
            trajectory_x, trajectory_y = trajectory_xy[:, 0], trajectory_xy[:, 1]

            # Determine max_y and add buffer
            max_y = max(all_points_y)
            max_y_buffered = max_y

            # Prepare flipped y-coordinates for visualization only
            all_points_y_flipped = max_y_buffered - all_points_y
            inted_longest_y_flipped = max_y_buffered - inted_longest_y
            trajectory_y_flipped = max_y_buffered - trajectory_y

            # Plotting
            plt.figure(figsize=(10, 6))
            plt.scatter(all_points_x, all_points_y_flipped, color='blue', label='All Points', s=10)
            plt.plot(inted_longest_x, inted_longest_y_flipped, color='red', linewidth=2, label='Longest Trajectory')
            plt.plot(trajectory_x, trajectory_y_flipped, color='green', linewidth=2, linestyle='--', label='Average Trajectory')

            # Adding the window patches with flipped coordinates
            for window in windows:
                flipped_window_coords = [(x, max_y_buffered - y) for x, y in window.exterior.coords]
                patch = patches.Polygon(flipped_window_coords, closed=True, edgecolor='purple', facecolor='none', linewidth=2)
                plt.gca().add_patch(patch)
                
                
            # Extract first two points of poly1 and poly2 for the plot title
            poly1_points = list(poly1.exterior.coords)[:2]  # Get the first two points of poly1
            poly2_points = list(poly2.exterior.coords)[:2]  # Get the first two points of poly2

            # Format the title with the first two points of poly1 and poly2
            title = f"Poly1: {poly1_points}, Poly2: {poly2_points}"

            plt.title(title)
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.legend()
            plt.savefig(f'trajectory_plot_Poly1_{poly1_points}_Poly2_{poly2_points}.png')


    return inted_trajectory

def average_DTW(items_list):
    all_points = []
    for item in items_list:
        all_points.extend(item)
    all_points = np.array(all_points)
    
    aligned_trajectories = align_trajectories(items_list)
    averaged_trajectory = average_trajectories(aligned_trajectories)
    return averaged_trajectory
    
def align_trajectories(trajectories):
    """
    Aligns multiple trajectories using DTW. The first trajectory in the list
    will be used as the reference trajectory, and the rest will be aligned to it.
    """
    # Initialize a list to store aligned trajectories
    aligned_trajectories = []

    # Use the first trajectory as the reference
    reference_trajectory = get_mode_list_by_length_kde(trajectories, return_list=True)
    ref_xy = [point[0] for point in reference_trajectory]
    ref_len = len(ref_xy)

    for traj in trajectories:
        traj_xy = [point[0] for point in traj]
        distance, path = fastdtw(ref_xy, traj_xy, dist=euclidean)

        aligned_traj = [None] * ref_len  # initialise with None or NaNs

        for ref_idx, traj_idx in path:
            aligned_traj[ref_idx] = traj[traj_idx]  # Copy full data point (x, y, meta...)

        # Replace any unfilled entries with NaNs
        for i in range(ref_len):
            if aligned_traj[i] is None:
                aligned_traj[i] = np.full_like(reference_trajectory[0], np.nan)

        aligned_trajectories.append(np.array(aligned_traj))
    return aligned_trajectories


def average_trajectories(aligned_trajectories):
    """
    Averages the aligned trajectories point-by-point.
    Assumes all trajectories have been aligned to the same length.
    """
    # Stack the aligned trajectories and calculate the mean along the axis 0 (point-wise average)
    stacked_trajectories = np.stack(aligned_trajectories)
    averaged_trajectory = np.nanmean(stacked_trajectories, axis=0)
    
    avg_traj_inted = smooth_density_resample(averaged_trajectory)

    return avg_traj_inted  

def create_vehicle_trajectories(label_file_path):
    """
    Creates vehicle trajectories from a label file, either in `.txt` or `.json` format, by parsing the bounding box data
    and organizing it by vehicle ID.

    Args:
        label_file_path (str): The path to the label file (either `.txt` or `.json` format).

    Returns:
        dict: A dictionary containing vehicle trajectories where keys are vehicle IDs and values are lists of tuples representing the vehicle's location and bounding box size.

    Raises:
        ValueError: If the input file format is not supported (neither `.txt` nor `.json`).
    """

    # Detect file extension
    _, file_extension = os.path.splitext(label_file_path)
    file_extension = file_extension.lower()

    # Supported file extensions
    txt_extension = '.txt'
    json_extension = '.json'

    if file_extension == txt_extension:
        # Process .txt file
        label_data = read_labels_from_txt(label_file_path)
        vehicle_trajectories = {}
        for frame_num in range(1, len(label_data) + 1):
            frame_labels = label_data[frame_num]
            for bbox in frame_labels:
                x1, y1, x2, y2, vehicle_id = bbox
                x, y, s, r = bbox_to_z([x1, y1, x2, y2])
                if vehicle_id in vehicle_trajectories:
                    vehicle_trajectories[vehicle_id].append([tuple([x, y]), tuple([s, r])])
                else:
                    vehicle_trajectories[vehicle_id] = [[tuple([x, y]), tuple([s, r])]]

    elif file_extension == json_extension:
        # Process .json file
        label_data = transform_labelstudio_input(label_file_path, 0)
        vehicle_trajectories = {}
        for frame_num in range(len(label_data)):
            frame_labels = get_true_labels(label_data, frame_num)
            for bbox in frame_labels:
                x1, y1, x2, y2, vehicle_id = bbox
                x, y, s, r = bbox_to_z([x1, y1, x2, y2])
                if vehicle_id in vehicle_trajectories:
                    vehicle_trajectories[vehicle_id].append([tuple([x, y]), tuple([s, r])])
                else:
                    vehicle_trajectories[vehicle_id] = [[tuple([x, y]), tuple([s, r])]]

    else:
        # Raise error for unsupported file format
        raise ValueError(f"Unsupported file format: {file_extension}. Supported formats are: {txt_extension}, {json_extension}")

    return vehicle_trajectories

# Function to create expected trajectories dictionary
def create_expected_trajectories(polygons, filepath, DTW=False):
    traj_dict = {}
    vehicle_trajectories = create_vehicle_trajectories(filepath)
    linked_polygons = find_linked_polygons(polygons, vehicle_trajectories)
    
    count = 0
    for poly1, inner_dict in linked_polygons.items():
        print(f'Running average: {count}', end='\r')
        for poly2, items_list in inner_dict.items():
            if DTW:
                averaged_traj = average_DTW(items_list)
            else:
                averaged_traj = average_similar_points(items_list, poly1, poly2, 150, show=False)
            final_trajs = add_item(traj_dict, poly1, poly2, averaged_traj)
        count += 1
    return final_trajs