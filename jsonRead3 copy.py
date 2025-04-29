import pandas as pd
from shapely.geometry import Polygon, Point
import ast
import cv2
import numpy as np
from scipy.interpolate import interp1d, UnivariateSpline, CubicSpline
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from transform_label_json import transform_labelstudio_input, x_to_bbox, bbox_to_z
import time
import pickle
import os

# Assuming label_json is already created
label_json = transform_labelstudio_input('label_studio_CAM-HAZELDELL-126THST.json', 0)

# Function to get true labels
def get_true_labels(label, frame_num):
    current_labels = label[frame_num]
    current_formatted_labels = []
    for elem in current_labels:
        current_formatted_labels.append(
            [elem["min_x"] * 38.4, elem["min_y"] * 21.6, elem["max_x"] * 38.4, elem["max_y"] * 21.6, elem["car_id"]]
        )
    return current_formatted_labels

# Function to read polygons from a CSV file
def read_polygons_from_csv(file_path):
    df = pd.read_csv(file_path)
    num_columns = len(df.columns)
    polygons = []
    for idx, row in df.iterrows():
        points = []
        for col_idx in range(num_columns):
            if pd.notna(row.iloc[col_idx]):  # Check if the value is not NaN
                point_str = row.iloc[col_idx]
                point_list = ast.literal_eval(point_str)
                points.append((point_list[0], point_list[1]))
        
        if points:
            polygons.append(Polygon(points))
    return polygons

# Function to draw polygons on a frame
def draw_polygons(frame, polygons):
    for polygon in polygons:
        points = np.array([list(coord) for coord in polygon.exterior.coords], np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)
        centroid = polygon.centroid
        label = f"({points[0][0][0]:.1f}, {points[0][0][1]:.1f})"
        label_position = (int(centroid.x), int(centroid.y))
        cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

# Function to draw bounding boxes on a frame
def draw_bounding_boxes(frame, labels):
    for label in labels:
        min_x, min_y, max_x, max_y, car_id = label
        cv2.rectangle(frame, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (255, 0, 0), 2)
        cv2.putText(frame, str(car_id), (int(min_x), int(min_y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Function to draw trajectories on a frame
def draw_trajectories(frame, expected_trajectories, coords='off'):
    label_colors = [
    (255, 0, 0),      # Bright Blue
    (0, 255, 0),      # Bright Green
    (0, 0, 255),      # Bright Red
    (255, 255, 0),    # Cyan
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Yellow
    (128, 0, 128),    # Purple
    (255, 165, 0),    # Orange
    (0, 128, 128),    # Teal
    (128, 128, 0),    # Olive
    (75, 0, 130),     # Indigo
    (255, 192, 203)   # Pink
]
    color_index = 0


    for out_index, (first_polygon_int, inner_dict) in enumerate(expected_trajectories.items()):
        color_index = (color_index + 1) % len(label_colors)
        color = label_colors[color_index % len(label_colors)]

        for in_index, (final_polygon, trajectory) in enumerate(inner_dict.items()):
            trajectory = trajectory[:,0]
            for point in trajectory:
                cv2.circle(frame, (int(point[0]), int(point[1])), 2, color, -1)
            for i in range(len(trajectory) - 1):
                cv2.line(frame, (int(trajectory[i][0]), int(trajectory[i][1])), (int(trajectory[i+1][0]), int(trajectory[i+1][1])), color, 1)
            x, y = trajectory[0][0], trajectory[0][1]
            points = np.array([list(coord) for coord in first_polygon_int.exterior.coords], np.int32).reshape((-1, 1, 2))
            label_text = f"Index: 1st: {out_index}, 2nd: {in_index}"
            label_position = (int(x), int(y))
            font_scale = 1.0
            font_color = (255, 255, 255)
            thickness = 2
            line_type = cv2.LINE_AA
            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            background_left = label_position[0]
            background_top = label_position[1] - text_height - baseline
            background_right = label_position[0] + text_width
            background_bottom = label_position[1]
            # cv2.rectangle(frame, (int(background_left), int(background_top)), (int(background_right), int(background_bottom)), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, label_text, label_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, thickness, line_type)

# Function to save a frame with annotations
def save_frame(video_path, output_path, polygons_csv, expected_trajectories, coords='off', frame_num=1):
    """
    Saves a frame from a video or an image file, overlaying it with bounding boxes, polygons, 
    and trajectories as specified.

    Args:
        video_path (str): The path to the video or image file.
        frame_num (int): The specific frame number to extract from the video (ignored if input is an image).
        output_path (str): The path where the processed frame/image will be saved.
        polygons_csv (str): The path to the CSV file containing polygon data.
        expected_trajectories (dict): The expected trajectories to be drawn on the frame.
        coords (str, optional): The format for coordinates ('off' for no coords, default is 'off').

    Raises:
        ValueError: If the input file format is not supported (neither video nor image).
    """

    # Detect file extension
    _, ext = os.path.splitext(video_path)
    ext = ext.lower()

    # Supported video file extensions
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    if ext in video_extensions:
        # Process as a video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video file.")
            return

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print(f"Error: Could not read frame {frame_num} from video.")
            return

    elif ext in image_extensions:
        # Process as an image
        frame = cv2.imread(video_path)
        if frame is None:
            print("Error: Could not open image file.")
            return

    else:
        # Raise error for unsupported format
        raise ValueError(f"Unsupported file format: {ext}. Supported formats are: {video_extensions + image_extensions}")

    # Now process the frame/image with polygons, labels, and trajectories
    polygons = read_polygons_from_csv(polygons_csv)
    # labels = get_true_labels(frame_num)
    
    # draw_bounding_boxes(frame, labels)
    draw_polygons(frame, polygons)
    draw_trajectories(frame, expected_trajectories, coords=coords)

    # Save the processed frame/image
    cv2.imwrite(output_path, frame)
    print(f"Frame {frame_num} saved as {output_path}.")

# Function to check if a point is within a list of polygons
def is_within(xy, polygons):
    x, y = xy
    center_point = Point(x, y)
    for polygon in polygons:
        if polygon.contains(center_point):
            return True, polygon
    return False, None


def contains_empty_or_nan(lst):
    for item in lst:
        if isinstance(item, (list, np.ndarray)):
            if any(pd.isna(x) or x == '' for x in item):
                return True
        elif pd.isna(item) or item == '':
            return True
    return False


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

# Function to get the next element in a list of points
def get_next_element(var, index):
    return var[index + 1] if index + 1 < len(var) else var[index-1]

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


# Function to interpolate a trajectory
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

# Function to resample a trajectory with smoother interpolation
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


def read_labels_from_txt(txt_path):
    label_txt = {}
    with open(txt_path, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            frame = int(parts[0])
            car_id = int(parts[1])
            x1 = float(parts[2])
            y1 = float(parts[3])
            x2 = float(parts[4])
            y2 = float(parts[5])
            if frame not in label_txt:
                label_txt[frame] = []
            label_txt[frame].append([x1, y1, x2, y2, car_id])
    return label_txt


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
        if start_poly != end_poly:
            if start_poly not in linked_polys:
                linked_polys[start_poly] = {}
            if end_poly not in linked_polys[start_poly]:
                linked_polys[start_poly][end_poly] = []
            linked_polys[start_poly][end_poly].append(items)
    
    return linked_polys


def add_item(dict, key1, key2, item):
    if key1 not in dict:
        dict[key1] = {}
    dict[key1][key2] = item
    return dict


# Function to create expected trajectories dictionary
def create_expected_trajectories(polygons, filepath):
    traj_dict = {}
    vehicle_trajectories = create_vehicle_trajectories(filepath)
    linked_polygons = find_linked_polygons(polygons, vehicle_trajectories)

    for poly1, inner_dict in linked_polygons.items():
        for poly2, items_list in inner_dict.items():
            averaged_traj = average_similar_points(items_list, poly1, poly2, 150, show=False)
            final_trajs = add_item(traj_dict, poly1, poly2, averaged_traj)
    return final_trajs


def show_structure(d, indent=0):
    for key, value in d.items():
        print(' ' * indent + f'{key} ({type(value).__name__})')
        if isinstance(value, dict):
            show_structure(value, indent + 2)

# Example usage
video_path = 'cam04.jpg'
frame_num = 1
output_path = 'test.jpg'
polygons_csv = 'cam04_poly.csv'
SCT_label = 'cam04_SCT.txt'

expected_trajectories = create_expected_trajectories(read_polygons_from_csv(polygons_csv), SCT_label)
# show_structure(expected_trajectories)

save_frame(video_path, output_path, polygons_csv, expected_trajectories)


# polygons = read_polygons_from_csv('polygons.csv')
# expected_trajectories['polygons'] = polygons
# with open('CAM_HAZEL_TRAJS.pkl', 'wb') as pkl_file:
#     pickle.dump(expected_trajectories, pkl_file)

# # Function to interpolate a trajectory
# def interpolate_trajectory(trajectory, num_points=100, smoothing_factor = 1000):
#     # Separate the coordinates into x and y arrays
#     x_original = np.array([coord[0][0] for coord in trajectory])
#     y_original = np.array([coord[0][1] for coord in trajectory])
#     extra = np.array([coord[1] for coord in trajectory])
    
#     time_original = np.arange(len(trajectory))
#     spline_x = UnivariateSpline(time_original, x_original, s=smoothing_factor)
#     spline_y = UnivariateSpline(time_original, y_original, s=smoothing_factor)

#     # Generate new time steps with exactly 100 points
#     time_new = np.linspace(0, len(trajectory) - 1, num_points)

#     # Interpolate the coordinates
#     x_new = spline_x(time_new)
#     y_new = spline_y(time_new)

#     # Combine the interpolated x and y coordinates back into a list of tuples
#     coords_new = np.array(list(zip(x_new, y_new)))
#     out = np.array(list(zip(coords_new, extra)))
#     return out

