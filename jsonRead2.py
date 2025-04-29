from transform_label_json import transform_labelstudio_input
import pandas as pd
from shapely.geometry import Polygon, Point
import ast
import cv2
import numpy as np
from scipy.interpolate import interp1d
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import time
from scipy.interpolate import UnivariateSpline
import pickle
import math
import matplotlib.patches as patches


label_json = transform_labelstudio_input('label_studio_CAM-HAZELDELL-126THST.json', 0)


def get_true_labels(frame_num):        
    # get labels from current frame
    current_labels = label_json[frame_num]
    # prepare output
    current_formatted_labels = []
    # extract data, format coordinates because label studio format is in percent of video width/height, we need pixels
    for elem in current_labels:
        current_formatted_labels.append(
            [elem["min_x"] * 38.4, elem["min_y"] * 21.6, elem["max_x"] * 38.4, elem["max_y"] * 21.6, elem["car_id"], ])
    return current_formatted_labels


def read_polygons_from_csv(file_path):
    """
    Read the polygons from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        List of shapely.geometry.Polygon objects.
    """
    df = pd.read_csv(file_path)
    polygons = []
    num_columns = len(df.columns)
    
    for idx, row in df.iterrows():
        points = []
        for col_idx in range(num_columns):
            if pd.notna(row.iloc[col_idx]):  # Check if the value is not NaN
                point_str = row.iloc[col_idx]
                point_dict = ast.literal_eval(point_str)
                points.append((point_dict['x'], point_dict['y']))
        
        if points:
            polygons.append(Polygon(points))
    return polygons


def show_structure(d, indent=0):
    for key, value in d.items():
        print(' ' * indent + f'{key} ({type(value).__name__})')
        if isinstance(value, dict):
            show_structure(value, indent + 2)


def draw_polygons(frame, polygons):
    """
    Draw polygons on the frame and label them with the first and second coordinates.
    
    Args:
        frame: The video frame.
        polygons: List of shapely.geometry.Polygon objects.
    """
    for polygon in polygons:
        # Convert polygon coordinates to numpy array format for OpenCV
        points = np.array([list(coord) for coord in polygon.exterior.coords], np.int32)
        points = points.reshape((-1, 1, 2))
        
        # Draw the polygon
        cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # Calculate the centroid (geometric center) of the polygon
        centroid = polygon.centroid
        
        # Create the label using the centroid coordinates
        label = f"({points[0][0][0]:.1f}, {points[0][0][1]:.1f})"
        
        # Determine the position to draw the label (using the centroid)
        label_position = (int(centroid.x), int(centroid.y))
        
        # Draw the label on the frame
        cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)



def draw_bounding_boxes(frame, labels):
    """
    Draw bounding boxes on the frame.
    
    Args:
        frame: The video frame.
        labels: List of bounding boxes with format [min_x, min_y, max_x, max_y, car_id].
    """
    for label in labels:
        # print(label)
        cv2.imwrite('useframe.jpg', frame)
        # time.sleep(60)
        min_x, min_y, max_x, max_y, car_id = label
        cv2.rectangle(frame, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (255, 0, 0), 2)
        cv2.putText(frame, str(car_id), (int(min_x), int(min_y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


def draw_trajectories(frame, expected_trajectories, coords='off'):
    """
    Draw expected trajectories on the frame.

    Args:
        frame: The video frame.
        expected_trajectories: Nested dictionary where keys are polygons and values are trajectories.
        coords: If 'off', draw trajectories normally. Otherwise, draw circles at specified coordinates.
    """
    # Define colors for labeling trajectories
    label_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    color_index = 0

    # Draw trajectories
    for first_polygon_int, inner_dict in expected_trajectories.items():
        # Increment color index for distinct coloring
        color_index = (color_index + 1) % len(label_colors)

        # Draw circles at specified coordinates if coords match the flag
        if coords != 'off' and color_index == coords:
            for final_polygon, trajectory in inner_dict.items():
                for output_coords in trajectory:
                    for coord in output_coords:
                        cv2.circle(frame, (int(coord[0]), int(coord[1])), 2, label_colors[color_index], -1)
        
        else:
            # Draw trajectories if no specific coordinates provided
            for final_polygon, trajectory in inner_dict.items():
                # Draw trajectory points
                for point in trajectory:
                    cv2.circle(frame, (int(point[0]), int(point[1])), 2, label_colors[color_index], -1)

                # Draw lines connecting trajectory points
                for i in range(len(trajectory) - 1):
                    cv2.line(frame, (int(trajectory[i][0]), int(trajectory[i][1])),
                             (int(trajectory[i+1][0]), int(trajectory[i+1][1])), label_colors[color_index], 1)
                if len(trajectory) > 0:
                    # Find the centroid of the trajectory (average of all points)
                    x = trajectory[0][0]
                    y = trajectory[0][1]

                    

                    points = np.array([list(coord) for coord in first_polygon_int.exterior.coords], np.int32)
                    points = points.reshape((-1, 1, 2))
                    # Create label text using the first polygon's name
                    label_text = f"({points[0][0][0]:.1f}, {points[0][0][1]:.1f})"
                    label_position = (int(x), int(y))  # Example position
                    font_scale = 1.0  # Example font scale
                    font_color = (255, 255, 255)  # Example font color (white)
                    thickness = 2  # Example thickness of the text
                    line_type = cv2.LINE_AA  # Anti-aliased line for smoother text


                    (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    background_left = label_position[0]
                    background_top = label_position[1] - text_height - baseline
                    background_right = label_position[0] + text_width
                    background_bottom = label_position[1]
                    cv2.rectangle(frame, (int(background_left), int(background_top)), (int(background_right), int(background_bottom)), (0, 0, 255), cv2.FILLED)
                    cv2.putText(frame, label_text, label_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, thickness, line_type)


def is_within(xy, polygons):
    """
    Check if the center of the bounding box is within any of the polygons.
    
    Args:
        bbox (tuple): Bounding box in the format (minx, miny, maxx, maxy).
        polygons (list): List of shapely.geometry.Polygon objects.
        
    Returns:
        bool: True if the center is within any polygon, False otherwise.
    """
    x,y = xy
    center_point = Point(x, y)
    
    for polygon in polygons:
        if polygon.contains(center_point):
            return True, polygon
    return False, None


def save_video_with_polygons(video_path, output_path, polygons_csv, expected_trajectories, coords = 'off'):
    # Load video
    cap = cv2.VideoCapture(video_path)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    # Read polygons
    polygons = read_polygons_from_csv(polygons_csv)

    # Process video
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get labels for the current frame
        labels = get_true_labels(frame_num)
        
        # Draw polygons
        draw_polygons(frame, polygons)
        
        # Draw bounding boxes
        draw_bounding_boxes(frame, labels)

        draw_trajectories(frame, expected_trajectories, coords='off')

        # Write the frame into the output file
        out.write(frame)
        
        frame_num += 1

    # Release video capture and writer
    cap.release()
    out.release()


def save_frame(video_path, frame_num, output_path, polygons_csv, expected_trajectories, coords = 'off'):
    # Load video
    cap = cv2.VideoCapture(video_path)

    # Read polygons
    polygons = read_polygons_from_csv(polygons_csv)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Set frame position to the desired frame number
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read frame {frame_num} from video.")
        cap.release()
        return

    # Get labels for the current frame
    labels = get_true_labels(frame_num)
    
    # Draw
    draw_bounding_boxes(frame, labels)
    draw_polygons(frame, polygons)
    draw_trajectories(frame, expected_trajectories, coords='off')
    # Write the frame into the output file
    cv2.imwrite(output_path, frame)
    cap.release()
    print(f"Frame {frame_num} saved as {output_path}.")


def interpolate_trajectory(trajectory, name='1', num_points=100, smoothing_factor = 75, show = False):
    # Separate the coordinates into x and y arrays
    trajectory = [np.array(traj) for traj in trajectory]
    x_original = np.array([coord[0] for coord in trajectory])
    y_original = np.array([coord[1] for coord in trajectory])
    
    time_original = np.arange(len(trajectory))

    spline_x = UnivariateSpline(time_original, x_original, s=smoothing_factor)
    spline_y = UnivariateSpline(time_original, y_original, s=smoothing_factor)


    # Generate new time steps with exactly 100 points
    time_new = np.linspace(0, len(trajectory) - 1, num_points)


    # Interpolate the coordinates
    x_new = spline_x(time_new)
    y_new = spline_y(time_new)

    # Combine the interpolated x and y coordinates back into a list of tuples
    coords_new = list(zip(x_new, y_new))

    if show:
        # Plot the original and interpolated coordinates for visualization
        plt.figure(figsize=(10, 5))
        plt.plot(x_original, y_original, 'o', label='Original Coordinates')
        plt.plot(x_new, y_new, '-', label='Interpolated Coordinates')
        plt.legend()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Interpolation of Coordinates to 100 Points: ' + str(name))
        plt.grid(True)
        plt.show()
        # time.sleep(60)

    return coords_new


def resample_trajectory(trajectory, name='1', num_points=50, smoothing_factor = 10, function='multiquadric', show = False):
    # Separate the coordinates into x and y arrays
    # trajectory = [np.array(traj) for traj in trajectory]
    x_original = np.array([coord[0] for coord in trajectory])
    y_original = np.array([coord[1] for coord in trajectory])

    arc_length = np.cumsum(np.sqrt(np.diff(x_original, prepend=x_original[0])**2 + np.diff(y_original, prepend=y_original[0])**2))
    resampled_arc_length = np.linspace(arc_length[0], arc_length[-1], num_points)

    time_original = np.arange(len(trajectory))
    # Interpolating functions for x and y coordinates
    interp_x = interp1d(arc_length, x_original, kind='linear')
    interp_y = interp1d(arc_length, y_original, kind='linear')

    # New resampled points
    x_resampled = interp_x(resampled_arc_length)
    y_resampled = interp_y(resampled_arc_length)



    # Generate new time steps with exactly 100 points
    time_new = np.linspace(0, len(trajectory) - 1, num_points)


    # Combine the interpolated x and y coordinates back into a list of tuples
    coords_new = list(zip(x_resampled, y_resampled))

    if show:
        # Plot the original and interpolated coordinates for visualization
        plt.figure(figsize=(10, 5))
        plt.plot(x_original, y_original, 'o', label='Original Coordinates')
        plt.plot(x_resampled, y_resampled, '-', label='Interpolated Coordinates')
        plt.legend()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Interpolation of Coordinates to 100 Points: ' + str(name))
        plt.grid(True)
        plt.show()
        # time.sleep(60)

    return coords_new


def get_next_element(var, index):
    return var[index + 1] if index + 1 < len(var) else var[index-1]


def rotate_rectangle(point, dir_point, width):

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


def average_similar_points(all_coords, width, show = True):
    all_points = []
    for coords in all_coords:
        for point in coords:
            all_points.append(point)
    all_points = sort_points(all_points)
    longest = max(all_coords, key=len)
    resampled_longest = resample_trajectory(longest)

    trajectory = []
    windows = []
    for i, point in enumerate(resampled_longest):
        dir_point = get_next_element(resampled_longest, i)
        rotated_corners = rotate_rectangle(point, dir_point, width)
        polygon = Polygon(rotated_corners)
        windows.append(polygon)
        points_within = [[point]]
        for loc in all_points:
            within, _ = is_within(loc, [polygon])
            if within:
                points_within.append(loc)
            print(points_within[0])
            # print(points_within)
        trajectory.append(np.average(points_within, axis=0))
    print(trajectory)
    time.sleep(60)
    inted_trajectory = interpolate_trajectory(trajectory, smoothing_factor=1000)


    
    all_points_x, all_points_y = zip(*all_points)
    inted_longest_x, inted_longest_y = zip(*resampled_longest)
    trajectory_x, trajectory_y = zip(*trajectory)

    if show:
        plt.figure(figsize=(10, 6))
        plt.scatter(all_points_x, all_points_y, color='blue', label='All Points', s=10)
        plt.plot(inted_longest_x, inted_longest_y, color='red', linewidth=2, label='Longest Trajectory')
        plt.plot(trajectory_x, trajectory_y, color='green', linewidth=2, linestyle='--', label='Average Trajectory')
        for window in windows:
            patch = patches.Polygon(list(window.exterior.coords), closed=True, edgecolor='purple', facecolor='none', linewidth=2)
            plt.gca().add_patch(patch)
        plt.title('Trajectory Plot')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.legend()

        # Show the plot
        plt.show()

    return inted_trajectory


def get_middle(x1,y1,x2,y2):
    x=(x1+x2) / 2
    y=(y2+y1)/2
    return (x,y)


def sort_points(points):
    """
    Sort points based on nearest neighbor heuristic.

    Args:
        points (list of tuple): Input trajectory points [(x1, y1), (x2, y2), ...]

    Returns:
        np.ndarray: Sorted trajectory points [(x1, y1), (x2, y2), ...]
    """
    points = np.array(points)
    sorted_points = [points[0]]
    points = np.delete(points, 0, axis=0)
    
    while len(points) > 0:
        last_point = sorted_points[-1]
        distances = np.linalg.norm(points - last_point, axis=1)
        nearest_index = np.argmin(distances)
        sorted_points.append(points[nearest_index])
        points = np.delete(points, nearest_index, axis=0)

    return np.array(sorted_points)


def filter_linked_polygons(d):
    for first_polygon_int, final_polygons in d.items():
        if first_polygon_int in final_polygons:
            del d[first_polygon_int][first_polygon_int]
    
    to_del = []
    for first_polygon_int, final_polygons in d.items():
        if len(final_polygons) == 0:
            to_del.append(first_polygon_int)

    for first_polygon_int in to_del:
        del d[first_polygon_int]

    return d


def add_item(dict, key1, key2, item):
    if key1 not in dict:
        dict[key1] = {}
    dict[key1][key2] = item
    return dict


labels = []
for frame_number in range(1791):
    out = get_true_labels(frame_number)
    labels.append(out)


items = {}
for frame in labels:
    # Iterate through each object (list) in the frame
    for obj in frame:
        # Extract coordinates and id
        x1, y1, x2, y2, obj_id = obj
        # Check if the id already exists in the dictionary
        if obj_id in items:
            # Append coordinates to the existing list for the id
            items[obj_id].append(get_middle(x1,y1,x2,y2))
        else:
            # Create a new list with coordinates for the id
            items[obj_id] = [get_middle(x1,y1,x2,y2)]


results = {}
polygons = read_polygons_from_csv('polygons.csv')
for obj_id in items:
    entered_polygons = set()
    for item in items[obj_id]:
        flag, polygon = is_within(item, polygons)
        if flag:
            entered_polygons.add(polygon)
    results[obj_id] = list(entered_polygons)

first_polygons = {}
last_polygons = {}
coords_transitions = {}

for obj_id, polygons in results.items():
    if polygons:  # Ensure the list is not empty
        first_polygons[obj_id] = polygons[0]
        last_polygons[obj_id] = polygons[-1]

for obj_id in first_polygons:
    first_polygon = first_polygons[obj_id]
    last_polygon = last_polygons[obj_id]
    
    if first_polygon not in coords_transitions:
        coords_transitions[first_polygon] = {}
    if last_polygon not in coords_transitions[first_polygon]:
        coords_transitions[first_polygon][last_polygon] = {"count": 0, "coordinates": []}
    
    coords_transitions[first_polygon][last_polygon]["count"] += 1
    coords_transitions[first_polygon][last_polygon]["coordinates"].append(items[obj_id])
coords_transitions = filter_linked_polygons(coords_transitions)

expected_trajectories = {}
counter = 0
first_polygon = None
for first_polygon, final_polygons in coords_transitions.items():
    for final_polygon, details in final_polygons.items():
        all_coords = details["coordinates"]
        trajectory = average_similar_points(all_coords, 500)
        expected_trajectories = add_item(expected_trajectories, first_polygon, final_polygon, trajectory)

# show_structure(expected_trajectories)


print('saving video')
video_path = 'CAM-HAZELDELL-126THST.mp4'
polygons_csv = 'polygons.csv'
# output_path = 'tester.mp4'
output_path = "tester.jpg"
frame_num = 20  # Frame number to capture (0-indexed)

# save_video_with_polygons(video_path, output_path, polygons_csv, expected_trajectories)
save_frame(video_path, frame_num, output_path, polygons_csv, expected_trajectories)

# polygons = read_polygons_from_csv('polygons.csv')
# expected_trajectories['polygons'] = polygons
# with open('expected_trajectories3.pkl', 'wb') as pkl_file:
#     pickle.dump(expected_trajectories, pkl_file)
# # Example: Loading from a pickle file
# filename = 'data.pkl'

# # Reading data from pickle file
# with open(filename, 'rb') as pkl_file:
#     loaded_data = pickle.load(pkl_file)