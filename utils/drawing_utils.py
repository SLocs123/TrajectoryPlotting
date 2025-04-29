import cv2
import numpy as np
from .io_utils import read_image_from_path, read_polygons_from_csv

def draw_all_points(image, points, thickness=2, color=(255, 0, 0), circle_radius=-1):
    """
    Draws circles on the image at the specified points.
    
    Args:
    - image (numpy.ndarray): The image on which to draw the circles.
    - points (list): A list of points, where each point is a tuple (x, y).
    
    Returns:
    - None: The function modifies the input image in-place.
    """
    for point in points:
        x, y = point[0][0], point[0][1]
        # Draw a filled circle at (x, y)
        cv2.circle(image, (int(x), int(y)), thickness, color, circle_radius)

def draw_bounding_boxes(frame, labels):
    """
    Draws bounding boxes on the frame and labels them with the car ID.
    
    Args:
    - frame (numpy.ndarray): The image/frame on which to draw the bounding boxes.
    - labels (list): A list of labels, where each label is a list [min_x, min_y, max_x, max_y, car_id].
    
    Returns:
    - None: The function modifies the input frame in-place.
    """
    for label in labels:
        min_x, min_y, max_x, max_y, car_id = label
        # Draw a bounding box
        cv2.rectangle(frame, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (255, 0, 0), 2)
        # Add car ID text
        cv2.putText(frame, str(car_id), (int(min_x), int(min_y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
def draw_polygons(frame, polygons):
    """
    Draws polygons on the frame and labels them with their centroid coordinates.
    
    Args:
    - frame (numpy.ndarray): The image/frame on which to draw the polygons.
    - polygons (list): A list of Shapely Polygon objects.
    
    Returns:
    - None: The function modifies the input frame in-place.
    """
    for polygon in polygons:
        # Convert polygon points to the required format for OpenCV
        points = np.array([list(coord) for coord in polygon.exterior.coords], np.int32).reshape((-1, 1, 2))
        # Draw the polygon on the frame
        cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # Get the centroid of the polygon
        centroid = polygon.centroid
        label = f"({centroid.x:.1f}, {centroid.y:.1f})"
        label_position = (int(centroid.x), int(centroid.y))
        
        # Label the centroid position
        cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        
def draw_trajectories(frame, expected_trajectories, coords=False):
    """
    Draws trajectories on the frame, and labels each trajectory with an index.
    
    Args:
    - frame (numpy.ndarray): The image/frame on which to draw the trajectories.
    - expected_trajectories (dict): A dictionary of trajectories where keys are trajectory identifiers 
                                    and values are dictionaries of polygon and corresponding trajectory points.
    - coords (Bool): Optional. If True, will display trajectory coordinates. Defaults to False.
    
    Returns:
    - None: The function modifies the input frame in-place.
    """
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
        color = label_colors[color_index]

        for in_index, (final_polygon, trajectory) in enumerate(inner_dict.items()):
            trajectory = trajectory[:, 0]
            # Draw the trajectory points
            for point in trajectory:
                cv2.circle(frame, (int(point[0]), int(point[1])), 2, color, -1)
            # Draw lines between consecutive points
            for i in range(len(trajectory) - 1):
                cv2.line(frame, (int(trajectory[i][0]), int(trajectory[i][1])), (int(trajectory[i+1][0]), int(trajectory[i+1][1])), color, 1)
            
            # Draw the label for the trajectory
            x, y = trajectory[0][0], trajectory[0][1]
            label_text = f"Index: 1st: {out_index}, 2nd: {in_index}"
            label_position = (int(x), int(y))
            font_scale = 1.0
            font_color = (255, 255, 255)
            thickness = 2
            line_type = cv2.LINE_AA
            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # Compute background for text
            background_left = label_position[0]
            background_top = label_position[1] - text_height - baseline
            background_right = label_position[0] + text_width
            background_bottom = label_position[1]
            # cv2.rectangle(frame, (int(background_left), int(background_top)), (int(background_right), int(background_bottom)), (0, 0, 255), cv2.FILLED)
            
            # Draw the text label
            cv2.putText(frame, label_text, label_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, thickness, line_type)
            
            # Show coordinates if 'coords' is 'on'
            if coords:
                for point in trajectory:
                    coord_text = f"({int(point[0])}, {int(point[1])})"
                    coord_position = (int(point[0]), int(point[1]) - 10)  # Place the label just above the point
                    cv2.putText(frame, coord_text, coord_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    
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
    # Read the frame/image from the specified path (if video, extract the frame_num)
    frame = read_image_from_path(video_path)

    # Now process the frame/image with polygons, labels, and trajectories
    polygons = read_polygons_from_csv(polygons_csv)
    # labels = get_true_labels(frame_num)
    
    # draw_bounding_boxes(frame, labels)
    draw_polygons(frame, polygons)
    draw_trajectories(frame, expected_trajectories, coords=coords)

    # Save the processed frame/image
    cv2.imwrite(output_path, frame)
    print(f"Frame {frame_num} saved as {output_path}.")