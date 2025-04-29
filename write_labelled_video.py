from transform_label_json import transform_labelstudio_input, x_to_bbox, bbox_to_z
import cv2

cap = cv2.VideoCapture('CAM-HAZELDELL-126THST.mp4')
# Get the width and height of the frames
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
output_video_path = 'SAE-PhD-overveiw-labelled.mp4'
fps = 30  # Frames per second
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

label_json = transform_labelstudio_input('label_studio_CAM-HAZELDELL-126THST.json', 0)
def get_true_labels(frame_num):
    current_labels = label_json[frame_num]
    current_formatted_labels = []
    for elem in current_labels:
        current_formatted_labels.append(
            [elem["min_x"] * 38.4, elem["min_y"] * 21.6, elem["max_x"] * 38.4, elem["max_y"] * 21.6, elem["car_id"]]
        )
    return current_formatted_labels


def draw_bounding_boxes(frame, labels):
    for label in labels:
        min_x, min_y, max_x, max_y, car_id = label
        cv2.rectangle(frame, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (255, 0, 0), 2)
        # cv2.putText(frame, str(car_id), (int(min_x), int(min_y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

frame_number = 0
while cap.isOpened():
    print(f'Processing frame {frame_number}', end='\r')
    ret, frame = cap.read()
    if not ret:
        break
    labels = get_true_labels(frame_number)
    frame_number += 1
    draw_bounding_boxes(frame, labels)
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()