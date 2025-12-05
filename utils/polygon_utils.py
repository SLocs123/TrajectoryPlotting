import cv2
import numpy as np
from shapely.geometry import Point, Polygon
from .io_utils import read_image_from_path, read_polygons_from_csv, write_polygons_to_csv
from .trajectory_utils import get_all_points
from .drawing_utils import draw_polygons, draw_all_points
from .trajectory_analysis import create_vehicle_trajectories

def redraw_poly(polygons_csv, video_path, label_path):
    image = read_image_from_path(video_path)
    backup_image = image.copy()
    polygons = read_polygons_from_csv(polygons_csv)
    vehicle_traj = create_vehicle_trajectories(label_path)
    Every_Point = get_all_points(vehicle_traj)
    draw_all_points(image, Every_Point)

    editing_state = {
        "index": None,
        "new_coords": []
    }

    # Get screen resolution and compute initial zoom
    screen_res = (1280, 720)  # fallback
    # try:
    #     cv2.namedWindow("_temp", cv2.WINDOW_NORMAL)
    #     cv2.setWindowProperty("_temp", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    #     dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    #     cv2.imshow("_temp", dummy)
    #     cv2.waitKey(1)
    #     screen_res = cv2.getWindowImageRect("_temp")[2:]
    #     cv2.destroyWindow("_temp")
    # except:
    #     pass

    screen_width, screen_height = screen_res
    img_height, img_width = image.shape[:2]
    zoom = min(screen_width / img_width, screen_height / img_height, 1.0)

    pan_x, pan_y = 0, 0
    dragging = False
    last_mouse_pos = (0, 0)

    def transform_point(x, y):
        return int((x - pan_x) / zoom), int((y - pan_y) / zoom)

    def inverse_transform_point(x, y):
        return int(x * zoom + pan_x), int(y * zoom + pan_y)

    def click_event(event, x, y, flags, param):
        nonlocal dragging, last_mouse_pos

        if event == cv2.EVENT_LBUTTONDOWN:
            dragging = True
            last_mouse_pos = (x, y)

            x_img, y_img = transform_point(x, y)
            point = Point(x_img, y_img)

            if editing_state["index"] is not None:
                editing_state["new_coords"].append((x_img, y_img))
            else:
                for idx, poly in enumerate(polygons):
                    if poly.contains(point):
                        print(f"Polygon {idx} clicked! Start redefining it.")
                        editing_state["index"] = idx
                        editing_state["new_coords"] = []
                        break

        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False

        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            dx = x - last_mouse_pos[0]
            dy = y - last_mouse_pos[1]
            pan_x += dx
            pan_y += dy
            last_mouse_pos = (x, y)

    cv2.namedWindow("Shapely Polygon Editor")
    cv2.setMouseCallback("Shapely Polygon Editor", click_event)

    while True:
        h, w = image.shape[:2]
        display_image = cv2.resize(image, (int(w * zoom), int(h * zoom)))
        display_image = display_image.copy()

        draw_polygons(display_image, [Polygon([inverse_transform_point(*pt) for pt in poly.exterior.coords]) for poly in polygons])

        if editing_state["new_coords"]:
            for pt in editing_state["new_coords"]:
                px, py = inverse_transform_point(*pt)
                cv2.circle(display_image, (px, py), 3, (255, 0, 0), -1)

        cv2.imshow("Shapely Polygon Editor", display_image)
        key = cv2.waitKey(1)

        if key == 27:
            break
        elif key == 13:
            if editing_state["index"] is not None and len(editing_state["new_coords"]) >= 3:
                polygons[editing_state["index"]] = Polygon(editing_state["new_coords"])
                print(f"Polygon {editing_state['index']} updated.")
            else:
                print("Not enough points to create a valid polygon.")
            editing_state["index"] = None
            editing_state["new_coords"] = []
        elif key == 8:
            print(f"Editing of polygon {editing_state['index']} cancelled.")
            editing_state["index"] = None
            editing_state["new_coords"] = []
        elif key == ord('+') or key == ord('='):
            zoom *= 1.1
        elif key == ord('-'):
            zoom /= 1.1

    cv2.destroyAllWindows()
    yes = ['yes', 'y']
    save_polys = input("Do you want to save the polygons? (y/n): ").strip().lower()
    if save_polys in yes:
        polygons_csv = input("Enter the path to save the polygons: ").strip()
        write_polygons_to_csv(polygons, polygons_csv)
        print(f"Polygons saved to {polygons_csv}.")
        review= input('Review Polygons? (y/n): ').strip().lower()
        if review in yes:
            draw_polygons(backup_image, polygons)
            cv2.imshow("Shapely Polygon Editor", backup_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    return polygons, polygons_csv



def draw_occlusion(video_path, label_path):
    image = read_image_from_path(video_path)
    backup_image = image.copy()
    occlusion_polygons = []
    vehicle_traj = create_vehicle_trajectories(label_path)
    Every_Point = get_all_points(vehicle_traj)
    draw_all_points(image, Every_Point)

    editing_state = {
        "index": None,
        "new_coords": []
    }

    # Get screen resolution and compute initial zoom
    screen_res = (1280, 720)  # fallback
    # try:
    #     cv2.namedWindow("_temp", cv2.WINDOW_NORMAL)
    #     cv2.setWindowProperty("_temp", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    #     dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    #     cv2.imshow("_temp", dummy)
    #     cv2.waitKey(1)
    #     screen_res = cv2.getWindowImageRect("_temp")[2:]
    #     cv2.destroyWindow("_temp")
    # except:
    #     pass

    screen_width, screen_height = screen_res
    img_height, img_width = image.shape[:2]
    zoom = min(screen_width / img_width, screen_height / img_height, 1.0)

    pan_x, pan_y = 0, 0
    dragging = False
    last_mouse_pos = (0, 0)

    def transform_point(x, y):
        return int((x - pan_x) / zoom), int((y - pan_y) / zoom)

    def inverse_transform_point(x, y):
        return int(x * zoom + pan_x), int(y * zoom + pan_y)

    def click_event(event, x, y, flags, param):
        nonlocal dragging, last_mouse_pos

        if event == cv2.EVENT_LBUTTONDOWN:
            dragging = True
            last_mouse_pos = (x, y)

            x_img, y_img = transform_point(x, y)
            point = Point(x_img, y_img)

            if editing_state["index"] is not None:
                editing_state["new_coords"].append((x_img, y_img))
            else:
                for idx, poly in enumerate(occlusion_polygons):
                    if poly.contains(point):
                        print(f"Polygon {idx} clicked! Start redefining it.")
                        editing_state["index"] = idx
                        editing_state["new_coords"] = []
                        break

        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False

        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            dx = x - last_mouse_pos[0]
            dy = y - last_mouse_pos[1]
            pan_x += dx
            pan_y += dy
            last_mouse_pos = (x, y)

    cv2.namedWindow("Shapely occlusion drawer")
    cv2.setMouseCallback("Shapely Polygon Editor", click_event)

    while True:
        h, w = image.shape[:2]
        display_image = cv2.resize(image, (int(w * zoom), int(h * zoom)))
        display_image = display_image.copy()

        # draw_polygons(display_image, [Polygon([inverse_transform_point(*pt) for pt in poly.exterior.coords]) for poly in polygons])

        if editing_state["new_coords"]:
            for pt in editing_state["new_coords"]:
                px, py = inverse_transform_point(*pt)
                cv2.circle(display_image, (px, py), 3, (255, 0, 0), -1)

        cv2.imshow("Shapely Polygon Editor", display_image)
        key = cv2.waitKey(1)

        if key == 27:
            break
        elif key == 13:
            if editing_state["index"] is not None and len(editing_state["new_coords"]) >= 3:
                occlusion_polygons[editing_state["index"]] = Polygon(editing_state["new_coords"])
                print(f"Polygon {editing_state['index']} updated.")
            else:
                print("Not enough points to create a valid polygon.")
            editing_state["index"] = None
            editing_state["new_coords"] = []
        elif key == 8:
            print(f"Editing of polygon {editing_state['index']} cancelled.")
            editing_state["index"] = None
            editing_state["new_coords"] = []
        elif key == ord('+') or key == ord('='):
            zoom *= 1.1
        elif key == ord('-'):
            zoom /= 1.1

    cv2.destroyAllWindows()
    yes = ['yes', 'y']
    save_polys = input("Do you want to save the polygons? (y/n): ").strip().lower()
    if save_polys in yes:
        polygons_csv = input("Enter the path to save the polygons: ").strip()
        write_polygons_to_csv(occlusion_polygons, polygons_csv)
        print(f"Polygons saved to {polygons_csv}.")
        review= input('Review Polygons? (y/n): ').strip().lower()
        if review in yes:
            draw_polygons(backup_image, occlusion_polygons)
            cv2.imshow("Shapely Polygon Editor", backup_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    return occlusion_polygons, polygons_csv