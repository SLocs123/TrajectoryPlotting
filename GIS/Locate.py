import cv2
import numpy as np
import osmnx as ox
import osmnx.plot as ox_plot
import matplotlib.pyplot as plt

def get_osm_from_gps(gps, dist=1000):
    """
    Fetches OSM graph for a given GPS coordinate.
    """
    lat, lon = gps
    graph = ox.graph_from_point((lat, lon), dist=dist, network_type='all')
    return graph

def draw_dot(img, pixel, colour=(0, 0, 255)):
    """
    Draw a dot on the image at the given pixel.
    """
    x, y = pixel
    cv2.circle(img, (x, y), 5, colour, -1)

def show_view_points(camera_img, graph, satellite_img=None):
    """
    Shows camera image in one window and OSM (or satellite) image in another window.
    Allows user to click on OSM image and returns clicked GPS coordinates.
    """

    # Convert OSM graph to image
    fig, ax = ox_plot.plot_graph(graph, show=False, close=False)
    fig.canvas.draw()
    osm_img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    osm_img = osm_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    osm_img = osm_img[..., :3]  # Drop alpha channel
    plt.close(fig)

    # Resize OSM image to match camera image height
    cam_h, cam_w = camera_img.shape[:2]
    osm_h, osm_w = osm_img.shape[:2]
    scale = cam_h / osm_h
    osm_img_resized = cv2.resize(osm_img, (int(osm_w * scale), cam_h))

    # Satellite alternative
    if satellite_img is not None:
        sat_img_resized = cv2.resize(satellite_img, (osm_img_resized.shape[1], cam_h))
        osm_or_sat_img = sat_img_resized
        osm_window_name = "Satellite View"
    else:
        osm_or_sat_img = osm_img_resized
        osm_window_name = "OSM View"

    camera_window_name = "Camera View"

    # Track clicks
    clicked_points = []   # [(pixel_x, pixel_y), ...]
    clicked_gps = []      # [(lat, lon), ...]

    # Get OSM bounds to map pixels -> lat/lon
    nodes, edges = ox.graph_to_gdfs(graph)
    lat_min, lat_max = nodes['y'].min(), nodes['y'].max()
    lon_min, lon_max = nodes['x'].min(), nodes['x'].max()
    osm_h_resized, osm_w_resized = osm_or_sat_img.shape[:2]

    # Mouse callback for OSM/Satellite window
    def mouse_callback_osm(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # clicked_points.append((x, y))
            draw_dot(param["img"], (x, y))
            cv2.imshow(osm_window_name, param["img"])
            # Convert pixel -> lat/lon
            lat = lat_max - (y / osm_h_resized) * (lat_max - lat_min)
            lon = lon_min + (x / osm_w_resized) * (lon_max - lon_min)
            clicked_gps.append((lat, lon))
            print(f"Clicked pixel: {(x, y)}, GPS: {(lat, lon)}")

    def mouse_callback_camera(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_points.append((x, y))
            draw_dot(camera_img, (x, y))
            cv2.imshow(camera_window_name, camera_img)
            print(f"Clicked pixel on camera: {(x, y)}")

    # Show both windows
    cv2.imshow(camera_window_name, camera_img)
    osm_img_container = {"img": osm_or_sat_img.copy()}
    cv2.imshow(osm_window_name, osm_img_container["img"])
    cv2.setMouseCallback(osm_window_name, mouse_callback_osm, param=osm_img_container)
    cv2.setMouseCallback(camera_window_name, mouse_callback_camera, param=camera_img)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        if satellite_img is not None:
            if key == ord('s'):
                osm_img_container["img"] = sat_img_resized.copy()
                cv2.imshow(osm_window_name, osm_img_container["img"])
            elif key == ord('o'):
                osm_img_container["img"] = osm_img_resized.copy()
                cv2.imshow(osm_window_name, osm_img_container["img"])

    cv2.destroyAllWindows()
    return clicked_points, clicked_gps

def main():
    # Example usage
    gps = (37.7749, -122.4194)  # San Francisco
    dist = 1000  # 1 km

    graph = get_osm_from_gps(gps, dist)
    
    # Load a sample camera image (replace with actual camera capture)
    camera_img = cv2.imread('GIS/cam01.jpg')
    cv2.putText(camera_img, "Camera View", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    clicked_points, clicked_gps = show_view_points(camera_img, graph)
    print("Clicked points (pixel):", clicked_points)
    print("Clicked GPS coordinates:", clicked_gps)


main()    
