from utils.io_utils import read_polygons_from_csv
from utils.trajectory_analysis import create_expected_trajectories
from utils.drawing_utils import save_frame
from utils.polygon_utils import redraw_poly
import time

print("Imports completed.")
video_path = 'assets/cam04.jpg'
output_path = 'output/test.jpg'
polygons_csv = 'assets/cam04_poly_new.csv'
SCT_label = 'assets/cam04_SCT.txt'

print('Reading polygons from CSV...')


flag = input("Do you want to adjust polygons? (y/n): ").strip().lower()
yes = ['yes', 'y']
if flag in yes:
    polygons = redraw_poly(polygons_csv, video_path, SCT_label)
else:
    polygons = read_polygons_from_csv(polygons_csv)

start = time.time()
print('creating expected trajectories...')
expected_trajectories = create_expected_trajectories(polygons, SCT_label)
end = time.time()
print(f'Expected trajectories created in {end - start:.2f} seconds.')
print('Saving frame...')
save_frame(video_path, output_path, polygons_csv, expected_trajectories)
