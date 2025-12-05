from utils.io_utils import read_polygons_from_csv, save_trajs
from utils.trajectory_analysis import create_expected_trajectories
from utils.drawing_utils import save_frame
from utils.polygon_utils import redraw_poly, draw_occlusion
import time

print("Imports completed.")
video_path = 'assets/cam04.jpg'
output_path = 'output/cam04_occlusion_test.jpg'
polygons_csv = 'assets/cam04_poly_new.csv'
SCT_label = 'assets/cam04_SCT_no108.txt'

print('Reading polygons from CSV...')


flag = input("Do you want to adjust polygons? (y/n): ").strip().lower()
yes = ['yes', 'y']
if flag in yes:
    polygons, polygons_csv = redraw_poly(polygons_csv, video_path, SCT_label) 
else:
    polygons = read_polygons_from_csv(polygons_csv)
    
flag_2 = input('Do you want to define occlusion areas? (y/n): ').strip().lower()
yes = ['yes', 'y']
if flag_2 in yes:
    occlusion = draw_occlusion(video_path, SCT_label)
else:
    flag_2 = input('Do you want to attempt automatic occlusion areas? (y/n): ').strip().lower()
    if flag_2 in yes:
        pass # automatic occlusion area detection code here

# If no occlusion areas are defined, proceed without them, if areas are defined, pass them to create_expected_trajectories, any trajectories passing through occlusion areas will labeled with occlusion zones.

# polygons = read_polygons_from_csv(polygons_csv)

start = time.time()
print('creating expected trajectories...')
expected_trajectories = create_expected_trajectories(polygons, SCT_label, DTW=True) # occlusion_areas=occlusion_areas
end = time.time()
print(f'Expected trajectories created in {end - start:.2f} seconds.')
print('Saving frame...')
save_frame(video_path, output_path, polygons_csv, expected_trajectories)

save_flag = input("Do you want to save the expected trajectories? (y/n): ").strip().lower()
if save_flag in yes:
    save_trajs(expected_trajectories, polygons)
