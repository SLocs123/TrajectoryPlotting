from utils.io_utils import read_trajs, read_image_from_path
from utils.drawing_utils import save_frame


pkl_dir = ''
image_path = ''
out_dir = 'output/test.jpg'
polygons_csv = 'assets/cam04_poly_new.csv'

img = read_image_from_path(image_path)
trajs = read_trajs(pkl_dir)

save_frame(image_path, out_dir, trajs, polygons_csv)