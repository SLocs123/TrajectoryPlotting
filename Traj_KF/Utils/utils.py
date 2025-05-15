import pickle
from shapely.geometry import Point

def read_pkl(self, traj_dir):
        with open(traj_dir, 'rb') as pkl_file:
            loaded_data = pickle.load(pkl_file) 
        polygon_set = loaded_data
        return polygon_set

def is_within(xy, polygons):
    """
    Determine if a given point lies within any of the provided polygons.
    
    Args:
        xy (tuple): Coordinates of the point as (x, y).
        polygons (list): A list of shapely.geometry.Polygon objects.
    
    Returns:
        shapely.geometry.Polygon or None: The polygon containing the point, 
        or None if the point is not within any polygon.
    """
    point = Point(xy[0], xy[1])
    for polygon in polygons:
        if polygon.contains(point):
            return polygon