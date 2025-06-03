import pickle
from shapely.geometry import Point

def read_pkl(traj_dir):
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
    return None

def get_obj_att(obj, att):
    """
    Retrieve the attribute value from an object.

    Args:
        obj: The object from which to retrieve the attribute.
        att (str): The name of the attribute to retrieve.

    Returns:
        The value of the specified attribute, or None if the attribute does not exist.
    """
    if hasattr(obj, att):
        return getattr(obj, att)
    else:
        raise AttributeError(f"Track object has no attribute '{att}'")