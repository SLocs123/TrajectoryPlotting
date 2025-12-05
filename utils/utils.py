from shapely.geometry import Point, Polygon

def add_item(nested_dict, key1, key2, item, local_occ, deltas):
    """
    Adds an item to a nested dictionary structure.

    Parameters:
        nested_dict (dict): The main dictionary to update.
        key1 (hashable): First-level key.
        key2 (hashable): Second-level key under key1.
        item (Any): The item to store at nested_dict[key1][key2].

    Returns:
        dict: Updated nested dictionary.
    """
    
    if key1 not in nested_dict:
        nested_dict[key1] = {}
    nested_dict[key1][key2] = {'trajectory': item, 'local_occlusion': local_occ, 'deltas': deltas}
    return nested_dict

def show_structure(nested_dict, indent=0):
    """
    Recursively prints the structure and types of values in a nested dictionary.

    Parameters:
        nested_dict (dict): The dictionary to inspect.
        indent (int): Current indentation level (used internally for recursion).
    """
    for key, value in nested_dict.items():
        print(' ' * indent + f'{key} ({type(value).__name__})')
        if isinstance(value, dict):
            show_structure(value, indent + 2)

def get_next_element(seq, index):
    """
    Returns the next element in a sequence if available; otherwise, returns the previous one.

    Parameters:
        seq (list): The list or sequence to access.
        index (int): The current index in the list.

    Returns:
        Any: The next element if possible, otherwise the previous element. If index is 0 and the list has only one element, returns that element.
    """
    if index + 1 < len(seq):
        return seq[index + 1]
    elif index > 0:
        return seq[index - 1]
    else:
        return seq[0]

def is_within(xy, polygons):
    """
    Checks if a 2D point lies within any polygon in a list.

    Parameters:
        xy (tuple): The (x, y) coordinates of the point.
        polygons (list): A list of shapely.geometry.Polygon objects.

    Returns:
        tuple:
            - bool: True if the point is within any polygon, False otherwise.
            - Polygon or None: The containing polygon, or None if not found.
    """
    x, y = xy
    point = Point(x, y)
    for polygon in polygons:
        if polygon.contains(point):
            return True, polygon
    return False, None

# def contains_empty_or_nan(lst):
#     for item in lst:
#         if isinstance(item, (list, np.ndarray)):
#             if any(pd.isna(x) or x == '' for x in item):
#                 return True
#         elif pd.isna(item) or item == '':
#             return True
#     return False