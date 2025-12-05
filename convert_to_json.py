import json
import numpy as np
from shapely.geometry import shape, mapping
from shapely.wkt import loads as wkt_loads, dumps as wkt_dumps
import pickle

def serialize_shapely(obj):
    # Convert shapely geometry to WKT string (easier to save as key)
    return wkt_dumps(obj)

def deserialize_shapely(wkt_str):
    return wkt_loads(wkt_str)

def serialize_data(data):
    """
    data: dict where keys are shapely objects (input boxes),
          values are dicts with shapely keys (output boxes),
          and values that include numpy arrays.
    """
    ser_outer = {}
    for input_box, inner_dict in data.items():
        input_key = serialize_shapely(input_box)  # convert outer key to WKT string
        
        ser_inner = {}
        for output_box, traj_array in inner_dict.items():
            output_key = serialize_shapely(output_box)  # convert inner key to WKT string
            
            # Convert numpy array to list
            if isinstance(traj_array, np.ndarray):
                traj_list = traj_array.tolist()
            else:
                traj_list = traj_array  # fallback, just in case
            
            ser_inner[output_key] = traj_list
        
        ser_outer[input_key] = ser_inner
    
    return ser_outer

def save_to_json(data, filename):
    """
    Save the serialized data to a JSON file.
    """
    serialized_data = serialize_data(data)

    # Add polygon list as WKT strings
    serialized_data['polygons'] = [wkt_dumps(poly) for poly in polygons]

    with open(filename, 'w') as f:
        json.dump(serialized_data, f, indent=2)

def read_pkl(traj_dir):
        with open(traj_dir, 'rb') as pkl_file:
            loaded_data = pickle.load(pkl_file) 
        polygon_set = loaded_data
        return polygon_set

pkl_dir = 'cam04_traj_redo.pkl'
polygon_set = read_pkl(pkl_dir)
polygons = polygon_set.pop('polygons')

save_to_json(polygon_set, 'cam04_traj_redo.json')







# def deserialize_data(data):
#     """
#     Revert the serialized dict back to original:
#     - convert WKT strings to shapely objects as keys
#     - convert lists back to numpy arrays as values
#     """
#     des_outer = {}
#     for input_key, inner_dict in data.items():
#         input_box = deserialize_shapely(input_key)
        
#         des_inner = {}
#         for output_key, traj_list in inner_dict.items():
#             output_box = deserialize_shapely(output_key)
            
#             traj_array = np.array(traj_list)
            
#             des_inner[output_box] = traj_array
        
#         des_outer[input_box] = des_inner
    
#     return des_outer