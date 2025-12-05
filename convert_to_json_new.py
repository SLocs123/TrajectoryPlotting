import json
import numpy as np
from shapely.wkt import loads as wkt_loads, dumps as wkt_dumps
from shapely.geometry.base import BaseGeometry
import pickle


def serialize_shapely(obj):
    return wkt_dumps(obj)


def deserialize_shapely(wkt_str):
    return wkt_loads(wkt_str)


def _to_json_friendly(value):
    """Recursively convert numpy + shapely types inside dicts/lists/tuples."""
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, BaseGeometry):
        return wkt_dumps(value)

    if isinstance(value, dict):
        return {k: _to_json_friendly(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_to_json_friendly(v) for v in value]

    return value


def serialize_data(data):
    ser_outer = {}
    for input_box, inner_dict in data.items():
        input_key = serialize_shapely(input_box)

        ser_inner = {}
        for output_box, inner_value in inner_dict.items():
            output_key = serialize_shapely(output_box)
            ser_inner[output_key] = _to_json_friendly(inner_value)

        ser_outer[input_key] = ser_inner

    return ser_outer


def save_to_json(data, filename):
    serialized_data = serialize_data(data)
    serialized_data['polygons'] = [wkt_dumps(poly) for poly in polygons]

    with open(filename, 'w') as f:
        json.dump(serialized_data, f, indent=2)


def read_pkl(traj_dir):
    with open(traj_dir, 'rb') as pkl_file:
        return pickle.load(pkl_file)

pkl_dir = 'cam04_with_occlusion_delta.pkl'
polygon_set = read_pkl(pkl_dir)
polygons = polygon_set.pop('polygons')

save_to_json(polygon_set, 'cam04_traj_occlusion_delta.json')
