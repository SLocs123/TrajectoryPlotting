from shapely.geometry import Point

def read_json(filename):
    """
    Rebuilds the saved JSON into the same structure returned by pickle:

    {
        input_box (Polygon): {
            output_box (Polygon): {
                'traj': np.ndarray,
                'deltas': np.ndarray,
                'local_occlusion': [Polygon, ...]
            }
        },
        'polygons': [Polygon, Polygon, ...]
    }
    """
    import json
    import numpy as np
    from shapely.wkt import loads as wkt_loads
    with open(filename, "r") as f:
        json_data = json.load(f)

    restored = {}

    # ---------- restore polygons ----------
    polygons = [wkt_loads(wkt) for wkt in json_data.pop("polygons")]
    restored["polygons"] = polygons

    # ---------- restore main structure ----------
    for input_wkt, inner_dict in json_data.items():
        input_box = wkt_loads(input_wkt)

        restored_inner = {}

        for output_wkt, payload in inner_dict.items():
            output_box = wkt_loads(output_wkt)

            rebuilt = {}

            for key, value in payload.items():

                if key in ("traj", "deltas"):
                    rebuilt[key] = np.array(value)

                elif key == "local_occlusion":
                    rebuilt[key] = [wkt_loads(poly_wkt) for poly_wkt in value]

                else:
                    rebuilt[key] = value

            restored_inner[output_box] = rebuilt

        restored[input_box] = restored_inner
    return restored