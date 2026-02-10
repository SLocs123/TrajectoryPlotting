import cameratransform as ct
from read_in_json import read_json

def transform_trajectory(trajectory, camera_path):
    trajectory = read_json('cam04_traj_seperated.json')
    polys = trajectory.pop('polygons')
    camera = ct.load_camera(camera_path)    

    # for poly in trajectory.values():
    #     for inner in poly.values():
            # TRAJECTORY IS HERE
            
       
    print(camera.gpsFromImage([2000,1800], Z=0.5))  # type: ignore
    

    return trajectory

if __name__ == "__main__":
    transform_trajectory(None, 'cam04_fitted_cam.json')