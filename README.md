# Trajectory Plotting with QuickTrack

This repository supports the QuickTrack method for trajectory-based tracking. It processes video or image inputs along with corresponding polygon zoning (created in-app) and SCT labeling.

## Usage Instructions

1. **Convert Labelbox Polygon Zones**  
    If using Labelbox polygon zones, run `labelbox_to_polygon.py` with the input Labelbox `.njson` file (excluding metadata). This will convert the labels into a CSV file.

2. **Prepare Ground Truth Labels**  
    Ensure you have ground truth labels for tracking, either in `yolo.txt` format or using Labelbox.

3. **Set Up Paths**  
    Update the paths at the start of `main.py` to point to your input files.

4. **Edit Polygons (Optional)**  
    - When prompted, you can edit the polygons. This will display all detections and allow you to refine your zones.
        - Click in a polygon to start editing, then click again to add polygon points.
        - Enter will save the change to polygon variable, Backspace will cancel any changes.
        - Press Escape to close the window, this will not remove the changes made.
        - Currently new polygons cannot be made, only edited.

5. **Save Changes**  
    - If you save your changes, the `.csv` file will be updated.  
    - If you choose not to save, the updated zones will still be used for the current run but won't persist.

6. **Save Results**  
    At the end of the process, you will be asked if you want to save the results. This will save the trajectories and polygon zones in a `.pkl` file.

7. **Running Tracking**
    KF_Traj contains the necassary code to run tracking with these trajectories, but many options are available (remember to always pop polygons before running to ensure correct format).

## Notes

- Ensure all required dependencies are installed before running the scripts, poetry.lock is provided.
- Refer to the code comments for additional details and troubleshooting.
