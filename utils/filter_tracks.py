# filter broken tracks

"""
identifiers:
- intitialise track in middle of scene
- track ends in middle of scene
- track is too short, must find best way to determine, some tracks will be naturally shorter
- track ends close to another track that continues, suggesting error
- track starts at middle of another track
"""

import cv2

def start_in_middle(track, image_size, buffer_coeff=0.2):
    x,y = track[0]
    buffer_x = image_size[0] * buffer_coeff
    buffer_y = image_size[1] * buffer_coeff

    if x > buffer_x and x < image_size[0] - buffer_x and y > buffer_y and y < image_size[1] - buffer_y:
        return True
    return False

def end_in_middle(track, image_size, buffer_coeff=0.2):
    x,y = track[-1]
    buffer_x = image_size[0] * buffer_coeff
    buffer_y = image_size[1] * buffer_coeff

    if x > buffer_x and x < image_size[0] - buffer_x and y > buffer_y and y < image_size[1] - buffer_y:
        return True
    return False

def too_short(track, min_length=20):
    """
    Could use current min_length threshold, but a MAD approach would likely b emore generalised.
    def too_short(all_tracks, track, threshold_coeff=1.5):
    """
    if len(track) < min_length:
        return True
    return False

def end_close_to_other(track, other_tracks, threshold):
    """
    Checks to see if the end point of the track is within the middle 30% of another track,
    suggesting that the track has ended in the middle of the scene
    """
    end_point = track[-1]
    for other_track in other_tracks:
        total_points = len(other_track)
        closest_point, distance, index = get_closest_point(end_point, other_track)
        if distance < threshold and index > total_points * 0.2 and index < total_points * 0.8:
            return True
    return False

def start_close_to_other(track, other_tracks, threshold):
    """
    Checks to see if the start point of the track is within the middle of another track,
    that it is close to. This would suggest that the track has started in the middle of the scene, and is likely broken.
    """
    start_point = track[0]
    for other_track in other_tracks:
        total_points = len(other_track)
        closest_point, distance, index = get_closest_point(start_point, other_track)
        if distance < threshold and index > total_points * 0.2 and index < total_points * 0.8:
            return True
    return False
    
def get_closest_point(point, track):
    """
    Return the closet point along a track, the distance and the index of the point along the track
    """
    closest_point = None
    min_distance = float('inf')
    closest_index = -1
    index = 0
    for track_point in track:
        distance = ((point[0] - track_point[0])**2 + (point[1] - track_point[1])**2)**0.5
        if distance < min_distance:
            min_distance = distance
            closest_point = track_point
            closest_index = index
        index += 1
    return closest_point, min_distance, closest_index


# Extra functions:
# Arc length
# num_points (too short copy)
# start to end displacnement (might need tight threshold here to avoid false positives)
# eff = disp / (L + eps) path efficiency (jitter detector) (displacement over arc length, if the track is very jittery, the efficiency will be low, and it may be a broken track, similar to displacement)
# logging metrics can help to reduce the "heavy tail" and get a nicer distribution of track lengths
# logL = log(L + eps)
# logDisp = log(disp + eps)
# logN = log(n_pts)

# This could be changed similar to clustering, where tracks are discarded if they are not similar to any other,
# careful if multiple tracks are broken in the same way, they may be similar to each other but not to any good tracks, so this would need to be carefully designed, and may not be worth it.
# after initial rule-based filtering, could then remove outliers based on track similarity, some track may be unique but still good, 
# clustering need at least 2 tracks anyway, so could work as purely unique would still be removed later.

def ask_yes_no(prompt="Continue? [y/n]: "):
    while True:
        choice = input(prompt).strip().lower()
        if choice in ("y", "yes"):
            return True
        if choice in ("n", "no"):
            return False
        print("Please enter 'y' or 'n'.")

def filter_broken_tracks(tracks, image_size, min_length, threshold):
    """
    checks a number of broken track gates, if track is too short it is filtered, 
    if it is consered a broken/half track it is added to occlusion group
    """
    filtered_tracks = []
    broken_tracks = []
    short_tracks = []
    for track in tracks:
        start, end, end_close, start_close, short = start_in_middle(track, image_size), end_in_middle(track, image_size), end_close_to_other(track, tracks, threshold), start_close_to_other(track, tracks, threshold), too_short(track, min_length)
        print(f'Short: {short}, start_middle_img: {start}, end_middle_img: {end}, start_mid_traj: {start_close}, end_mid_traj: {end_close}')
        
        img_temp = cv2.imread('assets/cam04.jpg')
        for point in track:
            cv2.circle(img_temp, (int(point[0]), int(point[1])), 3, (0,0,155), -1)
        cv2.imwrite(f'temp_check.jpg', img_temp)

        choise = ask_yes_no()

        if not choise:
            raise KeyboardInterrupt

        if short:
            short_tracks.append(track)
            continue
        if start_close and start:
            broken_tracks.append(track)
            continue
        if end_close and end:
            broken_tracks.append(track)
            continue
        
        # if start and end: # this may be invalid
        #     continue

        filtered_tracks.append(track)
    return filtered_tracks, broken_tracks, short_tracks


