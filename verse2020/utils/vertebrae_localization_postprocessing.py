
from copy import deepcopy

import numpy as np
from utils.landmark.common import Landmark


def reshift_landmarks(curr_landmarks):
    if (not curr_landmarks[0].is_valid) and curr_landmarks[7].is_valid:
        if (not curr_landmarks[6].is_valid) and curr_landmarks[5].is_valid:
            # shift c indizes up
            print('shift c indizes up')
            curr_landmarks = [Landmark([np.nan] * 3, is_valid=False)] + curr_landmarks[0:5] + curr_landmarks[6:26]
    if (not curr_landmarks[7].is_valid) and curr_landmarks[19].is_valid:
        if (not curr_landmarks[18].is_valid) and curr_landmarks[17].is_valid:
            # shift l indizes up
            print('shift t indizes up')
            curr_landmarks = curr_landmarks[0:7] + [Landmark([np.nan] * 3, is_valid=False)] + curr_landmarks[7:18] + curr_landmarks[19:26]
        elif curr_landmarks[25].is_valid:
            # shift l indizes down
            print('shift t indizes down')
            curr_landmarks = curr_landmarks[0:7] + curr_landmarks[8:19] + [curr_landmarks[25]] + curr_landmarks[19:25] + [Landmark([np.nan] * 3, is_valid=False)]
    return curr_landmarks


def filter_landmarks_top_bottom(curr_landmarks, input_image):
    image_extent = [spacing * size for spacing, size in zip(input_image.GetSpacing(), input_image.GetSize())]
    filtered_landmarks = []
    z_distance_top_bottom = 10
    for l in curr_landmarks:
        if z_distance_top_bottom < l.coords[2] < image_extent[2] - z_distance_top_bottom:
            filtered_landmarks.append(l)
        else:
            filtered_landmarks.append(Landmark(coords=[np.nan] * 3, is_valid=False))
    return filtered_landmarks


def add_landmarks_from_neighbors(local_maxima_landmarks):
    local_maxima_landmarks = deepcopy(local_maxima_landmarks)
    duplicate_penalty = 0.1
    for i in range(2, 6):
        local_maxima_landmarks[i + 1].extend([Landmark(coords=l.coords, value=l.value * duplicate_penalty) for l in local_maxima_landmarks[i]])
        local_maxima_landmarks[i].extend([Landmark(coords=l.coords, value=l.value * duplicate_penalty) for l in local_maxima_landmarks[i + 1]])
    for i in range(8, 18):
        local_maxima_landmarks[i + 1].extend([Landmark(coords=l.coords, value=l.value * duplicate_penalty) for l in local_maxima_landmarks[i]])
        local_maxima_landmarks[i].extend([Landmark(coords=l.coords, value=l.value * duplicate_penalty) for l in local_maxima_landmarks[i + 1]])
    local_maxima_landmarks[25].extend([Landmark(coords=l.coords, value=l.value) for l in local_maxima_landmarks[18]])
    local_maxima_landmarks[18].extend([Landmark(coords=l.coords, value=l.value) for l in local_maxima_landmarks[25]])
    for i in range(20, 24):
        local_maxima_landmarks[i + 1].extend([Landmark(coords=l.coords, value=l.value * duplicate_penalty) for l in local_maxima_landmarks[i]])
        local_maxima_landmarks[i].extend([Landmark(coords=l.coords, value=l.value * duplicate_penalty) for l in local_maxima_landmarks[i + 1]])
    return local_maxima_landmarks
