import json
import os
from glob import glob

import numpy as np
from utils.io.image import read_meta_data
from utils.io.landmark import save_points_csv
from utils.io.text import save_dict_csv
from utils.landmark.common import Landmark


def save_valid_landmarks_list(landmarks_dict, filename):
    """
    Saves the valid landmarks per image id to a file.
    :param landmarks_dict: A dictionary of valid landmarks per image id.
    :param filename: The filename to where to save.
    """
    valid_landmarks = {}
    for image_id, landmarks in landmarks_dict.items():
        current_valid_landmarks = []
        for landmark_id, landmark in enumerate(landmarks):
            if landmark.is_valid:
                current_valid_landmarks.append(landmark_id)
        valid_landmarks[image_id] = current_valid_landmarks
    save_dict_csv(valid_landmarks, filename)


if __name__ == '__main__':
    # this script converts the landmark json files from the VerSe 2019 challenge dataset to
    # a landmark.csv file with physical coordinates
    verse_dataset_folder = '../verse2019_dataset'
    files = glob(os.path.join(verse_dataset_folder, 'images', '*.json'))
    landmarks_dict = {}
    num_landmarks = 25
    for filename in sorted(files):
        # get image id
        filename_wo_folder = os.path.basename(filename)
        filename_wo_folder_and_ext = filename_wo_folder[:filename_wo_folder.find('_ctd')]
        image_id = filename_wo_folder_and_ext
        print(filename_wo_folder_and_ext)
        # get image meta data
        image_meta_data = read_meta_data(os.path.join(verse_dataset_folder, 'images_reoriented', image_id + '.nii.gz'))
        spacing = np.array(image_meta_data.GetSpacing())
        origin = np.array(image_meta_data.GetOrigin())
        direction = np.array(image_meta_data.GetDirection()).reshape([3, 3])
        size = np.array(image_meta_data.GetSize())
        # placeholder for landmarks
        current_landmarks = [Landmark([np.nan] * 3, False, 1.0, 0.0) for _ in range(num_landmarks)]
        with open(filename, 'r') as f:
            # load json file
            json_data = json.load(f)
            for landmark in json_data:
                # convert verse coordinate system to physical coordinates
                coords = np.array([float(landmark['Z']), float(landmark['Y']), size[2] * spacing[2] - float(landmark['X'])])
                # labels in verse start at 1, our indexing starts at 0
                index = int(landmark['label']) - 1
                current_landmarks[index].coords = coords
                current_landmarks[index].is_valid = True
                print(coords)
        landmarks_dict[image_id] = current_landmarks

    save_points_csv(landmarks_dict, 'landmarks.csv')
    save_valid_landmarks_list(landmarks_dict, 'valid_landmarks.csv')
