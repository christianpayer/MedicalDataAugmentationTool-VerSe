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
    # set to true if generating files for verse2020
    verse2020 = True
    if verse2020:
        verse_dataset_folder = '../verse2020_dataset'
        landmark_mapping = dict([(i + 1, i) for i in range(25)] + [(28, 25)])
    else:
        verse_dataset_folder = '../verse2019_dataset'
        landmark_mapping = dict([(i + 1, i) for i in range(25)])
    num_landmarks = len(landmark_mapping)
    landmarks_dict = {}
    files = glob(os.path.join(verse_dataset_folder, 'images', '*.json'))
    for filename in sorted(files):
        # get image id
        filename_wo_folder = os.path.basename(filename)
        if verse2020:
            ext_length = len('_ctd-iso.json')
        else:
            ext_length = len('_ctd.json')
        filename_wo_folder_and_ext = filename_wo_folder[:-ext_length]
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
                if verse2020:
                    coords = np.array([size[0] * spacing[0] - float(landmark['Z']), float(landmark['X']), size[2] * spacing[2] - float(landmark['Y'])])
                else:
                    coords = np.array([float(landmark['Z']), float(landmark['Y']), size[2] * spacing[2] - float(landmark['X'])])
                # labels in verse start at 1, our indexing starts at 0
                index = landmark_mapping[int(landmark['label'])]
                current_landmarks[index].coords = coords
                current_landmarks[index].is_valid = True
                print(coords)
        landmarks_dict[image_id] = current_landmarks

    save_points_csv(landmarks_dict, os.path.join(verse_dataset_folder, 'setup', 'landmarks.csv'))
    save_valid_landmarks_list(landmarks_dict, os.path.join(verse_dataset_folder, 'setup', 'valid_landmarks.csv'))
