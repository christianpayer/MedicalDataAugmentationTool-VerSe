#!/usr/bin/python

from collections import OrderedDict

import argparse
import json
import numpy as np
import os
import tensorflow as tf
import traceback
from copy import deepcopy
from glob import glob

import utils.io.common
import utils.io.image
import utils.io.landmark
import utils.io.text
import utils.sitk_image
import utils.sitk_np
from dataset import Dataset
from network import UnetClassicAvgLinear3d, spatial_configuration_net
from tensorflow_train.train_loop import MainLoopBase
from tensorflow_train.utils.tensorflow_util import create_placeholders_tuple
from utils.image_tiler import ImageTiler
from utils.landmark.heatmap_test import HeatmapTest
from utils.landmark.spine_postprocessing import SpinePostprocessing
import utils.np_image


class MainLoop(MainLoopBase):
    def __init__(self, network, unet, network_parameters, image_size, image_spacing, cropped_inc, data_format):
        super().__init__()
        self.num_labels = 25
        self.data_format = data_format
        self.channel_axis = 1
        self.network = network
        self.unet = unet
        self.network_parameters = network_parameters
        self.padding = 'same'
        self.image_size = image_size
        self.image_spacing = image_spacing
        self.cropped_inc = cropped_inc

    def init_networks(self):
        network_image_size = list(reversed(self.image_size))

        if self.data_format == 'channels_first':
            data_generator_entries = OrderedDict([('image', [1] + network_image_size)])
        else:
            data_generator_entries = OrderedDict([('image', network_image_size + [1])])

        data_generator_types = {'image': tf.float32}

        # create model with shared weights between train and val
        training_net = tf.make_template('net', self.network)

        # build val graph
        self.data_val = create_placeholders_tuple(data_generator_entries, data_types=data_generator_types, shape_prefix=[1])
        self.prediction_val, self.local_prediction_val, self.spatial_prediction_val = training_net(self.data_val, num_labels=self.num_labels, is_training=False, actual_network=self.unet, padding=self.padding, data_format=self.data_format, **self.network_parameters)

    def test_cropped_image(self, full_image):
        image_size_np = [1] + list(reversed(self.image_size)) if self.data_format == 'channels_first' else list(reversed(self.image_size)) + [1]
        labels_size_np = [self.num_labels] + list(reversed(self.image_size)) if self.data_format == 'channels_first' else list(reversed(self.image_size)) + [self.num_labels]
        predictions_full_size_np = [self.num_labels] + list(full_image.shape[1:]) if self.data_format == 'channels_first' else list(full_image.shape[:-1]) + [self.num_labels]
        cropped_inc = [0] + self.cropped_inc if self.data_format == 'channels_first' else self.cropped_inc + [0]
        image_tiler = ImageTiler(full_image.shape, image_size_np, cropped_inc, True, -1)
        prediction_tiler = ImageTiler(predictions_full_size_np, labels_size_np, cropped_inc, True, 0)

        for image_tiler, prediction_tiler in zip(image_tiler, prediction_tiler):
            current_image = image_tiler.get_current_data(full_image)
            feed_dict = {self.data_val: np.expand_dims(current_image, axis=0)}
            run_tuple = self.sess.run((self.prediction_val,), feed_dict=feed_dict)
            prediction = np.squeeze(run_tuple[0], axis=0)
            image_tiler.set_current_data(current_image)
            prediction_tiler.set_current_data(prediction)

        return prediction_tiler.output_image


class InferenceLoop(object):
    def __init__(self, network, unet, network_parameters, image_base_folder, setup_base_folder, load_model_filenames, output_base_folder):
        super().__init__()
        self.image_base_folder = image_base_folder
        self.setup_base_folder = setup_base_folder
        self.load_model_filenames = load_model_filenames
        self.data_format = 'channels_last'
        self.save_debug_images = False
        self.num_landmarks = 25
        self.image_size = [96, 96, 128]
        self.image_spacing = [2] * 3
        self.cropped_inc = [64, 0, 0]
        self.save_debug_images = False
        self.output_folder = os.path.join(output_base_folder, 'vertebrae_localization')
        self.landmark_file_output_folder = os.path.join(self.output_folder, 'verse_landmarks')
        utils.io.common.create_directories(self.landmark_file_output_folder)
        dataset_parameters = {'cv': 'inference',
                              'image_base_folder': self.image_base_folder,
                              'setup_base_folder': self.setup_base_folder,
                              'image_size': self.image_size,
                              'image_spacing': self.image_spacing,
                              'input_gaussian_sigma': 0.75,
                              'load_spine_landmarks': True,
                              'translate_to_center_landmarks': True,
                              'translate_by_random_factor': True,
                              'data_format': self.data_format,
                              'save_debug_images': self.save_debug_images}

        dataset = Dataset(**dataset_parameters)
        self.dataset_val = dataset.dataset_val()

        self.network_loop = MainLoop(network, unet, network_parameters, self.image_size, self.image_spacing, self.cropped_inc, self.data_format)
        self.network_loop.init_networks()
        self.network_loop.init_variables()
        self.network_loop.init_saver()
        self.init_image_list()

    def init_image_list(self):
        images_files = sorted(glob(os.path.join(self.image_base_folder, '*[0-9].nii.gz')))
        self.image_id_list = map(lambda filename: os.path.basename(filename)[:-len('.nii.gz')], images_files)
        print(self.image_id_list)

    def output_file_for_current_iteration(self, file_name):
        return os.path.join(self.output_folder, file_name)

    def test_cropped_image(self, dataset_entry):
        generators = dataset_entry['generators']
        transformations = dataset_entry['transformations']
        predictions = []
        for load_model_filename in self.load_model_filenames:
            if len(self.load_model_filenames) > 1:
                self.network_loop.load_model_filename = load_model_filename
                self.network_loop.load_model()
            prediction = self.network_loop.test_cropped_image(generators['image'])
            predictions.append(prediction)

        prediction = np.mean(predictions, axis=0)
        transformation = transformations['image']
        image = generators['image']

        return image, prediction, transformation

    def convert_landmarks_to_verse_indexing(self, landmarks, image):
        new_landmarks = []
        spacing = np.array(image.GetSpacing())
        size = np.array(image.GetSize())
        for landmark in landmarks:
            new_landmark = deepcopy(landmark)
            if not landmark.is_valid:
                new_landmarks.append(new_landmark)
                continue
            index = np.array(image.TransformPhysicalPointToContinuousIndex(landmark.coords.tolist()))
            index_with_spacing = index * spacing
            new_coord = np.array([size[2] * spacing[2] - index_with_spacing[2], index_with_spacing[1], index_with_spacing[0]])
            new_landmark.coords = new_coord
            new_landmarks.append(new_landmark)
        return new_landmarks

    def save_landmarks_verse_json(self, landmarks, filename):
        verse_landmarks_list = []
        for i, landmark in enumerate(landmarks):
            if landmark.is_valid:
                verse_landmarks_list.append({'Y': landmark.coords[1],
                                             'X': landmark.coords[0],
                                             'Z': landmark.coords[2],
                                             'label': i + 1})
        with open(filename, 'w') as f:
            json.dump(verse_landmarks_list, f)

    def save_valid_landmarks_list(self, landmarks_dict, filename):
        valid_landmarks = {}
        for image_id, landmarks in landmarks_dict.items():
            current_valid_landmarks = []
            for landmark_id, landmark in enumerate(landmarks):
                if landmark.is_valid:
                    current_valid_landmarks.append(landmark_id)
            valid_landmarks[image_id] = current_valid_landmarks
        utils.io.text.save_dict_csv(valid_landmarks, filename)

    def test(self):
        print('Testing...')

        channel_axis = 0
        if self.data_format == 'channels_last':
            channel_axis = 3

        if len(self.load_model_filenames) == 1:
            self.network_loop.load_model_filename = self.load_model_filenames[0]
            self.network_loop.load_model()

        heatmap_maxima = HeatmapTest(channel_axis, False, return_multiple_maxima=True, min_max_distance=7, min_max_value=0.25, multiple_min_max_value_factor=0.1)
        spine_postprocessing = SpinePostprocessing(num_landmarks=self.num_landmarks, image_spacing=self.image_spacing)

        landmarks = {}
        for image_id in self.image_id_list:
            try:
                print(image_id)
                dataset_entry = self.dataset_val.get({'image_id': image_id})
                current_id = dataset_entry['id']['image_id']
                datasources = dataset_entry['datasources']
                input_image = datasources['image']

                image, prediction, transformation = self.test_cropped_image(dataset_entry)
                
                if self.save_debug_images:
                    origin = transformation.TransformPoint(np.zeros(3, np.float64))
                    heatmap_normalization_mode = (0, 1)
                    utils.io.image.write_multichannel_np(image, self.output_file_for_current_iteration(current_id + '_input.mha'), normalization_mode='min_max', split_channel_axis=True, sitk_image_mode='default', data_format=self.data_format, image_type=np.uint8, spacing=self.image_spacing, origin=origin)
                    utils.io.image.write_multichannel_np(prediction, self.output_file_for_current_iteration(current_id + '_prediction.mha'), normalization_mode=heatmap_normalization_mode, split_channel_axis=True, data_format=self.data_format, image_type=np.uint8, spacing=self.image_spacing, origin=origin)

                predicted_landmarks = heatmap_maxima.get_landmarks(prediction, input_image, self.image_spacing, transformation)
                landmark_sequence = spine_postprocessing.postprocess_landmarks(predicted_landmarks, prediction.shape)
                landmarks[current_id] = landmark_sequence
                verse_landmarks = self.convert_landmarks_to_verse_indexing(landmark_sequence, input_image)
                self.save_landmarks_verse_json(verse_landmarks, os.path.join(self.landmark_file_output_folder, image_id + '_ctd.json'))
            except:
                print(traceback.format_exc())
                print('ERROR predicting', image_id)
                pass

        utils.io.landmark.save_points_csv(landmarks, self.output_file_for_current_iteration('landmarks.csv'))
        self.save_valid_landmarks_list(landmarks, self.output_file_for_current_iteration('valid_landmarks.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, required=True)
    parser.add_argument('--setup_folder', type=str, required=True)
    parser.add_argument('--model_files', nargs='+', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)
    parser_args = parser.parse_args()
    network_parameters = OrderedDict([('num_filters_base', 64), ('double_features_per_level', False), ('num_levels', 5), ('activation', 'relu'), ('spatial_downsample', 4)])
    loop = InferenceLoop(spatial_configuration_net, UnetClassicAvgLinear3d, network_parameters, parser_args.image_folder, parser_args.setup_folder, parser_args.model_files, parser_args.output_folder)
    loop.test()
