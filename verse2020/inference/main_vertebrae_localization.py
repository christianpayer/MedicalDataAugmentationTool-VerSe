#!/usr/bin/python
import argparse
import json
import os
import pickle
import sys
import traceback
from collections import OrderedDict
from copy import deepcopy
from glob import glob

import numpy as np
import tensorflow as tf
import utils.io.image
import utils.io.landmark
import utils.io.text
import utils.sitk_image
import utils.sitk_np
from dataset import Dataset
from network import SpatialConfigurationNet, Unet
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow_train_v2.train_loop import MainLoopBase
from tensorflow_train_v2.utils.output_folder_handler import OutputFolderHandler
from tqdm import tqdm
from utils.image_tiler import ImageTiler
from utils.landmark.common import Landmark
from utils.landmark.heatmap_test import HeatmapTest
from utils.landmark.spine_postprocessing_graph import SpinePostprocessingGraph
from utils.landmark.visualization.landmark_visualization_matplotlib import LandmarkVisualizationMatplotlib
from vertebrae_localization_postprocessing import add_landmarks_from_neighbors, filter_landmarks_top_bottom, reshift_landmarks


class MainLoop(MainLoopBase):
    def __init__(self, config):
        """
        Initializer.
        :param cv: The cv fold. 0, 1, 2 for CV; 'train_all' for training on whole dataset.
        :param config: config dictionary
        """
        super().__init__()
        gpu_available = tf.test.gpu_device_name() != ''
        self.use_mixed_precision = gpu_available
        if self.use_mixed_precision:
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_policy(policy)
        self.cv = config.cv
        self.config = config
        self.batch_size = 1
        self.num_landmarks = 26
        self.data_format = 'channels_last'
        self.network_parameters = OrderedDict(num_filters_base=config.num_filters_base,
                                              activation=config.activation,
                                              spatial_downsample=config.spatial_downsample,
                                              local_activation=config.local_activation,
                                              spatial_activation=config.spatial_activation,
                                              num_levels=config.num_levels,
                                              data_format=self.data_format)
        if config.model == 'scn':
            self.network = SpatialConfigurationNet
        if config.model == 'unet':
            self.network = Unet

        self.evaluate_landmarks_postprocessing = True
        self.save_output_images = True
        self.save_debug_images = False
        self.image_folder = config.image_folder
        self.setup_folder = config.setup_folder
        self.output_folder = config.output_folder
        self.load_model_filenames = config.load_model_filenames
        self.image_size = [None, None, None]
        self.image_spacing = [config.spacing] * 3
        self.max_image_size_for_cropped_test = [128, 128, 448]
        self.cropped_inc = [0, 128, 0, 0]
        self.heatmap_size = self.image_size
        images_files = sorted(glob(os.path.join(self.image_folder, '*.nii.gz')))
        self.image_id_list = list(map(lambda filename: os.path.basename(filename)[:-len('.nii.gz')], images_files))

        self.landmark_labels = [i + 1 for i in range(25)] + [28]
        self.landmark_mapping = dict([(i, self.landmark_labels[i]) for i in range(26)])
        self.landmark_mapping_inverse = dict([(self.landmark_labels[i], i) for i in range(26)])

    def init_model(self):
        self.model = self.network(num_labels=self.num_landmarks, **self.network_parameters)

    def init_checkpoint(self):
        self.checkpoint = tf.train.Checkpoint(model=self.model)

    def init_output_folder_handler(self):
        self.output_folder_handler = OutputFolderHandler(self.output_folder, use_timestamp=False, files_to_copy=[])

    def init_datasets(self):
        dataset_parameters = dict(image_base_folder=self.image_folder,
                                  setup_base_folder=self.setup_folder,
                                  image_size=self.image_size,
                                  image_spacing=self.image_spacing,
                                  num_landmarks=self.num_landmarks,
                                  normalize_zero_mean_unit_variance=False,
                                  cv=self.cv,
                                  input_gaussian_sigma=0.75,
                                  crop_image_top_bottom=True,
                                  use_variable_image_size=True,
                                  load_spine_bbs=True,
                                  valid_output_sizes_x=[64, 96],
                                  valid_output_sizes_y=[64, 96],
                                  valid_output_sizes_z=[64, 96, 128, 160, 192, 224, 256, 288, 320],
                                  translate_to_center_landmarks=True,
                                  translate_by_random_factor=True,
                                  data_format=self.data_format,
                                  save_debug_images=self.save_debug_images)

        dataset = Dataset(**dataset_parameters)
        self.dataset_val = dataset.dataset_val()
        self.network_image_size = list(reversed(self.image_size))

    def call_model(self, image):
        return self.model(image, training=False)

    def convert_landmarks_to_verse_indexing(self, landmarks, image):
        new_landmarks = []
        spacing = np.array(image.GetSpacing())
        size = np.array(image.GetSize())
        for landmark in landmarks:
            new_landmark = deepcopy(landmark)
            if not landmark.is_valid:
                new_landmarks.append(new_landmark)
                continue
            coords = np.array(landmark.coords.tolist())
            verse_coords = np.array([coords[1], size[2] * spacing[2] - coords[2], size[0] * spacing[0] - coords[0]])
            new_landmark.coords = verse_coords
            new_landmarks.append(new_landmark)
        return new_landmarks

    def save_landmarks_verse_json(self, landmarks, filename):
        verse_landmarks_list = []
        for i, landmark in enumerate(landmarks):
            if landmark.is_valid:
                verse_landmarks_list.append({'label': self.landmark_mapping[i],
                                             'X': landmark.coords[0],
                                             'Y': landmark.coords[1],
                                             'Z': landmark.coords[2]})
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

    def test_cropped_image(self, dataset_entry):
        """
        Perform inference on a dataset_entry with the validation network. Performs cropped prediction and merges outputs as maxima.
        :param dataset_entry: A dataset entry from the dataset.
        :return: input image (np.array), target heatmaps (np.array), predicted heatmaps,  transformation (sitk.Transform)
        """
        generators = dataset_entry['generators']
        transformations = dataset_entry['transformations']
        transformation = transformations['image']

        full_image = generators['image']

        if self.data_format == 'channels_first':
            image_size_for_tilers = np.minimum(full_image.shape[1:], list(reversed(self.max_image_size_for_cropped_test))).tolist()
            image_size_np = [1] + image_size_for_tilers
            labels_size_np = [self.num_landmarks] + image_size_for_tilers
            image_tiler = ImageTiler(full_image.shape, image_size_np, self.cropped_inc, True, -1)
            prediction_tiler = ImageTiler((self.num_landmarks,) + full_image.shape[1:], labels_size_np, self.cropped_inc, True, -np.inf)
            prediction_local_tiler = ImageTiler((self.num_landmarks,) + full_image.shape[1:], labels_size_np, self.cropped_inc, True, -np.inf)
            prediction_spatial_tiler = ImageTiler((self.num_landmarks,) + full_image.shape[1:], labels_size_np, self.cropped_inc, True, -np.inf)
        else:
            image_size_for_tilers = np.minimum(full_image.shape[:-1], list(reversed(self.max_image_size_for_cropped_test))).tolist()
            image_size_np = image_size_for_tilers + [1]
            labels_size_np = image_size_for_tilers + [self.num_landmarks]
            image_tiler = ImageTiler(full_image.shape, image_size_np, self.cropped_inc, True, -1)
            prediction_tiler = ImageTiler(full_image.shape[:-1] + (self.num_landmarks,), labels_size_np, self.cropped_inc, True, -np.inf)
            prediction_local_tiler = ImageTiler(full_image.shape[:-1] + (self.num_landmarks,), labels_size_np, self.cropped_inc, True, -np.inf)
            prediction_spatial_tiler = ImageTiler(full_image.shape[:-1] + (self.num_landmarks,), labels_size_np, self.cropped_inc, True, -np.inf)

        for image_tiler, prediction_tiler, prediction_local_tiler, prediction_spatial_tiler in zip(image_tiler, prediction_tiler, prediction_local_tiler, prediction_spatial_tiler):
            current_image = image_tiler.get_current_data(full_image)
            predictions = []
            predictions_local = []
            predictions_spatial = []
            for load_model_filename in self.load_model_filenames:
                if len(self.load_model_filenames) > 1:
                    self.load_model(load_model_filename)
                prediction, prediction_local, prediction_spatial = self.call_model(np.expand_dims(current_image, axis=0))
                predictions.append(prediction.numpy())
                predictions_local.append(prediction_local.numpy())
                predictions_spatial.append(prediction_spatial.numpy())
            prediction = np.mean(predictions, axis=0)
            prediction_local = np.mean(predictions_local, axis=0)
            prediction_spatial = np.mean(predictions_spatial, axis=0)
            image_tiler.set_current_data(current_image)
            prediction_tiler.set_current_data(np.squeeze(prediction, axis=0))
            prediction_local_tiler.set_current_data(np.squeeze(prediction_local, axis=0))
            prediction_spatial_tiler.set_current_data(np.squeeze(prediction_spatial, axis=0))

        return image_tiler.output_image, prediction_tiler.output_image, prediction_local_tiler.output_image, prediction_spatial_tiler.output_image, transformation

    def test(self):
        """
        The test function. Performs inference on the the validation images and calculates the loss.
        """
        print('Testing...')

        if len(self.load_model_filenames) == 1:
            self.load_model(self.load_model_filenames[0])

        vis = LandmarkVisualizationMatplotlib(dim=3,
                                              annotations=dict([(i, f'C{i + 1}') for i in range(7)] +        # 0-6: C1-C7
                                                               [(i, f'T{i - 6}') for i in range(7, 19)] +    # 7-18: T1-12
                                                               [(i, f'L{i - 18}') for i in range(19, 25)] +  # 19-24: L1-6
                                                               [(25, 'T13')]))                               # 25: T13

        channel_axis = 0
        if self.data_format == 'channels_last':
            channel_axis = 3
        heatmap_maxima = HeatmapTest(channel_axis,
                                     False,
                                     return_multiple_maxima=True,
                                     min_max_value=0.05,
                                     smoothing_sigma=2.0)

        with open('possible_successors.pickle', 'rb') as f:
            possible_successors = pickle.load(f)
        with open('units_distances.pickle', 'rb') as f:
            offsets_mean, distances_mean, distances_std = pickle.load(f)
        spine_postprocessing = SpinePostprocessingGraph(num_landmarks=self.num_landmarks,
                                                        possible_successors=possible_successors,
                                                        offsets_mean=offsets_mean,
                                                        distances_mean=distances_mean,
                                                        distances_std=distances_std,
                                                        bias=2.0,
                                                        l=0.2)

        landmarks = {}
        landmarks_no_postprocessing = {}
        for current_id in tqdm(self.image_id_list, desc='Testing'):
            try:
                dataset_entry = self.dataset_val.get({'image_id': current_id})
                print(current_id)
                datasources = dataset_entry['datasources']
                input_image = datasources['image']

                image, prediction, prediction_local, prediction_spatial, transformation = self.test_cropped_image(dataset_entry)

                origin = transformation.TransformPoint(np.zeros(3, np.float64))
                if self.save_output_images:
                    heatmap_normalization_mode = (-1, 1)
                    image_type = np.uint8
                    utils.io.image.write_multichannel_np(image, self.output_folder_handler.path('output', current_id + '_input.mha'), output_normalization_mode='min_max', sitk_image_output_mode='vector', data_format=self.data_format, image_type=image_type, spacing=self.image_spacing, origin=origin)
                    utils.io.image.write_multichannel_np(prediction, self.output_folder_handler.path('output', current_id + '_prediction.mha'), output_normalization_mode=heatmap_normalization_mode, sitk_image_output_mode='vector', data_format=self.data_format, image_type=image_type, spacing=self.image_spacing, origin=origin)
                    utils.io.image.write_multichannel_np(prediction, self.output_folder_handler.path('output', current_id + '_prediction_rgb.mha'), output_normalization_mode=(0, 1), channel_layout_mode='channel_rgb', sitk_image_output_mode='vector', data_format=self.data_format, image_type=image_type, spacing=self.image_spacing, origin=origin)
                    utils.io.image.write_multichannel_np(prediction_local, self.output_folder_handler.path('output', current_id + '_prediction_local.mha'), output_normalization_mode=heatmap_normalization_mode, sitk_image_output_mode='vector', data_format=self.data_format, image_type=image_type, spacing=self.image_spacing, origin=origin)
                    utils.io.image.write_multichannel_np(prediction_spatial, self.output_folder_handler.path('output', current_id + '_prediction_spatial.mha'), output_normalization_mode=heatmap_normalization_mode, sitk_image_output_mode='vector', data_format=self.data_format, image_type=image_type, spacing=self.image_spacing, origin=origin)

                local_maxima_landmarks = heatmap_maxima.get_landmarks(prediction, input_image, self.image_spacing, transformation)
                curr_landmarks_no_postprocessing = [l[0] if len(l) > 0 else Landmark(coords=[np.nan] * 3, is_valid=False)  for l in local_maxima_landmarks]
                landmarks_no_postprocessing[current_id] = curr_landmarks_no_postprocessing

                try:
                    local_maxima_landmarks = add_landmarks_from_neighbors(local_maxima_landmarks)
                    curr_landmarks = spine_postprocessing.solve_local_heatmap_maxima(local_maxima_landmarks)
                    curr_landmarks = reshift_landmarks(curr_landmarks)
                    curr_landmarks = filter_landmarks_top_bottom(curr_landmarks, input_image)
                except Exception:
                    print('error in postprocessing', current_id)
                    traceback.print_exc(file=sys.stdout)
                    curr_landmarks = curr_landmarks_no_postprocessing
                landmarks[current_id] = curr_landmarks

                if self.save_output_images:
                    vis.visualize_landmark_projections(input_image, curr_landmarks_no_postprocessing, filename=self.output_folder_handler.path('output', current_id + '_landmarks.png'))
                    vis.visualize_landmark_projections(input_image, curr_landmarks, filename=self.output_folder_handler.path('output', current_id + '_landmarks_pp.png'))

                verse_landmarks = self.convert_landmarks_to_verse_indexing(curr_landmarks, input_image)
                self.save_landmarks_verse_json(verse_landmarks, self.output_folder_handler.path(current_id + '_ctd.json'))
            except Exception:
                print('ERROR predicting', current_id)
                traceback.print_exc(file=sys.stdout)
                pass

        utils.io.landmark.save_points_csv(landmarks, self.output_folder_handler.path('landmarks.csv'))
        utils.io.landmark.save_points_csv(landmarks_no_postprocessing, self.output_folder_handler.path('landmarks_no_postprocessing.csv'))
        self.save_valid_landmarks_list(landmarks, self.output_folder_handler.path('valid_landmarks.csv'))


class dotdict(dict):
    """
    Dict subclass that allows dot.notation to access attributes.
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, required=True)
    parser.add_argument('--setup_folder', type=str, required=True)
    parser.add_argument('--model_files', nargs='+', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)
    parser_args = parser.parse_args()

    # Set hyperparameters, which can be overwritten with a W&B Sweep
    hyperparameters = dotdict(
        load_model_filenames=parser_args.model_files,
        image_folder=parser_args.image_folder,
        setup_folder=parser_args.setup_folder,
        output_folder=parser_args.output_folder,
        num_filters_base=96,
        activation='lrelu',
        spatial_downsample=4,
        learning_rate=0.0001,
        model='scn',
        loss='l2',
        local_activation='tanh',
        spatial_activation='tanh',
        num_levels=4,
        spacing=2.0,
        cv='inference',
    )
    with MainLoop(hyperparameters) as loop:
        loop.init_model()
        loop.init_output_folder_handler()
        loop.init_checkpoint()
        loop.init_datasets()
        print('Starting main test loop')
        loop.test()
