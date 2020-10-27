#!/usr/bin/python

from collections import OrderedDict

import argparse
import SimpleITK as sitk
import numpy as np
import os
import tensorflow as tf
import traceback
from glob import glob

import utils.io.image
import utils.io.landmark
import utils.io.text
import utils.np_image
import utils.sitk_image
import utils.sitk_np
from dataset import Dataset
from network import network_u, UnetClassicAvgLinear3d
from tensorflow_train.train_loop import MainLoopBase
from tensorflow_train.utils.tensorflow_util import create_placeholders_tuple
from utils.landmark.common import Landmark


class MainLoop(MainLoopBase):
    def __init__(self, network, unet, network_parameters, image_size, image_spacing, data_format):
        super().__init__()
        self.num_labels = 1
        self.data_format = data_format
        self.channel_axis = 1
        self.network = network
        self.unet = unet
        self.network_parameters = network_parameters
        self.padding = 'same'
        self.image_size = image_size
        self.image_spacing = image_spacing

    def init_networks(self):
        network_image_size = list(reversed(self.image_size))

        if self.data_format == 'channels_first':
            data_generator_entries = OrderedDict([('image', [1] + network_image_size),
                                                  ('spine_heatmap', [1] + network_image_size)
                                                  ])
        else:
            data_generator_entries = OrderedDict([('image', network_image_size + [1]),
                                                  ('spine_heatmap', network_image_size + [1])])

        data_generator_types = {'image': tf.float32,
                                'spine_heatmap': tf.float32}

        # create model with shared weights between train and val
        training_net = tf.make_template('net', self.network)

        # build val graph
        self.data_val, self.target_spine_heatmap_val = create_placeholders_tuple(data_generator_entries, data_types=data_generator_types, shape_prefix=[1])
        self.prediction_val = training_net(self.data_val, num_labels=self.num_labels, is_training=False, actual_network=self.unet, padding=self.padding, data_format=self.data_format, **self.network_parameters)

    def test_full_image(self, image):
        feed_dict = {self.data_val: np.expand_dims(image, axis=0)}
        # run loss and update loss accumulators
        run_tuple = self.sess.run((self.prediction_val,), feed_dict=feed_dict)
        prediction = np.squeeze(run_tuple[0], axis=0)
        return prediction


class InferenceLoop(object):
    def __init__(self, network, unet, network_parameters, image_base_folder, setup_base_folder, load_model_filenames, output_base_folder):
        self.image_base_folder = image_base_folder
        self.setup_base_folder = setup_base_folder
        self.load_model_filenames = load_model_filenames
        self.data_format = 'channels_last'
        self.image_size = [64, 64, 128]
        self.image_spacing = [8] * 3
        self.output_folder = os.path.join(output_base_folder, 'spine_localization')
        self.save_debug_images = False
        dataset_parameters = {'cv': 'inference',
                              'image_base_folder': self.image_base_folder,
                              'setup_base_folder': self.setup_base_folder,
                              'image_size': self.image_size,
                              'image_spacing': self.image_spacing,
                              'input_gaussian_sigma': 3.0,
                              'data_format': self.data_format,
                              'save_debug_images': self.save_debug_images}

        dataset = Dataset(**dataset_parameters)
        self.dataset_val = dataset.dataset_val()

        self.network_loop = MainLoop(network, unet, network_parameters, self.image_size, self.image_spacing, self.data_format)
        self.network_loop.init_networks()
        self.network_loop.init_variables()
        self.network_loop.init_saver()
        self.init_image_list()

    def init_image_list(self):
        images_files = sorted(glob(os.path.join(self.image_base_folder, '*[0-9].nii.gz')))
        self.image_id_list = map(lambda filename: os.path.basename(filename)[:-len('.nii.gz')], images_files)

    def output_file_for_current_iteration(self, file_name):
        return os.path.join(self.output_folder, file_name)

    def test_full_image(self, dataset_entry):
        generators = dataset_entry['generators']
        transformations = dataset_entry['transformations']
        predictions = []
        for load_model_filename in self.load_model_filenames:
            if len(self.load_model_filenames) > 1:
                self.network_loop.load_model_filename = load_model_filename
                self.network_loop.load_model()
            prediction = self.network_loop.test_full_image(generators['image'])
            predictions.append(prediction)

        prediction = np.mean(np.stack(predictions, axis=0), axis=0)
        transformation = transformations['image']
        image = generators['image']

        return image, prediction, transformation

    def test(self):
        print('Testing...')

        channel_axis = 0
        if self.data_format == 'channels_last':
            channel_axis = 3

        if len(self.load_model_filenames) == 1:
            self.network_loop.load_model_filename = self.load_model_filenames[0]
            self.network_loop.load_model()

        landmarks = {}
        for image_id in self.image_id_list:
            try:
                print(image_id)
                dataset_entry = self.dataset_val.get({'image_id': image_id})
                current_id = dataset_entry['id']['image_id']
                datasources = dataset_entry['datasources']
                input_image = datasources['image']
                
                image, prediction, transformation = self.test_full_image(dataset_entry)
                
                predictions_sitk = utils.sitk_image.transform_np_output_to_sitk_input(output_image=prediction,
                                                                                    output_spacing=self.image_spacing,
                                                                                    channel_axis=channel_axis,
                                                                                    input_image_sitk=input_image,
                                                                                    transform=transformation,
                                                                                    interpolator='linear',
                                                                                    output_pixel_type=sitk.sitkFloat32)
                
                if self.save_debug_images:
                    origin = transformation.TransformPoint(np.zeros(3, np.float64))
                    heatmap_normalization_mode = (0, 1)
                    utils.io.image.write_multichannel_np(image, self.output_file_for_current_iteration(current_id + '_input.mha'), normalization_mode='min_max', split_channel_axis=True, sitk_image_mode='default', data_format=self.data_format, image_type=np.uint8, spacing=self.image_spacing, origin=origin)
                    utils.io.image.write_multichannel_np(prediction, self.output_file_for_current_iteration(current_id + '_prediction.mha'), normalization_mode=heatmap_normalization_mode, split_channel_axis=True, data_format=self.data_format, image_type=np.uint8, spacing=self.image_spacing, origin=origin)
                    utils.io.image.write(predictions_sitk[0], self.output_file_for_current_iteration(current_id + '_prediction_original.mha'))

                predictions_com = input_image.TransformContinuousIndexToPhysicalPoint(list(reversed(utils.np_image.center_of_mass(utils.sitk_np.sitk_to_np_no_copy(predictions_sitk[0])))))
                landmarks[current_id] = [Landmark(predictions_com)]
            except:
                print(traceback.format_exc())
                print('ERROR predicting', image_id)
                pass

        utils.io.landmark.save_points_csv(landmarks, self.output_file_for_current_iteration('landmarks.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, required=True)
    parser.add_argument('--setup_folder', type=str, required=True)
    parser.add_argument('--model_files', nargs='+', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)
    parser_args = parser.parse_args()
    network_parameters = OrderedDict([('num_filters_base', 64), ('double_features_per_level', False), ('num_levels', 5), ('activation', 'relu')])
    loop = InferenceLoop(network_u, UnetClassicAvgLinear3d, network_parameters, parser_args.image_folder, parser_args.setup_folder, parser_args.model_files, parser_args.output_folder)
    loop.test()

