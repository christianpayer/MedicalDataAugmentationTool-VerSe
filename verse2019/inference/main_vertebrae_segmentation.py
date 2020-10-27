#!/usr/bin/python

import argparse
from collections import OrderedDict
from glob import glob

import SimpleITK as sitk
import numpy as np
import os
import traceback
import tensorflow as tf

import utils.io.image
import utils.io.text
import utils.sitk_image
import utils.sitk_np
from dataset import Dataset
from network import network_u, UnetClassicAvgLinear3d
from tensorflow_train.train_loop import MainLoopBase
from tensorflow_train.utils.tensorflow_util import create_placeholders_tuple
from utils.segmentation.segmentation_test import SegmentationTest
import utils.np_image


class MainLoop(MainLoopBase):
    def __init__(self, network, unet, network_parameters, image_size, image_spacing, data_format):
        super().__init__()
        self.num_labels = 1
        self.num_labels_all = 26
        self.data_format = data_format
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
                                                  ('single_heatmap', [1] + network_image_size)])
        else:
            data_generator_entries = OrderedDict([('image', network_image_size + [1]),
                                                  ('single_heatmap', network_image_size + [1])])

        data_generator_types = {'image': tf.float32, 'single_heatmap': tf.float32}


        # create model with shared weights between train and val
        training_net = tf.make_template('net', self.network)

        # build val graph
        self.data_val, self.single_heatmap_val = create_placeholders_tuple(data_generator_entries, data_types=data_generator_types, shape_prefix=[1])
        concat_axis = 1 if self.data_format == 'channels_first' else 4
        self.data_heatmap_concat_val = tf.concat([self.data_val, self.single_heatmap_val], axis=concat_axis)
        self.prediction_val = training_net(self.data_heatmap_concat_val, num_labels=self.num_labels, is_training=False, actual_network=self.unet, padding=self.padding, data_format=self.data_format, **self.network_parameters)
        self.prediction_softmax_val = tf.nn.sigmoid(self.prediction_val)

    def test_full_image(self, image, heatmap):
        feed_dict = {self.data_val: np.expand_dims(image, axis=0),
                     self.single_heatmap_val: np.expand_dims(heatmap, axis=0)}
        # run loss and update loss accumulators
        run_tuple = self.sess.run((self.prediction_softmax_val,), feed_dict=feed_dict)
        prediction = np.squeeze(run_tuple[0], axis=0)

        return prediction


class InferenceLoop(object):
    def __init__(self, network, unet, network_parameters, image_base_folder, setup_base_folder, load_model_filenames, output_base_folder):
        super().__init__()
        #self.load_model_filenames = ['/models/vertebrae_segmentation/model']
        #self.image_base_folder = '/tmp/data_reoriented'
        #self.setup_base_folder = '/tmp/'
        self.image_base_folder = image_base_folder
        self.setup_base_folder = setup_base_folder
        self.load_model_filenames = load_model_filenames
        self.num_labels = 1
        self.num_labels_all = 26
        self.data_format = 'channels_last'
        self.network = network
        self.unet = unet
        self.network_parameters = network_parameters
        self.padding = 'same'
        self.image_size = [128, 128, 96]
        self.image_spacing = [1] * 3
        self.save_debug_images = False
        self.output_folder = os.path.join(output_base_folder, 'vertebrae_segmentation')
        dataset_parameters = {'cv': 'inference',
                              'image_base_folder': self.image_base_folder,
                              'setup_base_folder': self.setup_base_folder,
                              'image_size': self.image_size,
                              'image_spacing': self.image_spacing,
                              'input_gaussian_sigma': 0.75,
                              'label_gaussian_sigma': 1.0,
                              'heatmap_sigma': 3.0,
                              'generate_single_vertebrae_heatmap': True,
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
        self.valid_landmarks_file = os.path.join(self.setup_base_folder, 'vertebrae_localization/valid_landmarks.csv')
        self.valid_landmarks = utils.io.text.load_dict_csv(self.valid_landmarks_file)

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
            prediction = self.network_loop.test_full_image(generators['image'], generators['single_heatmap'])
            predictions.append(prediction)

        prediction = np.mean(predictions, axis=0)
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

        labels = list(range(self.num_labels_all))
        interpolator = 'linear'
        filter_largest_cc = True
        segmentation_test = SegmentationTest(labels,
                                             channel_axis=channel_axis,
                                             interpolator=interpolator,
                                             largest_connected_component=False,
                                             all_labels_are_connected=False)

        for image_id in self.image_id_list:
            try:
                print(image_id)
                first = True
                prediction_resampled_np = None
                input_image = None
                for landmark_id in self.valid_landmarks[image_id]:
                    dataset_entry = self.dataset_val.get({'image_id': image_id, 'landmark_id' : landmark_id})
                    datasources = dataset_entry['datasources']
                    if first:
                        input_image = datasources['image']
                        if self.data_format == 'channels_first':
                            prediction_resampled_np = np.zeros([self.num_labels_all] + list(reversed(input_image.GetSize())), dtype=np.float16)
                            prediction_resampled_np[0, ...] = 0.5
                        else:
                            prediction_resampled_np = np.zeros(list(reversed(input_image.GetSize())) + [self.num_labels_all], dtype=np.float16)
                            prediction_resampled_np[..., 0] = 0.5
                        first = False
                    
                    image, prediction, transformation = self.test_full_image(dataset_entry)

                    origin = transformation.TransformPoint(np.zeros(3, np.float64))
                    if filter_largest_cc:
                        prediction_thresh_np = (prediction > 0.5).astype(np.uint8)
                        if self.data_format == 'channels_first':
                            largest_connected_component = utils.np_image.largest_connected_component(prediction_thresh_np[0])
                            prediction_thresh_np[largest_connected_component[None, ...] == 1] = 0
                        else:
                            largest_connected_component = utils.np_image.largest_connected_component(prediction_thresh_np[..., 0])
                            prediction_thresh_np[largest_connected_component[..., None] == 1] = 0
                        prediction[prediction_thresh_np == 1] = 0
                    
                    if self.save_debug_images:
                        utils.io.image.write_multichannel_np(image, self.output_file_for_current_iteration(image_id + '_' + landmark_id + '_input.mha'), normalization_mode='min_max', split_channel_axis=True, sitk_image_mode='default', data_format=self.data_format, image_type=np.uint8, spacing=self.image_spacing, origin=origin)
                        utils.io.image.write_multichannel_np(prediction, self.output_file_for_current_iteration(image_id + '_' + landmark_id + '_prediction.mha'), normalization_mode=(0, 1), split_channel_axis=True, sitk_image_mode='default', data_format=self.data_format, image_type=np.uint8, spacing=self.image_spacing, origin=origin)

                    prediction_resampled_sitk = utils.sitk_image.transform_np_output_to_sitk_input(output_image=prediction,
                                                                                        output_spacing=self.image_spacing,
                                                                                        channel_axis=channel_axis,
                                                                                        input_image_sitk=input_image,
                                                                                        transform=transformation,
                                                                                        interpolator=interpolator,
                                                                                        output_pixel_type=sitk.sitkFloat32)
                    if self.data_format == 'channels_first':
                        prediction_resampled_np[int(landmark_id) + 1] = utils.sitk_np.sitk_to_np(prediction_resampled_sitk[0])
                    else:
                        prediction_resampled_np[..., int(landmark_id) + 1] = utils.sitk_np.sitk_to_np(prediction_resampled_sitk[0])
                prediction_labels = segmentation_test.get_label_image(prediction_resampled_np, reference_sitk=input_image, image_type=np.uint16)
                # delete to save memory
                del prediction_resampled_np
                utils.io.image.write(prediction_labels, self.output_file_for_current_iteration(image_id + '_seg.nii.gz'))
            except:
                print(traceback.format_exc())
                print('ERROR predicting', image_id)
                pass


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

