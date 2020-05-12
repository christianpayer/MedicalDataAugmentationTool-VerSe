#!/usr/bin/python

import os
from collections import OrderedDict

import SimpleITK as sitk
import numpy as np
import tensorflow as tf
import utils.io.image
import utils.io.landmark
import utils.io.text
import utils.np_image
import utils.sitk_image
import utils.sitk_np
from datasets.pyro_dataset import PyroClientDataset
from tensorflow_train.data_generator import DataGenerator
from tensorflow_train.train_loop import MainLoopBase
from tensorflow_train.utils.data_format import get_batch_channel_image_size
from tensorflow_train.utils.summary_handler import create_summary_placeholder
from tensorflow_train.utils.tensorflow_util import get_reg_loss, create_placeholders_tuple, print_progress_bar
from utils.landmark.common import Landmark, get_mean_landmark
from utils.landmark.landmark_statistics import LandmarkStatistics

from dataset import Dataset
from network import network_u, UnetClassicAvgLinear3d


class MainLoop(MainLoopBase):
    def __init__(self, cv, network, unet, network_parameters, learning_rate, output_folder_name=''):
        """
        Initializer.
        :param cv: The cv fold. 0, 1, 2 for CV; 'train_all' for training on whole dataset.
        :param network: The used network. Usually network_u.
        :param unet: The specific instance of the U-Net. Usually UnetClassicAvgLinear3d.
        :param network_parameters: The network parameters passed to unet.
        :param learning_rate: The initial learning rate.
        :param output_folder_name: The output folder name that is used for distinguishing experiments.
        """
        super().__init__()
        self.batch_size = 1
        self.learning_rates = [learning_rate, learning_rate * 0.5, learning_rate * 0.1]
        self.learning_rate_boundaries = [10000, 15000]
        self.max_iter = 20000
        self.test_iter = 5000
        self.disp_iter = 100
        self.snapshot_iter = 5000
        self.test_initialization = False
        self.current_iter = 0
        self.reg_constant = 0.0005
        self.use_background = True
        self.num_labels = 1
        self.heatmap_sigma = 2.0
        self.data_format = 'channels_first'
        self.network = network
        self.unet = unet
        self.network_parameters = network_parameters
        self.padding = 'same'

        self.use_pyro_dataset = False
        self.save_output_images = True
        self.save_output_images_as_uint = True  # set to False, if you want to see the direct network output
        self.save_debug_images = False
        self.has_validation_groundtruth = cv in [0, 1, 2]
        self.local_base_folder = '../verse2019_dataset'
        self.image_size = [64, 64, 128]
        self.image_spacing = [8] * 3
        self.output_folder = os.path.join('./output/spine_localization/', network.__name__, unet.__name__, output_folder_name, str(cv), self.output_folder_timestamp())
        dataset_parameters = {'base_folder': self.local_base_folder,
                              'image_size': self.image_size,
                              'image_spacing': self.image_spacing,
                              'cv': cv,
                              'input_gaussian_sigma': 3.0,
                              'generate_spine_heatmap': True,
                              'save_debug_images': self.save_debug_images}

        dataset = Dataset(**dataset_parameters)
        if self.use_pyro_dataset:
            server_name = '@localhost:51232'
            uri = 'PYRO:verse_dataset' + server_name
            print('using pyro uri', uri)
            self.dataset_train = PyroClientDataset(uri, **dataset_parameters)
        else:
            self.dataset_train = dataset.dataset_train()
        self.dataset_val = dataset.dataset_val()

        self.point_statistics_names = ['pe_mean', 'pe_stdev', 'pe_median']
        self.additional_summaries_placeholders_val = dict([(name, create_summary_placeholder(name)) for name in self.point_statistics_names])

    def loss_function(self, pred, target):
        """
        L2 loss function calculated with prediction and target.
        :param pred: The predicted image.
        :param target: The target image.
        :return: L2 loss of (pred - target) / batch_size
        """
        batch_size, _, _ = get_batch_channel_image_size(pred, self.data_format)
        return tf.nn.l2_loss(pred - target) / batch_size

    def init_networks(self):
        """
        Initialize networks and placeholders.
        """
        network_image_size = list(reversed(self.image_size))

        if self.data_format == 'channels_first':
            data_generator_entries = OrderedDict([('image', [1] + network_image_size),
                                                  ('spine_heatmap', [1] + network_image_size)])
        else:
            data_generator_entries = OrderedDict([('image', network_image_size + [1]),
                                                  ('spine_heatmap', [1] + network_image_size)])

        data_generator_types = {'image': tf.float32,
                                'spine_heatmap': tf.float32}


        # create model with shared weights between train and val
        training_net = tf.make_template('net', self.network)

        # build train graph
        self.train_queue = DataGenerator(coord=self.coord, dataset=self.dataset_train, data_names_and_shapes=data_generator_entries, data_types=data_generator_types, batch_size=self.batch_size)
        data, target_spine_heatmap = self.train_queue.dequeue()

        prediction = training_net(data, num_labels=self.num_labels, is_training=True, actual_network=self.unet, padding=self.padding, **self.network_parameters)
        self.loss_net = self.loss_function(target=target_spine_heatmap, pred=prediction)
        self.loss_reg = get_reg_loss(self.reg_constant)
        self.loss = self.loss_net + tf.cast(self.loss_reg, tf.float32)

        # solver
        global_step = tf.Variable(self.current_iter, trainable=False)
        learning_rate = tf.train.piecewise_constant(global_step, self.learning_rate_boundaries, self.learning_rates)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss, global_step=global_step)
        self.train_losses = OrderedDict([('loss', self.loss_net), ('loss_reg', self.loss_reg)])

        # build val graph
        self.data_val, self.target_spine_heatmap_val = create_placeholders_tuple(data_generator_entries, data_types=data_generator_types, shape_prefix=[1])
        self.prediction_val = training_net(self.data_val, num_labels=self.num_labels, is_training=False, actual_network=self.unet, padding=self.padding, **self.network_parameters)

        if self.has_validation_groundtruth:
            self.loss_val = self.loss_function(target=self.target_spine_heatmap_val, pred=self.prediction_val)
            self.val_losses = OrderedDict([('loss', self.loss_val), ('loss_reg', self.loss_reg)])

    def test_full_image(self, dataset_entry):
        """
        Perform inference on a dataset_entry with the validation network.
        :param dataset_entry: A dataset entry from the dataset.
        :return: input image (np.array), network prediction (np.array), transformation (sitk.Transform)
        """
        generators = dataset_entry['generators']
        transformations = dataset_entry['transformations']
        if self.has_validation_groundtruth:
            feed_dict = {self.data_val: np.expand_dims(generators['image'], axis=0),
                         self.target_spine_heatmap_val: np.expand_dims(generators['spine_heatmap'], axis=0)}
            # run loss and update loss accumulators
            run_tuple = self.sess.run((self.prediction_val, self.loss_val) + self.val_loss_aggregator.get_update_ops(),
                                      feed_dict=feed_dict)
        else:
            feed_dict = {self.data_val: np.expand_dims(generators['image'], axis=0)}
            # run loss and update loss accumulators
            run_tuple = self.sess.run((self.prediction_val,), feed_dict=feed_dict)

        prediction = np.squeeze(run_tuple[0], axis=0)
        transformation = transformations['image']
        image = generators['image']

        return image, prediction, transformation

    def test(self):
        """
        The test function. Performs inference on the the validation images and calculates the loss.
        """
        print('Testing...')

        channel_axis = 0
        if self.data_format == 'channels_last':
            channel_axis = 3

        landmark_statistics = LandmarkStatistics()
        landmarks = {}
        num_entries = self.dataset_val.num_entries()
        for i in range(num_entries):
            dataset_entry = self.dataset_val.get_next()
            current_id = dataset_entry['id']['image_id']
            datasources = dataset_entry['datasources']
            if self.has_validation_groundtruth:
                groundtruth_landmarks = datasources['landmarks']
                groundtruth_landmark = [get_mean_landmark(groundtruth_landmarks)]
            input_image = datasources['image']

            image, prediction, transformation = self.test_full_image(dataset_entry)
            predictions_sitk = utils.sitk_image.transform_np_output_to_sitk_input(output_image=prediction,
                                                                                  output_spacing=self.image_spacing,
                                                                                  channel_axis=channel_axis,
                                                                                  input_image_sitk=input_image,
                                                                                  transform=transformation,
                                                                                  interpolator='linear',
                                                                                  output_pixel_type=sitk.sitkFloat32)
            if self.save_output_images:
                if self.save_output_images_as_uint:
                    image_normalization = 'min_max'
                    heatmap_normalization = (0, 1)
                    output_image_type = np.uint8
                else:
                    image_normalization = None
                    heatmap_normalization = None
                    output_image_type = np.float32
                origin = transformation.TransformPoint(np.zeros(3, np.float64))
                utils.io.image.write_multichannel_np(image, self.output_file_for_current_iteration(current_id + '_input.mha'), normalization_mode=image_normalization, split_channel_axis=True, sitk_image_mode='default', data_format=self.data_format, image_type=output_image_type, spacing=self.image_spacing, origin=origin)
                utils.io.image.write_multichannel_np(prediction, self.output_file_for_current_iteration(current_id + '_prediction.mha'), normalization_mode=heatmap_normalization, split_channel_axis=True, sitk_image_mode='default', data_format=self.data_format, image_type=output_image_type, spacing=self.image_spacing, origin=origin)
                #utils.io.image.write(predictions_sitk[0], self.output_file_for_current_iteration(current_id + '_prediction_original.mha'))

            predictions_com = input_image.TransformContinuousIndexToPhysicalPoint(list(reversed(utils.np_image.center_of_mass(utils.sitk_np.sitk_to_np_no_copy(predictions_sitk[0])))))
            current_landmark = [Landmark(predictions_com)]
            landmarks[current_id] = current_landmark

            if self.has_validation_groundtruth:
                landmark_statistics.add_landmarks(current_id, current_landmark, groundtruth_landmark)

            print_progress_bar(i, num_entries, prefix='Testing ', suffix=' complete')

        utils.io.landmark.save_points_csv(landmarks, self.output_file_for_current_iteration('points.csv'))

        # finalize loss values
        if self.has_validation_groundtruth:
            print(landmark_statistics.get_pe_overview_string())
            summary_values = OrderedDict(zip(self.point_statistics_names, list(landmark_statistics.get_pe_statistics())))

            # finalize loss values
            self.val_loss_aggregator.finalize(self.current_iter, summary_values)
            overview_string = landmark_statistics.get_overview_string([2, 2.5, 3, 4, 10, 20], 10)
            utils.io.text.save_string_txt(overview_string, self.output_file_for_current_iteration('eval.txt'))


if __name__ == '__main__':
    network_parameters = OrderedDict([('num_filters_base', 64), ('double_features_per_level', False), ('num_levels', 5), ('activation', 'relu')])
    for cv in ['train_all', 0, 1, 2]:
        loop = MainLoop(cv, network_u, UnetClassicAvgLinear3d, network_parameters, 0.0001, output_folder_name='baseline')
        loop.run()
