#!/usr/bin/python

import os
from collections import OrderedDict

import numpy as np
import tensorflow as tf

import utils.io.image
import utils.io.landmark
import utils.io.text
import utils.sitk_image
import utils.sitk_np
from dataset import Dataset
from network import spatial_configuration_net, UnetClassicAvgLinear3d
from datasets.pyro_dataset import PyroClientDataset
from tensorflow_train.data_generator import DataGenerator
from tensorflow_train.train_loop import MainLoopBase
from tensorflow_train.utils.data_format import get_batch_channel_image_size
from tensorflow_train.utils.heatmap_image_generator import generate_heatmap_target
from tensorflow_train.utils.summary_handler import create_summary_placeholder
from tensorflow_train.utils.tensorflow_util import get_reg_loss, create_placeholders_tuple, print_progress_bar
from utils.image_tiler import ImageTiler, LandmarkTiler
from utils.landmark.heatmap_test import HeatmapTest
from utils.landmark.landmark_statistics import LandmarkStatistics
from utils.landmark.spine_postprocessing import SpinePostprocessing


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
        self.learning_rate_boundaries = [50000, 75000]
        self.max_iter = 100000
        self.test_iter = 10000
        self.disp_iter = 100
        self.snapshot_iter = 5000
        self.test_initialization = False
        self.current_iter = 0
        self.reg_constant = 0.0005
        self.use_background = True
        self.num_landmarks = 25
        self.heatmap_sigma = 4.0
        self.learnable_sigma = True
        self.data_format = 'channels_first'
        self.network = network
        self.unet = unet
        self.network_parameters = network_parameters
        self.padding = 'same'
        self.clip_gradient_global_norm = 100000.0

        self.use_pyro_dataset = False
        self.use_spine_postprocessing = True
        self.save_output_images = True
        self.save_output_images_as_uint = True  # set to False, if you want to see the direct network output
        self.save_debug_images = False
        self.has_validation_groundtruth = cv in [0, 1, 2]
        self.local_base_folder = '../verse2019_dataset'
        self.image_size = [96, 96, 128]
        self.image_spacing = [2] * 3
        self.cropped_inc = [0, 96, 0, 0]
        self.heatmap_size = self.image_size
        self.sigma_regularization = 100
        self.sigma_scale = 1000.0
        self.cropped_training = True
        self.output_folder = os.path.join('./output/vertebrae_localization/', network.__name__, unet.__name__ , output_folder_name, str(cv), self.output_folder_timestamp())
        dataset_parameters = {'base_folder': self.local_base_folder,
                              'image_size': self.image_size,
                              'image_spacing': self.image_spacing,
                              'cv': cv,
                              'input_gaussian_sigma': 0.75,
                              'generate_landmarks': True,
                              'generate_landmark_mask': True,
                              'translate_to_center_landmarks': True,
                              'translate_by_random_factor': True,
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

        self.point_statistics_names = ['pe_mean', 'pe_stdev', 'pe_median', 'num_correct']
        self.additional_summaries_placeholders_val = dict([(name, create_summary_placeholder(name)) for name in self.point_statistics_names])

    def loss_function(self, pred, target, mask=None):
        """
        L2 loss function calculated with prediction and target.
        :param pred: The predicted image.
        :param target: The target image.
        :param mask: If not none, calculate loss only pixels, where mask == 1
        :return: L2 loss of (pred - target) / batch_size
        """
        batch_size, _, _ = get_batch_channel_image_size(pred, self.data_format)
        if mask is not None:
            return tf.nn.l2_loss((pred - target) * mask) / batch_size
        else:
            return tf.nn.l2_loss(pred - target) / batch_size

    def loss_function_sigmas(self, sigmas, valid_landmarks):
        """
        L2 loss function for sigmas. Only calculated for values ver valid_landmarks == 1.
        :param sigmas: Sigma variables.
        :param valid_landmarks: Valid landmarks. Needs to have same shape as sigmas.
        :return: L2 loss of sigmas * valid_landmarks.
        """
        return self.sigma_regularization * tf.nn.l2_loss(sigmas * valid_landmarks)

    def init_networks(self):
        """
        Initialize networks and placeholders.
        """
        network_image_size = list(reversed(self.image_size))

        if self.data_format == 'channels_first':
            data_generator_entries = OrderedDict([('image', [1] + network_image_size),
                                                  ('landmarks', [self.num_landmarks, 4]),
                                                  ('landmark_mask', [1] + network_image_size)])
        else:
            data_generator_entries = OrderedDict([('image', network_image_size + [1]),
                                                  ('landmarks', [self.num_landmarks, 4]),
                                                  ('landmark_mask', network_image_size + [1])])

        data_generator_types = {'image': tf.float32}

        # create sigmas variable
        sigmas = tf.get_variable('sigmas', [self.num_landmarks], initializer=tf.constant_initializer(self.heatmap_sigma))
        if not self.learnable_sigma:
            sigmas = tf.stop_gradient(sigmas)
        mean_sigmas = tf.reduce_mean(sigmas)

        # create model with shared weights between train and val
        training_net = tf.make_template('net', self.network)

        # build train graph
        self.train_queue = DataGenerator(coord=self.coord, dataset=self.dataset_train, data_names_and_shapes=data_generator_entries, data_types=data_generator_types, batch_size=self.batch_size)
        data, target_landmarks, landmark_mask = self.train_queue.dequeue()
        target_heatmaps = generate_heatmap_target(list(reversed(self.heatmap_size)), target_landmarks, sigmas, scale=self.sigma_scale, normalize=True, data_format=self.data_format)
        prediction, local_prediction, spatial_prediction = training_net(data, num_labels=self.num_landmarks, is_training=True, actual_network=self.unet, padding=self.padding, **self.network_parameters)
        # losses
        self.loss_net = self.loss_function(target=target_heatmaps, pred=prediction, mask=landmark_mask)
        self.loss_sigmas = self.loss_function_sigmas(sigmas, target_landmarks[0, :, 0])
        self.loss_reg = get_reg_loss(self.reg_constant)
        self.loss = self.loss_net + self.loss_reg + self.loss_sigmas

        # solver
        global_step = tf.Variable(self.current_iter, trainable=False)
        learning_rate = tf.train.piecewise_constant(global_step, self.learning_rate_boundaries, self.learning_rates)
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.99, use_nesterov=True)
        unclipped_gradients, variables = zip(*optimizer.compute_gradients(self.loss))
        norm = tf.global_norm(unclipped_gradients)
        if self.clip_gradient_global_norm > 0:
            gradients, _ = tf.clip_by_global_norm(unclipped_gradients, self.clip_gradient_global_norm)
        else:
            gradients = unclipped_gradients
        self.optimizer = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)
        self.train_losses = OrderedDict([('loss', self.loss_net), ('loss_reg', self.loss_reg), ('loss_sigmas', self.loss_sigmas), ('mean_sigmas', mean_sigmas), ('gradient_norm', norm)])

        # build val graph
        self.data_val, self.target_landmarks_val, self.landmark_mask_val = create_placeholders_tuple(data_generator_entries, data_types=data_generator_types, shape_prefix=[1])
        self.target_heatmaps_val = generate_heatmap_target(list(reversed(self.heatmap_size)), self.target_landmarks_val, sigmas, scale=self.sigma_scale, normalize=True, data_format=self.data_format)
        self.prediction_val, self.local_prediction_val, self.spatial_prediction_val = training_net(self.data_val, num_labels=self.num_landmarks, is_training=False, actual_network=self.unet, padding=self.padding, **self.network_parameters)

        if self.has_validation_groundtruth:
            self.loss_val = self.loss_function(target=self.target_heatmaps_val, pred=self.prediction_val)
            self.val_losses = OrderedDict([('loss', self.loss_val), ('loss_reg', self.loss_reg), ('loss_sigmas', tf.constant(0, tf.float32)), ('mean_sigmas', tf.constant(0, tf.float32)), ('gradient_norm', tf.constant(0, tf.float32))])

    def test_cropped_image(self, dataset_entry):
        """
        Perform inference on a dataset_entry with the validation network. Performs cropped prediction and merges outputs as maxima.
        :param dataset_entry: A dataset entry from the dataset.
        :return: input image (np.array), target heatmaps (np.array), predicted heatmaps,  transformation (sitk.Transform)
        """
        generators = dataset_entry['generators']
        transformations = dataset_entry['transformations']
        transformation = transformations['image']

        image_size_np = [1] + list(reversed(self.image_size))
        labels_size_np = [self.num_landmarks] + list(reversed(self.image_size))
        full_image = generators['image']
        landmarks = generators['landmarks']
        image_tiler = ImageTiler(full_image.shape, image_size_np, self.cropped_inc, True, -1)
        landmark_tiler = LandmarkTiler(full_image.shape, image_size_np, self.cropped_inc)
        prediction_tiler = ImageTiler((self.num_landmarks,) + full_image.shape[1:], labels_size_np, self.cropped_inc, True, 0)

        for image_tiler, landmark_tiler, prediction_tiler in zip(image_tiler, landmark_tiler, prediction_tiler):
            current_image = image_tiler.get_current_data(full_image)
            current_landmarks = landmark_tiler.get_current_data(landmarks)
            if self.has_validation_groundtruth:
                feed_dict = {self.data_val: np.expand_dims(current_image, axis=0),
                             self.target_landmarks_val: np.expand_dims(current_landmarks, axis=0)}
                run_tuple = self.sess.run((self.prediction_val, self.loss_val) + self.val_loss_aggregator.get_update_ops(),
                                          feed_dict=feed_dict)
            else:
                feed_dict = {self.data_val: np.expand_dims(current_image, axis=0)}
                run_tuple = self.sess.run((self.prediction_val,), feed_dict=feed_dict)
            prediction = np.squeeze(run_tuple[0], axis=0)
            image_tiler.set_current_data(current_image)
            prediction_tiler.set_current_data(prediction)

        return image_tiler.output_image, prediction_tiler.output_image, transformation

    def test(self):
        """
        The test function. Performs inference on the the validation images and calculates the loss.
        """
        print('Testing...')

        channel_axis = 0
        if self.data_format == 'channels_last':
            channel_axis = 3
        if self.use_spine_postprocessing:
            heatmap_maxima = HeatmapTest(channel_axis, False, return_multiple_maxima=True, min_max_distance=7, min_max_value=0.25, multiple_min_max_value_factor=0.1)
            spine_postprocessing = SpinePostprocessing(num_landmarks=self.num_landmarks, image_spacing=self.image_spacing)
        else:
            heatmap_maxima = HeatmapTest(channel_axis, False)

        landmark_statistics = LandmarkStatistics()
        landmarks = {}
        num_entries = self.dataset_val.num_entries()
        for i in range(num_entries):
            dataset_entry = self.dataset_val.get_next()
            current_id = dataset_entry['id']['image_id']
            datasources = dataset_entry['datasources']
            input_image = datasources['image']
            target_landmarks = datasources['landmarks']

            image, prediction, transformation = self.test_cropped_image(dataset_entry)

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
                utils.io.image.write_multichannel_np(prediction, self.output_file_for_current_iteration(current_id + '_prediction.mha'), normalization_mode=heatmap_normalization, split_channel_axis=True, sitk_image_mode='vector', data_format=self.data_format, image_type=output_image_type, spacing=self.image_spacing, origin=origin)

            if self.use_spine_postprocessing:
                local_maxima_landmarks = heatmap_maxima.get_landmarks(prediction, input_image, self.image_spacing, transformation)
                landmark_sequence = spine_postprocessing.postprocess_landmarks(local_maxima_landmarks, prediction.shape)
                landmarks[current_id] = landmark_sequence
            else:
                maxima_landmarks = heatmap_maxima.get_landmarks(prediction, input_image, self.image_spacing, transformation)
                landmarks[current_id] = maxima_landmarks

            if self.has_validation_groundtruth:
                landmark_statistics.add_landmarks(current_id, landmark_sequence, target_landmarks)

            print_progress_bar(i, num_entries, prefix='Testing ', suffix=' complete')

        utils.io.landmark.save_points_csv(landmarks, self.output_file_for_current_iteration('points.csv'))

        # finalize loss values
        if self.has_validation_groundtruth:
            print(landmark_statistics.get_pe_overview_string())
            print(landmark_statistics.get_correct_id_string(20.0))
            summary_values = OrderedDict(zip(self.point_statistics_names, list(landmark_statistics.get_pe_statistics()) + [landmark_statistics.get_correct_id(20)]))

            # finalize loss values
            self.val_loss_aggregator.finalize(self.current_iter, summary_values)
            overview_string = landmark_statistics.get_overview_string([2, 2.5, 3, 4, 10, 20], 10, 20.0)
            utils.io.text.save_string_txt(overview_string, self.output_file_for_current_iteration('eval.txt'))


if __name__ == '__main__':
    network_parameters = OrderedDict([('num_filters_base', 64), ('double_features_per_level', False), ('num_levels', 5), ('activation', 'relu'), ('spatial_downsample', 4)])
    for cv in ['train_all', 0, 1, 2]:
        loop = MainLoop(cv, spatial_configuration_net, UnetClassicAvgLinear3d, network_parameters, 0.00000001, output_folder_name='baseline')
        loop.run()
