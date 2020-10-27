#!/usr/bin/python

from collections import OrderedDict
import numpy as np
import tensorflow as tf
import utils.io.image
from tensorflow_train.data_generator import DataGenerator
from tensorflow_train.losses.semantic_segmentation_losses import sigmoid_cross_entropy_with_logits
from tensorflow_train.train_loop import MainLoopBase
import utils.sitk_image
import utils.sitk_np
from utils.segmentation.segmentation_test import SegmentationTest
from utils.segmentation.segmentation_statistics import SegmentationStatistics
from utils.segmentation.metrics import DiceMetric, HausdorffDistanceMetric
from dataset import Dataset
from network import network_u, UnetClassicAvgLinear3d
from datasets.pyro_dataset import PyroClientDataset
from tensorflow_train.utils.summary_handler import create_summary_placeholder
from tensorflow_train.utils.tensorflow_util import get_reg_loss, create_placeholders_tuple, print_progress_bar
import os
import SimpleITK as sitk
import utils.io.text
import utils.np_image


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
        self.learning_rate_boundaries = [20000, 30000]
        self.max_iter = 50000
        self.test_iter = 5000
        self.disp_iter = 100
        self.snapshot_iter = 5000
        self.test_initialization = False
        self.current_iter = 0
        self.reg_constant = 0.000001
        self.num_labels = 1
        self.num_labels_all = 26
        self.data_format = 'channels_first'
        self.channel_axis = 1
        self.network = network
        self.unet = unet
        self.network_parameters = network_parameters
        self.padding = 'same'
        self.clip_gradient_global_norm = 1.0

        self.use_pyro_dataset = False
        self.save_output_images = True
        self.save_output_images_as_uint = True  # set to False, if you want to see the direct network output
        self.save_debug_images = False
        self.has_validation_groundtruth = cv in [0, 1, 2]
        self.local_base_folder = '../verse2019_dataset'
        self.image_size = [128, 128, 96]
        self.image_spacing = [1] * 3
        self.output_folder = os.path.join('./output/vertebrae_segmentation/', network.__name__, unet.__name__, output_folder_name, str(cv), self.output_folder_timestamp())
        dataset_parameters = {'base_folder': self.local_base_folder,
                              'image_size': self.image_size,
                              'image_spacing': self.image_spacing,
                              'cv': cv,
                              'input_gaussian_sigma': 0.75,
                              'label_gaussian_sigma': 1.0,
                              'heatmap_sigma': 3.0,
                              'generate_single_vertebrae_heatmap': True,
                              'generate_single_vertebrae': True,
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

        self.dice_names = ['mean_dice'] + list(map(lambda x: 'dice_{}'.format(x), range(self.num_labels_all)))
        self.hausdorff_names = ['mean_h'] + list(map(lambda x: 'h_{}'.format(x), range(self.num_labels)))
        self.additional_summaries_placeholders_val = dict([(name, create_summary_placeholder(name)) for name in (self.dice_names + self.hausdorff_names)])
        self.loss_function = sigmoid_cross_entropy_with_logits

        self.setup_base_folder = os.path.join(self.local_base_folder, 'setup')
        if cv in [0, 1, 2]:
            self.cv_folder = os.path.join(self.setup_base_folder, os.path.join('cv', str(cv)))
            self.test_file = os.path.join(self.cv_folder, 'val.txt')
        else:
            self.test_file = os.path.join(self.setup_base_folder, 'train_all.txt')
        self.valid_landmarks_file = os.path.join(self.setup_base_folder, 'valid_landmarks.csv')
        self.test_id_list = utils.io.text.load_list(self.test_file)
        self.valid_landmarks = utils.io.text.load_dict_csv(self.valid_landmarks_file)

    def init_networks(self):
        """
        Initialize networks and placeholders.
        """
        network_image_size = list(reversed(self.image_size))

        if self.data_format == 'channels_first':
            data_generator_entries = OrderedDict([('image', [1] + network_image_size),
                                                  ('single_label', [self.num_labels] + network_image_size),
                                                  ('single_heatmap', [1] + network_image_size)])
        else:
            data_generator_entries = OrderedDict([('image', network_image_size + [1]),
                                                  ('single_label', network_image_size + [self.num_labels]),
                                                  ('single_heatmap', network_image_size + [1])])

        data_generator_types = {'image':  tf.float32,
                                'labels': tf.uint8}

        # create model with shared weights between train and val
        training_net = tf.make_template('net', self.network)

        # build train graph
        self.train_queue = DataGenerator(coord=self.coord, dataset=self.dataset_train, data_names_and_shapes=data_generator_entries, data_types=data_generator_types, batch_size=self.batch_size)
        data, mask, single_heatmap = self.train_queue.dequeue()
        data_heatmap_concat = tf.concat([data, single_heatmap], axis=1)
        prediction = training_net(data_heatmap_concat, num_labels=self.num_labels, is_training=True, actual_network=self.unet, padding=self.padding, **self.network_parameters)
        # losses
        self.loss_net = self.loss_function(labels=mask, logits=prediction, data_format=self.data_format)
        self.loss_reg = get_reg_loss(self.reg_constant)
        self.loss = self.loss_net + self.loss_reg

        # solver
        global_step = tf.Variable(self.current_iter, trainable=False)
        learning_rate = tf.train.piecewise_constant(global_step, self.learning_rate_boundaries, self.learning_rates)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        unclipped_gradients, variables = zip(*optimizer.compute_gradients(self.loss))
        norm = tf.global_norm(unclipped_gradients)
        if self.clip_gradient_global_norm > 0:
            gradients, _ = tf.clip_by_global_norm(unclipped_gradients, self.clip_gradient_global_norm)
        else:
            gradients = unclipped_gradients
        self.optimizer = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)
        self.train_losses = OrderedDict([('loss', self.loss_net), ('loss_reg', self.loss_reg), ('gradient_norm', norm)])

        # build val graph
        self.data_val, self.mask_val, self.single_heatmap_val = create_placeholders_tuple(data_generator_entries, data_types=data_generator_types, shape_prefix=[1])
        self.data_heatmap_concat_val = tf.concat([self.data_val, self.single_heatmap_val], axis=1)
        self.prediction_val = training_net(self.data_heatmap_concat_val, num_labels=self.num_labels, is_training=False, actual_network=self.unet, padding=self.padding, **self.network_parameters)
        self.prediction_softmax_val = tf.nn.sigmoid(self.prediction_val)

        if self.has_validation_groundtruth:
            self.loss_val = self.loss_function(labels=self.mask_val, logits=self.prediction_val, data_format=self.data_format)
            self.val_losses = OrderedDict([('loss', self.loss_val), ('loss_reg', self.loss_reg), ('gradient_norm', tf.constant(0, tf.float32))])

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
                         self.mask_val: np.expand_dims(generators['single_label'], axis=0),
                         self.single_heatmap_val: np.expand_dims(generators['single_heatmap'], axis=0)}
            # run loss and update loss accumulators
            run_tuple = self.sess.run((self.prediction_softmax_val, self.loss_val) + self.val_loss_aggregator.get_update_ops(),
                                      feed_dict=feed_dict)
        else:
            feed_dict = {self.data_val: np.expand_dims(generators['image'], axis=0),
                         self.single_heatmap_val: np.expand_dims(generators['single_heatmap'], axis=0)}
            # run loss and update loss accumulators
            run_tuple = self.sess.run((self.prediction_softmax_val,), feed_dict=feed_dict)
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
        labels = list(range(self.num_labels_all))
        segmentation_test = SegmentationTest(labels,
                                             channel_axis=channel_axis,
                                             largest_connected_component=False,
                                             all_labels_are_connected=False)
        segmentation_statistics = SegmentationStatistics(list(range(self.num_labels_all)),
                                                         self.output_folder_for_current_iteration(),
                                                         metrics=OrderedDict([('dice', DiceMetric()),
                                                                              ('h', HausdorffDistanceMetric())]))
        filter_largest_cc = True

        # iterate over all images
        for i, image_id in enumerate(self.test_id_list):
            first = True
            prediction_resampled_np = None
            input_image = None
            groundtruth = None
            # iterate over all valid landmarks
            for landmark_id in self.valid_landmarks[image_id]:
                dataset_entry = self.dataset_val.get({'image_id': image_id, 'landmark_id' : landmark_id})
                datasources = dataset_entry['datasources']
                if first:
                    input_image = datasources['image']
                    if self.has_validation_groundtruth:
                        groundtruth = datasources['labels']
                    prediction_resampled_np = np.zeros([self.num_labels_all] + list(reversed(input_image.GetSize())), dtype=np.float16)
                    prediction_resampled_np[0, ...] = 0.5
                    first = False

                image, prediction, transformation = self.test_full_image(dataset_entry)

                if filter_largest_cc:
                    prediction_thresh_np = (prediction > 0.5).astype(np.uint8)
                    largest_connected_component = utils.np_image.largest_connected_component(prediction_thresh_np[0])
                    prediction_thresh_np[largest_connected_component[None, ...] == 1] = 0
                    prediction[prediction_thresh_np == 1] = 0

                if self.save_output_images:
                    if self.save_output_images_as_uint:
                        image_normalization = 'min_max'
                        label_normalization = (0, 1)
                        output_image_type = np.uint8
                    else:
                        image_normalization = None
                        label_normalization = None
                        output_image_type = np.float32
                    origin = transformation.TransformPoint(np.zeros(3, np.float64))
                    utils.io.image.write_multichannel_np(image, self.output_file_for_current_iteration(image_id + '_' + landmark_id + '_input.mha'), normalization_mode=image_normalization, split_channel_axis=True, sitk_image_mode='default', data_format=self.data_format, image_type=output_image_type, spacing=self.image_spacing, origin=origin)
                    utils.io.image.write_multichannel_np(prediction, self.output_file_for_current_iteration(image_id + '_' + landmark_id + '_prediction.mha'), normalization_mode=label_normalization, split_channel_axis=True, sitk_image_mode='default', data_format=self.data_format, image_type=output_image_type, spacing=self.image_spacing, origin=origin)

                prediction_resampled_sitk = utils.sitk_image.transform_np_output_to_sitk_input(output_image=prediction,
                                                                                               output_spacing=self.image_spacing,
                                                                                               channel_axis=channel_axis,
                                                                                               input_image_sitk=input_image,
                                                                                               transform=transformation,
                                                                                               interpolator='linear',
                                                                                               output_pixel_type=sitk.sitkFloat32)
                #utils.io.image.write(prediction_resampled_sitk[0],  self.output_file_for_current_iteration(image_id + '_' + landmark_id + '_resampled.mha'))
                if self.data_format == 'channels_first':
                    prediction_resampled_np[int(landmark_id) + 1, ...] = utils.sitk_np.sitk_to_np(prediction_resampled_sitk[0])
                else:
                    prediction_resampled_np[..., int(landmark_id) + 1] = utils.sitk_np.sitk_to_np(prediction_resampled_sitk[0])
            prediction_labels = segmentation_test.get_label_image(prediction_resampled_np, reference_sitk=input_image)
            # delete to save memory
            del prediction_resampled_np
            utils.io.image.write(prediction_labels, self.output_file_for_current_iteration(image_id + '.mha'))

            if self.has_validation_groundtruth:
                segmentation_statistics.add_labels(image_id, prediction_labels, groundtruth)

            print_progress_bar(i, len(self.test_id_list), prefix='Testing ', suffix=' complete')

        # finalize loss values
        if self.has_validation_groundtruth:
            segmentation_statistics.finalize()
            dice_list = segmentation_statistics.get_metric_mean_list('dice')
            mean_dice = np.nanmean(dice_list)
            dice_list = [mean_dice] + dice_list
            hausdorff_list = segmentation_statistics.get_metric_mean_list('h')
            mean_hausdorff = np.mean(hausdorff_list)
            hausdorff_list = [mean_hausdorff] + hausdorff_list
            summary_values = OrderedDict(list(zip(self.dice_names, dice_list)) + list(zip(self.hausdorff_names, hausdorff_list)))
            self.val_loss_aggregator.finalize(self.current_iter, summary_values=summary_values)


if __name__ == '__main__':
    network_parameters = OrderedDict([('num_filters_base', 64), ('double_features_per_level', False), ('num_levels', 5), ('activation', 'relu'), ('dropout_ratio', 0.25)])
    for cv in ['train_all']:
        loop = MainLoop(cv, network_u, UnetClassicAvgLinear3d, network_parameters, 0.0001, output_folder_name='baseline')
        loop.run()

