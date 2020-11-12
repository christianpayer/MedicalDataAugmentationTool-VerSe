#!/usr/bin/python

import os
import socket
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
from dataset import Dataset
from datasets.pyro_dataset import PyroClientDataset
from network import Unet
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow_train_v2.dataset.dataset_iterator import DatasetIterator
from tensorflow_train_v2.train_loop import MainLoopBase
from tensorflow_train_v2.utils.loss_metric_logger import LossMetricLogger
from tensorflow_train_v2.utils.output_folder_handler import OutputFolderHandler
from tqdm import tqdm
from utils.segmentation.metrics import DiceMetric
from utils.segmentation.segmentation_statistics import SegmentationStatistics


class MainLoop(MainLoopBase):
    def __init__(self, cv, config):
        """
        Initializer.
        :param cv: The cv fold. 0, 1, 2 for CV; 'train_all' for training on whole dataset.
        :param config: config dictionary
        """
        super().__init__()
        self.use_mixed_precision = True
        if self.use_mixed_precision:
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_policy(policy)
        self.cv = cv
        self.config = config
        self.batch_size = 1
        self.learning_rate = config.learning_rate
        self.max_iter = 50000
        self.test_iter = 5000
        self.disp_iter = 100
        self.snapshot_iter = 5000
        self.test_initialization = False
        self.reg_constant = 0.0 #005
        self.use_background = True
        self.num_labels = 1
        self.num_labels_all = 27
        self.data_format = 'channels_first'
        self.network_parameters = OrderedDict(num_filters_base=config.num_filters_base,
                                              activation=config.activation,
                                              dropout_ratio=config.dropout_ratio,
                                              num_levels=config.num_levels,
                                              data_format=self.data_format)
        if config.model == 'unet':
            self.network = Unet
        self.clip_gradient_global_norm = 100.0

        self.use_pyro_dataset = True
        self.save_output_images = True
        self.save_debug_images = False
        self.has_validation_groundtruth = cv in [0, 1, 2]
        self.local_base_folder = '../verse2020_dataset'
        self.image_size = [128, 128, 96]
        self.image_spacing = [config.spacing] * 3
        self.heatmap_size = self.image_size
        self.base_output_folder = './output/vertebrae_segmentation/'
        self.additional_output_folder_info = config.info

        if self.data_format == 'channels_first':
            self.call_model_and_loss = tf.function(self.call_model_and_loss, input_signature=[tf.TensorSpec(tf.TensorShape([1, 2] + list(reversed(self.image_size))), tf.float16 if self.use_mixed_precision else tf.float32),
                                                                                              tf.TensorSpec(tf.TensorShape([1, 1] + list(reversed(self.image_size))), tf.uint8),
                                                                                              tf.TensorSpec(tf.TensorShape(None), tf.bool)])
        else:
            self.call_model_and_loss = tf.function(self.call_model_and_loss, input_signature=[tf.TensorSpec(tf.TensorShape([1] + list(reversed(self.image_size))) + [2], tf.float16 if self.use_mixed_precision else tf.float32),
                                                                                              tf.TensorSpec(tf.TensorShape([1] + list(reversed(self.image_size))) + [1], tf.uint8),
                                                                                              tf.TensorSpec(tf.TensorShape(None), tf.bool)])

        self.dice_names = ['mean_dice'] + list(map(lambda x: 'dice_{}'.format(x), range(1, self.num_labels_all)))

        self.setup_base_folder = os.path.join(self.local_base_folder, 'setup')
        if cv in [0, 1, 2]:
            self.cv_folder = os.path.join(self.setup_base_folder, os.path.join('cv', str(cv)))
            self.test_file = os.path.join(self.cv_folder, 'val.txt')
        else:
            self.test_file = os.path.join(self.setup_base_folder, 'train_all.txt')
        self.valid_landmarks_file = os.path.join(self.setup_base_folder, 'valid_landmarks.csv')
        self.test_id_list = utils.io.text.load_list(self.test_file)
        self.valid_landmarks = utils.io.text.load_dict_csv(self.valid_landmarks_file, squeeze=False)

        self.landmark_labels = [i + 1 for i in range(25)] + [28]
        self.landmark_mapping = dict([(i, self.landmark_labels[i]) for i in range(26)])
        self.landmark_mapping_inverse = dict([(self.landmark_labels[i], i) for i in range(26)])

    def init_model(self):
        """
        Init self.model.
        """
        # create sigmas variable
        self.norm_moving_average = tf.Variable(1.0)
        self.model = self.network(num_labels=self.num_labels, **self.network_parameters)

    def init_optimizer(self):
        """
        Init self.optimizer.
        """
        self.learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(self.learning_rate, self.max_iter, 0.1)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, amsgrad=True)
        if self.use_mixed_precision:
            self.optimizer = mixed_precision.LossScaleOptimizer(self.optimizer, loss_scale=tf.mixed_precision.experimental.DynamicLossScale(initial_loss_scale=2 ** 15, increment_period=1000))
        #self.optimizer_sigma = tf.keras.optimizers.Adam(learning_rate=self.learning_rate*10)

    def init_checkpoint(self):
        """
        Init self.checkpoint.
        """
        self.checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)

    def init_output_folder_handler(self):
        """
        Init self.output_folder_handler.
        """
        self.output_folder_handler = OutputFolderHandler(self.base_output_folder, model_name=self.model.name, cv=str(self.cv), additional_info=self.additional_output_folder_info)

    def init_datasets(self):
        """
        Init self.dataset_train, self.dataset_train_iter, self.dataset_val.
        """
        dataset_parameters = dict(base_folder=self.local_base_folder,
                                  image_size=self.image_size,
                                  image_spacing=self.image_spacing,
                                  normalize_zero_mean_unit_variance=False,
                                  cv=self.cv,
                                  label_gaussian_sigma=1.0,
                                  random_translation=10.0,
                                  random_rotate=0.5,
                                  heatmap_sigma=3.0,
                                  generate_single_vertebrae_heatmap=True,
                                  generate_single_vertebrae=True,
                                  output_image_type=np.float16 if self.use_mixed_precision else np.float32,
                                  data_format=self.data_format,
                                  save_debug_images=self.save_debug_images)

        dataset = Dataset(**dataset_parameters)
        if self.use_pyro_dataset:
            # TODO: adapt hostname, in case this script runs on a remote server
            hostname = socket.gethostname()
            server_name = '@' + hostname + ':52132'
            uri = 'PYRO:verse2020_dataset' + server_name
            print('using pyro uri', uri)
            try:
                self.dataset_train = PyroClientDataset(uri, **dataset_parameters)
            except Exception as e:
                print('Error connecting to server dataset. Start server_dataset_loop.py and set correct hostname, or set self.use_pyro_dataset = False.')
                raise e
        else:
            self.dataset_train = dataset.dataset_train()

        self.dataset_val = dataset.dataset_val()
        self.network_image_size = list(reversed(self.image_size))

        if self.data_format == 'channels_first':
            data_generator_entries = OrderedDict([('image', [1] + self.network_image_size),
                                                  ('single_label', [self.num_labels] + self.network_image_size),
                                                  ('single_heatmap', [1] + self.network_image_size)])
        else:
            data_generator_entries = OrderedDict([('image', self.network_image_size + [1]),
                                                  ('single_label', self.network_image_size + [self.num_labels]),
                                                  ('single_heatmap', self.network_image_size + [1])])

        data_generator_types = {'image': tf.float16 if self.use_mixed_precision else tf.float32,
                                'single_heatmap': tf.float16 if self.use_mixed_precision else tf.float32,
                                'single_label': tf.uint8}
        self.dataset_train_iter = DatasetIterator(dataset=self.dataset_train, data_names_and_shapes=data_generator_entries, data_types=data_generator_types, batch_size=self.batch_size, n_threads=4)

    def init_loggers(self):
        """
        Init self.loss_metric_logger_train, self.loss_metric_logger_val.
        """
        self.loss_metric_logger_train = LossMetricLogger('train',
                                                         self.output_folder_handler.path('train'),
                                                         self.output_folder_handler.path('train.csv'))
        self.loss_metric_logger_val = LossMetricLogger('test',
                                                       self.output_folder_handler.path('test'),
                                                       self.output_folder_handler.path('test.csv'))

    @tf.function
    def loss_function(self, pred, target):
        """
        L2 loss function calculated with prediction and target.
        :param pred: The predicted image.
        :param target: The target image.
        :return: L2 loss of (pred - target) / batch_size
        """
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(target, tf.float32), logits=tf.cast(pred, tf.float32)))

    def call_model_and_loss(self, image, target_labels, training):
        """
        Call model and loss.
        :param image: The image to call the model with.
        :param target_labels: The target labels used for loss calculation.
        :param training: training parameter used for calling the model.
        :return (prediction, losses) tuple
        """
        prediction = self.model(image, training=training)
        losses = {}
        losses['loss_net'] = self.loss_function(target=target_labels, pred=prediction)
        return prediction, losses

    @tf.function
    def train_step(self):
        """
        Perform a training step.
        """
        image, single_label, single_heatmap = self.dataset_train_iter.get_next()
        image_heatmap_concat = tf.concat([image, single_heatmap], axis=1 if self.data_format == 'channels_first' else -1)
        with tf.GradientTape() as tape:
            _, losses = self.call_model_and_loss(image_heatmap_concat, single_label, training=True)
            if self.reg_constant > 0:
                losses['loss_reg'] = self.reg_constant * tf.reduce_sum(self.model.losses)
            loss = tf.reduce_sum(list(losses.values()))
            if self.use_mixed_precision:
                scaled_loss = self.optimizer.get_scaled_loss(loss)
        variables = self.model.trainable_weights
        metric_dict = losses
        clip_norm = self.norm_moving_average * 2
        if self.use_mixed_precision:
            scaled_grads = tape.gradient(scaled_loss, variables)
            grads = self.optimizer.get_unscaled_gradients(scaled_grads)
            grads, norm = tf.clip_by_global_norm(grads, clip_norm)
            loss_scale = self.optimizer.loss_scale()
            metric_dict.update({'loss_scale': loss_scale})
        else:
            grads = tape.gradient(loss, variables)
            grads, norm = tf.clip_by_global_norm(grads, clip_norm)
        if tf.math.is_finite(norm):
            alpha = 0.99
            self.norm_moving_average.assign((1-alpha) * tf.minimum(norm, clip_norm) + alpha * self.norm_moving_average)
        metric_dict.update({'norm': norm, 'norm_average': self.norm_moving_average})
        self.optimizer.apply_gradients(zip(grads, variables))
        self.loss_metric_logger_train.update_metrics(metric_dict)

    def test_full_image(self, dataset_entry):
        """
        Perform inference on a dataset_entry with the validation network.
        :param dataset_entry: A dataset entry from the dataset.
        :return: input image (np.array), network prediction (np.array), transformation (sitk.Transform)
        """
        generators = dataset_entry['generators']
        transformations = dataset_entry['transformations']
        image = np.expand_dims(generators['image'], axis=0)
        single_heatmap = np.expand_dims(generators['single_heatmap'], axis=0)
        image_heatmap_concat = tf.concat([image, single_heatmap], axis=1 if self.data_format == 'channels_first' else -1)
        if self.has_validation_groundtruth:
            single_label = np.expand_dims(generators['single_label'], axis=0)
            prediction, losses = self.call_model_and_loss(image_heatmap_concat, single_label, training=False)
            self.loss_metric_logger_val.update_metrics(losses)
        else:
            prediction = self.model(image_heatmap_concat, training=False)
        prediction = np.squeeze(prediction, axis=0)
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
        segmentation_statistics = SegmentationStatistics(list(range(1, self.num_labels_all)),
                                                         self.output_folder_handler.path_for_iteration(self.current_iter),
                                                         metrics=OrderedDict([('dice', DiceMetric())]))
        filter_largest_cc = True

        # iterate over all images
        for image_id in tqdm(self.test_id_list, desc='Testing'):
            first = True
            prediction_labels_np = None
            prediction_max_value_np = None
            input_image = None
            groundtruth = None
            # iterate over all valid landmarks
            for landmark_id in self.valid_landmarks[image_id]:
                dataset_entry = self.dataset_val.get({'image_id': image_id, 'landmark_id': landmark_id})
                if first:
                    input_image = dataset_entry['datasources']['image']
                    if self.has_validation_groundtruth:
                        groundtruth = dataset_entry['datasources']['labels']
                    prediction_labels_np = np.zeros(list(reversed(input_image.GetSize())), dtype=np.uint8)
                    prediction_max_value_np = np.ones(list(reversed(input_image.GetSize())), dtype=np.float32) * 0.5
                    first = False

                image, prediction, transformation = self.test_full_image(dataset_entry)
                del dataset_entry

                origin = transformation.TransformPoint(np.zeros(3, np.float64))
                max_index = transformation.TransformPoint(np.array(self.image_size, np.float64) * np.array(self.image_spacing, np.float64))

                if self.save_output_images:
                    utils.io.image.write_multichannel_np(image, self.output_folder_handler.path('output', image_id + '_' + landmark_id + '_input.mha'), output_normalization_mode='min_max', sitk_image_output_mode='vector', data_format=self.data_format, image_type=np.uint8, spacing=self.image_spacing, origin=origin)
                    utils.io.image.write_multichannel_np(prediction, self.output_folder_handler.path('output', image_id + '_' + landmark_id + '_prediction.mha'), output_normalization_mode=(0, 1), sitk_image_output_mode='vector', data_format=self.data_format, image_type=np.uint8, spacing=self.image_spacing, origin=origin)
                del image
                prediction = prediction.astype(np.float32)
                prediction_resampled_sitk = utils.sitk_image.transform_np_output_to_sitk_input(output_image=prediction,
                                                                                               output_spacing=self.image_spacing,
                                                                                               channel_axis=channel_axis,
                                                                                               input_image_sitk=input_image,
                                                                                               transform=transformation,
                                                                                               interpolator='cubic',
                                                                                               output_pixel_type=sitk.sitkFloat32)
                del prediction
                prediction_resampled_np = utils.sitk_np.sitk_to_np(prediction_resampled_sitk[0])
                if self.save_output_images:
                    utils.io.image.write_multichannel_np(prediction_resampled_np, self.output_folder_handler.path('output', image_id + '_' + landmark_id + '_prediction_resampled.mha'), output_normalization_mode=(0, 1), is_single_channel=True, sitk_image_output_mode='vector', data_format=self.data_format, image_type=np.uint8, spacing=prediction_resampled_sitk[0].GetSpacing(), origin=prediction_resampled_sitk[0].GetOrigin())
                bb_start = np.floor(np.flip(origin / np.array(input_image.GetSpacing())))
                bb_start = np.maximum(bb_start, [0, 0, 0])
                bb_end = np.ceil(np.flip(max_index / np.array(input_image.GetSpacing())))
                bb_end = np.minimum(bb_end, prediction_resampled_np.shape - np.ones(3))  # bb is inclusive -> subtract [1, 1, 1] from max size
                slices = tuple([slice(int(bb_start[i]), int(bb_end[i] + 1)) for i in range(3)])
                prediction_resampled_cropped_np = prediction_resampled_np[slices]
                if filter_largest_cc:
                    prediction_thresh_cropped_np = (prediction_resampled_cropped_np > 0.5).astype(np.uint8)
                    largest_connected_component = utils.np_image.largest_connected_component(prediction_thresh_cropped_np)
                    prediction_thresh_cropped_np[largest_connected_component == 1] = 0
                    prediction_resampled_cropped_np[prediction_thresh_cropped_np == 1] = 0
                prediction_max_value_cropped_np = prediction_max_value_np[slices]
                prediction_labels_cropped_np = prediction_labels_np[slices]
                prediction_max_index_np = utils.np_image.argmax(np.stack([prediction_max_value_cropped_np, prediction_resampled_cropped_np], axis=-1), axis=-1)
                prediction_max_index_new_np = prediction_max_index_np == 1
                prediction_max_value_cropped_np[prediction_max_index_new_np] = prediction_resampled_cropped_np[prediction_max_index_new_np]
                prediction_labels_cropped_np[prediction_max_index_new_np] = self.landmark_mapping[int(landmark_id)]
                prediction_max_value_np[slices] = prediction_max_value_cropped_np
                prediction_labels_np[slices] = prediction_labels_cropped_np
                del prediction_resampled_sitk

            # delete to save memory
            del prediction_max_value_np
            prediction_labels = utils.sitk_np.np_to_sitk(prediction_labels_np)
            prediction_labels.CopyInformation(input_image)
            del prediction_labels_np
            utils.io.image.write(prediction_labels, self.output_folder_handler.path_for_iteration(self.current_iter, image_id + '.mha'))
            if self.save_output_images:
                prediction_labels_resampled = utils.sitk_np.sitk_to_np(utils.sitk_image.resample_to_spacing(prediction_labels, [1.0, 1.0, 1.0], 'nearest'))
                prediction_labels_resampled = np.flip(prediction_labels_resampled, axis=0)
                utils.io.image.write_multichannel_np(prediction_labels_resampled, self.output_folder_handler.path('output', image_id + '_seg.png'), channel_layout_mode='label_rgb', output_normalization_mode=(0, 1), image_layout_mode='max_projection', is_single_channel=True, sitk_image_output_mode='vector', data_format=self.data_format, image_type=np.uint8)
                utils.io.image.write_multichannel_np(prediction_labels_resampled, self.output_folder_handler.path('output', image_id + '_seg_rgb.mha'), channel_layout_mode='label_rgb', output_normalization_mode=(0, 1), is_single_channel=True, sitk_image_output_mode='vector', data_format=self.data_format, image_type=np.uint8)
                input_resampled = utils.sitk_np.sitk_to_np(utils.sitk_image.resample_to_spacing(input_image, [1.0, 1.0, 1.0], 'linear'))
                input_resampled = np.flip(input_resampled, axis=0)
                utils.io.image.write_multichannel_np(input_resampled, self.output_folder_handler.path('output', image_id + '_input.png'), output_normalization_mode='min_max', image_layout_mode='max_projection', is_single_channel=True, sitk_image_output_mode='vector', data_format=self.data_format, image_type=np.uint8)

            if self.has_validation_groundtruth:
                segmentation_statistics.add_labels(image_id, prediction_labels, groundtruth)
            del prediction_labels

        # finalize loss values
        if self.has_validation_groundtruth:
            segmentation_statistics.finalize()
            dice_list = segmentation_statistics.get_metric_mean_list('dice')
            mean_dice = np.nanmean(dice_list)
            dice_list = [mean_dice] + dice_list
            summary_values = OrderedDict(list(zip(self.dice_names, dice_list)))
            self.loss_metric_logger_val.update_metrics(summary_values)

        self.loss_metric_logger_val.finalize(self.current_iter)


class dotdict(dict):
    """
    Dict subclass that allows dot.notation to access attributes.
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


if __name__ == '__main__':
    for cv in [0, 1, 2, 'train_all']:
        # Set hyperparameters, which can be overwritten with a W&B Sweep
        config = dotdict(
            num_filters_base=96,
            dropout_ratio=0.25,
            activation='lrelu',
            learning_rate=0.0001,
            model='unet',
            num_levels=5,
            spacing=1.0,
            cv=cv,
            info='d0_25_fin',
        )
        with MainLoop(cv, config) as loop:
            loop.run()
