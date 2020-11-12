#!/usr/bin/python
import socket
from collections import OrderedDict

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
from spine_localization_postprocessing import bb, bb_iou
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow_train_v2.dataset.dataset_iterator import DatasetIterator
from tensorflow_train_v2.train_loop import MainLoopBase
from tensorflow_train_v2.utils.data_format import get_batch_channel_image_size
from tensorflow_train_v2.utils.loss_metric_logger import LossMetricLogger
from tensorflow_train_v2.utils.output_folder_handler import OutputFolderHandler
from tqdm import tqdm


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
        self.learning_rates = [self.learning_rate, self.learning_rate * 0.5, self.learning_rate * 0.1]
        self.learning_rate_boundaries = [50000, 75000]
        self.max_iter = 10000
        self.test_iter = 5000
        self.disp_iter = 100
        self.snapshot_iter = 5000
        self.test_initialization = False
        self.reg_constant = 0.0
        self.data_format = 'channels_first'
        self.network_parameters = OrderedDict(num_filters_base=config.num_filters_base,
                                              activation=config.activation,
                                              dropout_ratio=config.dropout_ratio,
                                              num_levels=config.num_levels,
                                              heatmap_initialization=True,
                                              data_format=self.data_format)
        if config.model == 'unet':
            self.network = Unet
        self.clip_gradient_global_norm = 100000.0

        self.use_pyro_dataset = True
        self.save_output_images = True
        self.save_debug_images = False
        self.has_validation_groundtruth = cv in [0, 1, 2]
        self.local_base_folder = '../verse2020_dataset'
        self.image_size = [None, None, None]
        self.image_spacing = [config.spacing] * 3
        self.sigma_regularization = 100.0
        self.sigma_scale = 1000.0
        self.cropped_training = True
        self.base_output_folder = './output/spine_localization/'
        self.additional_output_folder_info = config.info

        self.call_model_and_loss = tf.function(self.call_model_and_loss,
                                               input_signature=[tf.TensorSpec(tf.TensorShape([1, 1] + list(reversed(self.image_size))), tf.float16 if self.use_mixed_precision else tf.float32),
                                                                tf.TensorSpec(tf.TensorShape([1, 1] + list(reversed(self.image_size))), tf.float32),
                                                                tf.TensorSpec(tf.TensorShape(None), tf.bool)])

    def init_model(self):
        """
        Init self.model.
        """
        self.norm_moving_average = tf.Variable(10.0)
        self.model = self.network(num_labels=1, **self.network_parameters)

    def init_optimizer(self):
        """
        Init self.optimizer.
        """
        self.learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(self.learning_rate, self.max_iter, 0.1)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        if self.use_mixed_precision:
            self.optimizer = mixed_precision.LossScaleOptimizer(self.optimizer,
                                                                loss_scale=tf.mixed_precision.experimental.DynamicLossScale(initial_loss_scale=2 ** 15, increment_period=1000))

    def init_checkpoint(self):
        """
        Init self.checkpoint.
        """
        self.checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)

    def init_output_folder_handler(self):
        """
        Init self.output_folder_handler.
        """
        self.output_folder_handler = OutputFolderHandler(self.base_output_folder,
                                                         model_name=self.model.name,
                                                         cv=str(self.cv),
                                                         additional_info=self.additional_output_folder_info)

    def init_datasets(self):
        """
        Init self.dataset_train, self.dataset_train_iter, self.dataset_val.
        """
        dataset_parameters = dict(base_folder=self.local_base_folder,
                                  image_size=self.image_size,
                                  image_spacing=self.image_spacing,
                                  normalize_zero_mean_unit_variance=False,
                                  cv=self.cv,
                                  heatmap_sigma=3.0,
                                  generate_spine_heatmap=True,
                                  use_variable_image_size=True,
                                  valid_output_sizes_x=[32, 64, 96, 128],
                                  valid_output_sizes_y=[32, 64, 96, 128],
                                  valid_output_sizes_z=[32, 64, 96, 128],
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
                                                  ('spine_heatmap', [1] + self.network_image_size),
                                                  ('image_id', tuple())])
        else:
            data_generator_entries = OrderedDict([('image', self.network_image_size + [1]),
                                                  ('spine_heatmap', self.network_image_size + [1]),
                                                  ('image_id', tuple())])

        data_generator_types = {'image': tf.float16 if self.use_mixed_precision else tf.float32, 'spine_heatmap': tf.float32, 'image_id': tf.string}
        self.dataset_train_iter = DatasetIterator(dataset=self.dataset_train,
                                                  data_names_and_shapes=data_generator_entries,
                                                  data_types=data_generator_types,
                                                  batch_size=self.batch_size)

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
    def loss_function(self, pred, target, mask=None):
        """
        L2 loss function calculated with prediction and target.
        :param pred: The predicted image.
        :param target: The target image.
        :param mask: If not none, calculate loss only pixels, where mask == 1
        :return: L2 loss of (pred - target) / batch_size
        """
        batch_size, channel_size, image_size = get_batch_channel_image_size(pred, self.data_format, as_tensor=True)
        if mask is not None:
            diff = (pred - target) * mask
        else:
            diff = pred - target
        return tf.nn.l2_loss(diff) / tf.cast(batch_size * 1024, tf.float32) #* channel_size * np.prod(image_size))

    def call_model_and_loss(self, image, target_heatmap, training):
        """
        Call model and loss.
        :param image: The image to call the model with.
        :param target_heatmap: The target heatmap used for loss calculation.
        :param training: training parameter used for calling the model.
        :return (prediction, losses) tuple
        """
        prediction = self.model(image, training=training)
        losses = {}
        losses['loss_net'] = self.loss_function(target=target_heatmap, pred=prediction)
        return prediction, losses

    @tf.function
    def train_step(self):
        """
        Perform a training step.
        """
        image, target_landmarks, image_id = self.dataset_train_iter.get_next()
        with tf.GradientTape() as tape:
            _, losses = self.call_model_and_loss(image, target_landmarks, training=True)
            if self.reg_constant > 0:
                losses['loss_reg'] = self.reg_constant * tf.reduce_sum(self.model.losses)
            loss = tf.reduce_sum(list(losses.values()))
            if self.use_mixed_precision:
                scaled_loss = self.optimizer.get_scaled_loss(loss)
        variables = self.model.trainable_weights
        metric_dict = losses
        clip_norm = self.norm_moving_average * 5
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
            alpha = 0.01
            self.norm_moving_average.assign(alpha * tf.minimum(norm, clip_norm) + (1 - alpha) * self.norm_moving_average)
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
        if self.has_validation_groundtruth:
            spine_heatmap = np.expand_dims(generators['spine_heatmap'], axis=0)
            prediction, losses = self.call_model_and_loss(image, spine_heatmap, training=False)
            self.loss_metric_logger_val.update_metrics(losses)
        else:
            prediction = self.model(image, training=False)
        prediction = np.squeeze(prediction, axis=0)
        transformation = transformations['image']
        image = generators['image']

        return image, prediction, transformation

    def test(self):
        """
        The test function. Performs inference on the the validation images and calculates the loss.
        """
        print('Testing...')

        num_entries = self.dataset_val.num_entries()
        ious = {}
        for _ in tqdm(range(num_entries), desc='Testing'):
            dataset_entry = self.dataset_val.get_next()
            current_id = dataset_entry['id']['image_id']
            print(current_id)
            image, prediction, transformation = self.test_full_image(dataset_entry)
            start_transformed, end_transformed = bb(prediction, transformation, self.image_spacing)
            if self.has_validation_groundtruth:
                groundtruth = dataset_entry['generators']['spine_heatmap']
                gt_start_transformed, gt_end_transformed = bb(groundtruth, transformation, self.image_spacing)
                iou = bb_iou((start_transformed, end_transformed), (gt_start_transformed, gt_end_transformed))
                ious[current_id] = iou

            if self.save_output_images:
                origin = transformation.TransformPoint(np.zeros(3, np.float64))
                utils.io.image.write_multichannel_np(image,
                                                     self.output_folder_handler.path_for_iteration(self.current_iter, current_id + '_input.mha'),
                                                     output_normalization_mode='min_max',
                                                     sitk_image_output_mode='vector',
                                                     data_format=self.data_format,
                                                     image_type=np.uint8,
                                                     spacing=self.image_spacing,
                                                     origin=origin)
                utils.io.image.write_multichannel_np(prediction,
                                                     self.output_folder_handler.path_for_iteration(self.current_iter, current_id + '_prediction.mha'),
                                                     output_normalization_mode=(0, 1),
                                                     sitk_image_output_mode='vector',
                                                     data_format=self.data_format,
                                                     image_type=np.uint8,
                                                     spacing=self.image_spacing,
                                                     origin=origin)

        # finalize loss values
        if self.has_validation_groundtruth:
            mean_iou = np.mean(list(ious.values()))
            self.loss_metric_logger_val.update_metrics({'mean_iou': mean_iou})

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
        # Set hyperparameters
        hyperparameter_defaults = dotdict(
            num_filters_base=96,
            dropout_ratio=0.25,
            activation='lrelu',
            learning_rate=0.0001,
            model='unet',
            num_levels=5,
            spacing=8.0,
            cv=cv,
            info='d0_25_fin',
        )

        with MainLoop(cv, hyperparameter_defaults) as loop:
            loop.run()
