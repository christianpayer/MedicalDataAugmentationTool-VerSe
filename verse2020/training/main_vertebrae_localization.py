#!/usr/bin/python

import pickle
import socket
from collections import OrderedDict

import numpy as np
import tensorflow as tf
import utils.io.image
import utils.io.landmark
import utils.io.text
import utils.sitk_image
import utils.sitk_np
from dataset import Dataset
from datasets.pyro_dataset import PyroClientDataset
from network import SpatialConfigurationNet, Unet
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow_train_v2.dataset.dataset_iterator import DatasetIterator
from tensorflow_train_v2.train_loop import MainLoopBase
from tensorflow_train_v2.utils.data_format import get_batch_channel_image_size
from tensorflow_train_v2.utils.heatmap_image_generator import generate_heatmap_target
from tensorflow_train_v2.utils.loss_metric_logger import LossMetricLogger
from tensorflow_train_v2.utils.output_folder_handler import OutputFolderHandler
from tqdm import tqdm
from utils.image_tiler import ImageTiler, LandmarkTiler
from utils.landmark.common import Landmark
from utils.landmark.heatmap_test import HeatmapTest
from utils.landmark.landmark_statistics import LandmarkStatistics
from utils.landmark.spine_postprocessing_graph import SpinePostprocessingGraph
from utils.landmark.visualization.landmark_visualization_matplotlib import LandmarkVisualizationMatplotlib
from vertebrae_localization_postprocessing import add_landmarks_from_neighbors, filter_landmarks_top_bottom, reshift_landmarks


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
        self.has_validation_groundtruth = cv in [0, 1, 2]
        self.max_iter = 50000
        self.test_iter = 5000 if self.has_validation_groundtruth else self.max_iter
        self.disp_iter = 100
        self.snapshot_iter = 5000
        self.test_initialization = False
        self.reg_constant = 0.0
        self.use_background = True
        self.num_landmarks = 26
        self.heatmap_sigma = config.heatmap_sigma
        self.learnable_sigma = config.learnable_sigma
        self.data_format = 'channels_first'
        self.network_parameters = OrderedDict(num_filters_base=config.num_filters_base,
                                              activation=config.activation,
                                              spatial_downsample=config.spatial_downsample,
                                              dropout_ratio=config.dropout_ratio,
                                              local_activation=config.local_activation,
                                              spatial_activation=config.spatial_activation,
                                              num_levels=config.num_levels,
                                              data_format=self.data_format)
        if config.model == 'scn':
            self.network = SpatialConfigurationNet
        if config.model == 'unet':
            self.network = Unet
        self.clip_gradient_global_norm = 10000.0

        self.evaluate_landmarks_postprocessing = True
        self.use_pyro_dataset = True
        self.save_output_images = True
        self.save_debug_images = False
        self.local_base_folder = '../verse2020_dataset'
        self.image_size = [None, None, None]
        self.image_spacing = [config.spacing] * 3
        self.max_image_size_for_cropped_test = [128, 128, 448]
        self.cropped_inc = [0, 128, 0, 0]
        self.heatmap_size = self.image_size
        self.sigma_regularization = 100.0
        self.sigma_scale = 1000.0
        self.cropped_training = True
        self.base_output_folder = './output/vertebrae_localization/'
        self.additional_output_folder_info = config.info

        if self.data_format == 'channels_first':
            self.call_model_and_loss = tf.function(self.call_model_and_loss,
                                                   input_signature=[tf.TensorSpec(tf.TensorShape([1, 1] + list(reversed(self.image_size))), tf.float16 if self.use_mixed_precision else tf.float32),
                                                                    tf.TensorSpec(tf.TensorShape([1, self.num_landmarks, 4]), tf.float32),
                                                                    tf.TensorSpec(tf.TensorShape(None), tf.bool)])
        else:
            self.call_model_and_loss = tf.function(self.call_model_and_loss,
                                                   input_signature=[tf.TensorSpec(tf.TensorShape([1] + list(reversed(self.image_size))) + [1], tf.float16 if self.use_mixed_precision else tf.float32),
                                                                    tf.TensorSpec(tf.TensorShape([1, self.num_landmarks, 4]), tf.float32),
                                                                    tf.TensorSpec(tf.TensorShape(None), tf.bool)])

    def init_model(self):
        """
        Init self.model.
        """
        self.norm_moving_average = tf.Variable(10.0)
        # create sigmas variable
        self.sigmas_variables = tf.Variable([self.heatmap_sigma] * self.num_landmarks, name='sigmas', trainable=True)
        self.sigmas = self.sigmas_variables
        if not self.learnable_sigma:
            self.sigmas = tf.stop_gradient(self.sigmas)
        self.model = self.network(num_labels=self.num_landmarks, **self.network_parameters)

    def init_optimizer(self):
        """
        Init self.optimizer.
        """
        self.learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(self.learning_rate, self.max_iter, 0.1)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, amsgrad=False)
        if self.use_mixed_precision:
            self.optimizer = mixed_precision.LossScaleOptimizer(self.optimizer,
                                                                loss_scale=tf.mixed_precision.experimental.DynamicLossScale(initial_loss_scale=8, increment_period=1000))

    def init_checkpoint(self):
        """
        Init self.checkpoint.
        """
        self.checkpoint = tf.train.Checkpoint(model=self.model,
                                              optimizer=self.optimizer,
                                              sigmas=self.sigmas_variables)

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
                                  num_landmarks=self.num_landmarks,
                                  normalize_zero_mean_unit_variance=False,
                                  cv=self.cv,
                                  generate_landmarks=True,
                                  generate_landmark_mask=False,
                                  crop_image_top_bottom=True,
                                  crop_randomly_smaller=False,
                                  generate_heatmaps=False,
                                  use_variable_image_size=True,
                                  valid_output_sizes_x=[64, 96],
                                  valid_output_sizes_y=[64, 96],
                                  valid_output_sizes_z=[32, 64, 96, 128, 160, 192, 224, 256],
                                  translate_to_center_landmarks=True,
                                  translate_by_random_factor=True,
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
                                                  ('landmarks', [self.num_landmarks, 4]),
                                                  ('image_id', tuple())])
        else:
            data_generator_entries = OrderedDict([('image', self.network_image_size + [1]),
                                                  ('landmarks', [self.num_landmarks, 4]),
                                                  ('image_id', tuple())])

        data_generator_types = {'image': tf.float16 if self.use_mixed_precision else tf.float32, 'image_id': tf.string}
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
        return tf.nn.l2_loss(diff) / tf.cast(batch_size, tf.float32)

    @tf.function
    def loss_function_sigmas(self, sigmas, valid_landmarks):
        """
        L2 loss function for sigmas. Only calculated for values ver valid_landmarks == 1.
        :param sigmas: Sigma variables.
        :param valid_landmarks: Valid landmarks. Needs to have same shape as sigmas.
        :return: L2 loss of sigmas * valid_landmarks.
        """
        return tf.nn.l2_loss(sigmas * valid_landmarks)

    def call_model_and_loss(self, image, target_landmarks, training):
        """
        Call model and loss.
        :param image: The image to call the model with.
        :param target_landmarks: The target landmarks used for loss calculation.
        :param training: training parameter used for calling the model.
        :return ((prediction, local_prediction, spatial_prediction), losses) tuple
        """
        prediction, local_prediction, spatial_prediction = self.model(image, training=training)
        heatmap_shape = tf.shape(image)[2:] if self.data_format == 'channels_first' else tf.shape(image)[1:-1]
        target_heatmaps = generate_heatmap_target(heatmap_shape,
                                                  target_landmarks,
                                                  self.sigmas,
                                                  scale=1.0,
                                                  normalize=False,
                                                  data_format=self.data_format)
        losses = {}
        losses['loss_net'] = self.loss_function(target=target_heatmaps, pred=prediction)
        if self.sigma_regularization > 0 and self.learnable_sigma:
            losses['loss_sigmas'] = self.sigma_regularization * self.loss_function_sigmas(self.sigmas, target_landmarks[0, :, 0])
        if self.reg_constant > 0:
            losses['loss_reg'] = self.reg_constant * tf.reduce_sum(self.model.losses)
        return (prediction, local_prediction, spatial_prediction), losses

    @tf.function
    def train_step(self):
        """
        Perform a training step.
        """
        image, target_landmarks, image_id = self.dataset_train_iter.get_next()
        with tf.GradientTape() as tape:
            _, losses = self.call_model_and_loss(image, target_landmarks, training=True)
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

        if self.learnable_sigma:
            self.optimizer_sigma.apply_gradients(zip(grads[-1:], [self.sigmas_variables]))
            metric_dict['mean_sigmas'] = tf.reduce_mean(self.sigmas)

        self.loss_metric_logger_train.update_metrics(metric_dict)

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
        if self.has_validation_groundtruth:
            landmarks = generators['landmarks']

        image_size_for_tilers = np.minimum(full_image.shape[1:], list(reversed(self.max_image_size_for_cropped_test))).tolist()

        image_size_np = [1] + image_size_for_tilers
        labels_size_np = [self.num_landmarks] + image_size_for_tilers
        image_tiler = ImageTiler(full_image.shape, image_size_np, self.cropped_inc, True, -1)
        landmark_tiler = LandmarkTiler(full_image.shape, image_size_np, self.cropped_inc)
        prediction_tiler = ImageTiler((self.num_landmarks,) + full_image.shape[1:], labels_size_np, self.cropped_inc, True, -np.inf)
        prediction_local_tiler = ImageTiler((self.num_landmarks,) + full_image.shape[1:], labels_size_np, self.cropped_inc, True, -np.inf)
        prediction_spatial_tiler = ImageTiler((self.num_landmarks,) + full_image.shape[1:], labels_size_np, self.cropped_inc, True, -np.inf)
        for image_tiler, landmark_tiler, prediction_tiler, prediction_local_tiler, prediction_spatial_tiler in zip(image_tiler, landmark_tiler, prediction_tiler, prediction_local_tiler, prediction_spatial_tiler):
            current_image = image_tiler.get_current_data(full_image)
            if self.has_validation_groundtruth:
                current_landmarks = landmark_tiler.get_current_data(landmarks)
                (prediction, prediction_local, prediction_spatial), losses = self.call_model_and_loss(np.expand_dims(current_image, axis=0),
                                                                                                      np.expand_dims(current_landmarks, axis=0), training=False)
                self.loss_metric_logger_val.update_metrics(losses)
            else:
                prediction, prediction_local, prediction_spatial = self.model(np.expand_dims(current_image, axis=0), training=False)
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

        landmark_statistics = LandmarkStatistics()
        landmarks = {}
        landmark_statistics_no_postprocessing = LandmarkStatistics()
        landmarks_no_postprocessing = {}
        all_local_maxima_landmarks = {}
        num_entries = self.dataset_val.num_entries()
        for _ in tqdm(range(num_entries), desc='Testing'):
            dataset_entry = self.dataset_val.get_next()
            current_id = dataset_entry['id']['image_id']
            datasources = dataset_entry['datasources']
            input_image = datasources['image']
            if self.has_validation_groundtruth:
                target_landmarks = datasources['landmarks']
            else:
                target_landmarks = None

            image, prediction, prediction_local, prediction_spatial, transformation = self.test_cropped_image(dataset_entry)

            origin = transformation.TransformPoint(np.zeros(3, np.float64))
            if self.save_output_images:
                heatmap_normalization_mode = (-1, 1)
                image_type = np.uint8
                utils.io.image.write_multichannel_np(image,self.output_folder_handler.path_for_iteration(self.current_iter, current_id + '_input.mha'), output_normalization_mode='min_max', sitk_image_output_mode='vector', data_format=self.data_format, image_type=image_type, spacing=self.image_spacing, origin=origin)
                utils.io.image.write_multichannel_np(prediction, self.output_folder_handler.path_for_iteration(self.current_iter, current_id + '_prediction.mha'), output_normalization_mode=heatmap_normalization_mode, sitk_image_output_mode='vector', data_format=self.data_format, image_type=image_type, spacing=self.image_spacing, origin=origin)
                utils.io.image.write_multichannel_np(prediction_local, self.output_folder_handler.path_for_iteration(self.current_iter, current_id + '_prediction_local.mha'), output_normalization_mode=heatmap_normalization_mode, sitk_image_output_mode='vector', data_format=self.data_format, image_type=image_type, spacing=self.image_spacing, origin=origin)
                utils.io.image.write_multichannel_np(prediction_spatial, self.output_folder_handler.path_for_iteration(self.current_iter, current_id + '_prediction_spatial.mha'), output_normalization_mode=heatmap_normalization_mode, sitk_image_output_mode='vector', data_format=self.data_format, image_type=image_type, spacing=self.image_spacing, origin=origin)

            local_maxima_landmarks = heatmap_maxima.get_landmarks(prediction, input_image, self.image_spacing, transformation)

            # landmarks without postprocessing are the first local maxima (with the largest value)
            curr_landmarks_no_postprocessing = [l[0] if len(l) > 0 else Landmark(coords=[np.nan] * 3, is_valid=False)  for l in local_maxima_landmarks]
            landmarks_no_postprocessing[current_id] = curr_landmarks_no_postprocessing

            if self.has_validation_groundtruth:
                landmark_statistics_no_postprocessing.add_landmarks(current_id, curr_landmarks_no_postprocessing, target_landmarks)
                vis.visualize_landmark_projections(input_image, target_landmarks, filename=self.output_folder_handler.path_for_iteration(self.current_iter, current_id + '_landmarks_gt.png'))
                vis.visualize_prediction_groundtruth_projections(input_image, curr_landmarks_no_postprocessing, target_landmarks, filename=self.output_folder_handler.path_for_iteration(self.current_iter, current_id + '_landmarks.png'))
            else:
                vis.visualize_landmark_projections(input_image, curr_landmarks_no_postprocessing, filename=self.output_folder_handler.path_for_iteration(self.current_iter, current_id + '_landmarks.png'))

            if self.evaluate_landmarks_postprocessing:
                try:
                    local_maxima_landmarks = add_landmarks_from_neighbors(local_maxima_landmarks)
                    curr_landmarks = spine_postprocessing.solve_local_heatmap_maxima(local_maxima_landmarks)
                    curr_landmarks = reshift_landmarks(curr_landmarks)
                    curr_landmarks = filter_landmarks_top_bottom(curr_landmarks, input_image)
                except Exception:
                    print('error in postprocessing', current_id)
                    curr_landmarks = curr_landmarks_no_postprocessing
                landmarks[current_id] = curr_landmarks

                if self.has_validation_groundtruth:
                    landmark_statistics.add_landmarks(current_id, curr_landmarks, target_landmarks)
                    vis.visualize_prediction_groundtruth_projections(input_image, curr_landmarks, target_landmarks, filename=self.output_folder_handler.path_for_iteration(self.current_iter, current_id + '_landmarks_pp.png'))
                else:
                    vis.visualize_landmark_projections(input_image, curr_landmarks, filename=self.output_folder_handler.path_for_iteration(self.current_iter, current_id + '_landmarks_pp.png'))

        utils.io.landmark.save_points_csv(landmarks, self.output_folder_handler.path_for_iteration(self.current_iter, 'points.csv'))
        utils.io.landmark.save_points_csv(landmarks_no_postprocessing, self.output_folder_handler.path_for_iteration(self.current_iter, 'points_no_postprocessing.csv'))

        # finalize loss values
        if self.has_validation_groundtruth:
            summary_values = OrderedDict()
            if self.evaluate_landmarks_postprocessing:
                print(landmark_statistics.get_pe_overview_string())
                print(landmark_statistics.get_correct_id_string(20.0))
                overview_string = landmark_statistics.get_overview_string([2, 2.5, 3, 4, 10, 20], 10, 20.0)
                utils.io.text.save_string_txt(overview_string, self.output_folder_handler.path_for_iteration(self.current_iter, 'eval.txt'))
                summary_values.update(OrderedDict(zip(['pe_mean', 'pe_stdev', 'pe_median', 'num_correct'], list(landmark_statistics.get_pe_statistics()) + [landmark_statistics.get_num_correct_id(20)])))
            print(landmark_statistics_no_postprocessing.get_pe_overview_string())
            print(landmark_statistics_no_postprocessing.get_correct_id_string(20.0))
            overview_string = landmark_statistics_no_postprocessing.get_overview_string([2, 2.5, 3, 4, 10, 20], 10, 20.0)
            utils.io.text.save_string_txt(overview_string, self.output_folder_handler.path_for_iteration(self.current_iter, 'eval_no_postprocessing.txt'))
            summary_values.update(OrderedDict(zip(['pe_mean_np', 'pe_stdev_np', 'pe_median_np', 'num_correct_np'], list(landmark_statistics_no_postprocessing.get_pe_statistics()) + [landmark_statistics_no_postprocessing.get_num_correct_id(20)])))
            self.loss_metric_logger_val.update_metrics(summary_values)

            # finalize loss values
        self.loss_metric_logger_val.finalize(self.current_iter)


class dotdict(dict):
    """
    Dict subclass that allows dot.notation to access attributes.
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


if __name__ == '__main__':
    for cv in [0, 1, 2]:
        # Set hyperparameters, which can be overwritten with a W&B Sweep
        config = dotdict(
            num_filters_base=96,
            dropout_ratio=0.25,
            activation='lrelu',
            spatial_downsample=4,
            heatmap_sigma=3.0,
            learnable_sigma=False,
            learning_rate=0.0001,
            model='scn',
            local_activation='tanh',
            spatial_activation='tanh',
            num_levels=4,
            spacing=2.0,
            cv=cv,
            info='fin_cv',
        )
        with MainLoop(cv, config) as loop:
            loop.run()
