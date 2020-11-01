#!/usr/bin/python
import argparse
import os
import sys
import traceback
from collections import OrderedDict
from glob import glob

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
from network import Unet
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow_train_v2.train_loop import MainLoopBase
from tensorflow_train_v2.utils.output_folder_handler import OutputFolderHandler
from tqdm import tqdm


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
        self.num_labels = 1
        self.num_labels_all = 27
        self.data_format = 'channels_last'
        self.network_parameters = OrderedDict(num_filters_base=config.num_filters_base,
                                              activation=config.activation,
                                              num_levels=config.num_levels,
                                              data_format=self.data_format)
        self.network = Unet
        self.save_output_images = True
        self.save_debug_images = False
        self.image_folder = config.image_folder
        self.setup_folder = config.setup_folder
        self.output_folder = config.output_folder
        self.load_model_filenames = config.load_model_filenames
        self.image_size = [128, 128, 96]
        self.image_spacing = [config.spacing] * 3
        self.heatmap_size = self.image_size

        images_files = sorted(glob(os.path.join(self.image_folder, '*.nii.gz')))
        self.image_id_list = list(map(lambda filename: os.path.basename(filename)[:-len('.nii.gz')], images_files))
        self.valid_landmarks_file = os.path.join(self.setup_folder, 'vertebrae_localization/valid_landmarks.csv')
        self.valid_landmarks = utils.io.text.load_dict_csv(self.valid_landmarks_file, squeeze=False)

        self.landmark_labels = [i + 1 for i in range(25)] + [28]
        self.landmark_mapping = dict([(i, self.landmark_labels[i]) for i in range(26)])
        self.landmark_mapping_inverse = dict([(self.landmark_labels[i], i) for i in range(26)])

    def init_model(self):
        # create sigmas variable
        self.model = self.network(num_labels=self.num_labels, **self.network_parameters)

    def init_checkpoint(self):
        self.checkpoint = tf.train.Checkpoint(model=self.model)

    def init_output_folder_handler(self):
        self.output_folder_handler = OutputFolderHandler(self.output_folder, use_timestamp=False, files_to_copy=[])

    def init_datasets(self):
        dataset_parameters = dict(image_base_folder=self.image_folder,
                                  setup_base_folder=self.setup_folder,
                                  image_size=self.image_size,
                                  image_spacing=self.image_spacing,
                                  normalize_zero_mean_unit_variance=False,
                                  cv=self.cv,
                                  input_gaussian_sigma=0.75,
                                  heatmap_sigma=3.0,
                                  generate_single_vertebrae_heatmap=True,
                                  output_image_type=np.float16 if self.use_mixed_precision else np.float32,
                                  data_format=self.data_format,
                                  save_debug_images=self.save_debug_images)

        dataset = Dataset(**dataset_parameters)
        self.dataset_val = dataset.dataset_val()
        self.network_image_size = list(reversed(self.image_size))

    def call_model(self, image):
        return self.model(image, training=False)

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
        predictions = []
        for load_model_filename in self.load_model_filenames:
            if len(self.load_model_filenames) > 1:
                self.load_model(load_model_filename)
            prediction = tf.sigmoid(self.call_model(image_heatmap_concat))
            predictions.append(prediction.numpy())
        prediction = np.mean(predictions, axis=0)
        prediction = np.squeeze(prediction, axis=0)
        transformation = transformations['image']
        image = generators['image']

        return image, prediction, transformation

    def test(self):
        """
        The test function. Performs inference on the the validation images and calculates the loss.
        """
        print('Testing...')

        if len(self.load_model_filenames) == 1:
            self.load_model(self.load_model_filenames[0])

        channel_axis = 0
        if self.data_format == 'channels_last':
            channel_axis = 3

        filter_largest_cc = True

        # iterate over all images
        for image_id in tqdm(self.image_id_list, desc='Testing'):
            try:
                first = True
                prediction_labels_np = None
                prediction_max_value_np = None
                input_image = None
                # iterate over all valid landmarks
                for landmark_id in self.valid_landmarks[image_id]:
                    dataset_entry = self.dataset_val.get({'image_id': image_id, 'landmark_id' : landmark_id})
                    if first:
                        input_image = dataset_entry['datasources']['image']
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
                    #del transformation
                    prediction_resampled_np = utils.sitk_np.sitk_to_np(prediction_resampled_sitk[0])
                    if self.save_output_images:
                        utils.io.image.write_multichannel_np(prediction_resampled_np, self.output_folder_handler.path('output', image_id + '_' + landmark_id + '_prediction_resampled.mha'), output_normalization_mode=(0, 1), is_single_channel=True, sitk_image_output_mode='vector', data_format=self.data_format, image_type=np.uint8, spacing=prediction_resampled_sitk[0].GetSpacing(), origin=prediction_resampled_sitk[0].GetOrigin())
                    bb_start = np.floor(np.flip(origin / np.array(input_image.GetSpacing())))
                    bb_start = np.maximum(bb_start, [0, 0, 0])
                    bb_end = np.ceil(np.flip(max_index / np.array(input_image.GetSpacing())))
                    bb_end = np.minimum(bb_end, prediction_resampled_np.shape - np.ones(3))  # bb is inclusive -> subtract [1, 1, 1] from max size
                    #print(bb_start, bb_end)
                    #bb_start, bb_end = utils.np_image.bounding_box(prediction_resampled_np)
                    slices = tuple([slice(int(bb_start[i]), int(bb_end[i]+1)) for i in range(3)])
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

                if first:
                    # first is only True, if there exist no valid landmark.
                    print('No landmarks found for image with id', image_id)
                    continue

                # delete to save memory
                del prediction_max_value_np
                prediction_labels = utils.sitk_np.np_to_sitk(prediction_labels_np)
                prediction_labels.CopyInformation(input_image)
                del prediction_labels_np
                utils.io.image.write(prediction_labels, self.output_folder_handler.path(image_id + '_seg.nii.gz'))
                if self.save_output_images:
                    prediction_labels_resampled = utils.sitk_np.sitk_to_np(utils.sitk_image.resample_to_spacing(prediction_labels, [1.0, 1.0, 1.0], 'nearest'))
                    prediction_labels_resampled = np.flip(prediction_labels_resampled, axis=0)
                    utils.io.image.write_multichannel_np(prediction_labels_resampled, self.output_folder_handler.path('output', image_id + '_seg.png'), channel_layout_mode='label_rgb', output_normalization_mode=(0, 1), image_layout_mode='max_projection', is_single_channel=True, sitk_image_output_mode='vector', data_format=self.data_format, image_type=np.uint8)
                    utils.io.image.write_multichannel_np(prediction_labels_resampled, self.output_folder_handler.path('output', image_id + '_seg_rgb.mha'), channel_layout_mode='label_rgb', output_normalization_mode=(0, 1), is_single_channel=True, sitk_image_output_mode='vector', data_format=self.data_format, image_type=np.uint8)
                    input_resampled = utils.sitk_np.sitk_to_np(utils.sitk_image.resample_to_spacing(input_image, [1.0, 1.0, 1.0], 'linear'))
                    input_resampled = np.flip(input_resampled, axis=0)
                    utils.io.image.write_multichannel_np(input_resampled, self.output_folder_handler.path('output', image_id + '_input.png'), output_normalization_mode='min_max', image_layout_mode='max_projection', is_single_channel=True, sitk_image_output_mode='vector', data_format=self.data_format, image_type=np.uint8)

                del prediction_labels
            except Exception:
                print('ERROR predicting', image_id)
                traceback.print_exc(file=sys.stdout)
                pass


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
        model='unet',
        num_levels=5,
        spacing=1.0,
        cv='inference'
    )
    with MainLoop(hyperparameters) as loop:
        loop.init_model()
        loop.init_output_folder_handler()
        loop.init_checkpoint()
        loop.init_datasets()
        print('Starting main test loop')
        loop.test()
