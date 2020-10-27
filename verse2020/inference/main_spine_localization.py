#!/usr/bin/python
import argparse
import os
import sys
import traceback
from collections import OrderedDict
from glob import glob

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
from spine_localization_postprocessing import bb
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
        self.data_format = 'channels_last'
        self.network_parameters = OrderedDict(num_filters_base=config.num_filters_base,
                                              activation=config.activation,
                                              num_levels=config.num_levels,
                                              data_format=self.data_format)
        if config.model == 'unet':
            self.network = Unet

        self.save_output_images = True
        self.save_debug_images = False
        self.image_folder = config.image_folder
        self.setup_folder = config.setup_folder
        self.output_folder = config.output_folder
        self.load_model_filenames = config.load_model_filenames
        self.image_size = [None, None, None]
        self.image_spacing = [config.spacing] * 3
        images_files = sorted(glob(os.path.join(self.image_folder, '*.nii.gz')))
        self.image_id_list = list(map(lambda filename: os.path.basename(filename)[:-len('.nii.gz')], images_files))

    def init_model(self):
        """
        Init self.model.
        """
        self.model = self.network(num_labels=1, **self.network_parameters)

    def init_checkpoint(self):
        """
        Init self.checkpoint.
        """
        self.checkpoint = tf.train.Checkpoint(model=self.model)

    def init_output_folder_handler(self):
        """
        Init self.output_folder_handler.
        """
        self.output_folder_handler = OutputFolderHandler(self.output_folder, use_timestamp=False, files_to_copy=[])

    def init_datasets(self):
        """
        Init self.dataset_val.
        """
        dataset_parameters = dict(image_base_folder=self.image_folder,
                                  setup_base_folder=self.setup_folder,
                                  image_size=self.image_size,
                                  image_spacing=self.image_spacing,
                                  normalize_zero_mean_unit_variance=False,
                                  valid_output_sizes_x=[32, 64, 96, 128],
                                  valid_output_sizes_y=[32, 64, 96, 128],
                                  valid_output_sizes_z=[32, 64, 96, 128],
                                  use_variable_image_size=True,
                                  cv=self.cv,
                                  input_gaussian_sigma=0.75,
                                  output_image_type=np.float16 if self.use_mixed_precision else np.float32,
                                  data_format=self.data_format,
                                  save_debug_images=self.save_debug_images)

        dataset = Dataset(**dataset_parameters)
        self.dataset_val = dataset.dataset_val()
        self.network_image_size = list(reversed(self.image_size))

    def call_model(self, image):
        """
        Call model.
        :param image: The image to call the model with.
        :return prediction
        """
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
        predictions = []
        for load_model_filename in self.load_model_filenames:
            if len(self.load_model_filenames) > 1:
                self.load_model(load_model_filename)
            prediction = self.call_model(image)
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

        bbs = {}
        for current_id in tqdm(self.image_id_list, desc='Testing'):
            try:
                dataset_entry = self.dataset_val.get({'image_id': current_id})
                image, prediction, transformation = self.test_full_image(dataset_entry)
                start, end = bb(prediction, transformation, self.image_spacing)
                bbs[current_id] = start + end

                if self.save_output_images:
                    origin = np.array(transformation.TransformPoint(np.zeros(3, np.float64)))
                    utils.io.image.write_multichannel_np(image,
                                                         self.output_folder_handler.path('output', current_id + '_input.mha'),
                                                         output_normalization_mode='min_max',
                                                         data_format=self.data_format,
                                                         image_type=np.uint8,
                                                         spacing=self.image_spacing,
                                                         origin=origin)
                    utils.io.image.write_multichannel_np(prediction,
                                                         self.output_folder_handler.path('output', current_id + '_prediction.mha'),
                                                         output_normalization_mode=(0, 1),
                                                         data_format=self.data_format,
                                                         image_type=np.uint8,
                                                         spacing=self.image_spacing,
                                                         origin=origin)
            except Exception:
                print('ERROR predicting', current_id)
                traceback.print_exc(file=sys.stdout)
                pass

        utils.io.text.save_dict_csv(bbs, self.output_folder_handler.path('bbs.csv'))


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
    hyperparameters = dotdict(
        load_model_filenames=parser_args.model_files,
        image_folder=parser_args.image_folder,
        setup_folder=parser_args.setup_folder,
        output_folder=parser_args.output_folder,
        num_filters_base=96,
        activation='lrelu',
        model='unet',
        num_levels=5,
        spacing=8.0,
        cv='inference',
    )
    with MainLoop(hyperparameters) as loop:
        loop.init_model()
        loop.init_output_folder_handler()
        loop.init_checkpoint()
        loop.init_datasets()
        print('Starting main test loop')
        loop.test()
