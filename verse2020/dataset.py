import os

import numpy as np
import SimpleITK as sitk
from random import choice

from datasets.graph_dataset import GraphDataset
from datasources.cached_image_datasource import CachedImageDataSource
from datasources.image_datasource import ImageDataSource
from datasources.landmark_datasource import LandmarkDataSource
from datasources.label_datasource import LabelDatasource
from generators.image_generator import ImageGenerator
from generators.landmark_generator import LandmarkGeneratorHeatmap, LandmarkGenerator
from generators.image_size_generator import ImageSizeGenerator
from iterators.id_list_iterator import IdListIterator
from iterators.resample_labels_id_list_iterator import ResampleLabelsIdListIterator
from graph.node import LambdaNode
from transformations.intensity.np.shift_scale_clamp import ShiftScaleClamp
from transformations.intensity.sitk.shift_scale_clamp import ShiftScaleClamp as ShiftScaleClampSitk
from transformations.spatial import translation, scale, composite, rotation, landmark, deformation, flip
from utils.np_image import split_label_image, smooth_label_images
from transformations.intensity.sitk.smooth import gaussian as gaussian_sitk
from transformations.intensity.np.smooth import gaussian
from transformations.intensity.np.normalize import normalize_zero_mean_unit_variance, normalize
from transformations.intensity.np.gamma import change_gamma_unnormalized
from utils.random import float_uniform
from utils.landmark.common import Landmark
import utils.io.text
import utils.sitk_np


class Dataset(object):
    """
    The dataset that processes files from the VerSe2020 challenge.
    """

    def __init__(self,
                 image_size,
                 image_spacing,
                 base_folder=None,
                 image_base_folder=None,
                 setup_base_folder=None,
                 cv='train_all',
                 input_gaussian_sigma=0.0,
                 label_gaussian_sigma=1.0,
                 load_spine_landmarks=False,
                 load_spine_bbs=False,
                 generate_labels=False,
                 generate_heatmaps=False,
                 generate_landmarks=False,
                 generate_single_vertebrae_heatmap=False,
                 generate_single_vertebrae=False,
                 generate_spine_heatmap=False,
                 generate_landmark_mask=False,
                 crop_image_top_bottom=False,
                 translate_by_random_factor=False,
                 translate_to_center_landmarks=False,
                 use_variable_image_size=False,
                 valid_output_sizes_x=None,
                 valid_output_sizes_y=None,
                 valid_output_sizes_z=None,
                 random_translation=30,
                 random_scale=0.15,
                 random_rotate=0.25,
                 random_deformation=25,
                 random_intensity_shift=0.25,
                 random_intensity_scale=0.25,
                 normalize_zero_mean_unit_variance=False,
                 random_translation_single_landmark=3.0,
                 random_flip=True,
                 resample_iterator=True,
                 num_landmarks=26,
                 num_labels=27,
                 heatmap_sigma=3.0,
                 single_heatmap_sigma=6.0,
                 spine_heatmap_sigma=3.0,
                 crop_randomly_smaller=False,
                 data_format='channels_first',
                 output_image_type=np.float32,
                 save_debug_images=False):
        """
        Initializer.
        :param image_size: Network input image size.
        :param image_spacing: Network input image spacing.
        :param base_folder: Dataset base folder.
        :param cv: Cross validation index (0, 1, 2), 'train_all' if full training/testing, 'inference' in inference mode only.
        :param input_gaussian_sigma: Sigma value for input smoothing.
        :param label_gaussian_sigma: Sigma value for label smoothing.
        :param load_spine_landmarks: If true, load spine landmark file.
        :param generate_labels: If true, generate multi-label vertebrae segmentation.
        :param generate_heatmaps: If true, generate vertebrae heatmaps.
        :param generate_landmarks: If true, generate landmark coordinates.
        :param generate_single_vertebrae_heatmap: If true, generate single vertebrae heatmap.
        :param generate_single_vertebrae: If true, generate single vertebrae segmentation.
        :param generate_spine_heatmap: If true, generate spine heatmap.
        :param generate_landmark_mask: If true, generate landmark mask with 25 mm black on top and bottom of volume.
        :param translate_by_random_factor: If true, perform random crop of input volume, if it does not fit into the image_size.
        :param translate_to_center_landmarks: If true, translate input image to center of landmarks.
        :param random_translation: Amount of random translation for training data augmentation.
        :param random_scale: Amount of random scale for training data augmentation.
        :param random_rotate: Amount of random rotation for training data augmentation.
        :param random_deformation: Amount of random elastic deformation for training data augmentation.
        :param random_intensity_shift: Amount of random intensity shift for training data augmentation.
        :param random_intensity_scale: Amount of random intensity scale for training data augmentation.
        :param random_translation_single_landmark: Amount of random translation of landmark, when generating a single landmark output.
        :param num_landmarks: number of landmarks.
        :param num_labels: number of labels.
        :param heatmap_sigma: Sigma value for landmark heatmap generation.
        :param spine_heatmap_sigma: Sigma value for spine heatmap generation.
        :param data_format: Either 'channels_first' or 'channels_last'.
        :param save_debug_images: If true, the generated images are saved to the disk.
        """
        self.image_size = image_size
        self.image_spacing = image_spacing
        self.base_folder = base_folder
        self.cv = cv
        self.input_gaussian_sigma = input_gaussian_sigma
        self.label_gaussian_sigma = label_gaussian_sigma
        self.load_spine_landmarks = load_spine_landmarks
        self.load_spine_bbs = load_spine_bbs
        self.generate_labels = generate_labels
        self.generate_heatmaps = generate_heatmaps
        self.generate_landmarks = generate_landmarks
        self.generate_single_vertebrae_heatmap = generate_single_vertebrae_heatmap
        self.generate_single_vertebrae = generate_single_vertebrae
        self.generate_spine_heatmap = generate_spine_heatmap
        self.generate_landmark_mask = generate_landmark_mask
        self.crop_image_top_bottom = crop_image_top_bottom
        self.translate_by_random_factor = translate_by_random_factor
        self.translate_to_center_landmarks = translate_to_center_landmarks
        self.use_variable_image_size = use_variable_image_size
        self.valid_output_sizes_x = valid_output_sizes_x or [32, 64, 96]
        self.valid_output_sizes_y = valid_output_sizes_y or [32, 64, 96]
        self.valid_output_sizes_z = valid_output_sizes_z or [32, 64, 96, 128, 160, 192]
        self.random_translation = random_translation
        self.random_scale = random_scale
        self.random_rotate = random_rotate
        self.random_deformation = random_deformation
        self.random_intensity_shift = random_intensity_shift
        self.random_intensity_scale = random_intensity_scale
        self.normalize_zero_mean_unit_variance = normalize_zero_mean_unit_variance
        self.random_translation_single_landmark = random_translation_single_landmark
        self.random_flip = random_flip
        self.resample_iterator = resample_iterator

        self.num_landmarks = num_landmarks
        self.num_labels = num_labels
        self.heatmap_sigma = heatmap_sigma
        self.single_heatmap_sigma = single_heatmap_sigma
        self.spine_heatmap_sigma = spine_heatmap_sigma
        self.crop_randomly_smaller = crop_randomly_smaller

        self.data_format = data_format
        self.output_image_type = output_image_type
        self.save_debug_images = save_debug_images
        self.dim = 3
        self.image_base_folder = image_base_folder or os.path.join(self.base_folder, 'images_reoriented')
        self.setup_base_folder = setup_base_folder or os.path.join(self.base_folder, 'setup')
        self.landmarks_file = os.path.join(self.setup_base_folder, 'landmarks.csv')
        self.valid_landmarks_file = os.path.join(self.setup_base_folder, 'valid_landmarks.csv')

        self.landmark_labels = [i + 1 for i in range(25)] + [28]
        self.landmark_mapping = dict([(i, self.landmark_labels[i]) for i in range(26)])
        self.landmark_mapping_inverse = dict([(self.landmark_labels[i], i) for i in range(26)])

        self.preprocessing = self.intensity_preprocessing_ct
        self.preprocessing_random = self.intensity_preprocessing_ct_random
        self.postprocessing_random = self.intensity_postprocessing_ct_random
        self.postprocessing = self.intensity_postprocessing_ct
        self.image_default_pixel_value = -1024
        if self.cv in [0, 1, 2]:
            self.cv_folder = os.path.join(self.setup_base_folder, os.path.join('cv', str(cv)))
            self.train_file = os.path.join(self.cv_folder, 'train.txt')
            self.test_file = os.path.join(self.cv_folder, 'val.txt')
        elif self.cv == 'train_all':
            self.train_file = os.path.join(self.setup_base_folder, 'train_all.txt')
            self.test_file = os.path.join(self.setup_base_folder, 'train_all.txt')
        else:  # if self.cv == 'inference':
            self.spine_landmarks_file = os.path.join(self.setup_base_folder, 'spine_localization/landmarks.csv')
            self.spine_bbs_file = os.path.join(self.setup_base_folder, 'spine_localization/bbs.csv')
            self.landmarks_file = os.path.join(self.setup_base_folder, 'vertebrae_localization/landmarks.csv')
            self.valid_landmarks_file = os.path.join(self.setup_base_folder, 'vertebrae_localization/valid_landmarks.csv')

    def iterator(self, id_list_filename, random):
        """
        Iterator used for iterating over the dataset.
        If not self.generate_single_vertebrae or generate_single_vertebrae_heatmap: use id_list_filename
        else: use image_id and landmark_id tuples for all valid_landmarks per image
        :param id_list_filename: The used id_list_filename of image_ids
        :param random: Shuffle if true.
        :return: IdListIterator used for image_id (and landmark_id) iteration.
        """
        if self.generate_single_vertebrae or self.generate_single_vertebrae_heatmap:
            valid_landmarks = utils.io.text.load_dict_csv(self.valid_landmarks_file, squeeze=False)

            def whole_list_postprocessing(id_list):
                new_id_list = []
                for image_id in id_list:
                    for landmark in valid_landmarks[image_id[0]]:
                        new_id_list.append([image_id[0], landmark])
                return new_id_list

            if not random and not self.resample_iterator:
                id_list_iterator = IdListIterator(id_list_filename,
                                                  random,
                                                  whole_list_postprocessing=whole_list_postprocessing,
                                                  keys=['image_id', 'landmark_id'],
                                                  name='iterator',
                                                  use_shuffle=False)
            else:
                #     0-6: C1-C7
                #     7-18: T1-12
                #     19-24: L1-6
                #     25: T13
                def id_to_label_function(curr_id):
                    landmark_id = int(curr_id[1])
                    if 0 <= landmark_id <= 6:
                        return 'c'
                    elif 7 <= landmark_id <= 18 or landmark_id == 25:
                        return 't'
                    elif 19 <= landmark_id <= 24:
                        return 'l'
                    return 'u'
                id_list_iterator = ResampleLabelsIdListIterator(id_list_filename,
                                                                None,
                                                                ['c', 't', 'l'],
                                                                whole_list_postprocessing=whole_list_postprocessing,
                                                                id_to_label_function=id_to_label_function,
                                                                keys=['image_id', 'landmark_id'],
                                                                name='iterator')
        else:
            id_list_iterator = IdListIterator(id_list_filename,
                                              random,
                                              keys=['image_id'],
                                              name='iterator',
                                              use_shuffle=False)
        return id_list_iterator

    def landmark_mask_preprocessing(self, image):
        """
        Creates a landmark mask of ones, but with 25 mm zeroes on the top and the bottom of the volumes.
        :param image: The sitk input image
        :return: A mask as an sitk image.
        """
        image_np = np.ones(list(reversed(image.GetSize())), np.uint8)
        spacing_z = image.GetSpacing()[2]
        # set 25 mm on top and bottom of image to 0
        size = 25
        image_np[:int(spacing_z * size), ...] = 0
        image_np[-int(spacing_z * size):, ...] = 0
        return utils.sitk_np.np_to_sitk(image_np)

    def datasources(self, iterator, image_cached, labels_cached, image_preprocessing, cache_size):
        """
        Returns the data sources that load data.
        {
        'image:' CachedImageDataSource that loads the image files.
        'labels:' CachedImageDataSource that loads the groundtruth labels.
        'landmarks:' LandmarkDataSource that loads the landmark coordinates.
        }
        :param iterator: The dataset iterator.
        :param image_cached: If true, use CachedImageDataSource, else ImageDataSource for image datasource.
        :param labels_cached: If true, use CachedImageDataSource, else ImageDataSource for labels datasource.
        :param image_preprocessing: Preprocessing function for image datasource.
        :param cache_size: The cache size for CachedImageDataSource.
        :return: A dict of data sources.
        """
        datasources_dict = {}
        if image_cached:
            image_data_source = CachedImageDataSource
            image_source_kwargs = {'cache_maxsize': cache_size}
        else:
            image_data_source = ImageDataSource
            image_source_kwargs = {}
        datasources_dict['image'] = image_data_source(self.image_base_folder,
                                                      '',
                                                      '',
                                                      '.nii.gz',
                                                      set_zero_origin=False,
                                                      set_identity_direction=False,
                                                      set_identity_spacing=False,
                                                      sitk_pixel_type=sitk.sitkInt16,
                                                      preprocessing=image_preprocessing,
                                                      name='image',
                                                      parents=[iterator],
                                                      **image_source_kwargs)
        if self.generate_landmark_mask:
            datasources_dict['landmark_mask'] = LambdaNode(self.landmark_mask_preprocessing,
                                                           name='landmark_mask',
                                                           parents=[datasources_dict['image']])
        if self.generate_labels or self.generate_single_vertebrae:
            if labels_cached:
                image_data_source = CachedImageDataSource
                image_source_kwargs = {'cache_maxsize': cache_size}
            else:
                image_data_source = ImageDataSource
                image_source_kwargs = {}
            datasources_dict['labels'] = image_data_source(self.image_base_folder,
                                                           '',
                                                           '_seg',
                                                           '.nii.gz',
                                                           set_zero_origin=False,
                                                           set_identity_direction=False,
                                                           set_identity_spacing=False,
                                                           sitk_pixel_type=sitk.sitkUInt8,
                                                           name='labels',
                                                           parents=[iterator],
                                                           **image_source_kwargs)
        if self.generate_landmarks or self.generate_heatmaps or self.generate_spine_heatmap or self.generate_single_vertebrae or self.generate_single_vertebrae_heatmap or (self.translate_to_center_landmarks and not (self.load_spine_landmarks or self.load_spine_bbs)):
            datasources_dict['landmarks'] = LandmarkDataSource(self.landmarks_file,
                                                               self.num_landmarks,
                                                               self.dim,
                                                               name='landmarks',
                                                               parents=[iterator])
            datasources_dict['landmarks_bb'] = LambdaNode(self.image_landmark_bounding_box, name='landmarks_bb', parents=[datasources_dict['image'], datasources_dict['landmarks']])
            datasources_dict['landmarks_bb_start'] = LambdaNode(lambda x: x[0], name='landmarks_bb_start', parents=[datasources_dict['landmarks_bb']])
            datasources_dict['landmarks_bb_extent'] = LambdaNode(lambda x: x[1], name='landmarks_bb_extent', parents=[datasources_dict['landmarks_bb']])
        if self.load_spine_landmarks:
            datasources_dict['spine_landmarks'] = LandmarkDataSource(self.spine_landmarks_file, 1, self.dim, name='spine_landmarks', parents=[iterator])
        if self.load_spine_bbs:
            datasources_dict['spine_bb'] = LabelDatasource(self.spine_bbs_file, name='spine_landmarks', parents=[iterator])
            datasources_dict['landmarks_bb'] = LambdaNode(self.image_bounding_box, name='landmarks_bb', parents=[datasources_dict['image'], datasources_dict['spine_bb']])
            datasources_dict['landmarks_bb_start'] = LambdaNode(lambda x: x[0], name='landmarks_bb_start', parents=[datasources_dict['landmarks_bb']])
            datasources_dict['landmarks_bb_extent'] = LambdaNode(lambda x: x[1], name='landmarks_bb_extent', parents=[datasources_dict['landmarks_bb']])
        return datasources_dict

    def data_generators(self, iterator, datasources, transformation, image_post_processing, random_translation_single_landmark, image_size, crop=False):
        """
        Returns the data generators that process one input. See datasources() for dict values.
        :param iterator: The iterator node.
        :param datasources: datasources dict.
        :param transformation: transformation.
        :param image_post_processing: The np postprocessing function for the image data generator.
        :param random_translation_single_landmark: If true, randomly translate single landmark.
        :param image_size: The image size node.
        :param crop: If true, use landmark_based_crop.
        :return: A dict of data generators.
        """
        generators_dict = {}
        kwparents = {'output_size': image_size}
        image_datasource = datasources['image'] if not crop else LambdaNode(self.landmark_based_crop, name='image_cropped', kwparents={'image': datasources['image'], 'landmarks': datasources['landmarks']})
        generators_dict['image'] = ImageGenerator(self.dim,
                                                  None,
                                                  self.image_spacing,
                                                  interpolator='linear',
                                                  post_processing_np=image_post_processing,
                                                  data_format=self.data_format,
                                                  resample_default_pixel_value=self.image_default_pixel_value,
                                                  np_pixel_type=self.output_image_type,
                                                  name='image',
                                                  parents=[image_datasource, transformation],
                                                  kwparents=kwparents)
        if self.generate_landmark_mask:
            generators_dict['landmark_mask'] = ImageGenerator(self.dim,
                                                              None,
                                                              self.image_spacing,
                                                              interpolator='nearest',
                                                              data_format=self.data_format,
                                                              resample_default_pixel_value=0,
                                                              name='landmark_mask',
                                                              parents=[datasources['landmark_mask'], transformation],
                                                              kwparents=kwparents)
        if self.generate_labels:
            generators_dict['labels'] = ImageGenerator(self.dim,
                                                       None,
                                                       self.image_spacing,
                                                       interpolator='nearest',
                                                       post_processing_np=self.split_labels,
                                                       data_format=self.data_format,
                                                       name='labels',
                                                       parents=[datasources['labels'], transformation],
                                                       kwparents=kwparents)
        if self.generate_heatmaps or self.generate_spine_heatmap:
            generators_dict['heatmaps'] = LandmarkGeneratorHeatmap(self.dim,
                                                                   None,
                                                                   self.image_spacing,
                                                                   sigma=self.heatmap_sigma,
                                                                   scale_factor=1.0,
                                                                   normalize_center=True,
                                                                   data_format=self.data_format,
                                                                   name='heatmaps',
                                                                   parents=[datasources['landmarks'], transformation],
                                                                   kwparents=kwparents)
        if self.generate_landmarks:
            generators_dict['landmarks'] = LandmarkGenerator(self.dim,
                                                             None,
                                                             self.image_spacing,
                                                             data_format=self.data_format,
                                                             name='landmarks',
                                                             parents=[datasources['landmarks'], transformation],
                                                             kwparents=kwparents)
        if self.generate_single_vertebrae_heatmap:
            single_landmark = LambdaNode(lambda id_dict, landmarks: landmarks[int(id_dict['landmark_id']):int(id_dict['landmark_id']) + 1],
                                         name='single_landmark',
                                         parents=[iterator, datasources['landmarks']])
            if random_translation_single_landmark:
                single_landmark = LambdaNode(lambda l: [Landmark(l[0].coords + float_uniform(-self.random_translation_single_landmark, self.random_translation_single_landmark, [self.dim]), True)],
                                             name='single_landmark_translation',
                                             parents=[single_landmark])
            generators_dict['single_heatmap'] = LandmarkGeneratorHeatmap(self.dim,
                                                                         None,
                                                                         self.image_spacing,
                                                                         sigma=self.single_heatmap_sigma,
                                                                         scale_factor=1.0,
                                                                         normalize_center=True,
                                                                         data_format=self.data_format,
                                                                         np_pixel_type=self.output_image_type,
                                                                         name='single_heatmap',
                                                                         parents=[single_landmark, transformation],
                                                                         kwparents=kwparents)
        if self.generate_single_vertebrae:
            if self.generate_labels:
                if self.data_format == 'channels_first':
                    generators_dict['single_label'] = LambdaNode(lambda id_dict, images: images[int(id_dict['landmark_id']) + 1:int(id_dict['landmark_id']) + 2, ...],
                                                                 name='single_label',
                                                                 parents=[iterator, generators_dict['labels']])
                else:
                    generators_dict['single_label'] = LambdaNode(lambda id_dict, images: images[..., int(id_dict['landmark_id']) + 1:int(id_dict['landmark_id']) + 2],
                                                                 name='single_label',
                                                                 parents=[iterator, generators_dict['labels']])
            else:
                labels_unsmoothed = ImageGenerator(self.dim,
                                                   None,
                                                   self.image_spacing,
                                                   interpolator='nearest',
                                                   post_processing_np=None,
                                                   data_format=self.data_format,
                                                   name='labels_unsmoothed',
                                                   parents=[datasources['labels'], transformation],
                                                   kwparents=kwparents)
                generators_dict['single_label'] = LambdaNode(lambda id_dict, labels: self.split_and_smooth_single_label(labels, int(id_dict['landmark_id'])),
                                                             name='single_label',
                                                             parents=[iterator, labels_unsmoothed])
        if self.generate_spine_heatmap:
            generators_dict['spine_heatmap'] = LambdaNode(lambda images: normalize(gaussian(np.sum(images, axis=0 if self.data_format == 'channels_first' else -1, keepdims=True), sigma=self.spine_heatmap_sigma), out_range=(0, 1)),
                                                          name='spine_heatmap',
                                                          parents=[generators_dict['heatmaps']])

        return generators_dict

    def split_labels(self, image):
        """
        Splits a groundtruth label image into a stack of one-hot encoded images.
        :param image: The groundtruth label image.
        :return: The one-hot encoded image.
        """
        channel_axis = 0 if self.data_format == 'channels_first' else -1
        label_images = split_label_image(np.squeeze(image, channel_axis), [0] + self.landmark_labels, np.uint8)
        label_images_smoothed = smooth_label_images(label_images, sigma=self.label_gaussian_sigma)
        return np.stack(label_images_smoothed, channel_axis)

    def split_and_smooth_single_label(self, image, landmark_index):
        """
        Splits a groundtruth label image into a stack of one-hot encoded images.
        :param image: The groundtruth label image.
        :param landmark_index: The landmark index.
        :return: The one-hot encoded image.
        """
        label_image = (image == self.landmark_mapping[landmark_index]).astype(np.float32)
        label_image_smoothed = gaussian(label_image, self.label_gaussian_sigma)
        return (label_image_smoothed > 0.5).astype(np.uint8)

    def intensity_preprocessing_ct(self, image):
        """
        Intensity preprocessing function, working on the loaded sitk image, before resampling.
        :param image: The sitk image.
        :return: The preprocessed sitk image.
        """
        image = ShiftScaleClampSitk(clamp_min=-1024)(image)
        if self.input_gaussian_sigma > 0:
            return gaussian_sitk(image, self.input_gaussian_sigma)
        return image

    def intensity_preprocessing_ct_random(self, image):
        """
        Intensity preprocessing function, working on the loaded sitk image, before resampling.
        :param image: The sitk image.
        :return: The preprocessed sitk image.
        """
        image = ShiftScaleClampSitk(clamp_min=-1024)(image)
        if self.input_gaussian_sigma > 0:
            return gaussian_sitk(image, self.input_gaussian_sigma)
        return image

    def image_landmark_bounding_box(self, image, landmarks):
        """
        Calculate the bounding box from an image and landmarks.
        :param image: The image.
        :param landmarks: The landmarks.
        :return: (image, extent) tuple
        """
        additional_extent_x_y = np.array([64, 64, 64])
        all_coords = [l.coords for l in landmarks if l.is_valid]
        image_min = np.array(image.GetOrigin())
        image_max = np.array([image.GetOrigin()[i] + image.GetSize()[i] * image.GetSpacing()[i] for i in range(3)])
        min_coords = np.min(all_coords, axis=0) - additional_extent_x_y
        min_coords = np.max([image_min, min_coords], axis=0)
        max_coords = np.max(all_coords, axis=0) + additional_extent_x_y
        max_coords = np.min([image_max, max_coords], axis=0)
        extent = max_coords - min_coords
        return min_coords, extent

    def image_bounding_box(self, image, bb):
        """
        Calculate the bounding box from an image and another bounding box.
        :param image: The image.
        :param bb: The bounding box.
        :return: (image, extent) tuple
        """
        additional_extent_x_y = np.array([64, 64, 64])
        all_coords = [np.array(list(map(float, bb[:3]))), np.array(list(map(float, bb[3:])))]
        image_min = np.array(image.GetOrigin())
        image_max = np.array([image.GetOrigin()[i] + image.GetSize()[i] * image.GetSpacing()[i] for i in range(3)])
        min_coords = np.min(all_coords, axis=0) - additional_extent_x_y
        min_coords = np.max([image_min, min_coords], axis=0)
        max_coords = np.max(all_coords, axis=0) + additional_extent_x_y
        max_coords = np.min([image_max, max_coords], axis=0)
        extent = max_coords - min_coords
        return min_coords, extent

    def landmark_based_crop(self, image, landmarks):
        """
        Crop an image based on the most upper and most lower landmarks.
        :param image: The sitk image.
        :param landmarks: The landmarks.
        :return: The cropped sitk image.
        """
        all_coords = [l.coords for l in landmarks if l.is_valid]
        if len(all_coords) < 2:
            return image
        median_distance = np.median([np.linalg.norm(all_coords[i] - all_coords[i+1]) for i in range(len(all_coords) - 1)])
        min_coords = np.min(all_coords, axis=0)
        max_coords = np.max(all_coords, axis=0)

        upper_crop = 0
        lower_crop = 0
        if not landmarks[0].is_valid:
            upper_crop = np.maximum(upper_crop, (image.GetSpacing()[2] * image.GetSize()[2]) - (max_coords[2] + median_distance * 0.5))
        if not not landmarks[22].is_valid and not landmarks[23].is_valid and not landmarks[24].is_valid:
            lower_crop = np.maximum(lower_crop, (min_coords[2] - median_distance * 0.5))
        spacing_z = image.GetSpacing()[2]
        image = sitk.Crop(image, [0, 0, int(lower_crop / spacing_z)], [0, 0, int(upper_crop / spacing_z)])
        return image

    def intensity_postprocessing_ct_random(self, image):
        """
        Intensity postprocessing for CT input. Random augmentation version.
        :param image: The np input image.
        :return: The processed image.
        """
        if not self.normalize_zero_mean_unit_variance:
            random_lambda = float_uniform(0.9, 1.1)
            image = change_gamma_unnormalized(image, random_lambda)
            output = ShiftScaleClamp(shift=0,
                                     scale=1 / 2048,
                                     random_shift=self.random_intensity_shift,
                                     random_scale=self.random_intensity_scale,
                                     clamp_min=-1.0,
                                     clamp_max=1.0)(image)
        else:
            random_lambda = float_uniform(0.9, 1.1)
            image = change_gamma_unnormalized(image, random_lambda)
            output = normalize_zero_mean_unit_variance(image)
        return output

    def intensity_postprocessing_ct(self, image):
        """
        Intensity postprocessing for CT input.
        :param image: The np input image.
        :return: The processed image.
        """
        if not self.normalize_zero_mean_unit_variance:
            output = ShiftScaleClamp(shift=0,
                                     scale=1 / 2048,
                                     clamp_min=-1.0,
                                     clamp_max=1.0)(image)
        else:
            output = normalize_zero_mean_unit_variance(image)
        return output

    def spatial_transformation_augmented(self, iterator, datasources, image_size):
        """
        The spatial image transformation with random augmentation.
        :param iterator: The iterator node.
        :param datasources: datasources dict.
        :param image_size: The image size node.
        :return: The transformation.
        """
        transformation_list = []
        kwparents = {'image': datasources['image'], 'output_size': image_size}
        if self.translate_to_center_landmarks:
            kwparents['start'] = datasources['landmarks_bb_start']
            kwparents['extent'] = datasources['landmarks_bb_extent']
            transformation_list.append(translation.BoundingBoxCenterToOrigin(self.dim, None, self.image_spacing))
        elif self.generate_single_vertebrae or self.generate_single_vertebrae_heatmap:
            single_landmark = LambdaNode(lambda id_dict, landmarks: [landmarks[int(id_dict['landmark_id'])]],
                                         parents=[iterator, datasources['landmarks']])
            kwparents['landmarks'] = single_landmark
            transformation_list.append(landmark.Center(self.dim, True))
            transformation_list.append(translation.Fixed(self.dim, [0, 20, 0]))
        else:
            transformation_list.append(translation.InputCenterToOrigin(self.dim))
        if self.translate_by_random_factor:
            transformation_list.append(translation.RandomCropBoundingBox(self.dim, None, self.image_spacing))
        transformation_list.extend([translation.Random(self.dim, [self.random_translation] * self.dim),
                                    rotation.Random(self.dim, [self.random_rotate] * self.dim),
                                    scale.RandomUniform(self.dim, self.random_scale),
                                    scale.Random(self.dim, [self.random_scale] * self.dim),
                                    flip.Random(self.dim, [0.5 if self.random_flip else 0.0, 0.0, 0.0]),
                                    translation.OriginToOutputCenter(self.dim, None, self.image_spacing),
                                    deformation.Output(self.dim, [6, 6, 6], [self.random_deformation] * self.dim, None, self.image_spacing)
                                    ])
        comp = composite.Composite(self.dim, transformation_list, name='image', kwparents=kwparents)
        return LambdaNode(lambda transformation, output_size: sitk.DisplacementFieldTransform(sitk.TransformToDisplacementField(transformation, sitk.sitkVectorFloat64, size=output_size, outputSpacing=self.image_spacing)),
                          name='image',
                          kwparents={'transformation': comp, 'output_size': image_size})

    def spatial_transformation(self, iterator, datasources, image_size):
        """
        The spatial image transformation without random augmentation.
        :param iterator: The iterator node.
        :param datasources: datasources dict.
        :param image_size: The image size node.
        :return: The transformation.
        """
        transformation_list = []
        kwparents = {'image': datasources['image'], 'output_size': image_size}
        if self.translate_to_center_landmarks:
            kwparents['start'] = datasources['landmarks_bb_start']
            kwparents['extent'] = datasources['landmarks_bb_extent']
            transformation_list.append(translation.BoundingBoxCenterToOrigin(self.dim, None, self.image_spacing))
        elif self.generate_single_vertebrae or self.generate_single_vertebrae_heatmap:
            single_landmark = LambdaNode(lambda id_dict, landmarks: [landmarks[int(id_dict['landmark_id'])]],
                                         parents=[iterator, datasources['landmarks']])
            kwparents['landmarks'] = single_landmark
            transformation_list.append(landmark.Center(self.dim, True))
            transformation_list.append(translation.Fixed(self.dim, [0, 20, 0]))
        else:
            transformation_list.append(translation.InputCenterToOrigin(self.dim))
        transformation_list.append(translation.OriginToOutputCenter(self.dim, None, self.image_spacing))
        comp = composite.Composite(self.dim, transformation_list, name='image', kwparents=kwparents)
        return comp

    def crop_randomly_smaller_image_size(self, image_size):
        """
        Randomly use a smaller image size for a given image size.
        :param image_size: The image size.
        :return: The image size.
        """
        if utils.random.bool_bernoulli(0.5):
            smaller_sizes = [s for s in self.valid_output_sizes_z if s < image_size[2]]
            return image_size[:2] + [choice(smaller_sizes)]
        else:
            return image_size

    def dataset_train(self):
        """
        Returns the training dataset. Random augmentation is performed.
        :return: The training dataset.
        """
        iterator = self.iterator(self.train_file, True)
        sources = self.datasources(iterator, False, False, self.preprocessing_random, 8192)
        if self.use_variable_image_size:
            image_size = ImageSizeGenerator(self.dim, [None] * 3, self.image_spacing, valid_output_sizes=[self.valid_output_sizes_x, self.valid_output_sizes_y, self.valid_output_sizes_z], name='output_size', kwparents={'extent': sources['landmarks_bb_extent']})
            if self.crop_randomly_smaller:
                image_size = LambdaNode(self.crop_randomly_smaller_image_size, name='output_size', parents=[image_size])
        else:
            image_size = LambdaNode(lambda: self.image_size, name='output_size')
        reference_transformation = self.spatial_transformation_augmented(iterator, sources, image_size)
        generators = self.data_generators(iterator, sources, reference_transformation, self.postprocessing_random, True, image_size, self.crop_image_top_bottom)
        generators['image_id'] = LambdaNode(lambda d: np.array(d['image_id']), name='image_id', parents=[iterator])

        return GraphDataset(data_generators=list(generators.values()),
                            data_sources=list(sources.values()),
                            transformations=[reference_transformation],
                            iterator=iterator,
                            debug_image_folder='debug_train' if self.save_debug_images else None)

    def dataset_val(self):
        """
        Returns the validation dataset. No random augmentation is performed.
        :return: The validation dataset.
        """
        if self.cv == 'inference':
            iterator = 'iterator'
        else:
            iterator = self.iterator(self.test_file, False)
        sources = self.datasources(iterator, True, True, self.preprocessing, 2048)
        if self.use_variable_image_size:
            if self.load_spine_bbs:
                image_size = ImageSizeGenerator(self.dim, [None] * 3, self.image_spacing, valid_output_sizes=[[32, 64, 96, 128], [32, 64, 96, 128], [32 + i * 32 for i in range(20)]], name='output_size', kwparents={'extent': sources['landmarks_bb_extent']})
            else:
                image_size = ImageSizeGenerator(self.dim, [None] * 3, self.image_spacing, valid_output_sizes=[[32, 64, 96, 128], [32, 64, 96, 128], [32 + i * 32 for i in range(20)]], name='output_size', kwparents={'image': sources['image']})
        else:
            image_size = LambdaNode(lambda: self.image_size, name='output_size')
        reference_transformation = self.spatial_transformation(iterator, sources, image_size)
        generators = self.data_generators(iterator, sources, reference_transformation, self.postprocessing, False, image_size, False)
        generators['image_id'] = LambdaNode(lambda d: np.array(d['image_id']), name='image_id', parents=[iterator])

        return GraphDataset(data_generators=list(generators.values()),
                            data_sources=list(sources.values()),
                            transformations=[reference_transformation],
                            iterator=iterator,
                            debug_image_folder='debug_val' if self.save_debug_images else None)
