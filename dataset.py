import os

import numpy as np
import SimpleITK as sitk

from datasets.graph_dataset import GraphDataset
from datasources.cached_image_datasource import CachedImageDataSource
from datasources.image_datasource import ImageDataSource
from datasources.landmark_datasource import LandmarkDataSource
from generators.image_generator import ImageGenerator
from generators.landmark_generator import LandmarkGeneratorHeatmap, LandmarkGenerator
from iterators.id_list_iterator import IdListIterator
from graph.node import LambdaNode
from transformations.intensity.np.shift_scale_clamp import ShiftScaleClamp
from transformations.intensity.sitk.shift_scale_clamp import ShiftScaleClamp as ShiftScaleClampSitk
from transformations.spatial import translation, scale, composite, rotation, landmark, deformation
from utils.np_image import split_label_image, smooth_label_images
from transformations.intensity.sitk.smooth import gaussian as gaussian_sitk
from transformations.intensity.np.smooth import gaussian
from transformations.intensity.np.normalize import normalize_robust
from transformations.intensity.np.gamma import change_gamma_unnormalized
from utils.random import float_uniform
from utils.landmark.common import Landmark
import utils.io.text
import utils.sitk_np


class Dataset(object):
    """
    The dataset that processes files from the MMWHS challenge.
    """

    def __init__(self,
                 image_size,
                 image_spacing,
                 base_folder=None,
                 image_base_folder=None,
                 setup_base_folder=None,
                 cv='train_all',
                 input_gaussian_sigma=1.0,
                 label_gaussian_sigma=1.0,
                 load_spine_landmarks=False,
                 generate_labels=False,
                 generate_heatmaps=False,
                 generate_landmarks=False,
                 generate_single_vertebrae_heatmap=False,
                 generate_single_vertebrae=False,
                 generate_spine_heatmap=False,
                 generate_landmark_mask=False,
                 translate_by_random_factor=False,
                 translate_to_center_landmarks=False,
                 random_translation=30,
                 random_scale=0.15,
                 random_rotate=0.25,
                 random_deformation=25,
                 random_intensity_shift=0.25,
                 random_intensity_scale=0.25,
                 random_translation_single_landmark=3.0,
                 num_landmarks=25,
                 num_labels=26,
                 heatmap_sigma=3.0,
                 spine_heatmap_sigma=3.0,
                 data_format='channels_first',
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
        :param data_format: Either 'channels_first' or 'channels_last'. TODO: adapt code for 'channels_last' to work.
        :param save_debug_images: If true, the generated images are saved to the disk.
        """
        self.image_size = image_size
        self.image_spacing = image_spacing
        self.base_folder = base_folder
        self.cv = cv
        self.input_gaussian_sigma = input_gaussian_sigma
        self.label_gaussian_sigma = label_gaussian_sigma
        self.load_spine_landmarks = load_spine_landmarks
        self.generate_labels = generate_labels
        self.generate_heatmaps = generate_heatmaps
        self.generate_landmarks = generate_landmarks
        self.generate_single_vertebrae_heatmap = generate_single_vertebrae_heatmap
        self.generate_single_vertebrae = generate_single_vertebrae
        self.generate_spine_heatmap = generate_spine_heatmap
        self.generate_landmark_mask = generate_landmark_mask
        self.translate_by_random_factor = translate_by_random_factor
        self.translate_to_center_landmarks = translate_to_center_landmarks
        self.random_translation = random_translation
        self.random_scale = random_scale
        self.random_rotate = random_rotate
        self.random_deformation = random_deformation
        self.random_intensity_shift = random_intensity_shift
        self.random_intensity_scale = random_intensity_scale
        self.random_translation_single_landmark = random_translation_single_landmark

        self.num_landmarks = num_landmarks
        self.num_labels = num_labels
        self.heatmap_sigma = heatmap_sigma
        self.spine_heatmap_sigma = spine_heatmap_sigma
        self.data_format = data_format
        self.save_debug_images = save_debug_images
        self.dim = 3
        self.image_base_folder = image_base_folder or os.path.join(self.base_folder, 'images_reoriented')
        self.setup_base_folder = setup_base_folder or os.path.join(self.base_folder, 'setup')
        self.landmarks_file = os.path.join(self.setup_base_folder, 'landmarks.csv')
        self.valid_landmarks_file = os.path.join(self.setup_base_folder, 'valid_landmarks.csv')

        self.preprocessing = self.intensity_preprocessing_ct
        self.postprocessing_random = self.intensity_postprocessing_ct_random
        self.postprocessing = self.intensity_postprocessing_ct
        self.image_default_pixel_value = -1024
        if self.cv in [0, 1, 2]:
            self.cv_folder = os.path.join(self.setup_base_folder, os.path.join('cv', str(cv)))
            self.train_file = os.path.join(self.cv_folder, 'train.txt')
            self.test_file = os.path.join(self.cv_folder, 'val.txt')
        elif self.cv == 'train_all':
            self.train_file = os.path.join(self.setup_base_folder, 'train_all.txt')
            self.test_file = self.train_file
        else:  # if self.cv == 'inference':
            self.spine_landmarks_file = os.path.join(self.setup_base_folder, 'spine_localization/landmarks.csv')
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
            valid_landmarks = utils.io.text.load_dict_csv(self.valid_landmarks_file)

            def whole_list_postprocessing(id_list):
                new_id_list = []
                for image_id in id_list:
                    for landmark in valid_landmarks[image_id[0]]:
                        new_id_list.append([image_id[0], landmark])
                return new_id_list

            id_list_iterator = IdListIterator(id_list_filename,
                                              random,
                                              whole_list_postprocessing=whole_list_postprocessing,
                                              keys=['image_id', 'landmark_id'],
                                              name='iterator')
        else:
            id_list_iterator = IdListIterator(id_list_filename,
                                              random,
                                              keys=['image_id'],
                                              name='iterator')
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

    def datasources(self, iterator, cached):
        """
        Returns the data sources that load data.
        {
        'image:' CachedImageDataSource that loads the image files.
        'labels:' CachedImageDataSource that loads the groundtruth labels.
        'landmarks:' LandmarkDataSource that loads the landmark coordinates.
        }
        :param iterator: The dataset iterator.
        :param cached: If true, use CachedImageDataSource, else ImageDataSource.
        :return: A dict of data sources.
        """
        datasources_dict = {}
        image_data_source = CachedImageDataSource if cached else ImageDataSource
        datasources_dict['image'] = image_data_source(self.image_base_folder,
                                                      '',
                                                      '',
                                                      '.nii.gz',
                                                      set_zero_origin=False,
                                                      set_identity_direction=False,
                                                      set_identity_spacing=False,
                                                      sitk_pixel_type=sitk.sitkInt16,
                                                      preprocessing=self.preprocessing,
                                                      name='image',
                                                      parents=[iterator])
        if self.generate_landmark_mask:
            datasources_dict['landmark_mask'] = LambdaNode(self.landmark_mask_preprocessing,
                                                           name='image',
                                                           parents=[datasources_dict['image']])
        if self.generate_labels or self.generate_single_vertebrae:
            datasources_dict['labels'] = image_data_source(self.image_base_folder,
                                                           '',
                                                           '_seg',
                                                           '.nii.gz',
                                                           set_zero_origin=False,
                                                           set_identity_direction=False,
                                                           set_identity_spacing=False,
                                                           sitk_pixel_type=sitk.sitkUInt8,
                                                           name='labels',
                                                           parents=[iterator])
        if self.generate_landmarks or self.generate_heatmaps or self.generate_spine_heatmap or self.generate_single_vertebrae or self.generate_single_vertebrae_heatmap or (self.translate_to_center_landmarks and not self.load_spine_landmarks):
            datasources_dict['landmarks'] = LandmarkDataSource(self.landmarks_file,
                                                               self.num_landmarks,
                                                               self.dim,
                                                               name='landmarks',
                                                               parents=[iterator])
        if self.load_spine_landmarks:
            datasources_dict['spine_landmarks'] = LandmarkDataSource(self.spine_landmarks_file, 1, self.dim, name='spine_landmarks', parents=[iterator])
        return datasources_dict

    def data_generators(self, iterator, datasources, transformation, image_post_processing, random_translation_single_landmark, image_size):
        """
        Returns the data generators that process one input. See datasources() for dict values.
        :param datasources: datasources dict.
        :param transformation: transformation.
        :param image_post_processing: The np postprocessing function for the image data generator.
        :return: A dict of data generators.
        """
        generators_dict = {}
        generators_dict['image'] = ImageGenerator(self.dim,
                                                  image_size,
                                                  self.image_spacing,
                                                  interpolator='linear',
                                                  post_processing_np=image_post_processing,
                                                  data_format=self.data_format,
                                                  resample_default_pixel_value=self.image_default_pixel_value,
                                                  name='image',
                                                  parents=[datasources['image'], transformation])
        if self.generate_landmark_mask:
            generators_dict['landmark_mask'] = ImageGenerator(self.dim,
                                                              image_size,
                                                              self.image_spacing,
                                                              interpolator='nearest',
                                                              data_format=self.data_format,
                                                              resample_default_pixel_value=0,
                                                              name='landmark_mask',
                                                              parents=[datasources['landmark_mask'], transformation])
        if self.generate_labels or self.generate_single_vertebrae:
            generators_dict['labels'] = ImageGenerator(self.dim,
                                                       image_size,
                                                       self.image_spacing,
                                                       interpolator='nearest',
                                                       post_processing_np=self.split_labels,
                                                       data_format=self.data_format,
                                                       name='labels',
                                                       parents=[datasources['labels'], transformation])
        if self.generate_heatmaps or self.generate_spine_heatmap:
            generators_dict['heatmaps'] = LandmarkGeneratorHeatmap(self.dim,
                                                                   image_size,
                                                                   self.image_spacing,
                                                                   sigma=self.heatmap_sigma,
                                                                   scale_factor=1.0,
                                                                   normalize_center=True,
                                                                   data_format=self.data_format,
                                                                   name='heatmaps',
                                                                   parents=[datasources['landmarks'], transformation])
        if self.generate_landmarks:
            generators_dict['landmarks'] = LandmarkGenerator(self.dim,
                                                             image_size,
                                                             self.image_spacing,
                                                             data_format=self.data_format,
                                                             name='landmarks',
                                                             parents=[datasources['landmarks'], transformation])
        if self.generate_single_vertebrae_heatmap:
            single_landmark = LambdaNode(lambda id_dict, landmarks: landmarks[int(id_dict['landmark_id']):int(id_dict['landmark_id']) + 1],
                                         name='single_landmark',
                                         parents=[iterator, datasources['landmarks']])
            if random_translation_single_landmark:
                single_landmark = LambdaNode(lambda l: [Landmark(l[0].coords + float_uniform(-self.random_translation_single_landmark, self.random_translation_single_landmark, [self.dim]), True)],
                                             name='single_landmark_translation',
                                             parents=[single_landmark])
            generators_dict['single_heatmap'] = LandmarkGeneratorHeatmap(self.dim,
                                                                         image_size,
                                                                         self.image_spacing,
                                                                         sigma=self.heatmap_sigma,
                                                                         scale_factor=1.0,
                                                                         normalize_center=True,
                                                                         data_format=self.data_format,
                                                                         name='single_heatmap',
                                                                         parents=[single_landmark, transformation])
        if self.generate_single_vertebrae:
            if self.data_format == 'channels_first':
                generators_dict['single_label'] = LambdaNode(lambda id_dict, images: images[int(id_dict['landmark_id']) + 1:int(id_dict['landmark_id']) + 2, ...],
                                                             name='single_label',
                                                             parents=[iterator, generators_dict['labels']])
            else:
                generators_dict['single_label'] = LambdaNode(lambda id_dict, images: images[..., int(id_dict['landmark_id']) + 1:int(id_dict['landmark_id']) + 2],
                                                             name='single_label',
                                                             parents=[iterator, generators_dict['labels']])
        if self.generate_spine_heatmap:
            generators_dict['spine_heatmap'] = LambdaNode(lambda images: gaussian(np.sum(images, axis=0 if self.data_format == 'channels_first' else -1, keepdims=True), sigma=self.spine_heatmap_sigma),
                                                          name='spine_heatmap',
                                                          parents=[generators_dict['heatmaps']])

        return generators_dict

    def split_labels(self, image):
        """
        Splits a groundtruth label image into a stack of one-hot encoded images.
        :param image: The groundtruth label image.
        :return: The one-hot encoded image.
        """
        label_images = split_label_image(np.squeeze(image, 0), list(range(self.num_labels)), np.uint8)
        label_images_smoothed = smooth_label_images(label_images, sigma=self.label_gaussian_sigma)
        return np.stack(label_images_smoothed, 0)

    def intensity_preprocessing_ct(self, image):
        """
        Intensity preprocessing function, working on the loaded sitk image, before resampling.
        :param image: The sitk image.
        :return: The preprocessed sitk image.
        """
        image = ShiftScaleClampSitk(clamp_min=-1024)(image)
        return gaussian_sitk(image, self.input_gaussian_sigma)

    def intensity_postprocessing_ct_random(self, image):
        """
        Intensity postprocessing for CT input. Random augmentation version.
        :param image: The np input image.
        :return: The processed image.
        """
        return ShiftScaleClamp(shift=0,
                               scale=1 / 2048,
                               random_shift=self.random_intensity_shift,
                               random_scale=self.random_intensity_scale,
                               clamp_min=-1.0,
                               clamp_max=1.0)(image)

    def intensity_postprocessing_ct(self, image):
        """
        Intensity postprocessing for CT input.
        :param image: The np input image.
        :return: The processed image.
        """
        return ShiftScaleClamp(shift=0,
                               scale=1 / 2048,
                               clamp_min=-1.0,
                               clamp_max=1.0)(image)

    def spatial_transformation_augmented(self, iterator, datasources, image_size):
        """
        The spatial image transformation with random augmentation.
        :param datasources: datasources dict.
        :return: The transformation.
        """
        transformation_list = []
        kwparents = {'image': datasources['image']}
        if self.translate_to_center_landmarks:
            kwparents['landmarks'] = datasources['landmarks']
            transformation_list.append(translation.InputCenterToOrigin(self.dim, used_dimensions=[False, False, True]))
            transformation_list.append(landmark.Center(self.dim, True, used_dimensions=[True, True, False]))
        elif self.generate_single_vertebrae or self.generate_single_vertebrae_heatmap:
            single_landmark = LambdaNode(lambda id_dict, landmarks: [landmarks[int(id_dict['landmark_id'])]],
                                         parents=[iterator, datasources['landmarks']])
            kwparents['landmarks'] = single_landmark
            transformation_list.append(landmark.Center(self.dim, True))
            transformation_list.append(translation.Fixed(self.dim, [0, 20, 0]))
        else:
            transformation_list.append(translation.InputCenterToOrigin(self.dim))
        if self.translate_by_random_factor:
            transformation_list.append(translation.RandomFactorInput(self.dim, [0, 0, 0.5], [0, 0, self.image_spacing[2] * image_size[2]]))
        transformation_list.extend([translation.Random(self.dim, [self.random_translation] * self.dim),
                                    rotation.Random(self.dim, [self.random_rotate] * self.dim),
                                    scale.RandomUniform(self.dim, self.random_scale),
                                    scale.Random(self.dim, [self.random_scale] * self.dim),
                                    translation.OriginToOutputCenter(self.dim, image_size, self.image_spacing),
                                    deformation.Output(self.dim, [6, 6, 6], [self.random_deformation] * self.dim, image_size, self.image_spacing)])
        comp = composite.Composite(self.dim, transformation_list, name='image', kwparents=kwparents)
        return LambdaNode(lambda comp: sitk.DisplacementFieldTransform(sitk.TransformToDisplacementField(comp, sitk.sitkVectorFloat64, size=self.image_size, outputSpacing=self.image_spacing)),
                          name='image',
                          kwparents={'comp': comp})

    def spatial_transformation(self, iterator, datasources, image_size):
        """
        The spatial image transformation without random augmentation.
        :param datasources: datasources dict.
        :return: The transformation.
        """
        transformation_list = []
        kwparents = {'image': datasources['image']}
        if self.translate_to_center_landmarks:
            if 'spine_landmarks' in datasources:
                kwparents['landmarks'] = datasources['spine_landmarks']
            else:
                kwparents['landmarks'] = datasources['landmarks']
            transformation_list.append(translation.InputCenterToOrigin(self.dim, used_dimensions=[False, False, True]))
            transformation_list.append(landmark.Center(self.dim, True, used_dimensions=[True, True, False]))
        elif self.generate_single_vertebrae or self.generate_single_vertebrae_heatmap:
            single_landmark = LambdaNode(lambda id_dict, landmarks: [landmarks[int(id_dict['landmark_id'])]],
                                         parents=[iterator, datasources['landmarks']])
            kwparents['landmarks'] = single_landmark
            transformation_list.append(landmark.Center(self.dim, True))
            transformation_list.append(translation.Fixed(self.dim, [0, 20, 0]))
        else:
            transformation_list.append(translation.InputCenterToOrigin(self.dim))
        transformation_list.append(translation.OriginToOutputCenter(self.dim, image_size, self.image_spacing))
        comp = composite.Composite(self.dim, transformation_list, name='image', kwparents=kwparents)
        return comp

    def dataset_train(self):
        """
        Returns the training dataset. Random augmentation is performed.
        :return: The training dataset.
        """
        iterator = self.iterator(self.train_file, True)
        sources = self.datasources(iterator, True)
        image_size = self.image_size
        reference_transformation = self.spatial_transformation_augmented(iterator, sources, image_size)
        generators = self.data_generators(iterator, sources, reference_transformation, self.postprocessing_random, True, image_size)
        if self.generate_single_vertebrae and not self.generate_labels:
            del generators['labels']

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
        sources = self.datasources(iterator, False)
        if self.translate_by_random_factor:
            image_size = self.image_size[:2] + [None]
        else:
            image_size = self.image_size
        reference_transformation = self.spatial_transformation(iterator, sources, image_size)
        generators = self.data_generators(iterator, sources, reference_transformation, self.postprocessing, False, image_size)
        if self.generate_single_vertebrae and not self.generate_labels:
            del generators['labels']

        return GraphDataset(data_generators=list(generators.values()),
                            data_sources=list(sources.values()),
                            transformations=[reference_transformation],
                            iterator=iterator,
                            debug_image_folder='debug_val' if self.save_debug_images else None)
