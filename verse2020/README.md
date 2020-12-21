# Coarse to Fine Vertebrae Localization and Segmentation with SpatialConfiguration-Net and U-Net

## Usage
This code has been used for the Verse2020 challenge as well as the paper [Coarse to Fine Vertebrae Localization and Segmentation with SpatialConfiguration-Net and U-Net](https://doi.org/10.5220/0008975201240133). The folder `training` contains the scripts used for training networks, the folder `inference` contains scripts used for inference only. The files for preparing the Docker image that was submitted to the Verse2020 challenge are in the folder `docker`. The folder `verse2020_dataset` contains the setup files, e.g., cross-validation setup and landmark files. The folder `other` contains the preprocessing and reorientation scripts.

You need to have the [MedicalDataAugmentationTool](https://github.com/christianpayer/MedicalDataAugmentationTool) framework downloaded and in you PYTHONPATH for the scripts to work.
The required python packages include `tensorflow-gpu==2.2.0`, `SimpleITK==1.2.4`, `itk==5.1.1`, `scikit-image==0.17.2`, `tqdm==4.51.0`, `networkx==2.5`, `Pyro4==4.80`.
If you have problems/questions/suggestions about the code, write me a [mail](mailto:christian.payer@gmx.net)!

### Dataset preprocessing
Download the files from the [challenge website](https://verse2020.grand-challenge.org/) and copy them to the folder `verse2020_dataset/images`. The framework expects that the `.nii.gz` files are directly in the folder `verse2020_dataset/images`, e.g., `verse2020_dataset/images/GL003.nii.gz`. In order for the framework to be able to load the data, every image needs to be reoriented to RAI. The following script in the folder `other` performs the reorientation to RAI and Gaussian smoothing for all images:

`python preprocess.py --image_folder ../verse2020_dataset/images --output_folder ../verse2020_dataset/images_reoriented --sigma 0.75`

### Train models
In the folder `training` there are the scripts for training the spine localization, vertebrae localization, and vertebrae segmentation networks. Currently they are set up to train the three cross-validation folds as well as train on the whole training set. If you want to see the augmented network input images, set `self.save_debug_images = True`. This will save the images into the folders `debug_train` and `debug_val`. However, as every augmented images will be saved to the hard disk, this could lead to longer training times on slow computers.
If you want to see more detailed network outputs when testing, set `self.save_output_images = True`.

The vertebrae segmentation script needs a lot of memory for testing. We used a workstation with 32GB RAM, where the scripts worked. If you have less memory and problems with the testing phase, disable the testing code by setting `self.test_iter` to a number larger than `self.max_iter`.

Per default, training uses a dedicated process for faster preprocessing (by reducing the influence of python GIL). Before running the training scripts in `training/main_*.py` start the script `training/server_dataset_loop.py`. You can also run this process on a remote server. You would need to adapt the base_folder and server address for that.
If you do not want to run a dedicated process for preprocessing, set `self.use_pyro_dataset = False` in `training/main_*.py`. 

### Inference
In the folder `inference` there are dedicated scripts for inference only. These scripts can be used to load trained models and evaluate networks on all images from a folder.

There are also files for generating a Docker image in the folder `docker`. Look at these files if you want to know, how the whole pipeline may be executed.

### Train and test other datasets
In order to train and test on other datasets, modify the `dataset.py` file. See the example files and documentation for the specific file formats. Set the parameter `save_debug_images = True` in order to see, if the network input images are reasonable.

## Citation
If you use this code for your research, please cite our [paper](https://doi.org/10.5220/0008975201240133) and the overview paper of the [Verse2019 challenge](https://arxiv.org/abs/2001.09193):

```
@inproceedings{Payer2020,
  title     = {Coarse to Fine Vertebrae Localization and Segmentation with SpatialConfiguration-Net and U-Net},
  author    = {Payer, Christian and {\v{S}}tern, Darko and Bischof, Horst and Urschler, Martin},
  booktitle = {Proceedings of the 15th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications - Volume 5: VISAPP},
  doi       = {10.5220/0008975201240133},
  pages     = {124--133},
  volume    = {5},
  year      = {2020}
}
```

```
@misc{Sekuboyina2020verse,
 title         = {VerSe: A Vertebrae Labelling and Segmentation Benchmark},
 author        = {Anjany Sekuboyina and Amirhossein Bayat and Malek E. Husseini and Maximilian Löffler and Markus Rempfler and Jan Kukačka and Giles Tetteh and Alexander Valentinitsch and Christian Payer and Martin Urschler and Maodong Chen and Dalong Cheng and Nikolas Lessmann and Yujin Hu and Tianfu Wang and Dong Yang and Daguang Xu and Felix Ambellan and Stefan Zachowk and Tao Jiang and Xinjun Ma and Christoph Angerman and Xin Wang and Qingyue Wei and Kevin Brown and Matthias Wolf and Alexandre Kirszenberg and Élodie Puybareauq and Björn H. Menze and Jan S. Kirschke},
 year          = {2020},
 eprint        = {2001.09193},
 archivePrefix = {arXiv},
 primaryClass  = {cs.CV}
}
```
