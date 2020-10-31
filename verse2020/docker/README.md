# Coarse to Fine Vertebrae Localization and Segmentation with SpatialConfiguration-Net and U-Net - Docker image

## Usage
This folder contains the files needed for preparing the Docker image that was submitted to the Verse2020 challenge. The docker image may be used for running inference on the CPU or GPU. The trained network weights are in the folder `models`.
If you have problems/questions/suggestions about the code, write me a [mail](mailto:christian.payer@gmx.net)!

### Build Docker image
Check out the whole `MedicalDataAugmentationTool` project and go to the base folder. The following command will build the Docker image:

`docker build -t verse2020 -f MedicalDataAugmentationTool-VerSe/verse2020/docker/Dockerfile .`

You can also use the pre-build Docker image from Dockerhub that we used for submitting to the challenge: `https://hub.docker.com/repository/docker/christianpayer/verse20`

### Run Docker image
The Docker image reads all *.nii.gz files in each subfolder from the folder `/data` and performs the whole pipeline of reorientation, spine localization, vertebrae localization, vertebrae segmentation, and reorientation. See the script `predict.sh` for details. The final localization and segmentation results are saved in `/data/results_christian_payer`. Note that the data layout changed from VerSe2019 to VerSe2020 (see https://verse2020.grand-challenge.org/Docker/). To perform inference on the test data, adapt `<FOLDER_TO_VERSE_DATASET>` and run the Docker image with:

For CPU:

`docker run -ti -v <FOLDER_TO_VERSE_DATASET>:/data verse2020 /predict.sh`

For GPU:

`docker run --gpus '"device=0"' -ti -v <FOLDER_TO_VERSE_DATASET>:/data verse2020 /predict.sh`

You can also pass parameters to docker to run individual steps of the pipeline, e.g.,

`docker run --gpus '"device=0"' -ti -v <FOLDER_TO_VERSE_DATASET>:/data verse2020 /predict.sh preprocessing spine_localization`

will run the preprocessing and spine localization. You can use this to correct intermediate outputs (see output folder tmp) to create a semi-automatic approach. Note that it will not be checked, whether the steps have been run in the correct order. So the error messages may not be meaningful. Possible parameter values are: `preprocessing`, `spine_localization`, `vertebrae_localization`, `vertebrae_segmentation`, `postprocessing`.

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
