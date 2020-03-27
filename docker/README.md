# Coarse to Fine Vertebrae Localization and Segmentation with SpatialConfiguration-Net and U-Net - Docker image

## Usage
This folder contains the files needed for preparing the Docker image that was submitted to the Verse2019 challenge. The docker image may only be used for running inference on the CPU. The trained network weights are in the folder `models`.
If you have problems/questions/suggestions about the code, write me a [mail](mailto:christian.payer@gmx.net)!

### Build Docker image
Check out the whole `MedicalDataAugmentationTool` project and go to the base folder. The following command will build the Docker image:

`docker build -t verse2019 -f MedicalDataAugmentationTool/bin/experiments/semantic_segmentation/verse2019/docker/Dockerfile .`

### Run Docker image
The Docker image reads all *.nii.gz files from the folder `/data` and performs the whole pipeline of reorientation, spine localization, vertebrae localization, vertebrae segmentation, and reorientation. See the script `predict.sh` for details. The final localization and segmentation results are saved in `/data/results`. To perform inference on the test data, adapt `<FOLDER_TO_VERSE_DATASET>` and run the Docker image with:

`docker run -v <FOLDER_TO_VERSE_DATASET>:/data verse2019 /predict.sh`

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
