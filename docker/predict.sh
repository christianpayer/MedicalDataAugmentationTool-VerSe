#!/bin/bash

export PYTHONPATH=/MedicalDataAugmentationTool
mkdir /data/results
cd /scripts/
python reorient_reference_to_rai.py --image_folder /data --output_folder /tmp/data_reoriented
python main_spine_localization.py --image_folder /tmp/data_reoriented --setup_folder /tmp/ --model_files /models/spine_localization/model --output_folder /tmp/
python main_vertebrae_localization.py --image_folder /tmp/data_reoriented --setup_folder /tmp/ --model_files /models/vertebrae_localization/model --output_folder /tmp/
cp /tmp/vertebrae_localization/verse_landmarks/* /data/results/
python main_vertebrae_segmentation.py --image_folder /tmp/data_reoriented --setup_folder /tmp/ --model_files /models/vertebrae_segmentation/model --output_folder /tmp/
python reorient_prediction_to_reference.py --image_folder /tmp/vertebrae_segmentation --reference_folder /data --output_folder /data/results
