FROM tensorflow/tensorflow:2.2.0-gpu as base
RUN pip3 install SimpleITK==1.2.4 itk==5.1.1 scikit-image==0.17.2 tqdm==4.51.0 networkx==2.5
ADD MedicalDataAugmentationTool /MedicalDataAugmentationTool
ADD MedicalDataAugmentationTool-VerSe/verse2020/other/preprocess.py /MedicalDataAugmentationTool/bin/
ADD MedicalDataAugmentationTool-VerSe/verse2020/other/reorient_prediction_to_reference.py /MedicalDataAugmentationTool/bin/
ADD MedicalDataAugmentationTool-VerSe/verse2020/inference/main_spine_localization.py /MedicalDataAugmentationTool/bin/
ADD MedicalDataAugmentationTool-VerSe/verse2020/inference/main_vertebrae_localization.py /MedicalDataAugmentationTool/bin/
ADD MedicalDataAugmentationTool-VerSe/verse2020/inference/main_vertebrae_segmentation.py /MedicalDataAugmentationTool/bin/
ADD MedicalDataAugmentationTool-VerSe/verse2020/dataset.py /MedicalDataAugmentationTool/bin/
ADD MedicalDataAugmentationTool-VerSe/verse2020/network.py /MedicalDataAugmentationTool/bin/
ADD MedicalDataAugmentationTool-VerSe/verse2020/pickle/possible_successors.pickle /MedicalDataAugmentationTool/bin/
ADD MedicalDataAugmentationTool-VerSe/verse2020/pickle/units_distances.pickle /MedicalDataAugmentationTool/bin/
ADD MedicalDataAugmentationTool-VerSe/verse2020/utils/spine_localization_postprocessing.py /MedicalDataAugmentationTool/bin/
ADD MedicalDataAugmentationTool-VerSe/verse2020/utils/vertebrae_localization_postprocessing.py /MedicalDataAugmentationTool/bin/
ADD MedicalDataAugmentationTool-VerSe/verse2020/docker/cp_landmark_files.py /MedicalDataAugmentationTool/bin/
ADD MedicalDataAugmentationTool-VerSe/verse2020/docker/start_cv.py /MedicalDataAugmentationTool/bin/
ADD MedicalDataAugmentationTool-VerSe/verse2020/docker/models /models
ADD MedicalDataAugmentationTool-VerSe/verse2020/docker/predict.sh /
