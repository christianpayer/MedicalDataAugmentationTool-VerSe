# use tensorflow base image
FROM tensorflow/tensorflow:1.14.0-py3 as base
# install python dependencies
RUN pip3 install SimpleITK itk scikit-image cachetools
ADD MedicalDataAugmentationTool /MedicalDataAugmentationTool
# copy verse files
ADD MedicalDataAugmentationTool-VerSe/docker/models /models
RUN mkdir /scripts
ADD MedicalDataAugmentationTool-VerSe/inference/*.py /scripts/
ADD MedicalDataAugmentationTool-VerSe/other/*.py /scripts/
ADD MedicalDataAugmentationTool-VerSe/*.py /scripts/
ADD MedicalDataAugmentationTool-VerSe/docker/predict.sh /
