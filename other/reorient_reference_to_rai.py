
import argparse
from glob import glob
import os
import numpy as np
import itk


def reorient_to_rai(image):
    """
    Reorient image to RAI orientation.
    :param image: Input itk image.
    :return: Input image reoriented to RAI.
    """
    filter = itk.OrientImageFilter.New(image)
    filter.UseImageDirectionOn()
    filter.SetInput(image)
    m = itk.GetMatrixFromArray(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64))
    filter.SetDesiredCoordinateDirection(m)
    filter.Update()
    reoriented = filter.GetOutput()
    return reoriented
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)
    parser_args = parser.parse_args()
    if not os.path.exists(parser_args.output_folder):
        os.makedirs(parser_args.output_folder)
    filenames = glob(os.path.join(parser_args.image_folder, '*.nii.gz'))
    for filename in sorted(filenames):
        basename = os.path.basename(filename)
        basename_wo_ext = basename[:basename.find('.nii.gz')]
        print(basename_wo_ext)
        image = itk.imread(filename)
        reoriented = reorient_to_rai(image)
        
        reoriented.SetOrigin([0, 0, 0])
        m = itk.GetMatrixFromArray(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64))
        reoriented.SetDirection(m)
        reoriented.Update()
        itk.imwrite(reoriented, os.path.join(parser_args.output_folder, basename_wo_ext + '.nii.gz'))
    

