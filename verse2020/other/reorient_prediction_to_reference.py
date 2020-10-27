
import argparse
import itk
from glob import glob
import os
import multiprocessing


def reorient_to_reference(image, reference):
    """
    Reorient image to reference. See itk.OrientImageFilter.
    :param image: Input itk image.
    :param reference: Reference itk image.
    :return: Input image reoriented to reference image.
    """
    filter = itk.OrientImageFilter[type(image), type(image)].New(image)
    filter.UseImageDirectionOn()
    filter.SetInput(image)
    filter.SetDesiredCoordinateDirection(reference.GetDirection())
    filter.Update()
    return filter.GetOutput()


def cast(image, reference):
    """
    Cast image to reference image type.
    :param image: Input itk image.
    :param reference: Reference itk image.
    :return: Input image cast to reference image type.
    """
    filter = itk.CastImageFilter[type(image), type(reference)].New()
    filter.SetInput(image)
    filter.Update()
    return filter.GetOutput()


def copy_information(image, reference):
    """
    Copy image information (spacing, origin, direction) from reference image to input image.
    :param image: Input itk image.
    :param reference: Reference itk image.
    :return: Input image with image information from reference image.
    """
    filter = itk.ChangeInformationImageFilter[type(image)].New()
    filter.SetInput(image)
    filter.SetReferenceImage(reference)
    filter.UseReferenceImageOn()
    filter.ChangeSpacingOn()
    filter.ChangeOriginOn()
    filter.ChangeDirectionOn()
    filter.Update()
    return filter.GetOutput()


def process_image(filename, output_folder, reference_folder):
    basename = os.path.basename(filename)
    basename_wo_ext = basename[:basename.find('.nii.gz')]
    basename_wo_ext_and_seg = basename_wo_ext[:basename_wo_ext.find('_seg')]
    print(basename_wo_ext)
    image = itk.imread(filename)
    reference = itk.imread(os.path.join(reference_folder, basename_wo_ext_and_seg + '.nii.gz'), itk.US)
    reoriented = cast(image, reference)
    reoriented = reorient_to_reference(reoriented, reference)
    reoriented = copy_information(reoriented, reference)
    itk.imwrite(reoriented, os.path.join(output_folder, basename_wo_ext + '.nii.gz'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, required=True)
    parser.add_argument('--reference_folder', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)
    parser_args = parser.parse_args()
    if not os.path.exists(parser_args.output_folder):
        os.makedirs(parser_args.output_folder)
    filenames = glob(os.path.join(parser_args.image_folder, '*_seg.nii.gz'))
    pool = multiprocessing.Pool(8)
    pool.starmap(process_image, [(filename, parser_args.output_folder, parser_args.reference_folder) for filename in sorted(filenames)])

