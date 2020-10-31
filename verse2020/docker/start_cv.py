
import subprocess
import os
import sys
from glob import glob


def main():
    base_image_folder = '/data'
    base_output_folder = os.path.join(base_image_folder, 'results_christian_payer')
    base_intermediate_folder = os.path.join(base_image_folder, 'tmp')
    models_folder = '/models'

    pipeline = sys.argv[1:] if len(sys.argv) > 1 else ['all']
    print('Using pipeline: ', pipeline)

    all_image_folders = [os.path.split(path)[-1] for path in glob(os.path.join(base_image_folder, '*')) if os.path.isdir(path) and path != base_output_folder]
    for current_image_folder in sorted(all_image_folders):
        print('Processing folder ', current_image_folder)
    
        image_folder = os.path.join(base_image_folder, current_image_folder)
        output_folder = os.path.join(base_output_folder, current_image_folder)
        intermediate_folder = os.path.join(base_intermediate_folder, current_image_folder)
        
        preprocessed_image_folder = os.path.join(intermediate_folder, 'data_preprocessed')
        spine_localization_folder = os.path.join(intermediate_folder, 'spine_localization')
        spine_localization_model = os.path.join(models_folder, 'spine_localization')
        vertebrae_localization_folder = os.path.join(intermediate_folder, 'vertebrae_localization')
        vertebrae_localization_model = os.path.join(models_folder, 'vertebrae_localization')
        vertebrae_segmentation_folder = os.path.join(intermediate_folder, 'vertebrae_segmentation')
        vertebrae_segmentation_model = os.path.join(models_folder, 'vertebrae_segmentation')

        if 'preprocessing' in pipeline or 'all' in pipeline:
            subprocess.run(['python', 'preprocess.py',
                            '--image_folder', image_folder,
                            '--output_folder', preprocessed_image_folder,
                            '--sigma', '0.75'])
        if 'spine_localization' in pipeline or 'all' in pipeline:
            subprocess.run(['python', 'main_spine_localization.py',
                            '--image_folder', preprocessed_image_folder,
                            '--setup_folder', intermediate_folder,
                            '--model_files', spine_localization_model,
                            '--output_folder', spine_localization_folder])
        if 'vertebrae_localization' in pipeline or 'all' in pipeline:
            subprocess.run(['python', 'main_vertebrae_localization.py',
                            '--image_folder', preprocessed_image_folder,
                            '--setup_folder', intermediate_folder,
                            '--model_files', vertebrae_localization_model,
                            '--output_folder', vertebrae_localization_folder])
        if 'vertebrae_segmentation' in pipeline or 'all' in pipeline:
            subprocess.run(['python', 'main_vertebrae_segmentation.py',
                            '--image_folder', preprocessed_image_folder,
                            '--setup_folder', intermediate_folder,
                            '--model_files', vertebrae_segmentation_model,
                            '--output_folder', vertebrae_segmentation_folder])
        if 'postprocessing' in pipeline or 'all' in pipeline:
            subprocess.run(['python', 'cp_landmark_files.py',
                            '--landmark_folder', vertebrae_localization_folder,
                            '--output_folder', output_folder])
            subprocess.run(['python', 'reorient_prediction_to_reference.py',
                            '--image_folder', vertebrae_segmentation_folder,
                            '--reference_folder', image_folder,
                            '--output_folder', output_folder])


if __name__ == '__main__':
    main()
