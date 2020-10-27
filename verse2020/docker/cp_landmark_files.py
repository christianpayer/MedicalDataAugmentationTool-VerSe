
import argparse
from glob import glob
import os
import shutil


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--landmark_folder', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)
    parser_args = parser.parse_args()
    
    if not os.path.exists(parser_args.output_folder):
        os.makedirs(parser_args.output_folder)
    
    filenames = glob(os.path.join(parser_args.landmark_folder, '*.json'))
    for f in filenames:
        shutil.copy(f, parser_args.output_folder)
