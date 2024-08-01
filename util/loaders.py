import os
import pandas as pd

from util.constants import INPUT_FOLDER, OUTPUT_FOLDER
from util.common import register_info, dataframe2tuplelist

def go_back_directory(curr_path, folder_level=1):
    for _ in range(folder_level):
        curr_path = os.path.dirname(curr_path)
    return curr_path

def get_input_data_path(file_name):
    return os.path.join(INPUT_FOLDER, file_name)

def create_folder(folder_levels: list):
    curr_path = os.getcwd()

    for folder_level in folder_levels:
        if folder_level == '..':
            curr_path = go_back_directory(curr_path)
        else:
            curr_path = os.path.join(curr_path, folder_level)
            if not os.path.isdir(curr_path):
                os.makedirs(curr_path)
                print('Output folder located at', curr_path, 'is created')
            else:
                print('Output folder located at', curr_path, 'already exists, so it does not have to be created')

def load_associations_from_csv(file_name, foldernames: list | None = None):
    """
        :return List of tuples containing monarch associations from csv file
    """
    if foldernames:
        curr_path = os.getcwd()
        for foldername in foldernames:
            if foldername == '..':
                curr_path = go_back_directory(curr_path)
            else:
                curr_path = os.path.join(curr_path, foldername)
        file_path = os.path.join(curr_path, file_name)
        associations = pd.read_csv(file_path)
        associations = dataframe2tuplelist(associations)
    else:
        data_path = os.path.join(OUTPUT_FOLDER, file_name)
        associations = pd.read_csv(data_path)
        associations = dataframe2tuplelist(associations)
    
    register_info(f'Loaded {len(associations)} associations')
    return associations