import os
import gzip
import pandas as pd
from tqdm import tqdm

def get_data_file_paths(dir_name: str = 'data'):
    file_paths = list()

    curr_wdir = os.getcwd()
    data_dir_path = os.path.join(curr_wdir, dir_name)

    for file_name in os.listdir(data_dir_path):
        file_path = os.path.join(curr_wdir, dir_name, file_name)
        if os.path.isfile(file_path):
            file_paths.append(file_path)

    return file_paths

def unzip_file(file_path: str):
    with gzip.open(file_path) as f:
        content_df = pd.read_csv(f, sep='\t', low_memory=False)
    return content_df

def create_empty_df(col_list: list):
    return pd.DataFrame(columns=col_list)

def get_all_assoc_df(dir_name: str):
    data_file_paths = get_data_file_paths(dir_name=dir_name)

    data_dfs = list()

    print(f'Loading files from directory {dir_name}:')
    for data_file_path in tqdm(data_file_paths):
        data_df = unzip_file(data_file_path)

        file_name = data_file_path.split('\\')[-1].split('.')[0]
        subject_category, object_category = file_name.split('_')

        #if 'object_taxon:1' in data_df.columns and 'object_taxon_label:1' in data_df.columns:
        #    data_df.drop(labels=['object_taxon:1', 'object_taxon_label:1'], axis=1, inplace=True)

        data_df['subject_category'] = subject_category
        data_df['object_category'] = object_category

        data_dfs.append(data_df)
    
    print(f'Populating dataframe with content of all files from directory {dir_name}:')
    all_assoc_df = create_empty_df(col_list=list(data_dfs[0].columns.values))
    for data_df in tqdm(data_dfs):
        all_assoc_df = pd.concat([all_assoc_df, data_df]).drop_duplicates().reset_index(drop=True)
    
    print(f'Loaded all associations having a total of {all_assoc_df.shape[0]} rows')

    categories = set()
    categories.update(list(all_assoc_df['subject_category'].unique()))
    categories.update(list(all_assoc_df['object_category'].unique()))
    print('All categories:', categories)

    print('All predicates:', all_assoc_df['relation_label'].unique())

    return all_assoc_df





