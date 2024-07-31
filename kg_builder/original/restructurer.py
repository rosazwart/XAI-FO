import sys
import os

# Get the directory of the current script (restructurer.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory by going up two levels
project_root = os.path.dirname(os.path.dirname(current_dir))
# Add the project root directory to the system path
sys.path.append(project_root)

from util.loaders import load_associations_from_csv, create_folder
from util.constants import assoc_tuple_values, OUTPUT_FOLDER
from util.common import tuplelist2dataframe

from constants import prefix2category

def convert_concepts(node_id: str):
    """
    """
    prefix, _ = node_id.split(':')
    return prefix2category(prefix)

def remove_tuplelist_duplicates(tuplelist: list):
    return [t for t in (set(tuple(i) for i in tuplelist))]

def get_nodes(assoc: list):
    """
    """
    nodes = set()

    for assoc_tuple in assoc:
        node_subject_id = assoc_tuple[assoc_tuple_values.index('subject_id')]
        node_object_id = assoc_tuple[assoc_tuple_values.index('object_id')]

        node_subject_category = assoc_tuple[assoc_tuple_values.index('subject_category')]
        node_object_category = assoc_tuple[assoc_tuple_values.index('object_category')]

        node_subject_label = assoc_tuple[assoc_tuple_values.index('subject_label')]
        node_object_label = assoc_tuple[assoc_tuple_values.index('object_label')]

        nodes.add(tuple([node_subject_id, node_subject_category, node_subject_label]))
        nodes.add(tuple([node_object_id, node_object_category, node_object_label]))

    nodes_tuplelist = list(nodes)
    remove_tuplelist_duplicates(nodes_tuplelist)

    return nodes_tuplelist

if __name__ == "__main__":
    DISEASE_PREFIX = input('For which disease is the knowledge graph built? Choose from "dmd", "hd" and "oi"')
    assert DISEASE_PREFIX == 'dmd' or 'hd' or 'oi'
    FILE_DATE = input('What is the date of creation (yyyy-mm-dd) of the knowledge graph build?')
    FILENAME = f'{DISEASE_PREFIX}_monarch_associations_{FILE_DATE}.csv'

    create_folder(folder_levels=[OUTPUT_FOLDER, DISEASE_PREFIX])
    create_folder(folder_levels=[current_dir, OUTPUT_FOLDER])

    monarch_assoc = load_associations_from_csv(file_name=FILENAME, foldernames=['localfetcher', 'output'])

    converted_monarch_assoc = []
    for monarch_assoc_tuple in monarch_assoc:
        monarch_assoc_list = list(monarch_assoc_tuple)

        node_subject_id = monarch_assoc_list[assoc_tuple_values.index('subject_id')]
        node_subject_category_index = assoc_tuple_values.index('subject_category')

        node_object_id = monarch_assoc_list[assoc_tuple_values.index('object_id')]
        node_object_category_index = assoc_tuple_values.index('object_category')

        monarch_assoc_list[node_subject_category_index] = convert_concepts(node_id=node_subject_id)
        monarch_assoc_list[node_object_category_index] = convert_concepts(node_id=node_object_id)

        converted_monarch_assoc.append(tuple(monarch_assoc_list))

    tuplelist2dataframe(converted_monarch_assoc).to_csv(f'{OUTPUT_FOLDER}/{DISEASE_PREFIX}/prev_{FILENAME}', index=False)

    monarch_nodes = get_nodes(assoc=converted_monarch_assoc)
    tuplelist2dataframe(monarch_nodes, 
                        column_values=tuple(['id', 'semantic_groups', 'name'])).to_csv(f'kg_builder/original/output/prev_{FILENAME.replace("associations", "nodes")}', index=False)

    