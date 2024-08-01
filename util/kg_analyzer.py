import pandas as pd

import util.constants as constants
from util.common import register_info
from util.graph import draw_graph

def get_concepts(nodes: pd.DataFrame, node_colmapping: dict):
    """
        Retrieve all unique semantics present in the nodes.
        :param nodes: Dataframe of nodes containing a column for semantic group
        :param node_colmapping: Dictionary indicating name of column holding semantic group
    """
    unique_semantics = nodes[node_colmapping['semantics']].unique()
    register_info(f'There are {len(unique_semantics)} semantic groups: {unique_semantics}')
    return unique_semantics
    
def get_relations(edges: pd.DataFrame, edge_colmapping: dict):
    """
        Retrieve all unique relations present in the edges.
        :param edges: Dataframe of nodes containing a column for relation
        :param edge_colmapping: Dictionary indicating name of column holding relation label
    """
    relations = edges[[edge_colmapping['relationids'], edge_colmapping['relations']]]
    unique_relations_df = relations.groupby(edge_colmapping['relationids']).first()
    register_info(f'There are {unique_relations_df.shape[0]} relation labels: {unique_relations_df}')
    return unique_relations_df.reset_index()
    
def get_connection_summary(edges: pd.DataFrame, nodes: pd.DataFrame, edge_colmapping: dict, node_colmapping: dict, img_name: str, file_name: str, foldername: str):
    """
        Get summary of how the concepts are connected to each other.
        :param nodes: Dataframe of nodes containing a column for semantic group and their identifier
        :param edges: Dataframe of nodes containing a column for relation and the identifier of the subject and object
        :param node_colmapping: Dictionary indicating name of column holding semantic group and identifier
        :param edge_colmapping: Dictionary indicating names of columns holding relation label, subject and object
    """
    edges_relevant_cols = edges[[edge_colmapping['subject'], edge_colmapping['relations'], edge_colmapping['object']]]
    
    joined_subjects = edges_relevant_cols.merge(nodes, left_on=edge_colmapping['subject'], right_on=node_colmapping['node_id'])[[node_colmapping['semantics'], edge_colmapping['relations'], edge_colmapping['object']]]
    joined_subjects.rename(columns={node_colmapping['semantics']: 'semantic_groups_subject'}, inplace=True)
    
    joined_objects = joined_subjects.merge(nodes, left_on=edge_colmapping['object'], right_on=node_colmapping['node_id'])[['semantic_groups_subject', edge_colmapping['relations'], node_colmapping['semantics']]]
    joined_objects.rename(columns={node_colmapping['semantics']: 'semantic_groups_object'}, inplace=True)
    
    joined = joined_objects
    
    subject_object_pairs = joined[['semantic_groups_subject', 'semantic_groups_object']].drop_duplicates().reset_index(drop=True)
    draw_graph(subject_object_pairs, 'semantic_groups_subject', 'semantic_groups_object', f'{constants.OUTPUT_FOLDER}/{foldername}/{img_name}')
    register_info(f'Graph of all connections between concepts saved to {img_name}')
    
    triplets = joined.drop_duplicates().reset_index(drop=True)
    triplets = triplets.sort_values(by=edge_colmapping['relations']).reset_index(drop=True)
    
    triplets.rename(columns={'semantic_groups_subject': 'subject', edge_colmapping['relations']: 'relation', 'semantic_groups_object': 'object'}, inplace=True)
    triplets.to_csv(f'{constants.OUTPUT_FOLDER}/{foldername}/{file_name}', index=False)
    register_info(f'List of triplets saved to {file_name}')