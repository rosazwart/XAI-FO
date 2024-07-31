"""
    Module that adds new semantics to associations.
"""

import pandas as pd

from util.constants import GENE, GENOTYPE, TAXON
from util.common import register_info

class Mapper:
    """
    
    """
    def __init__(self, all_edges: pd.DataFrame, all_nodes: pd.DataFrame):
        self.all_edges = all_edges
        self.all_nodes = all_nodes
        self.all_edges_with_nodes = self.join_nodes_edges(all_edges, all_nodes)
        
    def join_nodes_edges(self, all_edges: pd.DataFrame, all_nodes: pd.DataFrame):
        """
            Join given dataframes of all nodes and all edges into one dataframe.
        """
        # Rename columns for clarity
        all_edges_to_join = all_edges.rename(columns={'id': 'edge_id'}, inplace=False)
        all_nodes_to_join = all_nodes.rename(columns={'id': 'node_id', 'label': 'node_label', 'semantic': 'node_semantic'}, inplace=False)

        # Only include relevant columns during joining dataframes
        relevant_info_edges = all_edges_to_join[['edge_id', 'subject', 'object', 'relation_id', 'relation_label']]
        relevant_info_nodes = all_nodes_to_join[['node_id', 'node_label', 'node_semantic']]

        # Merge all information of subject nodes
        joined_subjects = relevant_info_edges.merge(relevant_info_nodes, left_on='subject', right_on='node_id', how='inner')
        joined_subjects.rename(columns={'node_id': 'subject_id', 'node_label': 'subject_label', 'node_semantic': 'subject_semantic'}, inplace=True)

        # Merge all information of object nodes
        joined_objects = joined_subjects.merge(relevant_info_nodes, left_on='object', right_on='node_id', how='inner')
        joined_objects.rename(columns={'node_id': 'object_id', 'node_label': 'object_label', 'node_semantic': 'object_semantic'}, inplace=True)

        all_edges_with_nodes = joined_objects.drop(columns=['subject', 'object'])

        return all_edges_with_nodes
    
    def include_genotype_gene_relations(self):
        """
            A `None` type relation has been found in given nodes and edges between GENOTYPE subject and GENE object. This type of relation can be considered as 'has_allele_of'.
        """
        relation_id = 'CUSTOM:R1'
        relation_type = 'has_allele_of'
        relation_iri = 'none'
        
        relevant_edges = self.all_edges_with_nodes.loc[(self.all_edges_with_nodes['subject_semantic'] == GENOTYPE) & (self.all_edges_with_nodes['object_semantic'] <= GENE)]
        association_ids = relevant_edges['edge_id'].tolist()
        
        for association_id in association_ids:
            self.all_edges.loc[self.all_edges['id'] == association_id, 'relation_id'] = relation_id
            self.all_edges.loc[self.all_edges['id'] == association_id, 'relation_label'] = relation_type
            self.all_edges.loc[self.all_edges['id'] == association_id, 'relation_iri'] = relation_iri
            
        register_info(f'A total of {len(association_ids)} edges have been modified to include the genotype-gene relation.')
    
    def add_taxon_relations(self):
        """
            Add origins of all GENE entities in terms of TAXON (http://purl.obolibrary.org/obo/NCIT_C40098).
        """
        relevant_nodes = self.all_nodes[self.all_nodes[['taxon_id', 'taxon_label']].notnull().all(1)]
        print(relevant_nodes)