"""
    @description: This module adds drugs and their targets based on information from DrugCentral and TTD to data acquired from the Monarch Initiative.
    @author: Modified version of Jupyter Notebook https://github.com/PPerdomoQ/rare-disease-explainer/blob/c2fe63037c9e6fa680850090790597470cddbb12/Notebook_2.ipynb 
"""

import os
import sys

import pandas as pd
import numpy as np
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_dir)

import util.constants as constants
import util.common as common

from util.common import extract_colvalues, register_info, dataframe2tuplelist
from ttd.idmapper import db_mapper, IdMapper, DEFAULT_TO_DB

def load_drug_targets():
    """
        Load drug-target interactions entries from tsv file.
        :return: Dataframe containing drug-target interactions
    """
    drug_targets = pd.read_csv('././input/drug.target.interaction.tsv', header=0, sep='\t', index_col=False)
    common.register_info(f'Loaded {drug_targets.shape[0]} drug-target interactions:\n{drug_targets.head(3)}')
    return drug_targets

def get_organisms(df):
    """
        Get list of all organism names present in given dataframe.
        :param df: Dataframe containing column `ORGANISM`
    """
    organisms = df['ORGANISM'].unique()
    common.register_info(f'There are {organisms.shape[0]} organisms:\n{organisms}')
    return organisms

def get_single_id(id):
    """
        Get from list of IDs split by `|`, one ID.
        :return Single ID value
    """
    split_id = id.split('|')
    if len(split_id) > 1:
        return split_id[0]
    else:
        return id
    
def add_new_ids(df, mapped_ids):
    """
        Add new IDs of entries in given dataframe based on their mappings.
        :param df: Dataframe containing column names `ACCESSION` and `NEW_ID`
        :param mapped_ids: Mappings of IDs containing list of dictionaries with keys `from` and `to`
    """
    for mapped_id in tqdm(mapped_ids):
        accession_id = mapped_id['from']
        new_id = mapped_id['to']
        df.loc[df['ACCESSION'] == accession_id, 'NEW_ID'] = new_id
        
    return df[df['NEW_ID'].notna()]

def fetch_id_mappings(entries: pd.DataFrame, map_to_db):
    """
        Get ID mappings of IDs present in given dataframe. Mappings are based on given database to which the IDs need to be mapped.
        :param entries: Dataframe containing column name `ACCESSION`
        :param map_to_db: Name of database to which the given IDs need to be mapped
    """
    if (entries.shape[0] > 0):
        id_entries_to_map = entries.copy()
        id_entries_to_map['ACCESSION'] = entries.apply(lambda row: get_single_id(row['ACCESSION']), axis=1)
        
        mapper = IdMapper(ids_to_map=id_entries_to_map['ACCESSION'].to_list(), to_db=map_to_db)
        if hasattr(mapper, 'results'):
            return mapper.results
        else:
            return []
    else:
        return []

def get_mapped_ids(drug_targets):
    """
        Get mapped IDs for all included databases.
        :param drug_targets: Dataframe that contains column name `ORGANISM` and `ACCESSION`
    """
    all_mapped_id_results = []
    all_taxon_names = list(db_mapper.keys())
    
    for taxon in all_taxon_names:
        relevant_entries = drug_targets[drug_targets['ORGANISM'].str.contains(taxon)]
        id_mappings = fetch_id_mappings(relevant_entries, db_mapper[taxon])
        all_mapped_id_results = all_mapped_id_results + id_mappings

    # Map entity ids of leftover organisms to default database
    other_relevant_entries = drug_targets[~drug_targets['ORGANISM'].isin(all_taxon_names)]
    other_id_mappings = fetch_id_mappings(other_relevant_entries, DEFAULT_TO_DB)
    all_mapped_id_results = all_mapped_id_results + other_id_mappings

    print(f'A total of {len(all_mapped_id_results)} ACCESSION IDs are mapped to other database IDs')
                
    return all_mapped_id_results

def format_drugtarget_associations(drug_targets_prev: pd.DataFrame, disease_prefix: str):
    """
        Format dataframe with drug target interactions such that it complies with formatting of associations `constants.assoc_tuple_values`.
        :param drug_targets: Dataframe with all drug target interactions
        :return Dataframe with correct column names and order
    """
    drug_targets = drug_targets_prev.drop_duplicates(inplace=False).copy()
    common.register_info(f'Total of {drug_targets_prev.shape[0]} drug-target associations changed to {drug_targets.shape[0]} by dropping duplicates.')
    
    new_assocs = pd.DataFrame()
    
    for i, row in drug_targets.iterrows():
        drug_id = str(row['STRUCT_ID'])
        drug_label = row['DRUG_NAME']
        
        prod_id = row['PROD_ID']
        prod_label = row['PROD_NAME']
        
        gene_id = row['GENE_ID']
        
        prod_gene_relation = constants.IS_PRODUCT_OF
        drug_prod_relation = constants.TARGETS
        
        prod_gene_edge_id = common.generate_edge_id(prod_gene_relation['id'], prod_id, gene_id)
        drug_prod_edge_id = common.generate_edge_id(drug_prod_relation['id'], drug_id, prod_id)
        
        new_edges = {
            'id': [prod_gene_edge_id, drug_prod_edge_id],
            'subject_id': [prod_id, drug_id],
            'subject_label': [prod_label, drug_label],
            'subject_iri': [np.nan, np.nan],
            'subject_category': [constants.GENE_PRODUCT, constants.DRUG],
            'subject_taxon_id': [np.nan, np.nan],
            'subject_taxon_label': [np.nan, np.nan],
            'object_id': [gene_id, prod_id],
            'object_label': [np.nan, prod_label],
            'object_iri': [np.nan, np.nan],
            'object_category': [np.nan, constants.GENE_PRODUCT],
            'object_taxon_id': [np.nan, np.nan],
            'object_taxon_label': [np.nan, np.nan],
            'relation_id': [prod_gene_relation['id'], drug_prod_relation['id']],
            'relation_label': [prod_gene_relation['label'], drug_prod_relation['label']],
            'relation_iri': [prod_gene_relation['iri'], drug_prod_relation['iri']]
        }
        
        #new_assocs = new_assocs.append(pd.DataFrame(new_edges))
        new_assocs = pd.concat([new_assocs, pd.DataFrame(new_edges)], ignore_index=True)

    drugtarget_associations_df = new_assocs[list(constants.assoc_tuple_values)]
    drugtarget_associations_df.to_csv(f'{constants.OUTPUT_FOLDER}/{disease_prefix}/restr_{disease_prefix}_ttd_associations.csv', index=None)
    register_info(f'All {len(drugtarget_associations_df)} TTD associations are saved into restr_{disease_prefix}_ttd_associations.csv')
    
    return dataframe2tuplelist(new_assocs) 

def get_drugtarget_associations(gene_nodes: pd.DataFrame, disease_prefix: str):
    """
        Get all drug target interaction associations
        :param gene_nodes: Dataframe containing existing nodes
        :return List of tuples complying with `constants.assoc_tuple_values`
    """
    # Nodes fetched from Monarch Initiative
    gene_ids = extract_colvalues(gene_nodes, 'id')
    register_info(f'A total of {len(gene_ids)} gene IDs has been extracted')
    
    # Data for drug-target interactions
    drug_targets = load_drug_targets()
    
    # Collect correct target IDs
    print('The IDs in the drug-target interaction database need to be mapped to the previously extracted gene IDs:')
    all_mapped_ids = get_mapped_ids(drug_targets)
    
    # Create dataframe to hold new IDs
    mapped_drug_targets = drug_targets.copy()
    mapped_drug_targets['NEW_ID'] = np.nan
    mapped_drug_targets = add_new_ids(mapped_drug_targets, all_mapped_ids)
    register_info(f'For a total of {mapped_drug_targets.shape[0]} drug-target interactions, new mapped IDs are found.')
    
    matched_drug_targets = mapped_drug_targets[mapped_drug_targets['NEW_ID'].isin(gene_ids)]
    
    # Prepare dataframe with correct columns/column names
    relevant_matched_drug_targets = matched_drug_targets.rename({'NEW_ID': 'GENE_ID', 'ACCESSION': 'PROD_ID', 'TARGET_NAME': 'PROD_NAME'}, axis=1)[['DRUG_NAME', 'STRUCT_ID', 'GENE_ID', 'PROD_ID', 'PROD_NAME']]
    register_info(f'Retrieved {relevant_matched_drug_targets.shape[0]} drug-target interactions with matched gene IDs:\n{relevant_matched_drug_targets.head(4)}')

    drugtargets_associations = format_drugtarget_associations(relevant_matched_drug_targets, disease_prefix)
    return drugtargets_associations