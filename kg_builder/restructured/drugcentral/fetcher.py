import os
import sys

import re
import pandas as pd
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import util.constants as constants
import util.common as common
import drugcentral.matcher as matcher

ID_LINE = 'TTDDRUID'
NAME_LINE = 'DRUGNAME'
DISEASE_PHASE_LINE = 'INDICATI'
EMPTY_LINE = '\n'
        
def load_drug_disease_entries():
    with open('././input/P1-05-Drug_disease.txt') as f:
        entry_lines = f.readlines()[22:]
        
    drug_disease_pairs = list()
    current_drug_disease_pair = dict()
    
    for line in entry_lines:
        if line.startswith(ID_LINE):
            found_str = re.search('\t(.*)\n', line)
            current_drug_disease_pair['DRUG_ID'] = found_str.group(1)
        elif line.startswith(NAME_LINE): 
            found_str = re.search('\t(.*)\n', line)
            current_drug_disease_pair['DRUG_NAME'] = found_str.group(1).lower()
        elif line.startswith(DISEASE_PHASE_LINE): 
            found_str1 = re.search(r'\t(.*)\[', line)
            found_str2 = re.search(r'\](.*)', line)
            
            str1 = found_str1.group(1)
            
            current_drug_disease_pair['DISEASE_NAME'] = re.sub("[^0-9a-zA-Z]+", " ", str1).lower().strip()
            current_drug_disease_pair['PHASE'] = found_str2.group(1) 
            
            drug_disease_pairs.append(current_drug_disease_pair.copy())
        elif line.startswith(EMPTY_LINE): 
            current_drug_disease_pair = dict()
    
    drug_disease_df = pd.DataFrame.from_dict(drug_disease_pairs, orient='columns')
    common.register_info(f'Loaded {drug_disease_df.shape[0]} drug-disease pairs:\n{drug_disease_df.head(3)}')
    return drug_disease_df

def join_disease_name_with_id(names_df, ids_df):
    joined_df = pd.merge(names_df, ids_df, left_on='DISEASE_NAME', right_on='Name', how='left')
    left_outer_joined_df = joined_df[joined_df['DISEASE_ID'].notna()][['DRUG_ID', 'DRUG_NAME', 'DISEASE_ID', 'DISEASE_NAME']]
    common.register_info(f'Total of {left_outer_joined_df.shape[0]} disease names mapped to their IDs:\n{left_outer_joined_df.head(10)}')
    return left_outer_joined_df

def format_drugdisease_associations(drug_disease_pairs_prev: pd.DataFrame, drug_nodes: pd.DataFrame, disease_prefix: str):
    drug_disease_pairs = drug_disease_pairs_prev.drop_duplicates(inplace=False).copy()
    common.register_info(f'Total of {drug_disease_pairs_prev.shape[0]} drug-disease associations changed to {drug_disease_pairs.shape[0]} by dropping duplicates.')
    
    for i, row in drug_disease_pairs.iterrows():
        drug_id = drug_nodes.loc[drug_nodes['label'] == row['DRUG_NAME']]['id'].values[0]
        subject_id = str(drug_id)
        object_id = row['DISEASE_ID']
        relation_id = constants.TREATS['id']
        
        drug_disease_pairs.loc[i,'id'] = common.generate_edge_id(relation_id, subject_id, object_id)
        
        drug_disease_pairs.loc[i,'subject_id'] = subject_id
        drug_disease_pairs.loc[i,'subject_label'] = row['DRUG_NAME']
        drug_disease_pairs.loc[i,'subject_iri'] = np.nan
        drug_disease_pairs.loc[i,'subject_category'] = constants.DRUG
        drug_disease_pairs.loc[i,'subject_taxon_id'] = np.nan
        drug_disease_pairs.loc[i,'subject_taxon_label'] = np.nan
        
        drug_disease_pairs.loc[i,'object_id'] = object_id
        drug_disease_pairs.loc[i,'object_label'] = np.nan
        drug_disease_pairs.loc[i,'object_iri'] = np.nan
        drug_disease_pairs.loc[i,'object_category'] = np.nan
        drug_disease_pairs.loc[i,'object_taxon_id'] = np.nan
        drug_disease_pairs.loc[i,'object_taxon_label'] = np.nan
        
        drug_disease_pairs.loc[i,'relation_id'] = relation_id
        drug_disease_pairs.loc[i,'relation_label'] = constants.TREATS['label']
        drug_disease_pairs.loc[i,'relation_iri'] = constants.TREATS['iri']
        
    drugdisease_associations_df = drug_disease_pairs[list(constants.assoc_tuple_values)]
    drugdisease_associations_df.to_csv(f'{constants.OUTPUT_FOLDER}/{disease_prefix}/restr_{disease_prefix}_drugcentral_associations.csv', index=None)
    common.register_info(f'All DrugCentral associations are saved into restr_{disease_prefix}_drugcentral_associations.csv')
    
    return common.dataframe2tuplelist(drugdisease_associations_df) 
    
def get_drugdisease_associations(drug_nodes: pd.DataFrame, diso_pheno_nodes: pd.DataFrame, disease_prefix: str):
    """
        Get 
    """
    drug_names = common.extract_colvalues(drug_nodes, 'label')
    common.register_info(f'There are {len(drug_names)} unique drug names')
    
    diso_pheno_ids = common.extract_colvalues(diso_pheno_nodes, 'id')
    common.register_info(f'There are {len(diso_pheno_ids)} unique disease/phenotype IDs')
    
    # Data for drug disease pairs
    drug_disease_pairs = load_drug_disease_entries()
    # Data with phenotype names and IDs
    phenotype_matches = matcher.load_phenotype_matcher()
    
    drug_disease_edges = join_disease_name_with_id(drug_disease_pairs, phenotype_matches)
    included_drug_disease_edges = drug_disease_edges.loc[(drug_disease_edges['DISEASE_ID'].isin(diso_pheno_ids)) & (drug_disease_edges['DRUG_NAME'].isin(drug_names))]
    common.register_info(f'A total of {included_drug_disease_edges.shape[0]} are matched with existing drugs and diseases/phenotypes')
    
    drugdisease_associations = format_drugdisease_associations(included_drug_disease_edges, drug_nodes, disease_prefix)
    return drugdisease_associations