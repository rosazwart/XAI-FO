import pandas as pd
import numpy as np

from tqdm import tqdm

import constants as const
import util as util
import association_filterer as assoc_filterer

def load_into_tuple(assoc_df: pd.DataFrame, seed_id_list: list = None):
    """
        Load values from dataframe into a tuple. Use defined tuple value names `constants.assoc_tuple_values` 
        to navigate through association dataframe. 
        :param assoc_df: dataframe of associations
        :param seed_id_list: when list of seeds is given, exclude all associations that introduce nodes 
        that are not found in this list of seeds
        :return: list of tuples of association information values
    """
    all_assoc = list()

    for _, row in assoc_df.iterrows():
        assoc_tuple = list()

        for tuple_value in const.assoc_tuple_values:
            if tuple_value == 'id':
                id_value = util.generate_edge_id(relation_id=str(row['relation']), subject_id=str(row['subject']), object_id=str(row['object']))
                assoc_tuple.append(id_value)
            elif 'iri' in tuple_value:
                assoc_tuple.append(np.nan)
            elif 'taxon' in tuple_value:
                if tuple_value not in row:
                    assoc_tuple.append(np.nan)
                else:
                    assoc_tuple.append(row[tuple_value])
            else:
                assoc_tuple.append(row[tuple_value])
        
        if seed_id_list:
            if {row['subject'], row['object']} <= set(seed_id_list): # check whether every element in set at left is in set at right
                all_assoc.append(tuple(assoc_tuple))
        else:
            all_assoc.append(tuple(assoc_tuple))

    return all_assoc

def get_from_associations(assoc_df: pd.DataFrame, subject_node: str, relation: str | None = None, id_list: list = None, max_rows: int = 2000):
    """
        Get all out edges of given node. 
        :param assoc_df: dataframe of associations
        :param subject_node: subject node from which the edges need to be directed
        :param relation: if given, only select associations with this relation as predicate
        :param id_list: if given, only get associations that do not introduce nodes that are not found in given list of nodes
    """
    select_df = assoc_df.loc[assoc_df['subject'] == subject_node]

    if relation:
        select_df = select_df.loc[select_df['relation'] == relation]
        # print(f'With node {subject_node} as subject and {relation} as predicate {select_df.shape[0]} associations found')
    # else:
    #    print(f'With node {subject_node} as subject {select_df.shape[0]} associations found')

    limited_select_df = select_df.head(max_rows)

    return load_into_tuple(assoc_df=limited_select_df, seed_id_list=id_list)

def get_to_associations(assoc_df: pd.DataFrame, object_node: str, relation: str | None = None, id_list: list = None, max_rows: int = 2000):
    """
        Get all out edges of given node. 
        :param assoc_df: dataframe of associations
        :param object_node: subject node to which the edges need to be directed
        :param relation: if given, only select associations with this relation as predicate
        :param id_list: if given, only get associations that do not introduce nodes that are not found in given list of nodes
    """
    select_df = assoc_df.loc[assoc_df['object'] == object_node]

    if relation:
        select_df = select_df.loc[select_df['relation'] == relation]
        # print(f'With node {object_node} as object and {relation} as predicate {select_df.shape[0]} associations found')
    # else:
    #     print(f'With node {object_node} as object {select_df.shape[0]} associations found')

    limited_select_df = select_df.head(max_rows)

    return load_into_tuple(assoc_df=limited_select_df, seed_id_list=id_list)

def get_associations(assoc_df: pd.DataFrame, direction: str, node: str, relation: str | None = None, id_list: list = None):
    """
        Get associations given a direction and seed node.
        :param assoc_df: dataframe of associations
        :param direction: direction of association being 'from' or 'to' seed
        :param node: seed node 
        :param relation: if given, only select associations with this relation as predicate
        :param id_list: if given, only get associations that do not introduce nodes that are not found in given list of nodes
    """
    assert direction == 'to' or direction == 'from', "Direction parameter needs to be either 'to' or 'from'"

    assoc_tuples_list = list()
    if direction == 'from':
        assoc_tuples_list = get_from_associations(assoc_df=assoc_df, subject_node=node, relation=relation, id_list=id_list)
    else:
        assoc_tuples_list = get_to_associations(assoc_df=assoc_df, object_node=node, relation=relation, id_list=id_list)

    return assoc_tuples_list

def get_neighbour_ids(seed_list: list, associations: list, include_semantic_groups: list = []):
    """
        Get all IDs of list of associations that are not the seed IDs. If given, only include IDs when 
        entity belongs to at least one of given semantic groups.
        :param seed_list: list of seed ids
        :param associations: list of tuple associations
        :param include_semantic_groups: list of semantic groups to which all neighbour ids need to belong
        :return: set of (filtered) neighbour ids 
    """
    neighbour_ids = set()

    for association in associations:
        subject_id = association[const.assoc_tuple_values.index('subject')] 
        object_id = association[const.assoc_tuple_values.index('object')] 

        if not(subject_id in seed_list) and (len(include_semantic_groups) == 0 or association[const.assoc_tuple_values.index('subject_category')] in include_semantic_groups):
            neighbour_ids.add(subject_id)
        
        if not(object_id in seed_list) and (len(include_semantic_groups) == 0 or association[const.assoc_tuple_values.index('object_category')] in include_semantic_groups):
            neighbour_ids.add(object_id)

    return neighbour_ids

def get_neighbour_associations(assoc_df: pd.DataFrame, id_list: list, relations: list = [], exclude_new_ids: bool = False):
    """
        Return the first layer of neighbours from a list of seed nodes.
        :param id_list: list of entities represented by their identifiers
        :param relations: when parsing a non-empty list, these elements are the relation ids such that only associations 
        are retrieved including these relations
        :return: list of first-order neighbours (list of tuples)
    """
    all_associations = set()
    all_seed_nodes = set(id_list)   # make sure there are no duplicate seed ids

    if exclude_new_ids:
        included_id_list = all_seed_nodes
    else:
        included_id_list = None

    for seed_node in tqdm(all_seed_nodes):
        if len(relations) > 0:
            for relation_id in relations:
                assoc_in = get_associations(assoc_df=assoc_df, direction='to', node=seed_node, relation=relation_id, id_list=included_id_list)
                assoc_out = get_associations(assoc_df=assoc_df, direction='from', node=seed_node, relation=relation_id, id_list=included_id_list)

                all_associations.update(assoc_in)
                all_associations.update(assoc_out)
        else:
            assoc_in = get_associations(assoc_df=assoc_df, direction='to', node=seed_node, id_list=included_id_list)
            assoc_out = get_associations(assoc_df=assoc_df, direction='from', node=seed_node, id_list=included_id_list)
            
            all_associations.update(assoc_in)
            all_associations.update(assoc_out)

    return all_associations

def get_seed_neighbour_node_ids(assoc_df: pd.DataFrame, seed_id_list: list):
    """
        Get a list of all node IDs of all first order neighbours of given seeds.
        :param seed_id_list: list of entities represented by their identifiers
        :return: list of neighbour node ids
    """
    print('Neighbours of seeds retrieval has started...')

    direct_neighbours_associations = get_neighbour_associations(assoc_df=assoc_df, id_list=seed_id_list)
    print(f'A total of {len(direct_neighbours_associations)} associations have been found between seeds and their neighbours.')

    neighbour_ids = get_neighbour_ids(seed_list=seed_id_list, associations=direct_neighbours_associations)
    print(f'A total of {len(neighbour_ids)} neighbour nodes have been found for the {len(seed_id_list)} given seeds.')

    return neighbour_ids

def get_orthopheno_node_ids(assoc_df: pd.DataFrame, first_seed_id_list: list, depth: int):
    """
        Get list of all nodes ids yielded from associations between an ortholog gene and phenotype. In the first 
        iteration, orthologs are found for given seed list.
        :param first_seed_id_list: list of entities that are the seeds of first iteration
        :param depth: number of iterations
    """
    print('Orthologs/phenotypes retrieval has started...')

    all_sets = list()   # list of all sets that eventually need to be merged into one set

    # Initial iteration seed list
    seed_list = first_seed_id_list

    for d in range(depth):  
        if (d+1 > 1):
            print(f'At depth {d+1}, replace previous list of seeds with all their first order neighbours.')
        print(f'For depth {d+1} seed list contains {len(seed_list)} ids')

        # Get associations between seeds and their first order neighbours
        direct_neighbours_associations = get_neighbour_associations(assoc_df=assoc_df, id_list=seed_list)
        # Get all ids of found first order neighbour nodes
        direct_neighbour_id_set = get_neighbour_ids(seed_list=seed_list, associations=direct_neighbours_associations)
        print(f'{len(direct_neighbour_id_set)} neighbours of given seeds')

        # Filter to only include associations related to orthology
        associations_with_orthologs = assoc_filterer.get_associations_on_relations(all_associations=direct_neighbours_associations, 
                                                                                   include_relation_ids_group='orthologous')
        
        # Get all orthologs of genes included in given list of ids
        ortholog_id_set = get_neighbour_ids(seed_list=seed_list, 
                                            associations=associations_with_orthologs,
                                            include_semantic_groups=['gene'])
        print(f'{len(ortholog_id_set)} orthologous genes of given seeds')

        # Get the first layer of neighbours of orthologs
        ortholog_associations = get_neighbour_associations(assoc_df=assoc_df, id_list=ortholog_id_set)
        # Filter to only include associations related to phenotype
        phenotype_id_set = get_neighbour_ids(seed_list=ortholog_id_set, associations=ortholog_associations, include_semantic_groups=['phenotype'])
        print(f'{len(phenotype_id_set)} phenotypes of orthologous genes')

        # Add set of ortholog nodes of seeds and set of phenotype nodes of ortholog nodes
        all_sets.append(ortholog_id_set)
        all_sets.append(phenotype_id_set)
        print(f'{len(ortholog_id_set)+len(phenotype_id_set)} orthologs/phenotypes')

        # Next iteration seed list
        seed_list = direct_neighbour_id_set

    all_ortho_pheno_node_ids = set().union(*all_sets)
    print(f'A total of {len(all_ortho_pheno_node_ids)} orthologs/phenotypes have been found using a depth of {depth}')
    
    return all_ortho_pheno_node_ids

def get_monarch_associations(assoc_df: pd.DataFrame, nodes_list: list, disease_file_name_ref: str):
    """
        Get all associations given seed nodes.
        :param assoc_df: dataframe of associations
        :param nodes_list: list of initial seeds of intended KG
    """
    seed_neighbours_id_set = get_seed_neighbour_node_ids(assoc_df=assoc_df, seed_id_list=nodes_list)
    orthopheno_id_list = get_orthopheno_node_ids(assoc_df=assoc_df, first_seed_id_list=nodes_list, depth=2)

    print(f'A total of {len(seed_neighbours_id_set)} first order neighbours of given seeds have been found')
    print(f'A total of {len(orthopheno_id_list)} orthologs/phenotypes have been found.')

    all_nodes_id_list = seed_neighbours_id_set.union(orthopheno_id_list)
    all_nodes_id_list.update(nodes_list)
    print(f'A total of {len(all_nodes_id_list)} nodes have been found for which from and to associations will be retrieved.')

    # Get all first order associations allowing only the given node IDs
    print('Associations of seeds retrieval has started...')
    all_associations = get_neighbour_associations(assoc_df=assoc_df, id_list=all_nodes_id_list, exclude_new_ids=True)
    print(f'A total of {len(all_associations)} associations have been found between all retrieved nodes.')

    util.tuplelist2dataframe(tuple_list=list(all_associations)).to_csv(f'localfetcher/output/{disease_file_name_ref}_monarch_associations.csv', index=False)
    
    return all_associations