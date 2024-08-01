from util.common import register_info

import util.constants as constants
import util.common as common

import pandas as pd
import numpy as np

from tqdm import tqdm
from copy import deepcopy

class Node:
    """
        Initialize nodes using this class carrying information about the entity it represents.
    """
    def __eq__(self, other):
        return self.id == other.id
    
    def __hash__(self):
        return hash(self.id)
    
class AssocNode(Node):
    """
        Initialize nodes using this class, carrying information about the id of the entity,
        the semantic groups that are associated with the entity, the label of the entity as well as
        the iri link. The taxon attribute is used to specify whether the entity belongs to a certain taxon
        when applicable.
        :param assoc_tuple: tuple of information about association
    """
    def __init__(self, assoc_tuple: tuple, node_role: str):
        self.id = assoc_tuple[constants.assoc_tuple_values.index(f'{node_role}_id')]
        self.semantic_groups = assoc_tuple[constants.assoc_tuple_values.index(f'{node_role}_category')]
        self.label = assoc_tuple[constants.assoc_tuple_values.index(f'{node_role}_label')]
        self.iri = assoc_tuple[constants.assoc_tuple_values.index(f'{node_role}_iri')]
        self.taxon_id = assoc_tuple[constants.assoc_tuple_values.index(f'{node_role}_taxon_id')]
        self.taxon_label = assoc_tuple[constants.assoc_tuple_values.index(f'{node_role}_taxon_label')]
    
    def to_dict(self):
        """
            Convert Node object to a dictionary of values relevant to the node.
            :return: dictionary with node information
        """
        node_dict = {
            'id': self.id,
            'label': self.label,
            'iri': self.iri,
            'semantic': self.semantic_groups,
            'taxon_id': self.taxon_id,
            'taxon_label': self.taxon_label
        }
        
        return node_dict
    
class NewNode(Node):
    """
        Initialize nodes using this class, carrying information about the id
        of the entity, the semantic groups that are associated with the entity,
        and the label.
        :param id: id of entity
        :param label: label of entity
        :param iri: id in iri format of entity
        :param semantic: semantic group to which entity belongs
    """
    def __init__(self, id, label, iri, semantic):
        self.id = id
        self.label = label
        self.iri = iri
        self.semantic_groups = semantic
    
    def to_dict(self):
        """
            Convert NewNode object to a dictionary of values relevant to the
            node.
            :return: dictionary with node information
        """
        node_dict = {
            'id': self.id,
            'label': self.label,
            'iri': self.iri,
            'semantic': self.semantic_groups
        }
        
        return node_dict
    
class Edge:
    """
        Initialize edges using this class carryinh information about the association it represents.
    """
    def __eq__(self, other):
        return self.id == other.id
    
    def __hash__(self):
        return hash(self.id)
    
class AssocEdge(Edge):
    """
        Initialize edges using this class, carrying information about the id of the association,
        the ids of the subject and object and the id, label and iri link of the relation.
        :param assoc_tuple: tuple of information about association
    """
    def __init__(self, assoc_tuple: tuple):
        self.id = assoc_tuple[constants.assoc_tuple_values.index('id')]
        self.subject = assoc_tuple[constants.assoc_tuple_values.index('subject_id')]
        self.object = assoc_tuple[constants.assoc_tuple_values.index('object_id')]
        self.relation = {
            'id': assoc_tuple[constants.assoc_tuple_values.index('relation_id')],
            'iri': assoc_tuple[constants.assoc_tuple_values.index('relation_iri')],
            'label': str(assoc_tuple[constants.assoc_tuple_values.index('relation_label')]).replace('_', ' ')
        }
    
    def to_dict(self):
        """
            Convert Edge object to a dictionary of values relevant to the edge.
            :return: dictionary with edge information
        """
        edge_dict = {
            'id': self.id,
            'subject': self.subject,
            'object': self.object,
            'relation_id': self.relation['id'],
            'relation_label': self.relation['label'],
            'relation_iri': self.relation['iri']
        }
        
        return edge_dict
    
class NewEdge(Edge):
    def __init__(self, id, subject, object, relation_id, relation_label, relation_iri):
        self.id = id
        self.subject = subject
        self.object = object
        self.relation = {
            'id': relation_id,
            'iri': relation_iri,
            'label': relation_label
        }
    
    def to_dict(self):
        """
            Convert NewEdge object to a dictionary of values relevant to the edge.
            :return: dictionary with edge information
        """
        edge_dict = {
            'id': self.id,
            'subject': self.subject,
            'object': self.object,
            'relation_id': self.relation['id'],
            'relation_label': self.relation['label'],
            'relation_iri': self.relation['iri']
        }
        
        return edge_dict
    
class KnowledgeGraph:
    """
        Initialize a knowledge graph that consists of a set of edges and a set of nodes.
    """
    def __init__(self):
        self.all_edges = set()  # all unique edges in knowledge graph
        self.all_nodes = set()  # all unique nodes in knowledge graph
        
    def generate_dataframes(self):
        """
            Generate dataframes that contain all edges and nodes.
            :return: dataframe with all edges and dataframe with all nodes
        """
        edges = pd.DataFrame.from_records([edge.to_dict() for edge in self.all_edges])
        nodes = pd.DataFrame.from_records([node.to_dict() for node in self.all_nodes])
        return edges, nodes
    
    def save_graph(self, foldername, filename_prefix):
        edges, nodes = self.generate_dataframes()
        
        edges_file_name = f'{filename_prefix}_edges.csv'
        edges.to_csv(f'{constants.OUTPUT_FOLDER}/{foldername}/{edges_file_name}', index=False)
        
        nodes_file_name = f'{filename_prefix}_nodes.csv'
        nodes.to_csv(f'{constants.OUTPUT_FOLDER}/{foldername}/{nodes_file_name}', index=False)
        
        print(f'Knowledge graph content saved into files {edges_file_name} and {nodes_file_name} in the {constants.OUTPUT_FOLDER}/{foldername} folder.')
        
    def analyze_graph(self):
        """
            Analyze the current graph by looking at all semantic groups found in the nodes as well as how many nodes and edges
            are contained in the graph.
        """
        # Find all semantic groups
        all_semantic_groups = set()
        for node in self.all_nodes:
            all_semantic_groups.add(node.semantic_groups)
        register_info(f'The graph contains {len(all_semantic_groups)} different semantic groups: {all_semantic_groups}')
        
        # Show total number of edges and nodes
        register_info(f'For the graph, a total of {len(self.all_edges)} edges and {len(self.all_nodes)} nodes have been generated.')
        
    def find_relation_labels(self, substring_relation_label):
        """ 
            Find all relations that have a label that contains the given substring. 
            :param substring_relation_label: substring that needs to be contained by the relation label
            :return: dictionary of relations where key represents id of relation and values labels of relations
        """
        found_relations = {}
        
        for edge in self.all_edges:
            relation_id = edge.relation['id']
            relation_label = edge.relation['label']
            if (relation_label and substring_relation_label in relation_label):
                found_relations[relation_id] = relation_label
        register_info(f'All {len(found_relations)} relations with substring "{substring_relation_label}":\n {found_relations}')
        
        return found_relations
            
    def get_extracted_nodes(self, extract_semantic_groups: list):
        """
            Get all nodes that belong to at least one of the given semantic group(s).
            :param extract_semantic_groups: list of semantic group names
            :return Dataframe containing extracted nodes
        """
        extracted_nodes = set()
        
        for node in self.all_nodes:
            if node.semantic_groups in extract_semantic_groups or len(extract_semantic_groups) == 0:
                extracted_nodes.add(node)
        register_info(f'Extracted a total of {len(extracted_nodes)} nodes that belong to at least one of the semantic groups {extract_semantic_groups}')

        return pd.DataFrame.from_records([node.to_dict() for node in extracted_nodes])

class AssocKnowledgeGraph(KnowledgeGraph):
    """
        Initialize a knowledge graph by giving a list of associations that is converted into
        a set of Edge objects and Node objects. 
        :param all_associations: list of tuples complying with `constants.assoc_tuple_values`, by default an empty list
    """
    def __init__(self, all_associations: list = []):
        KnowledgeGraph.__init__(self)
        
        self.add_edges_and_nodes(all_associations)
        self.analyze_graph()
    
    def add_edges_and_nodes(self, associations: list):
        """
            Add new edges and nodes to the graph given a list of association dictionaries.
            :param associations: list of association dictionaries
        """
        for association in associations:
            self.all_edges.add(AssocEdge(association))
            self.all_nodes.add(AssocNode(association, 'subject'))
            self.all_nodes.add(AssocNode(association, 'object'))
    
class RestructuredKnowledgeGraph(KnowledgeGraph):
    """
        Initialize a knowledge graph restructuring by giving a KnowledgeGraph entity
        containing a set of Edge and Node objects.
        :param prev_kg: KnowledgeGraph instance
    """
    def __init__(self, prev_kg: AssocKnowledgeGraph):
        KnowledgeGraph.__init__(self)
            
        self.restructure_kg(prev_kg)
        
    def add_edge(self, edge: NewEdge):
        """
            Add a NewEdge object to set of edges.
        """
        if edge:
            self.all_edges.add(edge)
    
    def add_node(self, node: NewNode):
        """
            Add a NewNode object to set of nodes.
        """
        if node:
            self.all_nodes.add(node)
            
    def remove_node(self, node: AssocNode):
        """
            Remove node when it belongs to one of the excluded semantic groups.
        """
        if node.semantic_groups == constants.ANAT or node.semantic_groups == constants.ANAT2:
            return True
        else:
            return False
    
    def edge_is_empty(self, prev_edge: AssocEdge):
        """
            Check whether the edge has an empty relation.
        """
        if pd.isnull(prev_edge.relation['id']):
            return True
        else:
            return False
            
    def remove_edge(self, prev_edge: AssocEdge, prev_nodes_df: pd.DataFrame):
        """
            Check whether the edge needs to removed due to one or both nodes needed to be removed as well.
        """
        subject_id = prev_edge.subject
        subject_semantic = prev_nodes_df.loc[prev_nodes_df['id'] == subject_id, 'semantic'].iloc[0]
        object_id = prev_edge.object
        object_semantic = prev_nodes_df.loc[prev_nodes_df['id'] == object_id, 'semantic'].iloc[0]
        
        if subject_semantic == constants.ANAT or object_semantic == constants.ANAT:
            return True
        elif subject_semantic == constants.ANAT2 or object_semantic == constants.ANAT2:
            return True
        else: 
            return False
        
    def remove_incorrect_edge(self, prev_edge: AssocEdge, prev_nodes_df: pd.DataFrame):
        """
            Check whether the edge needs to removed due to model inconsistencies.
        """
        subject_id = prev_edge.subject
        subject_semantic = prev_nodes_df.loc[prev_nodes_df['id'] == subject_id, 'semantic'].iloc[0]
        object_id = prev_edge.object
        object_semantic = prev_nodes_df.loc[prev_nodes_df['id'] == object_id, 'semantic'].iloc[0]

        if subject_semantic == constants.GENE and object_semantic == constants.GENE and prev_edge.relation['id'] == constants.EXPRESSES_GENE['id']:
            return True
        else:
            return False
        
    def deduce_edge(self, prev_edge: AssocEdge, prev_nodes_df: pd.DataFrame):
        """
            Deduce from the object and subject semantic groups of the edge, the relation.
        """
        subject_id = prev_edge.subject
        subject_semantic = prev_nodes_df.loc[prev_nodes_df['id'] == subject_id, 'semantic'].iloc[0]
        object_id = prev_edge.object
        object_semantic = prev_nodes_df.loc[prev_nodes_df['id'] == object_id, 'semantic'].iloc[0]
        
        if subject_semantic == constants.VAR and object_semantic in [constants.GENOTYPE, constants.MODEL]:
            new_relation = constants.IS_VARIANT_IN
            return NewEdge(prev_edge.id, prev_edge.subject, prev_edge.object, new_relation['id'], new_relation['label'], new_relation['iri'])
        elif subject_semantic == constants.CHEMICAL and object_semantic == constants.DISEASE:
            new_relation = constants.TREATS
            return NewEdge(prev_edge.id, prev_edge.subject, prev_edge.object, new_relation['id'], new_relation['label'], new_relation['iri'])
        elif subject_semantic in [constants.GENOTYPE, constants.MODEL] and object_semantic == constants.GENE:
            new_relation = constants.EXPRESSES_GENE
            return NewEdge(prev_edge.id, prev_edge.subject, prev_edge.object, new_relation['id'], new_relation['label'], new_relation['iri'])
        elif subject_semantic == constants.MODEL and object_semantic == constants.GENOTYPE:
            new_relation = constants.HAS_GENOTYPE
            return NewEdge(prev_edge.id, prev_edge.subject, prev_edge.object, new_relation['id'], new_relation['label'], new_relation['iri'])
        else:   
            print(f'Ignore edge with subject concept {subject_semantic} and object concept {object_semantic}')
            return None
    
    def rename_edge(self, edge: NewEdge, prev_nodes_df: pd.DataFrame):
        """
            Replace the relation of an edge with another relation.
        """
        subject_id = edge.subject
        subject_semantic = prev_nodes_df.loc[prev_nodes_df['id'] == subject_id, 'semantic'].iloc[0]
        object_id = edge.object
        object_semantic = prev_nodes_df.loc[prev_nodes_df['id'] == object_id, 'semantic'].iloc[0]
        
        if subject_semantic == constants.DISEASE and object_semantic == constants.PHENOTYPE:
            new_relation = constants.PHENOTYPE_ASSOCIATED
            return NewEdge(edge.id, edge.subject, edge.object, new_relation['id'], new_relation['label'], new_relation['iri'])
        
        elif subject_semantic == constants.GENE and object_semantic in [constants.FUNCTION, constants.FUNCTION2]:
            new_relation = constants.ENABLES
            return NewEdge(edge.id, edge.subject, edge.object, new_relation['id'], new_relation['label'], new_relation['iri'])
        
        elif subject_semantic == constants.MODEL and object_semantic == constants.GENE:
            new_relation = constants.EXPRESSES_GENE
            return NewEdge(edge.id, edge.subject, edge.object, new_relation['id'], new_relation['label'], new_relation['iri'])
        
        return edge
    
    def rename_edge_after(self, edge: NewEdge, prev_nodes_df: pd.DataFrame):
        """
            After changing the nodes, decide whether the edge still needs to change.
        """
        subject_id = edge.subject
        subject_semantic = prev_nodes_df.loc[prev_nodes_df['id'] == subject_id, 'semantic'].iloc[0]
        object_id = edge.object
        object_semantic = prev_nodes_df.loc[prev_nodes_df['id'] == object_id, 'semantic'].iloc[0]

        if subject_semantic == constants.GENE and object_semantic == constants.FUNCTION:
            new_relation = constants.ENABLES
            return NewEdge(edge.id, edge.subject, edge.object, new_relation['id'], new_relation['label'], new_relation['iri'])
        
        return edge
        
    def group_edge(self, prev_edge: AssocEdge):
        """
            Replace the relation of an edge with another relation due to grouping of relations.
        """
        if prev_edge.relation['id'] in constants.REL_GROUPING:
            new_relation = constants.REL_GROUPING[prev_edge.relation['id']]
            grouped_edge = NewEdge(prev_edge.id, prev_edge.subject, prev_edge.object, new_relation['id'], new_relation['label'], new_relation['iri'])
            return grouped_edge
        else:
            edge = NewEdge(prev_edge.id, prev_edge.subject, prev_edge.object, prev_edge.relation['id'], prev_edge.relation['label'], prev_edge.relation['iri'])
            return edge
    
    def get_node_associations(self, node_id, edges_df: pd.DataFrame):
        """
            Get all relations that are found at least once in an edge connected to the given node.
        """
        # Create heatmap for each semantic group, each column being a relation, count occurrence of each relation, ratio with total of entity of that semantic group
        associated_rows_df = edges_df.loc[(edges_df['subject'] == node_id) | (edges_df['object'] == node_id)]
        relations = associated_rows_df['relation_id'].unique().tolist()
        
        return relations
    
    def get_to_node_associations(self, node_id, edges_df: pd.DataFrame):
        """ 
            Get all relations that are found at least one in an edge directed towards the given node.
        """
        associated_rows_df = edges_df.loc[edges_df['object'] == node_id]
        relations = associated_rows_df['relation_id'].unique().tolist()
        
        return relations
    
    def get_from_node_associations(self, node_id, edges_df: pd.DataFrame):
        """ 
            Get all relations that are found at least one in an edge directed from the given node.
        """
        associated_rows_df = edges_df.loc[edges_df['subject'] == node_id]
        relations = associated_rows_df['relation_id'].unique().tolist()
        
        return relations
        
    def transform_node_semantic(self, node: AssocNode, all_relations: list, all_relations_to_node: list = [], all_relations_from_node: list = []):
        """
            Change the semantic group of the node based on its previous semantic group or its associated relations.
        """
        if node.semantic_groups in [constants.MODEL, constants.MARKER, constants.HOMOLOGY, constants.INTERACTION]:
            if any(i in [constants.INTERACTS_WITH['id'], constants.COLOCALIZES_WITH['id'], constants.IN_ORTH_REL_WITH['id']] for i in all_relations):
                return constants.GENE
            elif any(i in [constants.ENABLES['id'], constants.IS_PART_OF['id']] for i in all_relations_from_node):
                return constants.GENE
            elif constants.IS_VARIANT_IN['id'] in all_relations_to_node:
                return constants.GENOTYPE
            elif constants.HAS_AFFECTED_FEATURE['id'] in all_relations_to_node:
                return constants.GENE
            else:
                return constants.BIOLART
            
        if node.semantic_groups == constants.PATHWAY:
            return constants.BIOLPRO
        
        if node.semantic_groups == constants.CHEMICAL:
            return constants.DRUG
        
        if node.semantic_groups == constants.FUNCTION2:
            if constants.IS_PART_OF['id'] in all_relations_to_node and constants.ENABLES['id'] not in all_relations_to_node:
                return constants.CELLULAR_COMPONENT
            else:
                return constants.FUNCTION
        
        return node.semantic_groups
        
    def add_concept_taxon(self, node: AssocNode):
        """
            Add nodes as entities of semantic group TAXON.
        """
        if not pd.isnull(node.taxon_id) and node.semantic_groups in [constants.GENE, constants.BIOLART]:
            gene_node = node
            taxon_node = NewNode(id=gene_node.taxon_id, label=gene_node.taxon_label, iri=np.nan, semantic=constants.TAXON)
            
            if node.semantic_groups == constants.GENE:
                taxon_edge_id = common.generate_edge_id(constants.FOUND_IN['id'], gene_node.id, taxon_node.id)
                taxon_edge = NewEdge(taxon_edge_id, gene_node.id, taxon_node.id, constants.FOUND_IN['id'], constants.FOUND_IN['label'], constants.FOUND_IN['iri'])
            elif node.semantic_groups == constants.BIOLART:
                taxon_edge_id = common.generate_edge_id(constants.IS_OF['id'], gene_node.id, taxon_node.id)
                taxon_edge = NewEdge(taxon_edge_id, gene_node.id, taxon_node.id, constants.IS_OF['id'], constants.IS_OF['label'], constants.IS_OF['iri'])
            else:
                taxon_edge = None
            
            self.add_node(taxon_node)
            self.add_edge(taxon_edge)
            
    def restructure_kg(self, prev_kg: AssocKnowledgeGraph):
        """
            Restructure the knowledge graph by iterating over all edges and nodes of the given graph.
        """
        _, prev_nodes_df = prev_kg.generate_dataframes()
        
        print('Iterating over edges (grouping, deducing, renaming) ...')
        for prev_edge in tqdm(prev_kg.all_edges):
            if self.edge_is_empty(prev_edge):
                new_edge = self.deduce_edge(prev_edge, prev_nodes_df)
                self.add_edge(new_edge)
            elif not self.remove_edge(prev_edge, prev_nodes_df):
                new_edge = self.group_edge(prev_edge)
                new_edge = self.rename_edge(new_edge, prev_nodes_df)
                
                self.add_edge(new_edge)
            
        edges_df, _ = self.generate_dataframes()
        
        print('Iterating over nodes (transforming, adding) ...')
        for node in tqdm(prev_kg.all_nodes):
            if not self.remove_node(node):
                # Get all relations it is associated with
                node_relations = self.get_node_associations(node.id, edges_df)
                to_node_relations = self.get_to_node_associations(node.id, edges_df)
                from_node_relations = self.get_from_node_associations(node.id, edges_df)
                
                node.semantic_groups = self.transform_node_semantic(node, node_relations, to_node_relations, from_node_relations)
                
                # Add TAXON nodes
                self.add_concept_taxon(node)
                
                new_node = NewNode(node.id, node.label, node.iri, node.semantic_groups)
                self.add_node(new_node)

        _, nodes_df = self.generate_dataframes()

        print('Iterating over new edges to remove triples that are inconsistent with model...')
        edges_removed = 0
        all_new_edges = deepcopy(self.all_edges)
        for edge in tqdm(all_new_edges):
            if edge in self.all_edges:
                self.all_edges.remove(edge)

                if not self.remove_incorrect_edge(prev_edge=edge, prev_nodes_df=nodes_df):
                    new_edge = self.rename_edge_after(edge, nodes_df)
                    self.add_edge(new_edge)

                    edges_removed += 1

        print(f'A total of {edges_removed} edges are removed due to model inconsistencies.')
