import os
import sys

import ttd.fetcher as ttd_fetcher
import drugcentral.fetcher as drugcentral_fetcher

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import util.constants as constants
from util.loaders import load_associations_from_csv, create_folder, OUTPUT_FOLDER
from util.kg_builder import AssocKnowledgeGraph, RestructuredKnowledgeGraph

import util.kg_analyzer as kg_analyzer

def analyze_kg(kg: AssocKnowledgeGraph, concepts_filename, triples_filename):
    """
    """
    edges, nodes = kg.generate_dataframes()
    
    edge_colmapping = {
        'relations': 'relation_label',
        'relationids': 'relation_id',
        'subject': 'subject',
        'object': 'object'
    }
    
    node_colmapping = {
        'node_id': 'id',
        'semantics': 'semantic'
    }
    
    kg.analyze_graph()
    kg_analyzer.get_concepts(nodes, node_colmapping)
    kg_analyzer.get_relations(edges, edge_colmapping)
    kg_analyzer.get_connection_summary(edges, nodes, edge_colmapping, node_colmapping, concepts_filename, triples_filename, DISEASE_PREFIX)

def build_kg(disease_prefix: str):
    """
    """
    monarch_assoc = load_associations_from_csv(f'{disease_prefix}_monarch_associations.csv', foldernames=['localfetcher', OUTPUT_FOLDER])

    kg = AssocKnowledgeGraph(monarch_assoc)

    # --- Add associations from TTD ---
    
    gene_nodes = kg.get_extracted_nodes([])
    ttd_associations = ttd_fetcher.get_drugtarget_associations(gene_nodes, disease_prefix=DISEASE_PREFIX)
    
    kg.add_edges_and_nodes(ttd_associations)
    print(f'Added {len(ttd_associations)} drug-target associations')

    # --- Add associations from DrugCentral ---
    
    drug_nodes = kg.get_extracted_nodes([constants.DRUG])
    diso_pheno_nodes = kg.get_extracted_nodes([constants.DISEASE, constants.PHENOTYPE])
    drugcentral_associations = drugcentral_fetcher.get_drugdisease_associations(drug_nodes, diso_pheno_nodes, disease_prefix=DISEASE_PREFIX)
    
    kg.add_edges_and_nodes(drugcentral_associations)
    print(f'Added {len(drugcentral_associations)} drug-phenotype/disease associations')

    # Initial knowledge graph
    analyze_kg(kg, f'all_{DISEASE_PREFIX}_concepts.png', f'all_{DISEASE_PREFIX}_triples.csv')

    # Restructuring
    restr_kg = RestructuredKnowledgeGraph(kg)
    analyze_kg(restr_kg, f'restr_{DISEASE_PREFIX}_concepts.png', f'restr_{DISEASE_PREFIX}_triples.csv')

    restr_kg.save_graph(DISEASE_PREFIX, f'restr_{DISEASE_PREFIX}_kg')

if __name__ == "__main__":
    DISEASE_PREFIX = input('For which disease is the knowledge graph built? Choose from "dmd", "hd" and "oi"')
    assert DISEASE_PREFIX == 'dmd' or 'hd' or 'oi'

    create_folder([OUTPUT_FOLDER, DISEASE_PREFIX])

    build_kg(DISEASE_PREFIX)