import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from util.loaders import load_associations_from_csv, create_folder, OUTPUT_FOLDER
from util.kg_builder import AssocKnowledgeGraph

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
    monarch_assoc = load_associations_from_csv(f'prev_{disease_prefix}_monarch_associations.csv', foldernames=[OUTPUT_FOLDER, DISEASE_PREFIX])
    ttd_assoc = load_associations_from_csv(f'prev_{disease_prefix}_ttd_associations.csv', foldernames=[OUTPUT_FOLDER, DISEASE_PREFIX])
    drugcentral_assoc = load_associations_from_csv(f'prev_{disease_prefix}_drugcentral_associations.csv', foldernames=[OUTPUT_FOLDER, DISEASE_PREFIX])

    kg = AssocKnowledgeGraph(monarch_assoc)
    kg.add_edges_and_nodes(ttd_assoc)
    kg.add_edges_and_nodes(drugcentral_assoc)

    analyze_kg(kg, f'prev_{DISEASE_PREFIX}_concepts.png', f'prev_{DISEASE_PREFIX}_triples.csv')

    kg.save_graph(disease_prefix, f'prev_{disease_prefix}_kg')

if __name__ == "__main__":
    DISEASE_PREFIX = input('For which disease is the knowledge graph built? Choose from "dmd", "hd" and "oi"')
    assert DISEASE_PREFIX == 'dmd' or 'hd' or 'oi'

    create_folder([OUTPUT_FOLDER, DISEASE_PREFIX])

    build_kg(DISEASE_PREFIX)
