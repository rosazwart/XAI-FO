from data_loader import get_all_assoc_df
from association_getter import get_monarch_associations


if __name__ == "__main__":
    all_assoc_df = get_all_assoc_df(dir_name='localfetcher/deprecated_data')
    
    disease_prefix = input('For which disease will the knowledge graph be built? Choose from "dmd", "hd" and "oi"')
    assert disease_prefix == 'dmd' or 'hd' or 'oi'
    
    if disease_prefix == 'hd':
        seeds = [
            'MONDO:0007739',    # Huntington disease
            'HGNC:4851' # HTT, causal gene Huntington disease
        ]
        
    elif disease_prefix == 'dmd':
        seeds = [
            'MONDO:0010679',    # Duchenne muscular dystrophy disease
            'HGNC:2928'         # DMD, causal gene
        ]

    elif disease_prefix == 'oi':
        seeds = [
            'MONDO:0019019',    # Osteogenesis imperfecta
            'HGNC:2197',        # COL1A1, causal gene
            'HGNC:2198'         # COL1A2, causal gene
    ]
        
    else:
        
        seeds = []

    all_assoc = get_monarch_associations(assoc_df=all_assoc_df, nodes_list=seeds,
                                         disease_file_name_ref=disease_prefix)