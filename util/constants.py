import numpy as np

# Dependent on BioLink API responses defined at https://api.monarchinitiative.org/api/
# `_` indicates dictionary key level separation for example `dict['subject']['id']`
assoc_tuple_values = ('id', 
                      'subject_id', 'subject_label', 'subject_iri', 'subject_category', 'subject_taxon_id', 'subject_taxon_label',
                      'object_id', 'object_label', 'object_iri', 'object_category', 'object_taxon_id', 'object_taxon_label', 
                      'relation_id', 'relation_label', 'relation_iri')

GENOTYPE = 'genotype'
GENE = 'gene'
TAXON = 'taxon'
DRUG = 'drug'
DISEASE = 'disease'
PHENOTYPE = 'phenotype'
MODEL = 'model'
BIOLART = 'biological artifact'
ANAT = 'anatomical entity'
ANAT2 = 'anatomy'
VAR = 'variant'
PATHWAY = 'pathway'
BIOLPRO = 'biological process'
CHEMICAL = 'chemical'
GENE_PRODUCT = 'gene product'
MARKER = 'marker'
HOMOLOGY = 'homology'
INTERACTION = 'interaction'
FUNCTION = 'molecular function'
FUNCTION2 = 'function'
CELLULAR_COMPONENT = 'cellular component'

FOUND_IN = {
    'id': 'CustomRO:foundin',
    'label': 'found in',
    'iri': np.nan
}

IS_OF = {
    'id': 'CustomRO:isof',
    'label': 'is of',
    'iri': np.nan
}

IS_VARIANT_IN = {
    'id': 'CustomRO:isvariantin',
    'label': 'is variant in',
    'iri': np.nan
}

TARGETS = {
    'id': 'CustomRO:TTD2',
    'label': 'targets',
    'iri': np.nan
}

IS_PRODUCT_OF = {
    'id': 'CustomRO:TTD1',
    'label': 'is product of',
    'iri': np.nan
}

TREATS = {
    'id': 'CustomRO:DC',
    'label': 'is substance that treats',
    'iri': np.nan
}

EXPRESSES_GENE = {
    'id': 'CustomRO:expressesgene',
    'label': 'expresses gene',
    'iri': np.nan
}

PHENOTYPE_ASSOCIATED = {
    'id': 'CustomRO:associatedphenotype',
    'label': 'associated with phenotype',
    'iri': np.nan
}

INTERACTS_WITH = {
    'id': 'RO:0002434',
    'label': 'interacts with',
    'iri': 'http://purl.obolibrary.org/obo/RO_0002434'
}

ENABLES = {
    'id': 'RO:0002327',
    'label': 'enables',
    'iri': 'http://purl.obolibrary.org/obo/RO_0002327'
}

IS_PART_OF = {
    'id': 'BFO:0000050',
    'label': 'is part of',
    'iri': np.nan
}

HAS_AFFECTED_FEATURE = {
    'id': 'GENO:0000418',
    'label': 'has affected feature',
    'iri': 'http://purl.obolibrary.org/obo/GENO_0000418'
}

HAS_GENOTYPE = {
    'id': 'GENO:0000222',
    'label': 'has genotype',
    'iri': 'http://purl.obolibrary.org/obo/GENO_0000222'
}

COLOCALIZES_WITH = {
    'id': 'RO:0002325',
    'label': 'colocalizes with',
    'iri': 'http://purl.obolibrary.org/obo/RO_0002325'
}

IN_ORTH_REL_WITH = {
    'id': 'RO:HOM0000017',
    'label': 'in orthology relationship with',
    'iri': 'http://purl.obolibrary.org/obo/RO_HOM0000017'
}

REL_GROUPING = {
    'RO:HOM0000020': {
        'id': 'RO:HOM0000017',
        'label': 'in orthology relationship with',
        'iri': 'http://purl.obolibrary.org/obo/RO_HOM0000017'
    },
    'RO:0004016': {
        'id': 'RO:0003304',
        'label': 'contributes to condition',
        'iri': 'http://purl.obolibrary.org/obo/RO_0003304'
    },
    'RO:0002607': {
        'id': 'RO:0003304',
        'label': 'contributes to condition',
        'iri': 'http://purl.obolibrary.org/obo/RO_0003304'
    },
    'RO:0002326': {
        'id': 'RO:0003304',
        'label': 'contributes to condition',
        'iri': 'http://purl.obolibrary.org/obo/RO_0003304'
    },
    'GENO:0000841': {
        'id': 'CustomRO:likelycauses',
        'label': 'likely causes condition',
        'iri': np.nan
    },
    'RO:0004012': {
        'id': 'RO:0003303',
        'label': 'causes condition',
        'iri': 'http://purl.obolibrary.org/obo/RO_0003303'
    },
    'RO:0004013': {
        'id': 'RO:0003303',
        'label': 'causes condition',
        'iri': 'http://purl.obolibrary.org/obo/RO_0003303'
    },
    'GENO:0000840': {
        'id': 'RO:0003303',
        'label': 'causes condition',
        'iri': 'http://purl.obolibrary.org/obo/RO_0003303'
    },
    'RO:0002200': {
        'id': 'RO:0003303',
        'label': 'causes condition',
        'iri': 'http://purl.obolibrary.org/obo/RO_0003303'
    }
}

INPUT_FOLDER = 'data'
OUTPUT_FOLDER = 'output'
