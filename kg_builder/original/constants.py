def prefix2category(prefix: str):
    """
    """
    prefix = prefix.lower()

    if 'variant' in prefix:
        return 'VARI'
    elif 'phenotype' in prefix or 'mondo' in prefix or 'omim' in prefix or 'doid' in prefix or 'hp' in prefix or 'mp' in prefix or 'fbcv' in prefix or 'fbbt' in prefix or 'zp' in prefix or 'apo' in prefix or 'trait' in prefix:
        return 'DISO'
    elif 'gene' in prefix or 'hgnc' in prefix:
        return 'GENE'
    elif 'mgi' in prefix or 'flybase' in prefix or 'wormbase' in prefix or 'zfin' in prefix or 'xenbase' in prefix or 'rgd' in prefix or 'sgd' in prefix or 'ensembl' in prefix: 
        return 'ORTH'
    elif 'react' in prefix or 'kegg-path' in prefix or 'go' in prefix:
        return 'PHYS'
    elif 'uberon' in prefix or 'cl' in prefix:
        return 'ANAT'
    elif 'mesh' in prefix: 
        return 'DRUG'
    elif 'geno' in prefix or 'coriell' in prefix or 'monarch' in prefix or 'mmrrc' in prefix or '' in prefix or 'bnode' in prefix:
        return 'GENO'
    else:
        return 'CONC'