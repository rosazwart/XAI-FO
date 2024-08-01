"""
    @description: This module maps UniProt accession IDs to other databases depending on taxon. 
    @source: https://www.uniprot.org/help/id_mapping
    @author: Rosa Zwart
"""

import requests
import time

RETRIES = 3
POLLING_S_INTERVAL = 5

FROM_DB = 'UniProtKB_AC-ID'
DEFAULT_TO_DB = 'Ensembl'

db_mapper = {
    'Homo sapiens': 'HGNC',
    'Rattus norvegicus': 'RGD',
    'Mus musculus': 'MGI',
    'Drosophila melanogaster': 'FlyBase',
    'Caenorhabditis elegans': 'WormBase',
    'Danio rerio': 'ZFIN',
    'Escherichia coli': 'EnsemblGenome',
    'Xenopus tropicalis': 'Xenbase',
    'Dictyostelium discoideum': 'dictyBase',
    'Saccharomyces cerevisiae S288C': 'SGD',
    'Schizosaccharomyces pombe': 'PomBase'
}

class IdMapper:
    def __init__(self, ids_to_map: list, to_db = DEFAULT_TO_DB, from_db = FROM_DB):
        self.url = 'https://rest.uniprot.org'
        self.job_id = self.submit_id_mapping(ids_to_map, to_db, from_db)

        if self.check_job_ready():
            self.results = self.get_results()
        
    def submit_id_mapping(self, id_list, to_db, from_db):
        print(f'Map to database {to_db}')
        
        data_params = {
            'from': from_db,
            'to': to_db,
            'ids': ','.join(id_list)
        }
        
        for i in range(RETRIES):
            try:  
                response = requests.post(f'{self.url}/idmapping/run', data=data_params)
                response.raise_for_status()
                return response.json()['jobId']
            except Exception as e:
                if (i < RETRIES - 1):
                    print(f'Retrying in {POLLING_S_INTERVAL}s')
                    time.sleep(POLLING_S_INTERVAL)
                    continue
                else:
                    print(f'After all attempts, request could not be submitted due to {e}')
                    return None
                
    def check_job_ready(self):
        while self.job_id:
            try:
                response = requests.get(f'{self.url}/idmapping/status/{self.job_id}')
                response.raise_for_status()
                response_values = response.json()
                if 'jobStatus' in response_values:
                    if response_values['jobStatus'] == 'RUNNING':
                        print(f'Check again after {POLLING_S_INTERVAL}s')
                        time.sleep(POLLING_S_INTERVAL)
                    
                    elif response_values['jobStatus'] == 'FINISHED':
                        print('Job is finished')
                        return True
                    
                    else:
                        print(f'Job {self.job_id} had status {response_values["jobStatus"]}, stopped checking if job is ready.')
                        return False
                elif 'results' in response_values:
                        return True
                else:
                    return False
            except Exception as e:
                print(f'Failed to check whether job is finished due to {e}, try again after {POLLING_S_INTERVAL}s...')
                time.sleep(POLLING_S_INTERVAL)
                
    def get_results(self):
        for i in range(RETRIES):
            try:  
                response = requests.get(f'{self.url}/idmapping/stream/{self.job_id}')
                response.raise_for_status()
                return response.json()['results']
            except Exception as e:
                if (i < RETRIES - 1):
                    print(f'Retrying in {POLLING_S_INTERVAL}s')
                    time.sleep(POLLING_S_INTERVAL)
                    continue
                else:
                    print(f'After all attempts, request could not be submitted due to {e}')
                    return None