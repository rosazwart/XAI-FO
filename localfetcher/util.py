import pandas as pd
import hashlib

import constants as const

def dataframe2tuplelist(df: pd.DataFrame):
    """
        Convert dataframe to list of tuples.
    """
    tuple_list = list(df.itertuples(index=False, name=None))
    print(f'Created a list of tuples with {len(tuple_list)} entries')
    return tuple_list

def tuplelist2dataframe(tuple_list: list):
    """
        Convert list of tuples to a dataframe.
    """

    df = pd.DataFrame.from_records(tuple_list, columns=list(const.assoc_tuple_values))
    print(f'Created a dataframe with {df.shape[0]} entries and column values {df.columns.values}')
    return df

def generate_edge_id(relation_id: str, subject_id: str, object_id: str):
    """
        Generate ID given relation, subject and object.
    """
    strings_tuple = (relation_id, subject_id, object_id)
    hasher = hashlib.md5()
    for string_value in strings_tuple:
        hasher.update(string_value.encode())
    return hasher.hexdigest()