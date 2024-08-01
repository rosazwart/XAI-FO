import logging
logging.basicConfig(level=logging.INFO, filename='datafetcher.log', filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")

import pandas as pd
import hashlib

from util.constants import assoc_tuple_values

def register_info(message):
    """
        Print message into console as well as given logger.
        :param current_logger: logger with configuration
        :param message: message that needs to be printed and logged
    """
    print(message)
    logging.info(message)
    
def register_error(message):
    """
        Log given error.
        :param message: message that needs to be printed and logged
    """
    print(message)
    logging.error(message)
    
def tuplelist2dataframe(tuple_list: list, column_values: tuple = assoc_tuple_values):
    """

    """
    df = pd.DataFrame.from_records(tuple_list, columns=list(column_values))
    register_info(f'Created a dataframe with {df.shape[0]} entries and column values {df.columns.values}')
    return df

def dataframe2tuplelist(df: pd.DataFrame):
    """

    """
    tuple_list = list(df.itertuples(index=False, name=None))
    register_info(f'Created a list of tuples with {len(tuple_list)} entries')
    return tuple_list

def extract_colvalues(df: pd.DataFrame, extract_colname: str):
    """
        Extract values from column with given name.
        :param df: Dataframe from which column values need to be extracted
        :param extract_colname: Name of column that needs to be extracted
        :return List of extracted column values
    """
    colvalues = df[extract_colname].to_list()
    return colvalues

def generate_edge_id(relation_id, subject_id, object_id):
    strings_tuple = (relation_id, subject_id, object_id)
    hasher = hashlib.md5()
    for string_value in strings_tuple:
        hasher.update(string_value.encode())
    return hasher.hexdigest()