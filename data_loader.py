#data_loader.py
"""
This module is used to read txt files and return the data in a dictionary.
The keys are the column names and the values are a list of assiciated column values.

Functions:
    read_text_file: Load a text file and return a dictionary of column names and values.

"""
import numpy as np

def read_text_file(file_name: str, skiprows: int=1, column_list:list=[])-> dict:
    """ 
    Load a text file and return a dictionary of column names and values.
    
    Args:
        file_name: name of txt file
        skiprows: number of header rows to skip
        column_list: list of column names
    """
    data = {}
    values = np.loadtxt(file_name, skiprows=skiprows,unpack=True)

    if len(column_list) != len(values):
        print(f"Different number of columns in column_list({len(column_list)}) than "+\
               f" in file ({len(values)}).")
        return

    for idx, key in enumerate(column_list):
        data[key] = values[idx]

    return data
