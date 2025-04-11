import pandas as pd

def load_data(file_path):
    """
    Loads dataset from a CSV file and returns a pandas DataFrame.
    """
    
    df = pd.read_csv(file_path)
    return df
