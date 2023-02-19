import pandas as pd

'''
Read the TSV data file, skipping the first 4 rows of disclaimer, adding in missing column headers
'''
def read_datafile(file_meta: dict) -> pd.DataFrame:
    df = pd.read_csv(file_meta['filepath'], sep='\t', skiprows=3, header=None)
    df.columns = file_meta['columns']
    return df

'''
Read the CSV data file
'''
def read_split(file_meta: dict) -> pd.DataFrame:
    df = pd.read_csv(file_meta['filepath'])
    return df