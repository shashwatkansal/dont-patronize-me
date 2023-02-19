import pandas as pd

'''
Read the raw TSV data file, skipping the first 3 rows of disclaimer, adding in missing column headers
'''
def read_raw_datafile(file_meta: dict) -> pd.DataFrame:
    df = pd.read_csv(file_meta['filepath'], sep='\t', skiprows=3, header=None)
    df.columns = file_meta['columns']
    return df

'''
Read the CSV data indices split file
'''
def read_split(file_meta: dict) -> pd.DataFrame:
    return pd.read_csv(file_meta['filepath'])

'''
Print a dataframe to a TSV file
'''
def write_dataframe(file_meta: dict, dataframe: pd.DataFrame)->None:
    print("Writing to TSV:", file_meta['filepath'])
    dataframe.to_csv(file_meta['filepath'], sep='\t', index=False)
    
'''
Read a dataframe from a TSV file
'''
def read_datafile(file_meta: dict) -> pd.DataFrame:
    return pd.read_csv(file_meta['filepath'], sep='\t')