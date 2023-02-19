from helper.read_files import read_datafile, read_split
from helper.file_meta_data import CATEGORIES_DATA_FILE_META,OFFICIAL_DEV_SPLIT_FILE_META,PCL_DATA_FILE_META,TRAIN_SPLIT_FILE_META

def validate_raw_data():
    # Check for duplicates in each of PCL, or the split datasets
    print("Duplicate checking...", end='')
    if any([x['par_id'].duplicated().any() for x in [pcl_df, raw_training_data_split_df, official_dev_split_df]]):
        raise Exception("Duplicates found!")
    print("Success")
    
    # Check the split training and dev set are disjoint
    print("Split disjoint checking...", end='')
    if any(set(raw_training_data_split_df['par_id']).intersection(set(official_dev_split_df['par_id']))):
        raise Exception("Sets not disjoint")
    print("Success")
    
    # Check all PCL data is split
    print("Checking split size totals dataset...", end='')
    if any(set(raw_training_data_split_df['par_id']).union(set(official_dev_split_df['par_id'])).symmetric_difference(set(pcl_df['par_id']))):
        raise Exception('Not all data values in split', (len(raw_training_data_split_df['par_id']) + len(official_dev_split_df['par_id'])), "versus", len(pcl_df['par_id']))
    print("Success")

if __name__ == '__main__':
    categories_df = read_datafile(CATEGORIES_DATA_FILE_META)
    pcl_df = read_datafile(PCL_DATA_FILE_META)
    raw_training_data_split_df = read_split(TRAIN_SPLIT_FILE_META)
    official_dev_split_df = read_split(OFFICIAL_DEV_SPLIT_FILE_META)
    validate_raw_data()