import pandas as pd
import ast

from preprocessing.file_io import read_raw_datafile, read_split, write_dataframe
from preprocessing.file_metadata import (
    CATEGORIES_DATA_FILE_META,
    OFFICIAL_DEV_PCL_DATA_FILE_META,
    OFFICIAL_DEV_SPLIT_FILE_META,
    PCL_DATA_FILE_META,
    SPLIT_FILE_ANNOTATOR_COLUMN_NAMES,
    TASK4_PROCESSED_DATA_FILE_META,
    TASK4_RAW_DATA_FILE_META,
    TRAIN_SPLIT_FILE_META,
    TRAINING_PCL_DATA_FILE_META, FileMeta,
)


def validate_raw_data(
        pcl_df: pd.DataFrame,
        categories_df: pd.DataFrame,
        train_split_df: pd.DataFrame,
        official_dev_split_df: pd.DataFrame,
) -> None:
    print("Checking for duplicate `par_id`s in the pcl and split datasets...", end="")
    for df in (pcl_df, train_split_df, official_dev_split_df):
        if df["par_id"].duplicated().any():
            raise ValueError("Duplicates found!")
    print("Success")

    print("Checking whether train and dev splits are disjoint...", end="")
    train_par_ids: set[int] = set(train_split_df["par_id"])
    official_dev_par_ids: set[int] = set(official_dev_split_df["par_id"])
    if train_par_ids & official_dev_par_ids:
        raise ValueError("Train and Dev split are not disjoint!")
    print("Success")

    print("Checking the sum of split size is equal to the size of dataset...", end="")
    pcl_par_ids: set[int] = set(pcl_df["par_id"])
    if (train_par_ids | official_dev_par_ids) ^ pcl_par_ids:
        train_split_len = len(train_par_ids)
        official_dev_split_len = len(official_dev_par_ids)
        pcl_par_len = len(pcl_par_ids)

        raise ValueError(f"Sum of split size ({train_split_len=} + {official_dev_split_len=}) != {pcl_par_len=}")
    print("Success")

    # TODO: Add validation for Category data once we know if we care about it.


def read_expand_split_data_columns(file_meta: FileMeta):
    split_df = read_split(file_meta)
    # Literally evaluate the array string to an list
    label_as_array = [ast.literal_eval(x) for x in split_df['label']]
    # Split the array into booleans for each type of PCL
    expanded_df = pd.DataFrame(label_as_array, columns=SPLIT_FILE_ANNOTATOR_COLUMN_NAMES).astype(bool)
    # Add the paragraph id column back
    expanded_df['par_id'] = split_df['par_id'];
    return expanded_df


def split_pcl_data(
        pcl_df: pd.DataFrame,
        train_split_df: pd.DataFrame,
        official_dev_split_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    official_dev_data_df = pcl_df.merge(official_dev_split_df, on="par_id")
    official_training_data_df = pcl_df.merge(train_split_df, on="par_id")

    return official_dev_data_df, official_training_data_df


def main() -> None:
    # Read raw data files and validate them.
    pcl_df = read_raw_datafile(PCL_DATA_FILE_META)
    categories_df = read_raw_datafile(CATEGORIES_DATA_FILE_META)
    task4_df = read_raw_datafile(TASK4_RAW_DATA_FILE_META, skiprows=0)

    train_split_df = read_expand_split_data_columns(TRAIN_SPLIT_FILE_META)
    official_dev_split_df = read_expand_split_data_columns(OFFICIAL_DEV_SPLIT_FILE_META)
    validate_raw_data(pcl_df, categories_df, train_split_df, official_dev_split_df)

    # Split the pcl data according to the splits.
    official_dev_data_df, official_training_data_df = split_pcl_data(pcl_df, train_split_df, official_dev_split_df)

    # Sort the processed pcl data by its `par_id`.
    official_dev_data_df = official_dev_data_df.sort_values(by="par_id")
    official_training_data_df = official_training_data_df.sort_values(by="par_id")

    # Write the resulting dataframes.
    write_dataframe(official_dev_data_df, OFFICIAL_DEV_PCL_DATA_FILE_META)
    write_dataframe(official_training_data_df, TRAINING_PCL_DATA_FILE_META)
    write_dataframe(task4_df, TASK4_PROCESSED_DATA_FILE_META)


if __name__ == "__main__":
    main()
