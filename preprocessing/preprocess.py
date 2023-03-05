import pandas as pd

from preprocessing.file_io import read_raw_datafile, read_split, write_dataframe
from preprocessing.file_metadata import (
    CATEGORIES_DATA_FILE_META,
    OFFICIAL_DEV_PCL_DATA_FILE_META,
    OFFICIAL_DEV_SPLIT_FILE_META,
    PCL_DATA_FILE_META,
    SPLIT_FILE_ANNOTATOR_COLUMN_NAMES,
    TRAIN_SPLIT_FILE_META,
    TRAINING_PCL_DATA_FILE_META,
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
    
def expand_split_data_columns(split_df: pd.DataFrame):
    new_df = pd.DataFrame(split_df['label'].to_list(), SPLIT_FILE_ANNOTATOR_COLUMN_NAMES);
    print(new_df)

def split_pcl_data(
    pcl_df: pd.DataFrame,
    train_split_df: pd.DataFrame,
    official_dev_split_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    official_dev_data_df = pcl_df.merge(official_dev_split_df, on="par_id")
    official_dev_data_df = official_dev_data_df.rename(
        columns={"label_x": "class_labels", "label_y": "annotator_label"}
    )

    official_training_data_df = pcl_df.merge(train_split_df, on="par_id")
    official_training_data_df = official_training_data_df.rename(
        columns={"label_x": "class_labels", "label_y": "annotator_label"}
    )

    return official_dev_data_df, official_training_data_df


def main() -> None:
    # Read raw data files and validate them.
    pcl_df = read_raw_datafile(PCL_DATA_FILE_META)
    categories_df = read_raw_datafile(CATEGORIES_DATA_FILE_META)
    train_split_df = read_split(TRAIN_SPLIT_FILE_META)
    official_dev_split_df = read_split(OFFICIAL_DEV_SPLIT_FILE_META)
    validate_raw_data(pcl_df, categories_df, train_split_df, official_dev_split_df)
    
    expand_split_data_columns(train_split_df)
    # Split the pcl data according to the splits.
    official_dev_data_df, official_training_data_df = split_pcl_data(pcl_df, train_split_df, official_dev_split_df)

    # Sort the processed pcl data by its `par_id`.
    official_dev_data_df = official_dev_data_df.sort_values(by="par_id")
    official_training_data_df = official_training_data_df.sort_values(by="par_id")

    # Write the resulting dataframes.
    write_dataframe(official_dev_data_df, OFFICIAL_DEV_PCL_DATA_FILE_META)
    write_dataframe(official_training_data_df, TRAINING_PCL_DATA_FILE_META)


if __name__ == "__main__":
    main()
