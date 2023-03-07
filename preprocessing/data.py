from functools import cache

import numpy as np
import numpy.typing as npt
import pandas as pd

from preprocessing.file_metadata import (
    OFFICIAL_DEV_PCL_DATA_FILE_META,
    TRAINING_PCL_DATA_FILE_META,
    FileMeta,
)


def _get_data(file_meta: FileMeta) -> tuple[list[str], npt.NDArray[np.int_]]:
    data_df = pd.read_csv(file_meta.filepath, sep="\t")
    data_df = data_df.loc[data_df["text"].str.len() > 0, ["text", "label"]]

    X: list[str] = data_df["text"].tolist()
    y = (data_df["label"].to_numpy(dtype=np.int_) > 2).astype(np.int_)

    return X, y


@cache
def get_training_data() -> tuple[list[str], npt.NDArray[np.int_]]:
    return _get_data(TRAINING_PCL_DATA_FILE_META)


@cache
def get_dev_test_data() -> tuple[list[str], npt.NDArray[np.int_]]:
    return _get_data(OFFICIAL_DEV_PCL_DATA_FILE_META)
