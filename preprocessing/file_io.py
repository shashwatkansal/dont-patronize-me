import pandas as pd

from .file_metadata import FileMeta


def read_raw_datafile(file_meta: FileMeta, skiprows=3) -> pd.DataFrame:
    """Read the raw TSV data file, skipping the first 3 rows of disclaimers, adding in missing column headers."""

    df = pd.read_csv(file_meta.filepath, sep="\t", skiprows=skiprows, header=None)

    assert file_meta.columns is not None
    df.columns = file_meta.columns

    return df


def read_datafile(file_meta: FileMeta) -> pd.DataFrame:
    """Read a dataframe from a TSV file."""

    return pd.read_csv(file_meta.filepath, sep="\t")


def read_split(file_meta: FileMeta) -> pd.DataFrame:
    """Read the CSV data indices split file."""

    return pd.read_csv(file_meta.filepath)


def write_dataframe(dataframe: pd.DataFrame, file_meta: FileMeta) -> None:
    """Output a dataframe to a TSV file."""

    dataframe.to_csv(file_meta.filepath, sep="\t", index=False)
