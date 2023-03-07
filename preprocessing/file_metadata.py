from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path("./data")
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_SPLITS_DIR = DATA_DIR / "splits"


@dataclass(frozen=True, kw_only=True, slots=True)
class FileMeta:
    filepath: Path
    columns: list[str] | None = None


# The file where the raw DPM PCL data is stored.
PCL_DATA_FILE_META = FileMeta(
    filepath=DATA_RAW_DIR / "dontpatronizeme_pcl.tsv",
    columns=["par_id", "art_id", "keyword", "country_code", "text", "label"],
)


# The file where the raw DPM category data is stored.
CATEGORIES_DATA_FILE_META = FileMeta(
    filepath=DATA_RAW_DIR / "dontpatronizeme_categories.tsv",
    columns=[
        "par_id",
        "art_id",
        "text",
        "keyword",
        "country_code",
        "span_start",
        "span_finish",
        "span_text",
        "pcl_category", 
        "number_of_annotators",
    ],
)

# The file where the raw DPM category data is stored.
TASK4_RAW_DATA_FILE_META = FileMeta(
    filepath=DATA_RAW_DIR / "task4_test.tsv",
    columns=[
        "uuid",
        "art_id",
        "keyword",
        "country_code",
        "span_text",
    ],
)

# The name mappings for the array 'label' in labels split CSVs
SPLIT_FILE_ANNOTATOR_COLUMN_NAMES = [
    'Unbalanced_power_relations', 
    'Shallow_solution', 
    'Presupposition', 
    'Authority_voice', 
    'Metaphors', 
    'Compassion',
    'The_poorer_the_merrier'
]


# The file where the training data split by paragraph id (par_id) is stored.
TRAIN_SPLIT_FILE_META = FileMeta(
    filepath=DATA_SPLITS_DIR / "train_semeval_parids-labels.csv",
)


# The file where the dev data split by paragraph id (par_id) is stored.
OFFICIAL_DEV_SPLIT_FILE_META = FileMeta(
    filepath=DATA_SPLITS_DIR / "dev_semeval_parids-labels.csv",
)
 

# The file where the dev pcl data is stored.
OFFICIAL_DEV_PCL_DATA_FILE_META = FileMeta(
    filepath=DATA_DIR / "dontpatronizeme_pcl_dev.tsv",
)


# The file where the training pcl data is stored.
TRAINING_PCL_DATA_FILE_META = FileMeta(
    filepath=DATA_DIR / "dontpatronizeme_pcl_training.tsv",
)

TASK4_PROCESSED_DATA_FILE_META = FileMeta(
    filepath=DATA_DIR / "task4_test.tsv",
)
