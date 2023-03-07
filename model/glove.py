import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torchtext.data import Dataset, Field, Iterator, LabelField, TabularDataset
from torchtext.vocab import GloVe

from preprocessing.data import get_dev_test_data, get_training_data

logger = logging.getLogger(__name__)


def _build_tabular_dataset(
    text_field: Field,
    label_field: Field,
    *,
    train_split: float,
    use_dev_test: bool = True,
) -> tuple[Dataset, Dataset, Dataset]:
    # Prepare train, val, test data into a dataframe (to be stored temporarily in tsv file)
    X, y = get_training_data()
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=train_split, shuffle=True)
    # TODO: Change this to incorporate the final test dataset.
    X_test, y_test = get_dev_test_data() if use_dev_test else get_dev_test_data()

    train_df = pd.DataFrame.from_dict({"text": X_train, "label": y_train})
    val_df = pd.DataFrame.from_dict({"text": X_val, "label": y_val})
    test_df = pd.DataFrame.from_dict({"text": X_test, "label": y_test})

    # TabularDataset requires file paths, so create a temporary folder to put the data in files.
    with TemporaryDirectory() as tempdir_name:
        tempdir = Path(tempdir_name)

        # Store the train-validation-test DataFrames into temporary tsv files.
        train_df.to_csv(tempdir / "train.tsv", sep="\t", index=False)
        val_df.to_csv(tempdir / "val.tsv", sep="\t", index=False)
        test_df.to_csv(tempdir / "test.tsv", sep="\t", index=False)

        # Build TabularDataset objects for each train-val-test split using data on the temp files.
        train, val, test = TabularDataset.splits(
            path=tempdir,
            format="tsv",
            skip_header=True,
            train="train.tsv",
            validation="val.tsv",
            test="test.tsv",
            fields=[("text", text_field), ("label", label_field)],
        )

    return train, val, test


def build_glove_embedding_iters(
    *,
    device: torch.device,
    embedding_dim: int = 300,
    train_split: float = 0.8,
    batch_size: int = 32,
    max_text_length: int = 150,
    use_dev_test: bool = True,
) -> tuple[Iterator, Iterator, Iterator]:
    # Define Field objects for the input text and labels.
    text_field = Field(
        sequential=True,
        batch_first=True,
        tokenize="spacy",
        tokenizer_language="en_core_web_sm",
        fix_length=max_text_length,
        lower=True,
    )
    label_field = LabelField(
        batch_first=True,
        use_vocab=False,
    )

    # Build a Dataset object for each train-val-test split.
    train, val, test = _build_tabular_dataset(
        text_field,
        label_field,
        train_split=train_split,
        use_dev_test=use_dev_test,
    )

    # Construct the Vocab object in the `text_field` using GloVe embeddings.
    text_field.build_vocab(train, val, test, vectors=GloVe(name="6B", dim=embedding_dim))

    # Build iterators for each splits.
    train_iter, val_iter, test_iter = Iterator.splits(
        (train, val, test),
        sort_key=lambda x: len(x.text),
        batch_sizes=(batch_size, batch_size, batch_size),
        device=device,
    )

    return train_iter, val_iter, test_iter
