"""GloVe pre-trained embedding with SVM classifier."""

import logging
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.svm import SVC
from torch import nn
from torchtext.data import Iterator
from torchtext.vocab import GloVe

from model.glove import build_glove_embedding_iters

logger = logging.getLogger(__name__)

IntArrayT = npt.NDArray[np.int_]
FloatArrayT = npt.NDArray[np.float_]


def _process_dataset(*iters: Iterator) -> tuple[FloatArrayT, IntArrayT]:
    embedding_dim = 300
    embedding = nn.Embedding.from_pretrained(GloVe(name="6B", dim=embedding_dim).vectors, freeze=True)

    logger.info("Re-Processing the training data to fit the SVM classifier...")
    texts, labels = [], []
    for text, label in chain(*iters):
        text = embedding(text)
        text_arr, label_arr = text.cpu().detach().numpy(), label.cpu().detach().numpy()

        texts.append(text_arr)
        labels.append(label_arr)

    X, y = np.concatenate(texts), np.concatenate(labels)
    X = np.mean(X, axis=1)
    logger.info("Finished Re-Processing the training data to fit the SVM classifier.")

    return X, y


def train_glove_svm() -> SVC:
    logger.info("Fetching the GloVe embedding data...")
    train_iter, val_iter, _ = build_glove_embedding_iters(device=torch.device("cpu"))
    logger.info("Finished fetching the GloVe embedding data.")

    X, y_true = _process_dataset(train_iter, val_iter)

    # Define a Support Vector Machine Classifier.
    # Hyperparameters for SVM are chosen in accordance with the original task paper.
    classifier = SVC(C=10.0, kernel="rbf")

    # Fit the classifier model on the given training dataset.
    logger.info("Training the classifier...")
    classifier.fit(X, y_true)
    logger.info("Finished training the classifier.")

    # Evaluate F1-score on the training dataset prediction.
    logger.info("Predicting on the training dataset...")
    y_pred = classifier.predict(X)
    logger.info("Finished predicting on the training dataset.")

    f1_train = f1_score(y_true, y_pred)
    logger.info(f"Training performance: {f1_train:.4f}")

    return classifier


def test_glove_svm_model(classifier: SVC, *, display_confusion: bool = False) -> tuple[float, float, float]:
    logger.info("Fetching the GloVe embedding data...")
    _, _, test_iter = build_glove_embedding_iters(device=torch.device("cpu"))
    logger.info("Finished fetching the GloVe embedding data.")
    X, y_true = _process_dataset(test_iter)

    logger.info("Performing prediction on test dataset...")
    y_pred: npt.NDArray[np.int_] = classifier.predict(X)
    logger.info("Finished performing prediction on test dataset.")

    # Calculate Various Metrics.
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    logger.info(f"Metrics:\t{precision=:4f}, {recall=:4f}, {f1=:4f}")

    # Display confusion matrix.
    if display_confusion:
        disp = ConfusionMatrixDisplay.from_estimator(
            classifier,
            X,
            y_true,
            display_labels=["Not PCL", "PCL"],
        )
        disp.ax_.set_title("Confusion Matrix for TF-IDF + SVM")
        plt.show()

    return float(precision), float(recall), float(f1)


def main(use_preloaded_model: bool = True) -> None:
    if use_preloaded_model:
        import pickle
        from pathlib import Path

        classifier_path = Path("./results/glove-svm/classifier.pkl")
        with open(classifier_path, "rb") as f:
            classifier = pickle.load(f)
    else:
        classifier = train_glove_svm()

    test_glove_svm_model(classifier, display_confusion=True)


if __name__ == "__main__":
    main()

