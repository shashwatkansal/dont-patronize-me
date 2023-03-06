"""TF-IDF Weighted Bag-of-Words (BoW) embedding with SVM classifier."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from preprocessing.data import get_dev_test_data, get_training_data

logger = logging.getLogger(__name__)


def train_tf_idf_svm_model(*, evaluate_train: bool = False) -> Pipeline:
    X_train, y_true = get_training_data()

    # Define a Support Vector Machine Classifier with a TF-IDF Bag-of-Words embedding.
    # Hyperparameters for SVM are chosen in accordance with the original task paper.
    classifier = Pipeline(
        [
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("svm", SVC(C=10.0, kernel="rbf")),
        ]
    )

    # Fit the classifier model on the given training dataset.
    logger.info("Training the classifier...")
    classifier = classifier.fit(X_train, y_true)
    logger.info("Finished training the classifier.")

    # Evaluate F1-score on the training dataset prediction.
    if evaluate_train:
        logger.info("Predicting on the training dataset...")
        y_pred = classifier.predict(X_train)
        logger.info("Finished predicting on the training dataset.")

        f1_train = f1_score(y_true, y_pred)
        logger.info(f"Training performance: {f1_train:.4f}")

    return classifier


def test_tf_idf_svm_model(classifier: Pipeline, *, display_confusion: bool = False) -> tuple[float, float, float]:
    # Fetch test data and perform prediction on test inputs.
    X_test, y_true = get_dev_test_data()
    logger.info("Performing prediction on test dataset...")
    y_pred: npt.NDArray[np.int_] = classifier.predict(X_test)
    logger.info("Finished performing prediction on test dataset.")

    # Calculate Various Metrics.
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    logger.info(f"Metrics:\t{precision=:4f}, {recall=:4f}, {f1=:4f}")

    # Display confusion matrix.
    if display_confusion:
        disp = ConfusionMatrixDisplay.from_estimator(
            classifier,
            X_test,
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

        classifier_path = Path("./results/tf-idf-svm/classifier.pkl")
        with open(classifier_path, "rb") as f:
            classifier = pickle.load(f)
    else:
        classifier = train_tf_idf_svm_model(evaluate_train=False)

    test_tf_idf_svm_model(classifier, display_confusion=True)


if __name__ == "__main__":
    main()
