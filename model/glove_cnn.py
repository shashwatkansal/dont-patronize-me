"""CNN Model using GloVe word embeddings."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_fscore_support
from torch import nn, optim
from torchtext.data import Iterator
from torchtext.vocab import GloVe
from tqdm import tqdm

from model.glove import build_glove_embedding_iters

logger = logging.getLogger(__name__)


class GloVeCNNModel(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(GloVe(name="6B", dim=embedding_dim).vectors, freeze=True)

        latent_size = 64
        self.conv_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(in_channels=embedding_dim, out_channels=latent_size, kernel_size=kernel_size),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=kernel_size),
                )
                for kernel_size in (5, 7)
            ]
        )

        self.linear = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(64 * 32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_embed: torch.Tensor = self.embedding(x)
        x_embed = x_embed.transpose(1, 2)

        y_components: list[torch.Tensor] = []
        for conv_layer in self.conv_layers:
            y_component: torch.Tensor = conv_layer(x_embed)
            y_components.append(y_component)
        y = torch.concat(y_components, dim=2)
        y = y.reshape(y.shape[0], -1)

        y = self.linear(y)

        return y


class EarlyStopper:
    def __init__(
        self,
        device: torch.device,
        patience: int = 3,
        delta: float = 0.005,
        model_path: Path = Path("./results/glove-cnn/") / "model_best.pth",
    ) -> None:
        self.device = device
        self.patience = patience
        self.counter = 0
        self.min_loss = np.inf
        self.delta = delta
        self.model_path = model_path

    def early_stop(self, loss: float, model: GloVeCNNModel) -> bool:
        if loss < self.min_loss:
            # Update the minimum loss.
            self.min_loss = loss
            self.counter = 0

            # Save the new model with minimum loss.
            torch.save(model.state_dict(), self.model_path)
        elif (loss - self.min_loss) > self.delta:
            # Loss is greater than the minimum loss, so increment the early stop counter.
            self.counter += 1

            # If the early stop counter is bigger than its patience, load the best
            # model parameters back into the model and stop early.
            if self.counter >= self.patience:
                model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                return True
        return False


def train_glove_cnn(
    model: GloVeCNNModel,
    train_iter: Iterator,
    val_iter: Iterator,
    *,
    device: torch.device,
    num_epochs: int = 50,
    learning_rate: float = 3e-4,
) -> None:
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    early_stopper = EarlyStopper(device=device)

    train_losses: list[float] = []
    val_losses: list[float] = []
    for epoch in range(num_epochs):
        # Training.
        model.train()
        train_loss = 0.0
        for idx, (text, label) in enumerate(tbatch := tqdm(train_iter, desc=f"Epoch {epoch}")):
            text, label = text.to(device), label.to(device).type(torch.float)

            # Forward Pass.
            output = model(text).squeeze()
            loss: torch.Tensor = criterion(output, label)
            train_loss += loss.item()

            # Backpropagation.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % 20 == 0:
                tbatch.set_postfix(loss=loss.item())
        train_loss /= len(train_iter)
        train_losses.append(train_loss)

        # Validation.
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for text, label in val_iter:
                text, label = text.to(device), label.to(device).type(torch.float)

                output = model(text).squeeze()
                loss: torch.Tensor = criterion(output, label)
                val_loss += loss.item()
        val_loss /= len(val_iter)
        val_losses.append(val_loss)

        logger.info(f"Train loss: {train_loss}\tValidation loss: {val_loss}")

        # Early stopping.
        if early_stopper.early_stop(val_loss, model):
            logger.info(f"Stopping early after {epoch} epochs with validation loss of {early_stopper.min_loss:.3f}")
            break

    # Plot the loss curves.
    xaxis = np.arange(1, epoch + 2)
    plt.plot(xaxis, train_losses, "-b", label="Loss (Train)")
    plt.plot(xaxis, val_losses, "-r", label="Loss (Validation)")
    plt.title("Loss Curves for the GloVe Embedding + CNN")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def test_glove_cnn(model: GloVeCNNModel, test_iter: Iterator, *, device: torch.device) -> None:
    criterion = nn.BCELoss()

    labels: list[npt.NDArray[np.float_]] = []
    outputs: list[npt.NDArray[np.float_]] = []

    # Calculate Test Loss.
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for text, label in test_iter:
            text, label = text.to(device), label.to(device).type(torch.float)

            output = model(text).squeeze()
            labels.append(label.cpu().detach().numpy())
            outputs.append(output.cpu().detach().numpy())

            loss: torch.Tensor = criterion(output, label)
            test_loss += loss.item()
    test_loss /= len(test_iter)
    logger.info(f"Test loss: {test_loss}")

    # Calculate metrics.
    label = (np.concatenate(labels) > 0.5).astype(np.int_)
    output = (np.concatenate(outputs) > 0.5).astype(np.int_)
    precision, recall, f1, _ = precision_recall_fscore_support(label, output, average="binary")
    logger.info(f"Metrics:\t{precision=:4f}, {recall=:4f}, {f1=:4f}")

    # Display confusion matrix
    disp = ConfusionMatrixDisplay.from_predictions(label, output, display_labels=["Not PCL", "PCL"])
    disp.ax_.set_title("Confusion Matrix for TF-IDF + SVM")
    plt.show()


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Hyperparameters.
    batch_size = 32
    embedding_dim = 300
    assert embedding_dim in [50, 100, 200, 300]
    max_text_length = 100
    num_epochs = 50
    learning_rate = 2e-4

    # Get dataset.
    train_iter, val_iter, test_iter = build_glove_embedding_iters(
        device=device,
        embedding_dim=embedding_dim,
        batch_size=batch_size,
        max_text_length=max_text_length,
    )

    # Define model.
    model = GloVeCNNModel(embedding_dim)

    # Training loop.
    train_glove_cnn(model, train_iter, val_iter, device=device, num_epochs=num_epochs, learning_rate=learning_rate)

    # Testing.
    test_glove_cnn(model, test_iter, device=device)


if __name__ == "__main__":
    main()
