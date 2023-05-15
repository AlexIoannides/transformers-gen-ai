"""Training models."""
from datetime import datetime

from torch import device, manual_seed, tensor
from torch.backends import mps
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Adam
from torch.utils.data import DataLoader


def get_device() -> device:
    """Run on CPUs or GPUs."""
    if mps.is_available():
        return device("mps")
    else:
        device("cpu")


def train_lstm_on_next_word_prediction(
    model: Module,
    sequence_data: DataLoader,
    n_epochs: int,
    learning_rate: float = 0.0001,
    random_seed: int = 42,
) -> None:
    """Training loop for LTSM flavoured RNNs on sequence data."""
    manual_seed(random_seed)
    device = get_device()

    model.to(device)
    model.train()

    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_func = CrossEntropyLoss()

    for epoch in range(n_epochs):
        for x_batch, y_batch in sequence_data:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            batch_size = len(x_batch)
            sequence_lenth = len(x_batch[0])

            loss = tensor(0.0, device=device)
            optimizer.zero_grad()
            hidden, cell = model.initialise(batch_size, device)

            for n in range(sequence_lenth):
                y_pred, hidden, cell = model(x_batch[:, n], hidden, cell)
                loss += loss_func(y_pred, y_batch[:, n])
            loss.backward()
            optimizer.step()
        timestamp = datetime.now().isoformat(timespec="seconds")
        print(f"{timestamp} epoch {epoch} loss: {loss.item()/sequence_lenth:.4f}")


if __name__ == "__main__":
    from .data import FilmReviewSequences
    from .lstm import NextWordPrediction

    EPOCHS = 10
    BATCH_SIZE = 64
    SEQUENCE_LENGTH = 40

    data = FilmReviewSequences(sequence_length=SEQUENCE_LENGTH)
    data_loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    model = NextWordPrediction(data.vocab_size, size_embed=256, size_hidden=512)
    train_lstm_on_next_word_prediction(model, data_loader, n_epochs=EPOCHS)
