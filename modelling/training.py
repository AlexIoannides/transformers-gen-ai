"""Training models."""
from torch import manual_seed, tensor
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Adam
from torch.utils.data import DataLoader


def train_lstm_on_next_word_prediction(
    model: Module,
    sequence_data: DataLoader,
    n_epochs: int,
    learning_rate: float = 0.0001,
    random_seed: int = 42,
) -> None:
    """Training loop for LTSM flavoured RNNs on sequence data."""
    manual_seed(random_seed)
    model.train()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_func = CrossEntropyLoss()
    for epoch in range(n_epochs):
        for x_batch, y_batch in sequence_data:
            loss = tensor(0.0)
            optimizer.zero_grad()
            sequence_lenth = len(x_batch[0])
            hidden, cell = model.initialise(len(x_batch))
            for n in range(sequence_lenth):
                y_pred, hidden, cell = model(x_batch[:, n], hidden, cell)
                loss += loss_func(y_pred, y_batch[:, n])
            loss.backward()
            optimizer.step()
            print(f"{loss.item()/sequence_lenth:.4f}")
        if epoch % 1 == 0:
            print(f"Epoch {epoch} loss: {loss.item()/sequence_lenth}")


if __name__ == "__main__":
    from .data import FilmReviewSequences
    from .lstm import NextWordPrediction

    BATCH_SIZE = 64
    SEQUENCE_LENGTH = 40

    dataset = FilmReviewSequences(sequence_length=SEQUENCE_LENGTH)
    sequence_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    generative_model = NextWordPrediction(
        dataset.vocab_size, size_embed=256, size_hidden=512
    )
    train_lstm_on_next_word_prediction(generative_model, sequence_loader, n_epochs=1)
