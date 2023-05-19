"""Language modelling using RNNs."""
from datetime import datetime
from typing import Tuple

from torch import device, manual_seed, tensor, Tensor, zeros
from torch.distributions import Categorical
from torch.nn import CrossEntropyLoss, Embedding, Linear, Module, LSTM
from torch.utils.data import DataLoader
from torch.optim import Adam

from .data import _Tokenizer
from .utils import get_device


class NextWordPrediction(Module):
    """LSTM for predicting the next token in a sequence."""

    def __init__(self, size_vocab: int, size_embed: int, size_hidden: int):
        super().__init__()
        self._size_hidden = size_hidden
        self._embedding = Embedding(size_vocab, size_embed)
        self._lstm = LSTM(size_embed, size_hidden, batch_first=True)
        self._linear = Linear(size_hidden, size_vocab)

    def forward(self, x: Tensor, hidden: Tensor, cell: Tensor) -> Tensor:
        out = self._embedding(x).unsqueeze(1)
        out, (hidden, cell) = self._lstm(out, (hidden, cell))
        out = self._linear(out).reshape(out.shape[0], -1)
        return out, hidden, cell

    def initialise(self, batch_size: int, device_: device) -> Tuple[Tensor, Tensor]:
        hidden = zeros(1, batch_size, self._size_hidden, device=device_)
        cell = zeros(1, batch_size, self._size_hidden, device=device_)
        return hidden, cell


def train_next_word_prediction(
    model: Module,
    sequence_data: DataLoader,
    n_epochs: int,
    learning_rate: float = 0.001,
    random_seed: int = 42,
) -> float:
    """Training loop for LTSM flavoured RNNs on sequence data."""
    manual_seed(random_seed)
    device = get_device()

    model.to(device)
    model.train()

    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_func = CrossEntropyLoss()

    for epoch in range(n_epochs):
        # use random batch of sequences over iterating over all possible batches
        x_batch, y_batch = next(iter(sequence_data))
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

        if epoch % 10 == 0:
            timestamp = datetime.now().isoformat(timespec="seconds")
            print(f"{timestamp} epoch {epoch} loss: {loss.item()/sequence_lenth:.4f}")

    return loss.item() / sequence_lenth


def generate(
    model: NextWordPrediction,
    prompt: str,
    tokenizer: _Tokenizer,
    output_length: int = 40,
    temperature: float = 1.0,
    random_seed: int = 42,
    device_: device = device("cpu"),
) -> str:
    """Generate new text conditional on a text prompt."""
    manual_seed(random_seed)

    model.to(device_)
    model.eval()
    hidden, cell = model.initialise(1, device_)

    prompt_tokens = tensor(tokenizer(prompt), device=device_).reshape(-1, 1)
    for token in prompt_tokens[:-1]:
        _, hidden, cell = model(token, hidden, cell)

    new_token_sequence = [prompt_tokens[-1]]
    for _ in range(output_length):
        token_logits, hidden, cell = model(new_token_sequence[-1], hidden, cell)
        token_pred = Categorical(logits=temperature * token_logits).sample()
        new_token_sequence += [token_pred]

    new_text = prompt + " " + " ".join(tokenizer.tokens2text(new_token_sequence[1:]))
    return new_text.replace(" endofsentence ", ". ")


if __name__ == "__main__":
    # train model
    from .data import FilmReviewSequences
    from .utils import save_model

    MODEL_NAME = "lstm_next_word_gen"

    SIZE_EMBED = 256 * 2
    SIZE_HIDDEN = 512 * 2

    EPOCHS = 1000
    BATCH_SIZE = 64
    SEQUENCE_LENGTH = 40
    LEARNING_RATE = 0.005

    data = FilmReviewSequences(sequence_length=SEQUENCE_LENGTH)
    data_loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    model = NextWordPrediction(data.vocab_size, SIZE_EMBED, SIZE_HIDDEN)
    model_loss = train_next_word_prediction(model, data_loader, EPOCHS, LEARNING_RATE)
    save_model(model, name=MODEL_NAME, loss=model_loss)

    # generate text
    from .data import IMDBTokenizer
    from .utils import load_model

    MODEL_NAME = "lstm_next_word_gen"
    PROMPT = "This movie was a total waste of time"
    TEMPERATURE = 1.0

    tokenizer = IMDBTokenizer()
    model: NextWordPrediction = load_model(MODEL_NAME)
    new_text = generate(model, PROMPT, tokenizer, temperature=TEMPERATURE)
    print(new_text)
