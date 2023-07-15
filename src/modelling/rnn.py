"""Language modelling using RNNs."""
from typing import Callable, Dict, Literal, Tuple

from torch import Tensor, device, manual_seed, no_grad, tensor, zeros
from torch.nn import LSTM, CrossEntropyLoss, Embedding, Linear, Module
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from modelling.data import PAD_TOKEN_IDX, _Tokenizer
from modelling.utils import (
    ModelCheckpoint,
    _early_stop,
    decode,
    format_generated_words,
    get_best_device,
)


class NextWordPredictionRNN(Module):
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


def _train_step(
    x_batch: Tensor,
    y_batch: Tensor,
    model: Module,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    optimizer: Optimizer,
    device: device,
) -> Tensor:
    """One iteration of the training loop (for one batch)."""
    model.train()
    batch_size, sequence_length = x_batch.shape

    loss_batch = tensor(0.0, device=device)
    optimizer.zero_grad(set_to_none=True)

    hidden, cell = model.initialise(batch_size, device)
    for n in range(sequence_length):
        y_pred, hidden, cell = model(x_batch[:, n], hidden, cell)
        loss_batch += loss_fn(y_pred, y_batch[:, n])
    loss_batch.backward()
    optimizer.step()

    return loss_batch / sequence_length


@no_grad()
def _val_step(
    x_batch: Tensor,
    y_batch: Tensor,
    model: Module,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    device: device,
) -> Tensor:
    """One iteration of the validation loop (for one batch)."""
    model.eval()
    batch_size, sequence_length = x_batch.shape

    loss_batch = tensor(0.0, device=device)

    hidden, cell = model.initialise(batch_size, device)
    for n in range(sequence_length):
        y_pred, hidden, cell = model(x_batch[:, n], hidden, cell)
        loss_batch += loss_fn(y_pred, y_batch[:, n])

    return loss_batch / sequence_length


def train(
    model: Module,
    train_data: DataLoader,
    val_data: DataLoader,
    n_epochs: int,
    learning_rate: float = 0.001,
    random_seed: int = 42,
    device: device = get_best_device(),
) -> Tuple[Dict[int, float], Dict[int, float], ModelCheckpoint]:
    """Training loop for LTSM flavoured RNNs on sequence data."""
    manual_seed(random_seed)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = CrossEntropyLoss(ignore_index=PAD_TOKEN_IDX)

    train_losses: Dict[int, float] = {}
    val_losses: Dict[int, float] = {}

    for epoch in range(1, n_epochs + 1):
        loss_train = tensor(0.0).to(device)
        for i, (x_batch, y_batch) in enumerate((pbar := tqdm(train_data)), start=1):
            x = x_batch.to(device, non_blocking=True)
            y = y_batch.to(device, non_blocking=True)
            loss_train += _train_step(x, y, model, loss_fn, optimizer, device)
            pbar.set_description(f"epoch {epoch} training loss = {loss_train/i:.4f}")

        loss_val = tensor(0.0).to(device)
        for x_batch, y_batch in val_data:
            x = x_batch.to(device, non_blocking=True)
            y = y_batch.to(device, non_blocking=True)
            loss_val += _val_step(x, y, model, loss_fn, device)

        epoch_train_loss = loss_train.item() / len(train_data)
        epoch_val_loss = loss_val.item() / len(val_data)

        if epoch == 1 or epoch_val_loss < min(val_losses.values()):
            best_checkpoint = ModelCheckpoint(
                epoch, epoch_train_loss, epoch_val_loss, model.state_dict().copy()
            )
        train_losses[epoch] = epoch_train_loss
        val_losses[epoch] = epoch_val_loss

        if _early_stop(val_losses):
            break

    print("\nbest model:")
    print(f"|-- epoch: {best_checkpoint.epoch}")
    print(f"|-- validation loss: {best_checkpoint.val_loss:.4f}")

    model.load_state_dict(best_checkpoint.state_dict)
    return train_losses, val_losses, best_checkpoint


def generate(
    model: NextWordPredictionRNN,
    prompt: str,
    tokenizer: _Tokenizer,
    strategy: Literal["greedy", "sample", "topk"] = "greedy",
    output_length: int = 60,
    temperature: float = 1.0,
    random_seed: int = 42,
    device: device = get_best_device(),
    *,
    k: int = 2,
) -> str:
    """Generate new text conditional on a text prompt."""
    manual_seed(random_seed)

    model.to(device)
    model.eval()

    prompt_tokens = tokenizer(prompt)

    hidden, cell = model.initialise(1, device)
    for token in prompt_tokens[:-1]:
        x = tensor([token], device=device)
        _, hidden, cell = model(x, hidden, cell)

    token_sequence = prompt_tokens.copy()
    for _ in range(output_length):
        x = tensor([token_sequence[-1]], device=device)
        token_logits, hidden, cell = model(x, hidden, cell)
        token_pred = decode(token_logits, strategy, temperature, k=k)
        token_sequence += [token_pred.item()]

    new_token_sequence = token_sequence[len(prompt_tokens) :]
    return format_generated_words(tokenizer.tokens2text(new_token_sequence), prompt)
