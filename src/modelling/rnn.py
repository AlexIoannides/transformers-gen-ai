"""Language modelling using RNNs."""
from typing import Dict, Tuple

from torch import device, manual_seed, tensor, Tensor, zeros
from torch.distributions import Categorical
from torch.nn import CrossEntropyLoss, Embedding, Linear, Module, LSTM
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from .data import _Tokenizer, EOS_DELIM, PAD_TOKEN_IDX
from .utils import capitalise_sentences, get_device


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


def train(
    model: Module,
    sequence_data: DataLoader,
    n_epochs: int,
    learning_rate: float = 0.001,
    random_seed: int = 42,
) -> Dict[int, float]:
    """Training loop for LTSM flavoured RNNs on sequence data."""
    manual_seed(random_seed)
    device = get_device()

    model.to(device)
    model.train()

    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_func = CrossEntropyLoss(ignore_index=PAD_TOKEN_IDX)
    train_loss: Dict[int, float] = {}

    for epoch in range(1, n_epochs+1):
        for x_batch, y_batch in (pbar := tqdm(sequence_data)):
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            batch_size, sequence_length = x_batch.shape

            loss = tensor(0.0, device=device)
            optimizer.zero_grad()

            hidden, cell = model.initialise(batch_size, device)
            for n in range(sequence_length):
                y_pred, hidden, cell = model(x_batch[:, n], hidden, cell)
                loss += loss_func(y_pred, y_batch[:, n])
            loss.backward()
            optimizer.step()

            avg_loss = loss / sequence_length
            pbar.set_description(f"epoch {epoch} current loss = {avg_loss:.4f}")

        if epoch == 1 or avg_loss.item() < min(train_loss.values()):
            best_checkpoint = {
                "state_dict": model.state_dict().copy(),
                "loss": avg_loss.item(),
                "epoch": epoch
            }

        train_loss[epoch] = avg_loss.item()

    print("\nbest model:")
    print(f"|-- epoch: {best_checkpoint['epoch']}")
    print(f"|-- loss: {best_checkpoint['loss']:.4f}")

    model.load_state_dict(best_checkpoint["state_dict"])
    return train_loss


def generate(
    model: NextWordPredictionRNN,
    prompt: str,
    tokenizer: _Tokenizer,
    output_length: int = 60,
    temperature: float = 1.0,
    random_seed: int = 42,
    device_: device = device("cpu"),
) -> str:
    """Generate new text conditional on a text prompt."""
    manual_seed(random_seed)

    model.to(device_)
    model.eval()

    prompt_tokens = tokenizer(prompt)
    hidden, cell = model.initialise(1, device_)
    for token in prompt_tokens[:-1]:
        x = tensor([token], device=device_)
        _, hidden, cell = model(x, hidden, cell)

    token_sequence = prompt_tokens.copy()
    for _ in range(output_length):
        x = tensor([token_sequence[-1]], device=device_)
        token_logits, hidden, cell = model(x, hidden, cell)
        token_pred = Categorical(logits=temperature * token_logits).sample()
        token_sequence += [token_pred.item()]

    new_token_sequence = token_sequence[len(prompt_tokens):]
    new_text = " " + " ".join(tokenizer.tokens2text(new_token_sequence))
    new_text = capitalise_sentences(new_text, sentence_delimiter=EOS_DELIM)
    new_text = new_text.replace(EOS_DELIM, ". ")
    return "==> " + prompt.upper() + new_text + "..."
