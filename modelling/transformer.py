"""Language modelling using multi-head attention transformers."""
from datetime import datetime
from typing import Dict

from torch import (
    arange, cos, device, exp, log, manual_seed, tensor, sin, sqrt, Tensor, zeros
)
from torch.distributions import Categorical
from torch.nn import (
    CrossEntropyLoss,
    Dropout,
    Embedding,
    Linear,
    Module,
    TransformerDecoderLayer
)
from torch.utils.data import DataLoader
from torch.optim import Adam

from .data import _Tokenizer
from .utils import get_device


class NextWordPredictionTransformer(Module):
    """Transformer for predicting the next tokens in a sequence."""

    def __init__(self, size_vocab: int, size_embed: int, n_heads: int = 1):
        super().__init__()
        self._size_vocab = size_vocab
        self._size_embed = size_embed
        self._position_encoder = PositionalEncoding(size_embed)
        self._embedding = Embedding(size_vocab, size_embed)
        self._decoder = TransformerDecoderLayer(size_embed, n_heads, batch_first=True)
        self._linear = Linear(size_embed, size_vocab)

    def forward(self, x: Tensor) -> Tensor:
        out = self._embedding(x) * sqrt(tensor(self._size_embed))
        out = self._position_encoder(out)
        out = self._decoder(out, out, tgt_is_causal=True, memory_is_causal=True)
        out = self._linear(out)
        return out


class PositionalEncoding(Module):
    """Position encoder taken from 'Attention is all you Need'."""

    def __init__(self, size_embed: int, dropout: float = 0.1, max_seq_len: int = 1000):
        super().__init__()
        self._dropout = Dropout(p=dropout)

        position = arange(max_seq_len).unsqueeze(1)
        div_term = exp(arange(0, size_embed, 2) * (-log(tensor(10000.0)) / size_embed))
        pos_encoding = zeros(max_seq_len, size_embed)
        pos_encoding[:, 0::2] = sin(position * div_term)
        pos_encoding[:, 1::2] = cos(position * div_term)
        self.register_buffer('pos_encoding', pos_encoding)  # don't train these

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        seq_len = x.size(1)
        x = x + self.pos_encoding[:seq_len]
        return self.dropout(x)


def train(
    model: Module,
    sequence_data: DataLoader,
    n_epochs: int,
    learning_rate: float = 0.001,
    random_seed: int = 42,
) -> Dict[int, float]:
    """Training loop for transformer decoder."""
    manual_seed(random_seed)
    device = get_device()

    model.to(device)
    model.train()

    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_func = CrossEntropyLoss()
    train_loss: Dict[int, float] = {}

    for epoch in range(1, n_epochs+1):
        # use random batch of sequences over iterating over all possible batches
        x_batch, y_batch = next(iter(sequence_data))
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        y_pred = model(x_batch)
        loss = loss_func(y_pred.permute(0, 2, 1), y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss = loss.item()
        train_loss[epoch] = loss.item()

        if epoch % 10 == 0:
            timestamp = datetime.now().isoformat(timespec="seconds")
            print(f"{timestamp} epoch {epoch} loss: {avg_loss.item():.4f}")

    return train_loss


def generate(
    model: NextWordPredictionTransformer,
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

    prompt_tokens = tensor(tokenizer(prompt), device=device_).view(-1, 1)

    new_token_sequence = [prompt_tokens[-1]]
    for _ in range(output_length):
        token_logits = model(new_token_sequence[-1])
        token_pred = Categorical(logits=temperature * token_logits).sample()
        new_token_sequence += [token_pred]

    new_text = prompt + " " + " ".join(tokenizer.tokens2text(new_token_sequence[1:]))
    return new_text.replace(" endofsentence ", ". ")


if __name__ == "__main__":
    # train model
    from .data import FilmReviewSequences
    from .utils import save_model

    MODEL_NAME = "decoder_next_word_gen"

    SIZE_EMBED = 256 * 2

    N_EPOCHS = 1000
    BATCH_SIZE = 64
    SEQUENCE_LENGTH = 40
    LEARNING_RATE = 0.005

    data = FilmReviewSequences(sequence_length=SEQUENCE_LENGTH)
    data_loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    model = NextWordPredictionTransformer(data.vocab_size, SIZE_EMBED)
    train_loss = train(model, data_loader, N_EPOCHS, LEARNING_RATE)
    save_model(model, name=MODEL_NAME, loss=train_loss[N_EPOCHS])

    # generate text
    from .data import IMDBTokenizer
    from .utils import load_model

    MODEL_NAME = "decoder_next_word_gen"
    PROMPT = "This movie was a total waste of time"
    TEMPERATURE = 1.0

    tokenizer = IMDBTokenizer()
    model: NextWordPredictionTransformer = load_model(MODEL_NAME)
    new_text = generate(model, PROMPT, tokenizer, temperature=TEMPERATURE)
    print(new_text)
