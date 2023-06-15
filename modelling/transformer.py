"""Language modelling using multi-head attention transformers."""
from __future__ import annotations
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
from torch.nn.init import xavier_uniform_
from torch.utils.data import DataLoader
from torch.optim import Adam

from .data import _Tokenizer, EOS_DELIM
from .utils import capitalise_sentences, get_device


class NextWordPredictionTransformer(Module):
    """Transformer for predicting the next tokens in a sequence."""

    def __init__(self, size_vocab: int, size_embed: int, n_heads: int = 2):
        super().__init__()
        self._size_vocab = size_vocab
        self._size_embed = size_embed
        self._position_encoder = PositionalEncoding(size_embed)
        self._embedding = Embedding(size_vocab, size_embed)
        self._decoder = TransformerDecoderLayer(size_embed, n_heads, batch_first=True)
        self._linear = Linear(size_embed, size_vocab)
        self._init_weights()

    def forward(self, x: Tensor) -> Tensor:
        out = self._embedding(x) * sqrt(tensor(self._size_embed))
        out = self._position_encoder(out)
        out = self._decoder(out, out, tgt_is_causal=True, memory_is_causal=True)
        out = self._linear(out)
        return out

    def _init_weights(self) -> NextWordPredictionTransformer:
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
        return self


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
        return self._dropout(x)


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
        for x_batch, y_batch in sequence_data:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            y_pred = model(x_batch)
            loss = loss_func(y_pred.permute(0, 2, 1), y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss = loss
            train_loss[epoch] = avg_loss.item()

        timestamp = datetime.now().isoformat(timespec="seconds")
        print(f"{timestamp} epoch {epoch} loss: {train_loss[epoch]:.4f}")

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

    prompt_tokens = tokenizer(prompt)
    token_sequence = prompt_tokens.copy()
    for _ in range(output_length):
        x = tensor([token_sequence], device=device_)
        token_logits = model(x)
        token_pred = Categorical(logits=temperature * token_logits[0, -1]).sample()
        token_sequence += [token_pred.item()]

    new_token_sequence = token_sequence[len(prompt_tokens):]
    new_text = " " + " ".join(tokenizer.tokens2text(new_token_sequence))
    new_text = capitalise_sentences(new_text, sentence_delimiter=EOS_DELIM)
    new_text = new_text.replace(EOS_DELIM, ". ")
    return "==> " + prompt.upper() + new_text + "..."


if __name__ == "__main__":
    # train model
    from .data import FilmReviewSequences, pad_seq2seq_data
    from .utils import save_model

    MODEL_NAME = "decoder_next_word_gen"

    SIZE_EMBED = 256

    N_EPOCHS = 1
    BATCH_SIZE = 256
    SEQUENCE_LENGTH = 40
    LEARNING_RATE = 0.005

    print("-- training model --")

    data = FilmReviewSequences(split="all", sequence_length=SEQUENCE_LENGTH)
    data_loader = DataLoader(
        data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        collate_fn=pad_seq2seq_data
    )
    model = NextWordPredictionTransformer(data.vocab_size, SIZE_EMBED)
    train_loss = train(model, data_loader, N_EPOCHS, LEARNING_RATE)
    save_model(model, name=MODEL_NAME, loss=train_loss[N_EPOCHS])

    # generate text
    from .data import IMDBTokenizer
    from .utils import load_model

    MODEL_NAME = "decoder_next_word_gen"
    PROMPT = "I thought this movie was"
    TEMPERATURE = 1.0

    print("\n-- generating text --")

    tokenizer = IMDBTokenizer()
    model: NextWordPredictionTransformer = load_model(MODEL_NAME)
    new_text = generate(model, PROMPT, tokenizer, temperature=TEMPERATURE)
    print(new_text)
