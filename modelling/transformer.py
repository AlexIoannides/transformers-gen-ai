"""Language modelling using multi-head attention transformers."""
from __future__ import annotations

import math
from datetime import datetime
from functools import partial
from typing import Dict, Tuple

from torch import (
    arange,
    cos,
    device,
    exp,
    log,
    manual_seed,
    ones,
    tensor,
    sin,
    sqrt,
    Tensor,
    tril,
    zeros
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
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from .data import _Tokenizer, EOS_DELIM, PAD_TOKEN_IDX
from .utils import capitalise_sentences, get_device


class NextWordPredictionTransformer(Module):
    """Transformer for predicting the next tokens in a sequence."""

    def __init__(self, size_vocab: int, size_embed: int, n_heads: int = 2):
        super().__init__()
        self._size_vocab = size_vocab
        self._size_embed = size_embed
        self._n_heads = n_heads
        self._position_encoder = PositionalEncoding(size_embed)
        self._embedding = Embedding(size_vocab, size_embed)
        self._decoder = TransformerDecoderLayer(
            size_embed, n_heads, dim_feedforward=2*size_embed, batch_first=True
        )
        self._linear = Linear(size_embed, size_vocab)
        self._init_weights()

    def forward(self, x: Tensor) -> Tensor:
        x_causal_mask, x_padding_mask = self._make_mask(x)
        out = self._embedding(x) * sqrt(tensor(self._size_embed))
        out = self._position_encoder(out)
        out = self._decoder(
            out,
            out,
            tgt_mask=x_causal_mask,
            tgt_key_padding_mask=x_padding_mask,
            memory_mask=x_causal_mask,
            memory_key_padding_mask=x_padding_mask
        )
        out = self._linear(out)
        return out

    def _init_weights(self) -> NextWordPredictionTransformer:
        """Parameter initialisaion from Attention is all you Need."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
        return self

    def _make_mask(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Make causal and padding masks."""
        causal_mask = ones(x.size(0) * self._n_heads, x.size(1), x.size(1))
        causal_mask = (tril(causal_mask) == 0)
        padding_mask = (x == PAD_TOKEN_IDX)
        return causal_mask.to(x.device), padding_mask.to(x.device)


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
        self.register_buffer('_pos_encoding', pos_encoding)  # don't train these

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        seq_len = x.size(1)
        x = x + self._pos_encoding[:seq_len]
        return self._dropout(x)


def cosine_warmup_schedule(step: int, warmup_steps: int, max_steps: int):
    """Learning rate schedule function taken from Hugging Face."""
    lr_factor = 0.5 * (1 + math.cos(math.pi * step / max_steps))
    if step <= warmup_steps:
        lr_factor *= step / warmup_steps
    return lr_factor


def train(
    model: Module,
    sequence_data: DataLoader,
    n_epochs: int,
    learning_rate: float = 0.001,
    warmup_epochs: float = 0.5,
    clip_grads: float = None,
    random_seed: int = 42,
) -> Dict[int, float]:
    """Training loop for transformer decoder."""
    manual_seed(random_seed)
    device = get_device()

    model.to(device)
    model.train()

    optimizer = Adam(model.parameters(), lr=learning_rate)
    n_epoch_steps = len(sequence_data)
    n_warmup_steps = math.floor(warmup_epochs * n_epoch_steps)
    n_steps = n_epochs * n_epoch_steps
    learning_rate_scheduler = LambdaLR(
        optimizer,
        partial(cosine_warmup_schedule, warmup_steps=n_warmup_steps, max_steps=n_steps)
    )
    loss_func = CrossEntropyLoss(ignore_index=PAD_TOKEN_IDX)
    train_loss: Dict[int, float] = {}

    print(f"number of warmup steps: {n_warmup_steps} / {n_steps}")
    for epoch in range(1, n_epochs+1):
        for x_batch, y_batch in (pbar := tqdm(sequence_data)):
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            y_pred = model(x_batch)
            loss = loss_func(y_pred.permute(0, 2, 1), y_batch)

            optimizer.zero_grad()
            loss.backward()
            if clip_grads:
                clip_grad_norm_(model.parameters(), clip_grads)
            optimizer.step()
            learning_rate_scheduler.step()

            avg_loss = loss
            current_lr = learning_rate_scheduler.get_last_lr()[0]
            pbar.set_description(
                f"epoch {epoch} current loss = {avg_loss:.4f} (LR = {current_lr:.8f})"
            )

        if epoch == 1 or avg_loss.item() < min(train_loss.values()):
            best_checkpoint = {
                "state_dict": model.state_dict().copy(),
                "loss": avg_loss.item(),
                "epoch": epoch
            }

        train_loss[epoch] = avg_loss.item()
        timestamp = datetime.now().isoformat(timespec="seconds")
        print(f"{timestamp} epoch {epoch} loss: {train_loss[epoch]:.4f}")

    print("best model:")
    print(f"|-- epoch: {best_checkpoint['epoch']}")
    print(f"|-- loss: {best_checkpoint['loss']:.4f}")
    model.load_state_dict(best_checkpoint["state_dict"])

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

    N_EPOCHS = 20
    BATCH_SIZE = 32
    SEQ_LEN = 40
    MIN_WORD_FREQ = 2
    MAXIMUM_LEARNING_RATE = 0.001
    WARMUP_EPOCHS = 2
    GRADIENT_CLIP = 5

    print("-- training model --")

    data = FilmReviewSequences(split="all", seq_len=SEQ_LEN, min_freq=MIN_WORD_FREQ)
    data_loader = DataLoader(
        data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        collate_fn=pad_seq2seq_data
    )
    model = NextWordPredictionTransformer(data.vocab_size, SIZE_EMBED)
    train_losses = train(
        model,
        data_loader,
        N_EPOCHS,
        MAXIMUM_LEARNING_RATE,
        WARMUP_EPOCHS,
        GRADIENT_CLIP
    )
    save_model(model, name=MODEL_NAME, loss=min(train_losses.values()))

    # generate text
    from .data import IMDBTokenizer
    from .utils import load_model

    MODEL_NAME = "decoder_next_word_gen"
    PROMPT = "I thought this movie was"
    TEMPERATURE = 1.0

    print("\n-- generating text --")

    tokenizer = IMDBTokenizer()
    model: NextWordPredictionTransformer = load_model(MODEL_NAME, latest=True)
    new_text = generate(model, PROMPT, tokenizer, temperature=TEMPERATURE)
    print(new_text)
