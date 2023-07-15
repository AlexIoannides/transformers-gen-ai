"""Language modelling using multi-head attention transformers."""
from __future__ import annotations

import math
from functools import partial
from typing import Callable, Literal

from torch import (
    Tensor,
    arange,
    cos,
    device,
    exp,
    log,
    manual_seed,
    no_grad,
    ones,
    sin,
    sqrt,
    tensor,
    tril,
    zeros,
)
from torch.nn import (
    CrossEntropyLoss,
    Dropout,
    Embedding,
    Linear,
    Module,
    TransformerDecoderLayer,
)
from torch.nn.init import xavier_uniform_
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from modelling.data import PAD_TOKEN_IDX, _Tokenizer
from modelling.utils import _early_stop, ModelCheckpoint, decode, format_generated_words, get_best_device


class NextWordPredictionTransformer(Module):
    """Transformer for predicting the next tokens in a sequence."""

    def __init__(self, size_vocab: int, size_embed: int, n_heads: int = 4):
        super().__init__()
        self._size_vocab = size_vocab
        self._size_embed = size_embed
        self._n_heads = n_heads
        self._position_encoder = PositionalEncoding(size_embed)
        self._embedding = Embedding(size_vocab, size_embed)
        self._decoder = TransformerDecoderLayer(
            size_embed, n_heads, dim_feedforward=2 * size_embed, batch_first=True
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
            memory_key_padding_mask=x_padding_mask,
        )
        out = self._linear(out)
        return out

    def _init_weights(self) -> NextWordPredictionTransformer:
        """Parameter initialisaion from Attention is all you Need."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
        return self

    def _make_mask(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Make causal and padding masks."""
        causal_mask = ones(x.size(0) * self._n_heads, x.size(1), x.size(1))
        causal_mask = tril(causal_mask) == 0
        padding_mask = x == PAD_TOKEN_IDX
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
        self.register_buffer("_pos_encoding", pos_encoding)  # don't train these

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        seq_len = x.size(1)
        x = x + self._pos_encoding[:seq_len]
        return self._dropout(x)


def warmup_schedule(step: int, warmup_steps: int, max_steps: int):
    """Learning rate schedule function taken from GPT-1 paper."""
    lr_factor = 0.5 * (1 + math.cos(math.pi * step / max_steps))
    if step <= warmup_steps:
        lr_factor *= step / warmup_steps
    return lr_factor


def _train_step(
    x_batch: Tensor,
    y_batch: Tensor,
    model: Module,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    optimizer: Optimizer,
    lr_scheduler: LRScheduler,
    clip_grads: float | None = None,
) -> Tensor:
    """One iteration of the training loop (for one batch)."""
    model.train()
    y_pred = model(x_batch)
    loss_batch = loss_fn(y_pred.permute(0, 2, 1), y_batch)

    optimizer.zero_grad(set_to_none=True)
    loss_batch.backward()
    if clip_grads:
        clip_grad_norm_(model.parameters(), clip_grads)
    optimizer.step()
    lr_scheduler.step()

    return loss_batch


@no_grad()
def _val_step(
    x_batch: Tensor,
    y_batch: Tensor,
    model: Module,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
) -> Tensor:
    """One iteration of the validation loop (for one batch)."""
    model.eval()
    y_pred = model(x_batch)
    loss_batch = loss_fn(y_pred.permute(0, 2, 1), y_batch)
    return loss_batch


def train(
    model: Module,
    train_data: DataLoader,
    val_data: DataLoader,
    n_epochs: int,
    learning_rate: float = 0.001,
    warmup_epochs: float = 0.5,
    clip_grads: float | None = None,
    random_seed: int = 42,
    device: device = get_best_device()
) -> tuple[dict[int, float], dict[int, float], ModelCheckpoint]:
    """Training loop for transformer decoder."""
    manual_seed(random_seed)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = CrossEntropyLoss(ignore_index=PAD_TOKEN_IDX)

    n_batches = len(train_data)
    n_warmup_steps = math.floor(warmup_epochs * n_batches)
    n_steps = n_epochs * n_batches
    lrs_fn = partial(warmup_schedule, warmup_steps=n_warmup_steps, max_steps=n_steps)
    lrs = LambdaLR(optimizer, lrs_fn)

    train_losses: dict[int, float] = {}
    val_losses: dict[int, float] = {}

    print(f"number of warmup steps: {n_warmup_steps} / {n_steps}")
    for epoch in range(1, n_epochs + 1):
        loss_train = tensor(0.0).to(device)
        for i, (x_batch, y_batch) in enumerate((pbar := tqdm(train_data)), start=1):
            x = x_batch.to(device, non_blocking=True)
            y = y_batch.to(device, non_blocking=True)
            loss_train += _train_step(x, y, model, loss_fn, optimizer, lrs, clip_grads)
            lr = lrs.get_last_lr()[0]
            pbar.set_description(
                f"epoch {epoch} training loss = {loss_train/i:.4f} (LR = {lr:.8f})"
            )

        loss_val = tensor(0.0).to(device)
        for x_batch, y_batch in val_data:
            x = x_batch.to(device, non_blocking=True)
            y = y_batch.to(device, non_blocking=True)
            loss_val += _val_step(x, y, model, loss_fn)

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
    print(f"|-- loss: {best_checkpoint.val_loss:.4f}")
    model.load_state_dict(best_checkpoint.state_dict)

    return train_losses, val_losses, best_checkpoint


def generate(
    model: NextWordPredictionTransformer,
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
    token_sequence = prompt_tokens.copy()
    for _ in range(output_length):
        x = tensor([token_sequence], device=device)
        token_logits = model(x)
        token_pred = decode(token_logits[0, -1], strategy, temperature, k=k)
        token_sequence += [token_pred.item()]

    new_token_sequence = token_sequence[len(prompt_tokens) :]
    new_token_sequence = token_sequence[len(prompt_tokens) :]
    return format_generated_words(tokenizer.tokens2text(new_token_sequence), prompt)
