"""Language modelling using RNNs."""
from typing import Tuple

import torch.nn as nn
from torch import device, tensor, zeros


class NextWordPrediction(nn.Module):
    """LTSM for predicting the next token in a sequence."""

    def __init__(self, size_vocab: int, size_embed: int, size_hidden: int):
        super().__init__()
        self._size_hidden = size_hidden
        self._embedding = nn.Embedding(size_vocab, size_embed)
        self._lstm = nn.LSTM(size_embed, size_hidden, batch_first=True)
        self._linear = nn.Linear(size_hidden, size_vocab)

    def forward(self, x: tensor, hidden: tensor, cell: tensor) -> tensor:
        out = self._embedding(x).unsqueeze(1)
        out, (hidden, cell) = self._lstm(out, (hidden, cell))
        out = self._linear(out).reshape(out.shape[0], -1)
        return out, hidden, cell

    def initialise(self, batch_size: int, device_: device) -> Tuple[tensor, tensor]:
        hidden = zeros(1, batch_size, self._size_hidden, device=device_)
        cell = zeros(1, batch_size, self._size_hidden, device=device_)
        return hidden, cell
