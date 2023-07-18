"""Downloading and pre-processing IMDB film review data."""
from __future__ import annotations

import math
import re
import string
import warnings
from abc import ABC, abstractmethod
from collections import Counter, OrderedDict
from pathlib import Path
from random import randint
from typing import Iterable, Literal, NamedTuple

from pandas import DataFrame, concat
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPre
from tokenizers.processors import ByteLevel as ByteLevelPost
from tokenizers.trainers import BpeTrainer
from torch import Tensor, tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchtext.datasets import IMDB
from torchtext.vocab import vocab

EOS_TOKEN = "endofsentence"
QM_TOKEN = "questionmark"
PAD_TOKEN_IDX = 0
UNKOWN_TOKEN_IDX = 1
TORCH_DATA_STORAGE_PATH = Path(".data")


def get_data() -> DataFrame:
    """Download raw data and convert to Pandas DataFrame."""
    if not TORCH_DATA_STORAGE_PATH.exists():
        warnings.warn("Downloading IMDB data - this may take a minute or two.")
    train_data = DataFrame(
        IMDB(str(TORCH_DATA_STORAGE_PATH), split="train"),
        columns=["sentiment", "review"],
    )
    train_data["sentiment"] = train_data["sentiment"].apply(lambda e: e - 1)

    test_data = DataFrame(
        IMDB(str(TORCH_DATA_STORAGE_PATH), split="test"),
        columns=["sentiment", "review"],
    )
    test_data["sentiment"] = test_data["sentiment"].apply(lambda e: e - 1)

    all_data = concat([train_data, test_data], ignore_index=True)
    return all_data.sample(all_data.shape[0], random_state=42, ignore_index=True)


class FilmReviewSequences(Dataset):
    """IMDB film reviews training generative models."""

    def __init__(
        self,
        tokenized_reviews: list[list[int]],
        seq_len: int = 40,
        rnd_chunks: bool = False,
    ):
        self._tokenized_reviews = tokenized_reviews
        self._chunk_size = seq_len + 1
        self._rnd_chunks = rnd_chunks

    def __len__(self) -> int:
        return len(self._tokenized_reviews) - self._chunk_size

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        review = self._tokenized_reviews[idx]

        if self._rnd_chunks:
            chunk_start = randint(0, max(0, len(review) - self._chunk_size))
        else:
            chunk_start = 0
        chunk_end = chunk_start + self._chunk_size

        tokenized_chunk = review[chunk_start:chunk_end]
        return (tensor(tokenized_chunk[:-1]), tensor(tokenized_chunk[1:]))

    def __iter__(self) -> Iterable[tuple[Tensor, Tensor]]:
        for n in range(len(self)):
            yield self[n]


class SequenceDatasets(NamedTuple):
    """Container for all experiment data."""

    train_data: FilmReviewSequences
    test_data: FilmReviewSequences
    val_data: FilmReviewSequences
    tokenizer: IMDBTokenizer


def make_sequence_datasets(
    train_test_split: float = 0.1,
    train_val_split: float = 0.05,
    tokenizer_type: Literal["IMDBTokenizer", "GPTTokenizer"] = "IMDBTokenizer",
    seq_len: int = 40,
    min_freq: int = 2,
) -> SequenceDatasets:
    """Make train, validation and test datasets."""
    reviews = get_data()["review"].tolist()

    if tokenizer_type == "GPTTokenizer":
        tokenizer = GPTTokenizer(reviews, min_freq)
    else:
        tokenizer = IMDBTokenizer(reviews, min_freq)

    reviews_tok = [tokenizer(review) for review in reviews]

    n_reviews = len(reviews_tok)
    n_train = math.floor(n_reviews * (1 - train_test_split))
    n_val = math.floor(n_train * train_val_split)

    train_ds = FilmReviewSequences(reviews_tok[n_val:n_train], seq_len, rnd_chunks=True)
    val_ds = FilmReviewSequences(reviews_tok[:n_val], seq_len, rnd_chunks=False)
    test_ds = FilmReviewSequences(reviews_tok[n_train:], seq_len, rnd_chunks=True)

    return SequenceDatasets(train_ds, test_ds, val_ds, tokenizer)


def pad_seq2seq_data(batch: list[tuple[int, int]]) -> tuple[Tensor, Tensor]:
    """Pad sequence2sequence data tuples."""
    x = [e[0] for e in batch]
    y = [e[1] for e in batch]
    x_padded = pad_sequence(x, batch_first=True)
    y_padded = pad_sequence(y, batch_first=True)
    return x_padded, y_padded


class _Tokenizer(ABC):
    """Abstract base class for text tokenizers."""

    def __call__(self, text: str) -> list[int]:
        return self.text2tokens(text)

    @abstractmethod
    def text2tokens(self, text: str) -> list[int]:
        pass

    @abstractmethod
    def tokens2text(self, token: list[int]) -> str:
        pass


class IMDBTokenizer(_Tokenizer):
    """Word to integer tokenization for use with any dataset or model."""

    def __init__(self, reviews: list[str], min_word_freq: int = 2):
        reviews_doc = " ".join(reviews)
        token_counter = Counter(self._tokenize(reviews_doc))
        token_freqs = sorted(token_counter.items(), key=lambda e: e[1], reverse=True)
        _vocab = vocab(OrderedDict(token_freqs), min_freq=min_word_freq)
        _vocab.insert_token("<pad>", PAD_TOKEN_IDX)
        _vocab.insert_token("<unk>", UNKOWN_TOKEN_IDX)
        _vocab.set_default_index(1)
        self.vocab = _vocab
        self.vocab_size = len(self.vocab)

    def text2tokens(self, text: str) -> list[int]:
        return self.vocab(self._tokenize(text))

    def tokens2text(self, tokens: list[int]) -> str:
        text = " ".join(self.vocab.lookup_tokens(tokens))
        text = re.sub(rf"\s{EOS_TOKEN}", ".", text)
        text = re.sub(rf"\s{QM_TOKEN}", "?", text)
        return text.strip()

    @staticmethod
    def _standardise(text: str) -> str:
        """Remove punctuation, HTML and make lower case."""
        text = text.lower().strip()
        text = re.sub(r"<[^>]*>", "", text)
        text = [char for char in text if char not in string.punctuation]
        return "".join(text)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Basic tokenizer."""
        text = (". ".join(sentence.strip() for sentence in text.split("."))).strip()
        text = re.sub(r"\.", f" {EOS_TOKEN}", text)
        text = re.sub(r"\?", f" {QM_TOKEN}", text)
        text = IMDBTokenizer._standardise(text)
        return text.split()


class GPTTokenizer(_Tokenizer):
    """Implementation of GPT's tokenizer based on Bytpe Pair Encoding (BPE)."""

    def __init__(self, reviews: list[str], min_freq: int = 2) -> None:
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = ByteLevelPre(add_prefix_space=False)
        tokenizer.post_processor = ByteLevelPost()
        tokenizer.decoder = ByteLevelDecoder()
        tokenizer.train_from_iterator(reviews, BpeTrainer(special_tokens=["[UNK]"]))

        self._tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()

    def text2tokens(self, text: str) -> list[int]:
        return self._tokenizer.encode(text).ids

    def tokens2text(self, tokens: list[int]) -> str:
        return self._tokenizer.decode(tokens)
