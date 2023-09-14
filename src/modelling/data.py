"""Downloading and pre-processing IMDB film review data."""
from __future__ import annotations

import json
import math
import re
import warnings
from abc import ABC, abstractmethod
from collections import Counter, OrderedDict
from itertools import pairwise
from pathlib import Path
from typing import Iterable, Literal, NamedTuple

from pandas import DataFrame, concat
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPre
from tokenizers.processors import ByteLevel as ByteLevelPost
from tokenizers.trainers import BpeTrainer
from torch import Tensor, float32, tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset
from torchtext.datasets import IMDB
from torchtext.vocab import vocab
from unidecode import unidecode

EOS_TOKEN = "endofsentence"
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


class FilmReviewSequences(IterableDataset):
    """IMDB film reviews for training generative models."""

    def __init__(
        self,
        tokenized_reviews: list[list[int]],
        max_seq_len: int = 40,
        min_seq_length: int = 20,
        chunk_eos_token: int | None = None,
        chunk_overlap: bool = True,
        tag: str = "data",
    ):
        self._chunk_size = max_seq_len + 1
        self._data_file_path = TORCH_DATA_STORAGE_PATH / f"imdb_sequences_{tag}.json"

        with open(self._data_file_path, mode="w", ) as file:
            if chunk_eos_token:
                for tok_seq in tokenized_reviews:
                    chunks_itr = self.make_chunks(
                        tok_seq,
                        chunk_eos_token,
                        min_seq_length,
                        max_seq_len,
                        chunk_overlap
                    )
                    for chunk in chunks_itr:
                        file.write(json.dumps(chunk) + "\n")
            else:
                for tok_seq in tokenized_reviews:
                    file.write(json.dumps(tok_seq) + "\n")

    def __len__(self) -> int:
        with open(self._data_file_path) as file:
            num_chunks = sum(1 for line in file)
        return num_chunks

    def __iter__(self) -> Iterable[tuple[Tensor, Tensor]]:
        with open(self._data_file_path) as file:
            for line in file:
                review = json.loads(line)
                tokenized_chunk = review[:self._chunk_size]
                yield (tensor(tokenized_chunk[:-1]), tensor(tokenized_chunk[1:]))

    @staticmethod
    def make_chunks(
        tokenized_text: list[int],
        eos_token: int,
        min_chunk_size: int = 20,
        max_chunk_size: int = 40,
        overlapping: bool = True
    ) -> Iterable[list[int]]:
        """Split tokenised text into chunks of complete sentences."""
        sentence_start_indices = (
            [0] + [i+1 for i, token in enumerate(tokenized_text) if token == eos_token]
        )
        tokenized_sentences = [
            tokenized_text[i:j] for i, j in pairwise(sentence_start_indices)
        ]
        if overlapping:
            for i in range(len(tokenized_sentences)):
                chunk: list[int] = []
                for tok_sentence in tokenized_sentences[i:]:
                    if len(chunk + tok_sentence) <= max_chunk_size:
                        chunk += tok_sentence
                    else:
                        break
                if len(chunk) >= min_chunk_size:
                    yield chunk
        else:
            sentence_idx = 0
            while sentence_idx < len(tokenized_sentences):
                chunk: list[int] = []
                for tok_sentence in tokenized_sentences[sentence_idx:]:
                    if len(tok_sentence) > max_chunk_size:
                        sentence_idx += 1
                        break
                    if len(chunk + tok_sentence) <= max_chunk_size:
                        chunk += tok_sentence
                        sentence_idx += 1
                    else:
                        break
                yield chunk


class SequenceDatasets(NamedTuple):
    """Container for all sequence data."""

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
    eos_tok = tokenizer(".")[0]

    n_reviews = len(reviews_tok)
    n_train = math.floor(n_reviews * (1 - train_test_split))
    n_val = math.floor(n_train * train_val_split)

    train_ds = FilmReviewSequences(
        reviews_tok[n_val:n_train], seq_len, chunk_eos_token=eos_tok, tag="train"
    )
    val_ds = FilmReviewSequences(
        reviews_tok[:n_val], seq_len, chunk_eos_token=eos_tok, tag="validation"
    )
    test_ds = FilmReviewSequences(
        reviews_tok[n_train:], seq_len, chunk_eos_token=eos_tok, tag="test"
    )

    return SequenceDatasets(train_ds, test_ds, val_ds, tokenizer)


class FilmReviewSentiment(IterableDataset):
    """IMDB film reviews and associated sentiment."""

    def __init__(
        self,
        tokenized_reviews: list[list[int]],
        review_sentiment: list[int],
        seq_len: int = 40,
        tag: str = "data",
    ):
        if len(tokenized_reviews) != len(review_sentiment):
            raise ValueError("len(tokenized_reviews) != len(review_sentiment)")

        self._chunk_size = seq_len
        self._data_file_path = TORCH_DATA_STORAGE_PATH / f"imdb_sentiment_{tag}.json"

        with open(self._data_file_path, mode="w", ) as file:
            for tok_seq, sentiment in zip(tokenized_reviews, review_sentiment):
                row = {"tok_seq": tok_seq, "sentiment": sentiment}
                file.write(json.dumps(row) + "\n")

    def __len__(self) -> int:
        with open(self._data_file_path) as file:
            num_rows = sum(1 for line in file)
        return num_rows

    def __iter__(self) -> Iterable[tuple[Tensor, Tensor]]:
        with open(self._data_file_path) as file:
            for line in file:
                row = json.loads(line)
                tokenized_review = row["tok_seq"]
                sentiment = row["sentiment"]
                tokenized_chunk = tokenized_review[:self._chunk_size]
                yield (
                    tensor(tokenized_chunk),
                    tensor([sentiment], dtype=float32),
                )


class SentimentDatasets(NamedTuple):
    """Container for all sentiment classification data."""

    train_data: FilmReviewSentiment
    test_data: FilmReviewSentiment
    val_data: FilmReviewSentiment
    tokenizer: IMDBTokenizer


def make_sentiment_datasets(
    train_test_split: float = 0.1,
    train_val_split: float = 0.05,
    tokenizer_type: Literal["IMDBTokenizer", "GPTTokenizer"] = "IMDBTokenizer",
    seq_len: int = 40,
    min_freq: int = 2,
) -> SentimentDatasets:
    """Make train, validation and test datasets."""
    data = get_data()
    reviews = data["review"].tolist()
    sentiment = data["sentiment"].tolist()

    if tokenizer_type == "GPTTokenizer":
        tokenizer = GPTTokenizer(reviews, min_freq)
    else:
        tokenizer = IMDBTokenizer(reviews, min_freq)

    reviews_tok = [tokenizer(review) for review in reviews]

    n_reviews = len(reviews_tok)
    n_train = math.floor(n_reviews * (1 - train_test_split))
    n_val = math.floor(n_train * train_val_split)

    train_ds = FilmReviewSentiment(
        reviews_tok[n_val:n_train], sentiment[n_val:n_train], seq_len, "train"
    )
    val_ds = FilmReviewSentiment(
        reviews_tok[:n_val], sentiment[:n_val], seq_len, "validation"
    )
    test_ds = FilmReviewSentiment(
        reviews_tok[n_train:], sentiment[n_train:], seq_len, "test"
    )
    return SentimentDatasets(train_ds, test_ds, val_ds, tokenizer)


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

    def __init__(self, reviews: list[str], min_word_freq: int = 1):
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
        return text.strip()

    @staticmethod
    def _standardise(text: str) -> str:
        """Remove punctuation, HTML and make lower case."""
        text = text.lower().strip()
        text = unidecode(text)
        text = re.sub(r"<[^>]*>", "", text)
        text = re.sub(r"mr.", "mr", text)
        text = re.sub(r"mrs.", "mrs", text)
        text = re.sub(r"ms.", "ms", text)
        text = re.sub(r"(\!|\?)", ".", text)
        text = re.sub(r"-", " ", text)
        text = "".join(
            char for char in text if char not in "\"#$%&'()*+,/:;<=>@[\\]^_`{|}~"
        )
        text = re.sub(r"\.+", ".", text)
        return text

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Basic tokenizer."""
        text = IMDBTokenizer._standardise(text)
        text = (". ".join(sentence.strip() for sentence in text.split("."))).strip()
        text = re.sub(r"\.", f" {EOS_TOKEN} ", text)
        text = re.sub(r"\s+", " ", text)
        return text.split()


class GPTTokenizer(_Tokenizer):
    """Implementation of GPT's tokenizer based on Byte Pair Encoding (BPE)."""

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
