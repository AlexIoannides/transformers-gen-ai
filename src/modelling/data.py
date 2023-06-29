"""Downloading and pre-processing IMDB film review data."""
import re
import warnings
from abc import ABC, abstractmethod
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Iterable, List, Tuple

from pandas import concat, DataFrame
from torch import tensor, Tensor
from torch.nn.utils.rnn import pad_sequence
from torchtext.datasets import IMDB
from torchtext.vocab import vocab
from torch.utils.data import Dataset, DataLoader

EOS_DELIM = " endofsentence "
PAD_TOKEN_IDX = 0
UNKOWN_TOKEN_IDX = 1
TORCH_DATA_STORAGE_PATH = Path(".data")


def get_data() -> Tuple[DataFrame, DataFrame]:
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

    return train_data, test_data


class FilmReviewSentiment(Dataset):
    """IMDB film reviews and associated sentiment."""

    def __init__(self, split: str = "train"):
        if split not in ("train", "test", "all"):
            raise ValueError("split must be one of 'train', 'test' or 'all'.")
        train_data, test_data = get_data()
        match split:
            case "train":
                self._df = train_data
            case "test":
                self._df = test_data
            case "all":
                self._df = concat([train_data, test_data], ignore_index=True)

    def __len__(self) -> int:
        return self._df.shape[0]

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return (self._df["review"][idx], self._df["sentiment"][idx])

    def __iter__(self) -> Iterable[Tuple[str, int]]:
        for row in self._df.itertuples():
            yield row.review, row.sentiment


class FilmReviewSequences(Dataset):
    """IMDB film reviews training generative models."""

    def __init__(
            self, split: str = "train", seq_len: int = 40, min_freq: int = 1
    ):
        train_data, test_data = get_data()
        if split == "train":
            reviews = train_data["review"]
        elif split == "test":
            reviews = test_data["review"]
        elif split == "all":
            all_data = concat([train_data, test_data], ignore_index=True)
            reviews = all_data["review"]
        else:
            raise ValueError("split must be one of 'train' or 'test'.")
        tokenizer = IMDBTokenizer(min_freq)
        self._tokenised_reviews = [tokenizer(review) for review in reviews]
        self.vocab_size = tokenizer.vocab_size
        self._chunk_size = seq_len + 1

    def __len__(self) -> int:
        return len(self._tokenised_reviews) - self._chunk_size

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        tokenized_chunk = self._tokenised_reviews[idx][:self._chunk_size]
        return (tensor(tokenized_chunk[:-1]), tensor(tokenized_chunk[1:]))

    def __iter__(self) -> Iterable[Tuple[Tensor, Tensor]]:
        for n in range(len(self)):
            yield self[n]


def pad_seq2seq_data(batch: List[Tuple[int, int]]) -> Tuple[Tensor, Tensor]:
    """Pad sequence2sequence data tuples."""
    x = [e[0] for e in batch]
    y = [e[1] for e in batch]
    x_padded = pad_sequence(x, batch_first=True)
    y_padded = pad_sequence(y, batch_first=True)
    return x_padded, y_padded


class BasePreprocessor:
    """Basic pre-processor for use as a collate_fn for DataLoaders."""

    def __init__(self):
        self._tokenizer = IMDBTokenizer()

    def __call__(self, batch: List[Tuple[str, int]]) -> Tuple[Tensor, Tensor, Tensor]:
        y = [tensor(e) for _, e in batch]
        x = [tensor(self._tokenizer(review)) for review, _ in batch]
        x_padded = pad_sequence(x, batch_first=True)
        return x_padded, y


class _Tokenizer(ABC):
    """Abstract base class for text tokenizers."""

    def __call__(self, text: str) -> List[int]:
        return self.text2tokens(text)

    @abstractmethod
    def text2tokens(self, text: str) -> List[int]:
        pass

    @abstractmethod
    def tokens2text(self, token: List[int]) -> str:
        pass


class IMDBTokenizer(_Tokenizer):
    """Word to integer tokenisation for use with any dataset or model."""

    def __init__(self, min_freq: int = 2):
        train_and_test = concat(get_data(), ignore_index=True)
        reviews = " ".join(train_and_test["review"].tolist())

        token_counter = Counter(self._tokenize(reviews))
        token_freqs = sorted(token_counter.items(), key=lambda e: e[1], reverse=True)
        _vocab = vocab(OrderedDict(token_freqs), min_freq=min_freq)
        _vocab.insert_token("<pad>", PAD_TOKEN_IDX)
        _vocab.insert_token("<unk>", UNKOWN_TOKEN_IDX)
        _vocab.set_default_index(1)
        self.vocab = _vocab
        self.vocab_size = len(self.vocab)

    def text2tokens(self, text: str) -> List[int]:
        return self.vocab(self._tokenize(text))

    def tokens2text(self, tokens: List[int]) -> str:
        return self.vocab.lookup_tokens(tokens)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Basic tokenizer that can strip HTML."""
        text = re.sub(r"[\.\?](\s|$)", EOS_DELIM, text)
        text = re.sub(r"<[^>]*>", "", text)
        emoticons = re.findall(r"(?::|;|=)(?:-)?(?:\)|\(|D|P)", text.lower())
        text = re.sub(r"[\W]+", " ", text.lower()) + " ".join(emoticons).replace(
            "-", ""
        )
        tokenized = text.split()
        return tokenized


if __name__ == "__main__":
    # data pipeline: generative model training data
    train_data = FilmReviewSequences()
    train_data_loader = DataLoader(train_data, batch_size=4)
    for batch in train_data_loader:
        print(batch)
        break

    # data pipeline: sentiment classification data pipeline
    train_data = FilmReviewSentiment("train")
    preproc = BasePreprocessor()
    train_data_loader = DataLoader(train_data, batch_size=4, collate_fn=preproc)
    for batch in train_data_loader:
        print(batch)
        break
