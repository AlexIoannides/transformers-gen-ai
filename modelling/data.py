"""Downloading and pre-processing IMDB film review data."""
import re
import warnings
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Iterable, List, Tuple

from pandas import concat, DataFrame
from torch import tensor, Tensor
from torch.nn.utils.rnn import pad_sequence
from torchtext.datasets import IMDB
from torchtext.vocab import vocab
from torch.utils.data import Dataset, DataLoader

TORCH_DATA_STORAGE_PATH = Path(".data")


def get_data() -> Tuple[DataFrame, DataFrame]:
    """Download raw data and convert to Pandas DataFrame."""
    if not TORCH_DATA_STORAGE_PATH.exists():
        warnings.warn("Downloading IMDB data - this may take a minute or two.")
    train_data = DataFrame(
        IMDB(str(TORCH_DATA_STORAGE_PATH), split="train"),
        columns=["sentiment", "review"]
    )
    train_data["sentiment"] = train_data["sentiment"].apply(lambda e: e - 1)

    test_data = DataFrame(
        IMDB(str(TORCH_DATA_STORAGE_PATH), split="test"),
        columns=["sentiment", "review"]
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

    def __init__(self, split: str = "train", sequence_length: int = 40):
        train_data, test_data = get_data()
        if split == "train":
            reviews = " ".join(train_data["review"].tolist())
        elif split == "test":
            reviews = " ".join(test_data["review"].tolist())
        elif split == "all":
            all_data = concat([train_data, test_data], ignore_index=True)
            reviews = " ".join(all_data["review"].tolist())
        else:
            raise ValueError("split must be one of 'train' or 'test'.")
        tokenizer = IMDBTokenizer()
        self._tokenised_reviews = tokenizer(reviews)
        self.vocab_size = tokenizer.vocab_size
        self._chunk_size = sequence_length + 1

    def __len__(self) -> int:
        return len(self._tokenised_reviews) - self._chunk_size

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        tokenized_chunk = self._tokenised_reviews[idx:(idx+self._chunk_size)]
        return (Tensor(tokenized_chunk[:-1]), tensor(tokenized_chunk[1:]))

    def __iter__(self) -> Iterable[Tuple[Tensor, Tensor]]:
        for n in range(len(self)):
            yield self[n]


class BasePreprocessor:
    """Basic pre-processor for use as a collate_fn for DataLoaders."""

    def __init__(self):
        self._tokenizer = IMDBTokenizer()

    def __call__(self, batch: List[Tuple[str, int]]) -> Tuple[Tensor, Tensor, Tensor]:
        y = [tensor(e) for _, e in batch]
        x = [tensor(self._tokenizer(review)) for review, _ in batch]
        x_padded = pad_sequence(x, batch_first=True)
        x_lens = [e.shape[0] for e in x]
        return x_padded, y, x_lens


class IMDBTokenizer:
    """Word to integer tokenisation for use with any dataset or model."""

    def __init__(self):
        train_and_test = concat(get_data(), ignore_index=True)
        reviews = " ".join(train_and_test["review"].tolist())

        token_counter = Counter(self._tokenize(reviews))
        token_freqs = sorted(token_counter.items(), key=lambda e: e[1], reverse=True)
        _vocab = vocab(OrderedDict(token_freqs))
        _vocab.insert_token("<pad>", 0)
        _vocab.insert_token("<unk>", 1)
        _vocab.set_default_index(1)
        self.vocab = _vocab
        self.vocab_size = len(self.vocab)

    def __call__(self, text: str) -> List[int]:
        return self.vocab(self._tokenize(text))

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Basic tokenizer that can strip HTML."""
        text = re.sub(r"[\.\?](\s|$)", " endofsentence ", text)
        text = re.sub(r"<[^>]*>", "", text)
        emoticons = re.findall(
            r"(?::|;|=)(?:-)?(?:\)|\(|D|P)", text.lower()
        )
        text = re.sub(r"[\W]+", " ", text.lower()) +\
            " ".join(emoticons).replace("-", "")
        tokenized = text.split()
        return tokenized


if __name__ == "__main__":
    # data pipeline: generative model training data
    train_data = FilmReviewSequences()
    train_data_loader = DataLoader(train_data, batch_size=2)
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
