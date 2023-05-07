"""Downloading and pre-processing IMDB film review data."""
import re
import warnings
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Iterable, List, Tuple

from pandas import concat, DataFrame
from torch import tensor
from torch.nn.utils.rnn import pad_sequence
from torchtext.datasets import IMDB
from torchtext.vocab import vocab
from torch.utils.data import Dataset, DataLoader

TORCH_DATA_CACHE_PATH = Path(".data")


def get_data() -> Tuple[DataFrame, DataFrame]:
    """Download raw data and convert to Pandas DataFrame."""
    if not TORCH_DATA_CACHE_PATH.exists():
        warnings.warn("Downloading IMDB data - this may take a minute or two.")
    train_data = DataFrame(
        IMDB(str(TORCH_DATA_CACHE_PATH), split="train"),
        columns=["sentiment", "review"]
    )
    train_data["sentiment"] = train_data["sentiment"].apply(lambda e: e - 1)

    test_data = DataFrame(
        IMDB(str(TORCH_DATA_CACHE_PATH), split="test"),
        columns=["sentiment", "review"]
    )
    test_data["sentiment"] = test_data["sentiment"].apply(lambda e: e - 1)

    return train_data, test_data


class FilmReviewDataset(Dataset):
    """IMDB film review dataset."""

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


class BasePreprocessor:
    """Basic pre-processor for use as a collate_fn for DataLoaders."""

    def __init__(self, train_data: Dataset):
        token_counter = Counter()
        for x, _ in train_data:
            token_counter.update(self._tokenize(x))
        token_freqs = sorted(token_counter.items(), key=lambda e: e[1], reverse=True)
        _vocab = vocab(OrderedDict(token_freqs))
        _vocab.insert_token("<pad>", 0)
        _vocab.insert_token("<unk>", 1)
        _vocab.set_default_index(1)
        self.vocab = _vocab
        self.vocab_size = len(self.vocab)

    def __call__(self, batch: List[Tuple[str, int]]) -> Tuple[tensor, tensor, tensor]:
        y = [tensor(e) for _, e in batch]
        tokens = [self._tokenize(review) for review, _ in batch]
        x = [tensor(self.vocab(token)) for token in tokens]
        x_padded = pad_sequence(x, batch_first=True)
        x_lens = [e.shape[0] for e in x]
        return x_padded, y, x_lens

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = re.sub(r"<[^>]*>", "", text)
        emoticons = re.findall(
            r"(?::|;|=)(?:-)?(?:\)|\(|D|P)", text.lower()
        )
        text = re.sub(r"[\W]+", " ", text.lower()) +\
            " ".join(emoticons).replace("-", "")
        tokenized = text.split()
        return tokenized


if __name__ == "__main__":
    train_data = FilmReviewDataset("train")
    preproc = BasePreprocessor(train_data)
    train_data_loader = DataLoader(train_data, batch_size=4, collate_fn=preproc)
