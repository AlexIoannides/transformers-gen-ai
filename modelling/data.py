"""Downloading and pre-processing IMDB film review data."""
import re
import warnings
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Tuple

from pandas import concat, DataFrame
from torch import tensor
from torchtext.datasets import IMDB
from torch.utils.data import Dataset

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

    def __getitem__(self, idx: int) -> Tuple[int, str]:
        return (self._df["sentiment"][idx], self._df["review"][idx])


class BasePreProcessor:
    """Basic pre-processor for use as a collate_fn for DataLoaders."""

    def __init__(self, train_data: Dataset):
        pass

    def __call__(self, instance: Tuple[int, str]) -> Tuple[tensor, tensor, tensor]:
        pass

    @staticmethod
    def _tokenizer(text: str) -> str:
        text = re.sub(r"<[^>]*>", "", text)
        emoticons = re.findall(
            r"(?::|;|=)(?:-)?(?:\)|\(|D|P)", text.lower()
        )
        text = re.sub(r"[\W]+", " ", text.lower()) +\
            " ".join(emoticons).replace("-", "")
        tokenized = text.split()
        return tokenized
