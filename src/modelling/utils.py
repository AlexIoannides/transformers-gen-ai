"""Helper functions."""
from datetime import datetime
from pathlib import Path
from typing import Dict

from pandas import DataFrame
from seaborn import lineplot
from torch import device, load, save
from torch.backends import mps
from torch.nn import Module

TORCH_MODEL_STORAGE_PATH = Path(".models")


def get_device() -> device:
    """Run on CPUs or GPUs."""
    if mps.is_available():
        return device("mps")
    else:
        return device("cpu")


def save_model(model: Module, name: str, loss: float) -> None:
    """Save models to disk."""
    if not TORCH_MODEL_STORAGE_PATH.exists():
        TORCH_MODEL_STORAGE_PATH.mkdir()
    model_dir = TORCH_MODEL_STORAGE_PATH / name
    if not model_dir.exists():
        model_dir.mkdir()
    timestamp = datetime.now().isoformat(timespec="seconds")
    loss_str = f"{loss:.4f}".replace(".", "_") if loss else ""
    filename = f"trained@{timestamp};loss={loss_str}.pt"
    model.to(device("cpu"))
    save(model, model_dir / filename)


def load_model(name: str, latest: bool = False) -> Module:
    """Load model with best loss."""
    if not TORCH_MODEL_STORAGE_PATH.exists():
        TORCH_MODEL_STORAGE_PATH.mkdir()
    model_dir = TORCH_MODEL_STORAGE_PATH / name

    if not latest:
        stored_models = [
            (file_path, str(file_path).split("loss=")[1])
            for file_path in model_dir.glob("*.pt")
        ]
        model = sorted(stored_models, key=lambda e: e[1])[0][0]
    else:
        stored_models = [
            (file_path, str(file_path).split("trained@")[1][:19])
            for file_path in model_dir.glob("*.pt")
        ]
        model = sorted(stored_models, key=lambda e: datetime.fromisoformat(e[1]))[-1][0]

    print(f"loading {model}")
    model = load(model)
    return model


def capitalise_sentences(text: str, sentence_delimiter: str = ". ") -> str:
    """Capitalise the first letter of sentences in text passage."""
    sentences = text.split(sentence_delimiter)  
    sentences = [sentence.capitalize() for sentence in sentences]
    return sentence_delimiter.join(sentences)


def plot_train_losses(
    train_losses: Dict[int, float], val_losses: Dict[int, float]
) -> None:
    """Plot training and validation losses per-epoch."""
    train_rows = [(epoch, loss, "train") for epoch, loss in train_losses.items()]
    val_rows = [(epoch, loss, "val") for epoch, loss in val_losses.items()]
    df = DataFrame(train_rows + val_rows, columns=["epoch", "loss", "dataset"])
    lineplot(df, x="epoch", y="loss", hue="dataset")


def _early_stop(train_loss: Dict[int, float], epoch_window: int = 3) -> bool:
    """Flag when training no longer improves loss."""
    if len(train_loss) < epoch_window + 1:
        return False
    else:
        losses = list(train_loss.values())
        current_loss = losses[-1]
        avg_window_loss = sum(losses[-(epoch_window + 1) : -1]) / epoch_window
        if current_loss >= avg_window_loss:
            return True
        else:
            return False
