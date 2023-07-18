"""Helper functions."""
import re
from datetime import datetime
from pathlib import Path
from textwrap import wrap
from typing import Any, Dict, Literal, NamedTuple

from pandas import DataFrame
from seaborn import lineplot
from torch import Tensor, argmax, device, load, save, topk
from torch.backends import mps
from torch.distributions import Categorical
from torch.nn import Module

TORCH_MODEL_STORAGE_PATH = Path(".models")


class ModelCheckpoint(NamedTuple):
    "Model checkpoint data."
    epoch: int
    train_loss: float
    val_loss: float
    state_dict: Dict[str, Any]


def get_best_device() -> device:
    """Return the best device available on the machine."""
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


def _sample_decoding(logits: Tensor, temperature: float = 1.0) -> Tensor:
    """Generate next token using sample decoding strategy."""
    return Categorical(logits=logits.squeeze() / temperature).sample()


def _top_k_decoding(logits: Tensor, temperature: float = 1.0, k: int = 3) -> Tensor:
    """Generate next token using top-k decoding strategy."""
    token_probs = Categorical(logits=logits.squeeze() / temperature).probs
    top_k_tokens = topk(token_probs, k=k)
    sampled_token = Categorical(probs=top_k_tokens.values).sample()
    return top_k_tokens.indices[sampled_token]


def _greedy_decoding(logits: Tensor, temperature: float = 1.0) -> Tensor:
    """Generate next token using greedy decoding strategy."""
    token_probs = Categorical(logits=logits.squeeze() / temperature).probs
    return argmax(token_probs)


def decode(
    token_logits: Tensor,
    strategy: Literal["greedy", "sample", "topk"] = "greedy",
    temperature: float = 1.0,
    *,
    k: int = 5,
) -> Tensor:
    """Decode generative model output using the specified strategy."""
    match strategy:
        case "greedy":
            return _greedy_decoding(token_logits, temperature)
        case "topk":
            return _top_k_decoding(token_logits, temperature, k)
        case "sample":
            return _sample_decoding(token_logits, temperature)


def _capitalise_sentences(text: str, sentence_delimiter: str = ". ") -> str:
    """Capitalise the first letter of sentences in text passage."""
    sentences = text.split(sentence_delimiter)
    sentences = [sentence[:1].upper() + sentence[1:] for sentence in sentences]
    return sentence_delimiter.join(sentences)


def format_generated_words(text: str, prompt: str) -> str:
    """Format list of words into a readable paragraph."""
    text = re.sub(r" i ", " I ", text)
    text = _capitalise_sentences(text, sentence_delimiter=". ")
    text = text if text[0] == "I" else text[:1].lower() + text[1:]
    text = "==> " + prompt.upper().strip() + " " + text.strip() + "..."
    return "\n".join([line for line in wrap(text, width=89)])
