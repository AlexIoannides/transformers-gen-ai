"""Helper functions."""
from datetime import datetime
from pathlib import Path

from torch import device, load, save
from torch.backends import mps
from torch.nn import Module

TORCH_MODEL_STORAGE_PATH = Path(".models")


def get_device() -> device:
    """Run on CPUs or GPUs."""
    if mps.is_available():
        return device("mps")
    else:
        device("cpu")


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


def load_model(name: str) -> Module:
    """Load model with best loss."""
    if not TORCH_MODEL_STORAGE_PATH.exists():
        TORCH_MODEL_STORAGE_PATH.mkdir()
    model_dir = TORCH_MODEL_STORAGE_PATH / name
    stored_models = [
        (file_path, str(file_path).split("loss=")[1])
        for file_path in model_dir.glob("*.pt")
    ]
    best_model = sorted(stored_models, key=lambda e: e[1])[0][0]
    print(f"loading {best_model}")
    model = load(best_model)
    return model
