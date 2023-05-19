"""Generate text from language models."""
from torch import device, manual_seed, tensor
from torch.distributions import Categorical
from torch.nn import Module

from .data import _Tokenizer
from .lstm import NextWordPrediction


def generate(
    model: Module,
    prompt: str,
    tokenizer: _Tokenizer,
    output_length: int = 40,
    temperature: float = 1.0,
    random_seed: int = 42,
    device_: device = device("cpu"),
) -> str:
    """Generate new text conditional on a text prompt."""
    manual_seed(random_seed)

    model.to(device_)
    model.eval()
    hidden, cell = model.initialise(1, device_)

    prompt_tokens = tensor(tokenizer(prompt), device=device_).reshape(-1, 1)
    for token in prompt_tokens[:-1]:
        _, hidden, cell = model(token, hidden, cell)

    new_token_sequence = [prompt_tokens[-1]]
    for _ in range(output_length):
        token_logits, hidden, cell = model(new_token_sequence[-1], hidden, cell)
        token_pred = Categorical(logits=temperature * token_logits).sample()
        new_token_sequence += [token_pred]

    new_text = prompt + " " + " ".join(tokenizer.tokens2text(new_token_sequence[1:]))
    return new_text.replace(" endofsentence ", ". ")


if __name__ == "__main__":
    from .data import IMDBTokenizer
    from .utils import load_model

    MODEL_NAME = "lstm_next_word_gen"
    PROMPT = "This movie was a total waste of time"
    TEMPERATURE = 1.0

    tokenizer = IMDBTokenizer()
    model: NextWordPrediction = load_model(MODEL_NAME)
    new_text = generate(model, PROMPT, tokenizer, temperature=TEMPERATURE)
    print(new_text)
