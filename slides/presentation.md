---
title: Transformers & LLMs
author: Alex Ioannides
date: October 12th, 2023
---

## The story of one man's mission not to get left behind in the dust


"*Don't let the little fuckers generation gap you.*"

\- William Gibson, Neuromancer

## What I'm intending to talk about

::: incremental

1. The problem we're trying to solve.
2. How to compute multi-head attention.
3. Transformers: encoders, decoders, and all that.
4. How I developed a generative LLM.
5. Exciting things to try with this LLM.
6. Conclusions (heavily opinionated).

:::

## Before we get started

This presentation is based on the codebase at [github.com/AlexIoannides/transformer-gen-ai](https://github.com/AlexIoannides/transformers-gen-ai).

I'm not going to assume you've worked through it, but if you have and there are questions you want to ask, then please do 🙂

## The problem we're trying to solve

---

![](images/paradigm.png)

___

The role that attention plays in all this...

![](images/attention.png)

## How to compute multi-head attention

### Let's start with self-attention

---

```python
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE = 20000
EMBEDDING_DIM = 32


# Let's assume some tokenizer has tokenised our sentence.
tokenized_sentence = torch.randint(0, vocab_size, 8)
n_tokens = len(tokenized_sentence)

# We then map from token to embeddings.
embedding_layer = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
embedded_tokens = embedding_layer(tokenized_sentence)

# And compute self-attention weights.
attn_weights = torch.empty(n_tokens, n_tokens)
for i in range(n_tokens):
    for j in range(n_tokens):
        attn_weights[i, j] = torch.dot(embedded_tokens[i], embedded_tokens[j])

# Normalise the weights, so that they sun to 1.0.
attn_weights_norm = F.softmax(attn_weights / math.sqrt(EMBEDDING_DIM), dim=1)

# And finally use the weights to compute context-aware embeddings.
context_weighted_embeddings = torch.matmul(attn_weights_norm, embedded_tokens)
```

---

More formally...

$$
\vec{x_{i}} \to \vec{z_{i}} = \sum_{j=1}^{N}{a_{ij} \times \vec{x_{j}}}
$$

i.e., we build new embeddings using semantic similarity to selectively pool information from the original embeddings. Note, there aren't any attention-specific parameters that need to be learnt, only the original embeddings. We'll come back to this later.

### Time and causality

In the current setup, the attention-modulated embedding at time $t_1$ is a function of embeddings for tokens that come after. If we want to develop generative models, then this isn't appropriate. A common solution is to use **causal masking**.

___

![](images/causal_mask.png)

```python
causal_mask = torch.triu(torch.full((n_tokens, n_tokens), True), diagonal=1)
causal_attn_weights = attn_weights.masked_fill(causal_mask, -1e10)
causal_attn_weights_norm = F.softmax(causal_attn_weights / math.sqrt(EMBEDDING_DIM), dim=1)
```

---

### Learning how to attend

```python
# Define three linear transformations.
u_q = torch.rand(n_tokens, n_tokens)
u_k = torch.rand(n_tokens, n_tokens)
u_v = torch.rand(n_tokens, n_tokens)

# Use these to transform the embedded tokens.
q = torch.matmul(u_q, embedded_tokens)
k = torch.matmul(u_k, embedded_tokens)
v = torch.matmul(u_v, embedded_tokens)

# And then re-work the computation of the attention weights.
attn_weights_param = torch.empty(n_tokens, n_tokens)

for i in range(n_tokens):
    for j in range(n_tokens):
        attn_weights_param[i, j] = torch.dot(q[i], k[j])

attn_weights_param_norm = F.softmax(
    attn_weights_param / math.sqrt(EMBEDDING_DIM), dim=1
)
context_weighted_embeddings_param = torch.matmul(attn_weights_param_norm, v)
```

This is equivalent to passing `embedded_tokens` through three separate linear network layers and using the outputs within the self-attention mechanism.

### From single to multiple attention heads

![](images/multi_head_attention.png)

---

“*The 'Attention is all you need' paper was written at a time when the idea of factoring feature spaces into independent subspaces had been shown to provide great benefits for computer vision models... Multi-head attention is simply the application of the same idea to self-attention.*"

\- François Chollet (the author of Keras)

___

We have now arrived at

`torch.nn.MultiheadAttention`

## Transformers: encoders, decoders, and all that

___

How do we arrive at

`torch.nn.TransformerEncoderLayer`
`torch.nn.TransformerDecoderLayer`

?

---

Encoder-decoder for seq-to-seq translation

![](images/encoder_decoder.png){width=40%}

\- "*Attention is all you Need*", Vaswani et al. (2017)

___

"*... adding residual connections, adding normalization layers—all of these are standard architecture patterns that one would be wise to leverage in any complex model. Together, these bells and whistles form the Transformer encoder—one of two critical parts that make up the Transformer architecture*”

\- François Chollet (the author of Keras)

---

"*We stare into the void where our math fails us and try to write math papers anyway... We could then turn to the deepness itself and prove things about batch norm or dropout or whatever, but these just give us some nonpredictive post hoc justifications... deep learning seems to drive people completely insane.*"

\- [Ben Recht](https://argmin.substack.com/p/my-mathematical-mind) (Prof. of Computer Sciences, UC Berkley)

---

"*Deep learning is like riding a bicycle - it's not something you prove the existence of, it's just something you do.*"

\- Dr Alex Ioannides (ML Engineering Chapter Lead, Lloyd's Banking Group)

___

### When do we use encoders, decoders, or both?

::: incremental

- **Encoder**: pure embedding models.
- **Decoder**: generative models.
- **Encoder + Decoder**: sequence-to-sequence models.

:::

## How I developed a generative LLM

<!--
- the dataset
- preparing the data - tokenisation
- preparing the data - PyTorch DataLoaders
- an aside on GPUs
- benchmark - training an RNN.
- using a generative model to generate text, given a prompt. 
- training a transformer-decoder.
    - positional encoding
    - learning rate schedule
-->

### The data

50k film reviews and sentiment scores from IMDB.

```python
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, IterableDataset
from torchtext.datasets import IMDB
from torchtext.vocab import vocab

from modelling.data import (
    FilmReviewSequences,
    IMDBTokenizer,
    get_data,
    make_chunks,
    make_sequence_datasets,
    pad_seq2seq_data,
)


data = get_data()
data.head(10)

# sentiment	review
# 0	0	Forget what I said about Emeril. Rachael Ray i...
# 1	0	Former private eye-turned-security guard ditch...
# 2	0	Mann photographs the Alberta Rocky Mountains i...
# 3	0	Simply put: the movie is boring. Cliché upon c...
# 4	1	Now being a fan of sci fi, the trailer for thi...
# 5	1	In 'Hoot' Logan Lerman plays Roy Eberhardt, th...
# 6	0	This is the worst film I have ever seen.I was ...
# 7	1	I think that Toy Soldiers is an excellent movi...
# 8	0	I think Micheal Ironsides acting career must b...
# 9	0	This was a disgrace to the game FarCry i had m...
```

---

Example #4 in full:

*"Now being a fan of sci fi, the trailer for this film looked a bit too, how do i put it, hollywood. But after watching it i can gladly say it has impressed me greatly. Jude is a class actor and miss Leigh pulls it off better than she did in Delores Clairborne. It brings films like The Matrix, 12 Monkeys and The Cell into mind, which might not sound that appealing, but it truly is one of the best films i have seen."*

___

### Generating tokens

```python
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
    def _tokenize(text: str) -> list[str]:
        text = IMDBTokenizer._standardise(text)
        text = (". ".join(sentence.strip() for sentence in text.split("."))).strip()
        text = re.sub(r"\.", f" {EOS_TOKEN} ", text)
        text = re.sub(r"\s+", " ", text)
        return text.split()

    ...
```

---

```python
class IMDBTokenizer(_Tokenizer):
    """Word to integer tokenization for use with any dataset or model."""

    ...

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
```

---

`IMDBTokenizer` in action

```python
reviews = data["review"].tolist()
review = reviews[0]

tokenizer = IMDBTokenizer(reviews)
tokenized_review = tokenizer(review)
tokenised_review_decoded = tokenizer.tokens2text(tokenized_review[:10])

print(f"ORIGINAL TEXT: {review[:47]} ...")
print(f"TOKENS FROM TEXT: {', '.join(str(t) for t in tokenized_review[:10])} ...")
print(f"TEXT FROM TOKENS: {tokenised_review_decoded} ...")

# ORIGINAL TEXT: Forget what I said about Emeril. Rachael Ray is ...
# TOKENS FROM TEXT: 831, 49, 11, 300, 44, 37877, 3, 10505, 1363, 8 ...
# TEXT FROM TOKENS: forget what i said about emeril. rachael ray is ...
```

This is adequate for the current endeavor, but serious models use more sophisticated tokenisation algorithms, such as [Byte-Pair Encoding (BPE)](https://huggingface.co/learn/nlp-course/chapter6/5?fw=pt), which is one of the 'secret ingredients' of OpenAI's GPT models.

### Datasets and DataLoaders

We would benefit from standardized interface for delivering data to our models during training, which in this case requires two token sequences (offset by one as we are developing generative models). PyTorch provides such an interface:

`torch.utils.data.IterableDataset`

```python
tokenized_reviews = [tokenizer(review) for review in reviews]
dataset = FilmReviewSequences(tokenized_reviews)
x, y = next(iter(dataset))

print(f"x[:5]: {x[:5]}")
print(f"y[:5]: {y[:5]}")

# x[:5]: tensor([831,  49,  11, 300,  44])
# y[:5]: tensor([   49,    11,   300,    44, 37877])
```

### Chunking

Most reviews are too long to be used as one input sequence and need to be broken into chunks. I chose a strategy based on preserving sentence integrity to create overlapping chunks that fall within a maximum sequence length.

--- 

Example with maximum sequence length of 30 words:

```python
full_text = """I've seen things you people wouldn't believe. Attack ships on fire off
the shoulder of Orion. I watched C-beams glitter in the dark near the Tannhäuser Gate.
All those moments will be lost in time, like tears in rain."""

chunk_one = """I've seen things you people wouldn't believe. Attack ships on fire off
the shoulder of Orion. I watched C-beams glitter in the dark near the Tannhäuser Gate."""

chunk_three = """Attack ships on fire off the shoulder of Orion. I watched C-beams
glitter in the dark near the Tannhäuser Gate."""

chunk_four = """I watched C-beams glitter in the dark near the Tannhäuser Gate. All
those moments will be lost in time, like tears in rain."""
```

---

```python
class FilmReviewSequences(IterableDataset):
    """IMDB film reviews for training generative models."""

    def __init__(
        self,
        tokenized_reviews: Iterable[list[int]],
        max_seq_len: int = 40,
        min_seq_len: int = 20,
        chunk_eos_token: int | None = None,
        chunk_overlap: bool = True,
        tag: str = "data",
    ):
        self._data_file_path = TORCH_DATA_STORAGE_PATH / f"imdb_sequences_{tag}.json"

        with open(self._data_file_path, mode="w") as file:
            if chunk_eos_token:
                for tok_review in tokenized_reviews:
                    tok_chunks_itr = make_chunks(
                        tok_review,
                        chunk_eos_token,
                        max_seq_len,
                        min_seq_len,
                        chunk_overlap
                    )
                    for tok_chunk in tok_chunks_itr:
                        file.write(json.dumps(tok_chunk) + "\n")
            else:
                for tok_review in tokenized_reviews:
                    file.write(json.dumps(tok_review[:max_seq_len]) + "\n")

    ...
```

---

```python
class FilmReviewSequences(IterableDataset):
    """IMDB film reviews for training generative models."""

    ...

    def __iter__(self) -> Iterable[tuple[Tensor, Tensor]]:
        with open(self._data_file_path) as file:
            for line in file:
                tokenized_chunk = json.loads(line)
                yield (tensor(tokenized_chunk[:-1]), tensor(tokenized_chunk[1:]))

    def __len__(self) -> int:
        with open(self._data_file_path) as file:
            num_rows = sum(1 for line in file)
        return num_rows
```

Note, the entire dataset it not held in memory, but is loaded from disk on-demand in an attempt to optimize memory during training.

---

Use `DataLoader` to batch data and handle parallelism.

```python
def pad_seq2seq_data(batch: list[tuple[int, int]]) -> tuple[Tensor, Tensor]:
    """Pad sequence2sequence data tuples."""
    x = [e[0] for e in batch]
    y = [e[1] for e in batch]
    x_padded = pad_sequence(x, batch_first=True)
    y_padded = pad_sequence(y, batch_first=True)
    return x_padded, y_padded


data_loader = DataLoader(datasets.test_data, batch_size=10, collate_fn=pad_seq2seq_data)

data_batches = [batch for batch in data_loader]
x_batch, y_batch = data_batches[0]

print(f"x_batch_size = {x_batch.size()}")
print(f"y_batch_size = {y_batch.size()}")
# x_batch_size = torch.Size([10, 38])
# y_batch_size = torch.Size([10, 38])
```

### GPUs

Basic approach - use the best available device for a given model. Note that sometimes `mps` is slower than `cpu` (until Apple get their act together).

```python
from torch import device


def get_best_device(
        cuda_priority: Literal[1, 2, 3] = 1,
        mps_priority: Literal[1, 2, 3] = 2,
        cpu_priority: Literal[1, 2, 3] = 3,
    ) -> device:
    """Return the best device available on the machine."""
    device_priorities = sorted(
        (("cuda", cuda_priority), ("mps", mps_priority), ("cpu", cpu_priority)),
        key=lambda e: e[1]
    )
    for device_type, _ in device_priorities:
        if device_type == "cuda" and cuda.is_available():
            return device("cuda")
        elif device_type == "mps" and mps.is_available():
            return device("mps")
        elif device_type == "cpu":
            return device("cpu")
```

---

Models were trained using one of:

- Apple M1 Max
- AWS `p3.xlarge` EC2 instance with a single NVIDIA V100 Tensor Core GPU.

## Benchmarking with an RNN

TODO

## Generative transformer-decoder

TODO

## Exciting things to try with this LLM

<!--
- semantic search
- sentiment classification
-->

## Conclusions (heavily opinionated)

<!--
- We have achieved full AutoNLP.
- This is alchemy, not physics.
- Emergent capabilities is probably a fallacy.
- But it has been shown that deep learning can grok...
- ... but for now probably just VERY useful stochastic parrots.
-->
