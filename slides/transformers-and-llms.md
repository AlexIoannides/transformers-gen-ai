---
title: Transformers & LLMs
author: Alex Ioannides
date: October 12th, 2023
---

"*Don't let the little fuckers generation gap you.*"

\- William Gibson, Neuromancer

## What I'm intending to talk about

::: incremental

1. The problem(s) we're trying to solve.
2. How to compute multi-head attention.
3. Transformers: encoders, decoders, and all that.
4. How I developed a generative LLM.
5. Exciting things to try with this LLM.
6. Conclusions (heavily opinionated).

:::

---

<!-- A nice schematic of words to tokens, to embeddings, to context-aware embeddings. -->

## How to compute multi-head attention

<!-- step-through notebook code -->

## Transformers: encoders, decoders, and all that

<!--
- how to build an encoder using multi-head attention
- encoders vs. decoders
- generative LLMs vs. non-generative LLMs
-->

## How I developed a generative LLM

<!--
- the dataset
- preparing the data - tokenisation
- preparing the data - PyTorch DataLoaders
- an aside on GPUs
- benchmark - training an RNN.
- using a generative model to generate text, given a prompt. 
- training a transformer-decoder.
-->

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


## Appendix

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

```

---

```python
class FilmReviewSequences(IterableDataset):
    """IMDB film reviews for training generative models."""

    ...

    def __len__(self) -> int:
        with open(self._data_file_path) as file:
            num_chunks = sum(1 for line in file)
        return num_chunks

    def __iter__(self) -> Iterable[tuple[Tensor, Tensor]]:
        with open(self._data_file_path) as file:
            for line in file:
                tokenized_chunk = json.loads(line)
                yield (tensor(tokenized_chunk[:-1]), tensor(tokenized_chunk[1:]))
```

### Section Three Point Two
